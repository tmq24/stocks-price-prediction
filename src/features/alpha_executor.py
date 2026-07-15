"""
Safe sandboxed evaluation of LLM-generated alpha expressions.

All time-series operators are implemented as backward-looking rolling
operations over pandas Series.  The eval() namespace contains only the
explicitly whitelisted functions and variables - __builtins__ is empty.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from .technical import INDICATOR_COLS
from ..alpha.alpha_validator import validate_alpha_expression, AlphaValidationError


# ---------------------------------------------------------------------------
# Operator implementations (all backward-looking, no look-ahead)
# ---------------------------------------------------------------------------

def _ts_mean(series: pd.Series, d: int) -> pd.Series:
    return series.rolling(d, min_periods=1).mean()


def _ts_std(series: pd.Series, d: int) -> pd.Series:
    return series.rolling(d, min_periods=1).std().fillna(0.0)


def _ts_min(series: pd.Series, d: int) -> pd.Series:
    return series.rolling(d, min_periods=1).min()


def _ts_max(series: pd.Series, d: int) -> pd.Series:
    return series.rolling(d, min_periods=1).max()


def _ts_rank(series: pd.Series, d: int) -> pd.Series:
    return series.rolling(d, min_periods=1).rank(pct=True)


def _ts_argmax(series: pd.Series, d: int) -> pd.Series:
    def _argmax(x):
        return float(np.argmax(x.values))
    return series.rolling(d, min_periods=1).apply(_argmax, raw=False)


def _ts_argmin(series: pd.Series, d: int) -> pd.Series:
    def _argmin(x):
        return float(np.argmin(x.values))
    return series.rolling(d, min_periods=1).apply(_argmin, raw=False)


def _delay(series: pd.Series, d: int) -> pd.Series:
    return series.shift(d)


def _delta(series: pd.Series, d: int) -> pd.Series:
    return series.diff(d)


def _rank(series: pd.Series) -> pd.Series:
    """Expanding pct rank up to each time point - always backward-looking."""
    return series.expanding().rank(pct=True)


def _decay_linear(series: pd.Series, d: int) -> pd.Series:
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()

    def _apply(x):
        arr = x.values
        if len(arr) < d:
            w = np.arange(1, len(arr) + 1, dtype=float)
            w /= w.sum()
            return float(np.dot(arr, w))
        return float(np.dot(arr[-d:], weights))

    return series.rolling(d, min_periods=1).apply(_apply, raw=False)


def _correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=2).corr(y).fillna(0.0)


def _covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=2).cov(y).fillna(0.0)


def _scale(series: pd.Series) -> pd.Series:
    """Normalise by expanding L1 norm up to each time point - always backward-looking."""
    l1 = series.abs().expanding().sum().clip(lower=1e-10)
    return series / l1


def _sign(series: pd.Series) -> pd.Series:
    return series.apply(np.sign)


def _log(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: np.log(x) if x > 0 else np.nan)


def _if_else(cond, a, b) -> pd.Series:
    """Element-wise conditional. Accepts Series or scalar for a and b."""
    cond_vals = cond.values.astype(bool) if isinstance(cond, pd.Series) else np.asarray(cond, dtype=bool)
    a_vals = a.values if isinstance(a, pd.Series) else a
    b_vals = b.values if isinstance(b, pd.Series) else b
    idx = cond.index if isinstance(cond, pd.Series) else pd.RangeIndex(len(cond_vals))
    return pd.Series(np.where(cond_vals, a_vals, b_vals), index=idx)


# ---------------------------------------------------------------------------
# Namespace builder
# ---------------------------------------------------------------------------

# Full set of indicator columns exposed in the alpha namespace.
# Paper's 11 indicators plus legacy aliases used by previously-generated
# LLM expressions.
_INDICATOR_COL_SET = set(INDICATOR_COLS) | {
    'SMA_10', 'SMA_60',
    'EMA_12', 'EMA_26',
    'momentum_5', 'momentum_10',
    'MACD_diff', 'MACD_signal',
    'BB_upper', 'BB_lower', 'BB_middle', 'BB_pct_b', 'BB_width',
    'realized_vol_5d', 'realized_vol_20d',
    'ATR_14', 'volume_ratio',
    'log_return',
}


def build_alpha_namespace(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a restricted eval namespace mapping variable names to df Series
    and operator names to their implementations.

    OHLCV columns are shifted by 1 so that alpha expressions at row t only
    see data through close[t-1], consistent with the blanket shift applied to
    all indicator columns in compute_all_indicators().

    Technical indicator columns are already shifted in the df, so they are
    exposed directly (no additional shift).

    'vwap' is synthesised as (high + low + close) / 3 if not in df.
    'returns' maps to log_return.
    """
    ns: Dict[str, Any] = {}

    # OHLCV - shift by 1 for look-ahead prevention
    for col in ['open', 'high', 'low', 'close', 'volume']:
        ns[col] = df[col].shift(1) if col in df.columns else pd.Series(np.nan, index=df.index)

    if 'vwap' in df.columns:
        ns['vwap'] = df['vwap'].shift(1)
    else:
        _h = df['high'].shift(1) if 'high' in df.columns else pd.Series(0.0, index=df.index)
        _l = df['low'].shift(1) if 'low' in df.columns else pd.Series(0.0, index=df.index)
        _c = df['close'].shift(1) if 'close' in df.columns else pd.Series(0.0, index=df.index)
        ns['vwap'] = (_h + _l + _c) / 3.0

    ns['returns'] = df.get('log_return', pd.Series(np.nan, index=df.index))

    # Per-company polarity columns (from sentiment_features/<TICKER>.csv)
    # and polarity_avg (mean of all company polarities, used by TCEHY alphas).
    for col in df.columns:
        if col.endswith('_polarity') or col == 'polarity_avg':
            ns[col] = df[col].fillna(0.0)

    # Technical indicator columns - already shifted in df via compute_all_indicators().
    # Expose under both paper names (BB_Upper, MACD_Signal, …) and legacy names
    # (BB_upper, MACD_signal, …) so old LLM-generated expressions remain valid.
    for col in _INDICATOR_COL_SET:
        if col in df.columns:
            ns[col] = df[col]
        else:
            ns[col] = pd.Series(np.nan, index=df.index)

    # Operators
    ns.update({
        'abs': lambda s: s.abs(),
        'sign': _sign,
        'log': _log,
        'rank': _rank,
        'delay': _delay,
        'delta': _delta,
        'ts_rank': _ts_rank,
        'ts_mean': _ts_mean,
        'ts_std': _ts_std,
        'ts_min': _ts_min,
        'ts_max': _ts_max,
        'ts_argmax': _ts_argmax,
        'ts_argmin': _ts_argmin,
        'correlation': _correlation,
        'covariance': _covariance,
        'decay_linear': _decay_linear,
        'scale': _scale,
        'if_else': _if_else,
    })

    return ns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_alpha_expression(
    expression: str,
    df: pd.DataFrame,
    validate: bool = True,
) -> pd.Series:
    """
    Validate (optionally) and evaluate an alpha expression string.

    Returns a pd.Series aligned to df.index.
    Returns a NaN series on any runtime error rather than raising, so that
    a single bad alpha does not crash an experiment.
    """
    if validate:
        try:
            validate_alpha_expression(expression)
        except AlphaValidationError:
            return pd.Series(np.nan, index=df.index, name='invalid_alpha')

    namespace = build_alpha_namespace(df)

    try:
        result = eval(expression, {'__builtins__': {}}, namespace)  # noqa: S307
    except Exception:
        return pd.Series(np.nan, index=df.index)

    if isinstance(result, pd.Series):
        return result.reindex(df.index)
    return pd.Series(float(result), index=df.index)
