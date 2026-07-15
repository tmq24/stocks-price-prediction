import pandas as pd
import numpy as np
import ta
from typing import List

# Paper's 11 indicators (Table 1, Chen & Kawashima 2025).
# Naming convention matches paper exactly; lowercase aliases (BB_upper, MACD_signal,
# momentum_10, EMA_12, EMA_26) are also exposed so previously-generated LLM alpha
# expressions still parse against the allowed-variable whitelist.
# All columns are shifted by 1 day in compute_all_indicators() so that the
# value at row t uses only information available up to t-1 (no look-ahead).
INDICATOR_COLS: List[str] = [
    'SMA_5',
    'SMA_20',
    'EMA_10',
    'Momentum_3',
    'Momentum_10',
    'RSI_14',
    'MACD',
    'MACD_Signal',
    'BB_Upper',
    'BB_Lower',
    'OBV',
]

_OHLCV_COLS = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume'}


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators from raw OHLCV data.
    All indicator columns are shifted by 1 period so that features at time t
    only use information available up to t-1 (no look-ahead).

    Produces the 11 columns in INDICATOR_COLS plus lowercase aliases used by
    previously-generated LLM alpha expressions.
    """
    df = df.copy()
    close = df['close']
    pct_chg = close.pct_change()

    # --- Returns ---
    df['return_1d'] = pct_chg
    df['return_5d'] = close.pct_change(5)
    df['log_return'] = np.log(close / close.shift(1))

    # --- Moving averages ---
    df['SMA_5']  = close.rolling(5).mean()
    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_60'] = close.rolling(60).mean()

    # EMA - paper uses span=10; spans 12 and 26 are also exposed for cached
    # LLM expressions that reference standard MACD inputs.
    df['EMA_10'] = close.ewm(span=10, adjust=False).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

    df['price_dev_SMA_20'] = (close - df['SMA_20']) / (df['SMA_20'] + 1e-8)

    # --- Momentum (paper: 3-day and 10-day) ---
    df['Momentum_3']  = close - close.shift(3)
    df['Momentum_10'] = close - close.shift(10)
    # Lowercase aliases used by cached LLM expressions
    df['momentum_5']  = close - close.shift(5)
    df['momentum_10'] = df['Momentum_10']

    df['RSI_14'] = ta.momentum.rsi(close, window=14)

    # --- MACD ---
    macd = ta.trend.MACD(close)
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()   # paper name
    df['MACD_diff']   = macd.macd_diff()
    # Lowercase alias used by cached LLM expressions
    df['MACD_signal'] = df['MACD_Signal']

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20)
    df['BB_Upper']  = bb.bollinger_hband()   # paper name
    df['BB_Lower']  = bb.bollinger_lband()   # paper name
    df['BB_middle'] = bb.bollinger_mavg()
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_width']  = bb_range / (df['BB_middle'] + 1e-8)
    df['BB_pct_b']  = np.where(
        bb_range.abs() > 1e-8,
        (close - df['BB_Lower']) / bb_range,
        0.0,
    )
    # Lowercase aliases used by cached LLM expressions
    df['BB_upper']  = df['BB_Upper']
    df['BB_lower']  = df['BB_Lower']

    # --- Realised volatility & ATR ---
    df['rolling_std_5']   = pct_chg.rolling(5).std()
    df['rolling_std_20']  = pct_chg.rolling(20).std()
    df['realized_vol_5d'] = df['rolling_std_5']
    df['realized_vol_20d']= df['rolling_std_20']
    df['ATR_14'] = ta.volatility.average_true_range(
        df['high'], df['low'], close, window=14
    )

    # --- Volume ---
    df['Volume_MA_5']  = df['volume'].rolling(5).mean()
    df['Volume_ratio'] = df['volume'] / (df['Volume_MA_5'] + 1e-8)
    df['volume_ratio'] = df['Volume_ratio']
    df['OBV'] = ta.volume.on_balance_volume(close, df['volume'])

    # Shift ALL computed columns by 1 period (look-ahead prevention).
    # Raw OHLCV columns are excluded - they serve only as inputs to the
    # computations above and are NOT used as direct model features.
    indicator_cols = [c for c in df.columns if c not in _OHLCV_COLS]
    df[indicator_cols] = df[indicator_cols].shift(1)

    # close_feat: today's close as decoder input (paper §3.3 Figure 4).
    # Target is close.shift(-h), so close[t] is NOT a look-ahead for close[t+h].
    # Added AFTER the blanket shift so it does NOT get the indicator shift.
    df['close_feat'] = df['close']

    # Temporal features for paper's Temporal Embedding (§3.3). Calendar info
    # at the bar's own date - no look-ahead. Cyclic encoding (sin/cos) so MinMax
    # scaling preserves the cycle.
    dow = df['date'].dt.dayofweek.astype(float)        # 0=Mon..4=Fri
    month = df['date'].dt.month.astype(float) - 1.0    # 0..11
    df['tm_dow_sin']   = np.sin(2 * np.pi * dow / 5.0)
    df['tm_dow_cos']   = np.cos(2 * np.pi * dow / 5.0)
    df['tm_month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['tm_month_cos'] = np.cos(2 * np.pi * month / 12.0)

    return df


def validate_no_lookahead(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Sanity check: more than 10% NaN in the tail of any feature column raises."""
    tail = df[feature_cols].iloc[-100:] if len(df) > 100 else df[feature_cols]
    nan_fracs = tail.isna().mean()
    bad = nan_fracs[nan_fracs > 0.10]
    if len(bad) > 0:
        raise AssertionError(
            f"Suspicious NaN rate in tail for columns: {bad.to_dict()}"
        )
