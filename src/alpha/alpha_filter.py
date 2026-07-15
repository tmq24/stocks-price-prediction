"""
C3: Generate-and-filter alpha selection (DA-LLM contribution).

Generates N candidate alpha expressions and selects the top-5 by IC + diversity.
The filter rejects sign-flip duplicates (|corr| > 0.8) and greedily picks
the highest-|IC| candidate at each step that is sufficiently diverse from
already-selected alphas.

This module is stateless - call select_top_k() with the candidate list and
a context DataFrame.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .alpha_validator import validate_alpha_expression, AlphaValidationError
from ..features.alpha_executor import evaluate_alpha_expression

logger = logging.getLogger(__name__)

_MAX_CORR = 0.8   # greedy diversity threshold (includes sign-flip)
_MIN_IC_WINDOW = 15  # minimum observations for IC computation (per sub-window)


def _ic_on_range(
    alpha_series: pd.Series,
    ret_series: pd.Series,
    start: int,
    end: int,
) -> float:
    """Spearman IC of alpha vs next-day returns over rows [start, end)."""
    if end - start < _MIN_IC_WINDOW + 1:
        return np.nan
    alpha_vals = alpha_series.iloc[start:end].values
    ret_vals = ret_series.iloc[start + 1: end + 1].values  # next-day return
    mask = ~(np.isnan(alpha_vals) | np.isnan(ret_vals))
    if mask.sum() < _MIN_IC_WINDOW:
        return np.nan
    ic = pd.Series(alpha_vals[mask]).corr(
        pd.Series(ret_vals[mask]), method='spearman'
    )
    return float(ic) if not np.isnan(ic) else np.nan


def _compute_ic(
    expr: str,
    df: pd.DataFrame,
    window: int = 63,
    lag: int = 1,
) -> float:
    """
    Spearman IC of expr vs next-day returns on the last `window` rows of df.

    Backwards-compatible helper used by alpha_generator.py for storage.
    Returns np.nan on error or insufficient data.
    """
    ret_col = 'log_return' if 'log_return' in df.columns else 'returns'
    if ret_col not in df.columns or len(df) < window + lag + 1:
        return np.nan

    end = len(df) - lag
    start = end - window
    if start < 1:
        return np.nan
    try:
        alpha_series = evaluate_alpha_expression(expr, df.iloc[:end], validate=False)
    except Exception:
        return np.nan
    return _ic_on_range(alpha_series, df[ret_col], start, end)


def _compute_ic_split(
    expr: str,
    df: pd.DataFrame,
    select_days: int = 42,
    val_days: int = 21,
    lag: int = 1,
) -> Tuple[float, float, np.ndarray]:
    """
    Split-window IC (F1: OOS validation for C3).

    Trailing data of length `select_days + val_days` is split into:
      - SELECT window: [end - select_days - val_days,  end - val_days)  -> ranking
      - VAL    window: [end - val_days,                end)              -> OOS check

    Returns (ic_select, ic_val, series_vals_for_diversity).
    The validation window is the MOST RECENT data, closest to test distribution.
    """
    total = select_days + val_days
    ret_col = 'log_return' if 'log_return' in df.columns else 'returns'
    if ret_col not in df.columns or len(df) < total + lag + 1:
        return np.nan, np.nan, np.array([])

    end = len(df) - lag
    val_start = end - val_days
    sel_start = val_start - select_days
    if sel_start < 1:
        return np.nan, np.nan, np.array([])

    try:
        alpha_series = evaluate_alpha_expression(expr, df.iloc[:end], validate=False)
    except Exception:
        return np.nan, np.nan, np.array([])

    # Reject constant / near-constant series before computing IC. Degenerate
    # expressions like `expr * sign(sparse_sentiment)` collapse to 0 on most
    # days, so the IC of a constant series is undefined and post-hoc NaN
    # handling leaks them into fallback selection. Require ≥10 unique non-zero
    # values in the trailing window for the candidate to be eligible.
    trailing_vals = alpha_series.iloc[sel_start:end].dropna().values.astype(float)
    if len(trailing_vals) == 0 or np.unique(trailing_vals[np.nonzero(trailing_vals)]).size < 10:
        return np.nan, np.nan, np.array([])

    ic_sel = _ic_on_range(alpha_series, df[ret_col], sel_start, val_start)
    ic_val = _ic_on_range(alpha_series, df[ret_col], val_start, end)
    # Series for diversity check covers full trailing window
    series_vals = alpha_series.iloc[sel_start:end].values.astype(float)
    return ic_sel, ic_val, series_vals


def select_top_k(
    candidates: List[Dict],
    df: pd.DataFrame,
    k: int = 5,
    ic_window: int = 63,
    val_days: int = 21,
    min_val_abs_ic: float = 0.03,
) -> Tuple[List[Dict], List[float]]:
    """
    From validated candidates, select k alphas with sign-stability + diversity (F1).

    Split-window IC: trailing ic_window rows are split into SELECT (older) and
    VAL (last `val_days`) sub-windows. A candidate must:
        1. have sign(IC_select) == sign(IC_val)   (sign stability)
        2. have |IC_val| >= min_val_abs_ic         (genuine OOS signal)
    Survivors are then ranked by combined score = (|IC_sel| + |IC_val|) / 2.
    Diversity (|corr| < 0.8) is applied greedily.

    Fallback: if too few survivors, fill remaining slots with the highest
    |IC_select| candidates from the rejected pool (legacy behaviour).

    Returns (selected_alphas, per_alpha_ics).  Each selected dict gets
    `_ic_select`, `_ic_val`, `_score` recorded for downstream logging.
    """
    select_days = max(ic_window - val_days, _MIN_IC_WINDOW + 1)

    if len(candidates) <= k:
        ics = [_compute_ic(c['expression'], df, ic_window) for c in candidates]
        return candidates, ics

    # Step 1: split-window IC per candidate + cache time series for diversity
    series_cache: Dict[int, np.ndarray] = {}
    for i, c in enumerate(candidates):
        ic_sel, ic_val, vals = _compute_ic_split(
            c['expression'], df,
            select_days=select_days, val_days=val_days,
        )
        c['_ic_select'] = ic_sel
        c['_ic_val'] = ic_val
        c['_ic'] = ic_val if not np.isnan(ic_val) else ic_sel  # for legacy logging
        series_cache[i] = vals

    # Step 2: F1 filter - sign stability + minimum OOS magnitude
    survivors: List[Tuple[float, int]] = []
    for i, c in enumerate(candidates):
        ic_sel, ic_val = c['_ic_select'], c['_ic_val']
        if np.isnan(ic_sel) or np.isnan(ic_val):
            continue
        if np.sign(ic_sel) != np.sign(ic_val):
            continue
        if abs(ic_val) < min_val_abs_ic:
            continue
        score = (abs(ic_sel) + abs(ic_val)) / 2.0
        c['_score'] = score
        survivors.append((score, i))

    survivors.sort(key=lambda x: x[0], reverse=True)

    # Step 3: greedy diversity over survivors
    selected_indices: List[int] = []
    for _, i in survivors:
        if len(selected_indices) >= k:
            break
        vals_i = series_cache[i]
        diverse = True
        for j in selected_indices:
            vals_j = series_cache[j]
            mask = ~(np.isnan(vals_i) | np.isnan(vals_j))
            if mask.sum() < _MIN_IC_WINDOW:
                continue
            corr = float(np.corrcoef(vals_i[mask], vals_j[mask])[0, 1])
            if abs(corr) >= _MAX_CORR:
                diverse = False
                break
        if diverse:
            selected_indices.append(i)

    # Step 4: fallback if too few survivors - fill with highest |IC_select|
    # Skip candidates with NaN _ic_select (constant series flagged in step 1)
    # so degenerate `* sign(sentiment)` expressions never reach the pool.
    if len(selected_indices) < k:
        def _valid_ic_select(c):
            v = c.get('_ic_select')
            return v is not None and not (isinstance(v, float) and np.isnan(v))

        fallback_pool = [
            (abs(candidates[i]['_ic_select']), i)
            for i in range(len(candidates))
            if i not in selected_indices and _valid_ic_select(candidates[i])
        ]
        fallback_pool.sort(key=lambda x: x[0], reverse=True)
        for _, i in fallback_pool:
            if len(selected_indices) >= k:
                break
            selected_indices.append(i)

    # Final safety net: if still < k after all filters (e.g. first generation
    # event at low idx where split-window IC can't be computed - all candidates
    # have NaN ic_select), fill remaining slots with first available candidates.
    # Without this, the event ends with 0 stored alphas -> all-NaN feature column.
    if len(selected_indices) < k:
        for i in range(len(candidates)):
            if i not in selected_indices:
                selected_indices.append(i)
                if len(selected_indices) >= k:
                    break

    selected = [candidates[i] for i in selected_indices[:k]]
    # Report IC_val (OOS) for downstream logging
    ics = [candidates[i].get('_ic_val', np.nan) for i in selected_indices[:k]]
    ics = [ic if (ic is not None and not (isinstance(ic, float) and np.isnan(ic))) else
           candidates[selected_indices[idx]].get('_ic_select', np.nan)
           for idx, ic in enumerate(ics)]

    n_survivors = len(survivors)
    n_candidates = len(candidates)
    logger.info(
        f"C3 (F1): {n_candidates} cands to {n_survivors} pass sign-stability+|IC_val|>={min_val_abs_ic} "
        f"- {len(selected)} selected (IC_val: "
        f"{[round(ic, 3) if not (ic is None or (isinstance(ic,float) and np.isnan(ic))) else 'nan' for ic in ics]})"
    )
    return selected, ics
