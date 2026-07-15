"""
Comprehensive metric computation for the thesis evaluation.
"""
import numpy as np
import scipy.stats as sp
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..walkforward.fold_runner import FoldResult


def compute_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """Pearson Information Coefficient."""
    if len(preds) < 2 or preds.std() < 1e-8 or targets.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(preds, targets)[0, 1])


def compute_rank_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """Spearman Rank IC."""
    if len(preds) < 2:
        return 0.0
    result = sp.spearmanr(preds, targets)
    corr = result.correlation
    return float(corr) if not np.isnan(corr) else 0.0


def compute_ic_ir(ic_series: List[float]) -> float:
    """IC Information Ratio = mean(IC) / std(IC). Uses ddof=1 (sample std)."""
    arr = np.array(ic_series, dtype=float)
    if len(arr) < 2 or arr.std(ddof=1) < 1e-10:
        return 0.0
    return float(arr.mean() / arr.std(ddof=1))


def compute_directional_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 2:
        return 0.0
    return float(np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(targets))))


def compute_price_level_metrics(
    log_return_preds: np.ndarray,
    log_return_targets: np.ndarray,
    close_prices: np.ndarray,
) -> Dict[str, float]:
    """
    Reconstruct price series and compute price-level metrics.

    predicted_price[t+1] = close[t] * exp(predicted_log_return[t])
    """
    pred_prices = close_prices * np.exp(log_return_preds)
    actual_prices = close_prices * np.exp(log_return_targets)

    price_rmse = float(np.sqrt(np.mean((pred_prices - actual_prices) ** 2)))

    nonzero = actual_prices != 0
    mape = float(np.mean(np.abs((pred_prices[nonzero] - actual_prices[nonzero]) / actual_prices[nonzero]))) if nonzero.any() else 0.0

    return {'price_rmse': price_rmse, 'mape': mape}


def aggregate_fold_metrics(fold_results: List) -> Dict[str, float]:
    """
    Compute mean ± std over a list of FoldResult objects.
    Returns a dict with keys like 'mse_mean', 'mse_std', etc.
    """
    metric_names = ['mse', 'rmse', 'mae', 'directional_accuracy', 'pearson_ic', 'spearman_rank_ic']
    out = {}
    for name in metric_names:
        vals = [getattr(r, name, 0.0) for r in fold_results if len(r.test_preds) > 0]
        if vals:
            out[f'{name}_mean'] = float(np.mean(vals))
            out[f'{name}_std'] = float(np.std(vals))
        else:
            out[f'{name}_mean'] = 0.0
            out[f'{name}_std'] = 0.0
    return out
