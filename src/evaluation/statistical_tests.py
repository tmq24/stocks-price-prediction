"""
Statistical tests for pairwise model comparison.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Modified Diebold-Mariano test (Harvey-Leybourne-Newbold 1997) for equal
    predictive accuracy.

    H0: Equal MSE.  H1: model_a has strictly lower MSE.

    Uses a Newey-West HAC estimator (q = h - 1 lags) for the variance of the
    loss differential, applies the HLN small-sample correction factor, and
    compares the corrected statistic to a Student-t distribution with T-1
    degrees of freedom.

    Args:
        errors_a : Prediction errors for model A (e.g., pred - actual).
        errors_b : Prediction errors for model B.
        h        : Forecast horizon (determines the NW lag truncation).

    Returns:
        (dm_statistic, p_value)
    """
    d = errors_a ** 2 - errors_b ** 2  # loss differential
    T = len(d)

    if T < 4:
        return 0.0, 1.0

    d_mean = float(d.mean())
    d_centered = d - d_mean

    # Newey-West HAC variance of the sample mean with q = h - 1 lags.
    # Var(mean) = (1/T) * [ γ_0 + 2 * Σ_{k=1}^{q} (1 - k/(q+1)) * γ_k ]
    q = max(h - 1, 0)
    gamma_0 = float((d_centered ** 2).mean())
    nw_var_sum = gamma_0
    for k in range(1, q + 1):
        gamma_k = float((d_centered[k:] * d_centered[:-k]).mean())
        weight = 1.0 - k / (q + 1)
        nw_var_sum += 2.0 * weight * gamma_k

    if nw_var_sum <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(nw_var_sum / T)

    # Harvey-Leybourne-Newbold (1997) small-sample correction. For h=1 the
    # factor is sqrt((T-1)/T), a near-identity at these sample sizes.
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = dm_stat * hln_factor

    # H1: errors_a have strictly lower MSE (a better than b).
    # a better => d_mean = mean(e_a² − e_b²) < 0 => dm_stat < 0.
    # Reject H0 when dm_stat is sufficiently negative => p-value = left-tail = cdf(dm_stat).
    from scipy.stats import t as t_dist
    p_value = float(t_dist.cdf(dm_stat, df=T - 1))

    return float(dm_stat), p_value


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    block_size: int = 21,
) -> Tuple[float, float]:
    """
    Block bootstrap confidence interval for the annualised Sharpe Ratio.

    Uses non-overlapping blocks of size `block_size` (default 21 about 1 month)
    to preserve autocorrelation structure.

    Returns:
        (lower_bound, upper_bound) of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    arr = returns.values

    n = len(arr)
    if n < block_size * 2:
        sr = float((arr.mean() / arr.std()) * np.sqrt(252)) if arr.std() > 0 else 0.0
        return sr, sr

    # Build non-overlapping blocks
    n_blocks = n // block_size
    blocks = [arr[i * block_size: (i + 1) * block_size] for i in range(n_blocks)]

    sharpe_samples = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(len(blocks), size=n_blocks, replace=True)
        sample = np.concatenate([blocks[i] for i in chosen])
        std = sample.std()
        sr = float((sample.mean() / std) * np.sqrt(252)) if std > 1e-10 else 0.0
        sharpe_samples.append(sr)

    alpha = 1 - confidence
    lower = float(np.percentile(sharpe_samples, 100 * alpha / 2))
    upper = float(np.percentile(sharpe_samples, 100 * (1 - alpha / 2)))
    return lower, upper
