"""
Pairwise statistical tests for DA-LLM vs A4 STATIC.

For each ticker:
  - Diebold-Mariano test on MSE differential (prediction errors)
  - Bootstrap 95% CI for Sharpe ratio of trading strategy returns

Inputs : results/raw/_DALLM_predictions.pkl
         results/raw/_A4_predictions.pkl
         results/aggregated/_trading_returns.pkl
Outputs: results/aggregated/statistical_tests.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.statistical_tests import diebold_mariano_test, bootstrap_sharpe_ci

TICKERS = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']


def _ensemble_errors(fold_results) -> np.ndarray:
    """Prediction errors = ensemble_mean_pred - actual."""
    preds = np.array([r.test_preds for r in fold_results]).mean(axis=0)
    actual = fold_results[0].test_targets
    return preds - actual


def main():
    with open('results/raw/_DALLM_predictions.pkl', 'rb') as f:
        dallm = pickle.load(f)
    with open('results/raw/_A4_predictions.pkl', 'rb') as f:
        a4 = pickle.load(f)
    with open('results/aggregated/_trading_returns.pkl', 'rb') as f:
        returns_map = pickle.load(f)

    rows = []
    for ticker in TICKERS:
        # --- DM test on prediction errors (DA-LLM vs A4) ---
        err_dl = _ensemble_errors(dallm[(0, ticker)])
        err_a4 = _ensemble_errors(a4[(0, ticker)])
        dm_stat, dm_p = diebold_mariano_test(err_dl, err_a4, h=1)

        # --- Bootstrap Sharpe CI ---
        r_dl = returns_map[(ticker, 'DA-LLM')]
        r_a4 = returns_map[(ticker, 'A4 STATIC')]
        ci_dl_lo, ci_dl_hi = bootstrap_sharpe_ci(r_dl, n_bootstrap=1000, seed=42)
        ci_a4_lo, ci_a4_hi = bootstrap_sharpe_ci(r_a4, n_bootstrap=1000, seed=42)

        sharpe_dl = float(r_dl.mean() / r_dl.std() * np.sqrt(252)) if r_dl.std() > 0 else 0.0
        sharpe_a4 = float(r_a4.mean() / r_a4.std() * np.sqrt(252)) if r_a4.std() > 0 else 0.0

        rows.append({
            'ticker': ticker,
            'dm_stat': dm_stat,
            'dm_p_value': dm_p,
            'dm_verdict': ('DA-LLM lower MSE' if dm_p < 0.05 else
                           'A4 lower MSE' if dm_stat < 0 and (1 - dm_p) < 0.05 else
                           'not significant'),
            'sharpe_dallm': sharpe_dl,
            'sharpe_dallm_ci_lo': ci_dl_lo,
            'sharpe_dallm_ci_hi': ci_dl_hi,
            'sharpe_a4': sharpe_a4,
            'sharpe_a4_ci_lo': ci_a4_lo,
            'sharpe_a4_ci_hi': ci_a4_hi,
            'sharpe_ci_overlap': not (ci_dl_lo > ci_a4_hi or ci_a4_lo > ci_dl_hi),
        })

    df = pd.DataFrame(rows)
    out = 'results/aggregated/statistical_tests.csv'
    df.to_csv(out, index=False)
    print(f"Saved. {out}\n")

    print('-' * 40)
    print("DIEBOLD-MARIANO TEST - DA-LLM vs A4 STATIC (H0: equal MSE, H1: DA-LLM lower MSE)")
    print('-' * 40)
    print(f"{'Ticker':<8} {'DM stat':>10} {'p-value':>10} {'Verdict':<25}")
    for _, r in df.iterrows():
        print(f"{r['ticker']:<8} {r['dm_stat']:>10.3f} {r['dm_p_value']:>10.4f} {r['dm_verdict']:<25}")

    print()
    print('-' * 40)
    print("BOOTSTRAP 95% CI FOR SHARPE - DA-LLM vs A4 STATIC")
    print('-' * 40)
    print(f"{'Ticker':<8} {'DA-LLM (CI)':>26} {'A4 STATIC (CI)':>26} {'CI overlap?':>14}")
    for _, r in df.iterrows():
        dl_ci = f"{r['sharpe_dallm']:+.3f} [{r['sharpe_dallm_ci_lo']:+.2f},{r['sharpe_dallm_ci_hi']:+.2f}]"
        a4_ci = f"{r['sharpe_a4']:+.3f} [{r['sharpe_a4_ci_lo']:+.2f},{r['sharpe_a4_ci_hi']:+.2f}]"
        overlap = 'yes' if r['sharpe_ci_overlap'] else 'no (significant)'
        print(f"{r['ticker']:<8} {dl_ci:>26} {a4_ci:>26} {overlap:>14}")


if __name__ == '__main__':
    main()
