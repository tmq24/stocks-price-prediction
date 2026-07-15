"""
Diebold-Mariano test on next-day forecast errors (MSE), DA-LLM (A2) vs baselines.

Per-ticker DM (HAC, h=1). H1: DA-LLM has lower MSE than the comparison model.
Reproducible source of the DM numbers in the paper (Table 4.5).

Sources (A2 config = C1+C3, the paper config):
  DA-LLM        : results/raw/data_alphas_A2/method_m_checkpoint.pkl
  A5            : results/sweep_backup/transformer_encdec_a5_predictions.pkl
  LSTM/NBE      : results/arch_baselines_A2/predictions.pkl
  GRU/RF/XGB    : results/arch_baselines_extra/predictions.pkl

Output: results/aggregated/dm_test.csv

Usage: python scripts/run_dm_test.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.statistical_tests import diebold_mariano_test

TICKERS = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']


def _ensemble(results):
    preds = np.array([r.test_preds for r in results]).mean(axis=0)
    targ = np.asarray(results[0].test_targets, dtype=float)
    dates = pd.to_datetime(results[0].test_dates)
    n = min(len(preds), len(targ), len(dates))
    return pd.Series(preds[:n], index=dates[:n]), pd.Series(targ[:n], index=dates[:n])


def _load_simple(path):
    d = pickle.load(open(path, 'rb'))
    return {tk: _ensemble(res) for (fold, tk), res in d.items()}


def _load_arch(path, model_type):
    d = pickle.load(open(path, 'rb'))
    return {tk: _ensemble(res) for (fold, tk, m), res in d.items() if m == model_type}


def main():
    _extra = 'results/arch_baselines_extra/predictions.pkl'
    cfg = {
        'DA-LLM': _load_simple('results/raw/data_alphas_A2/method_m_checkpoint.pkl'),
        'A5': _load_simple('results/sweep_backup/transformer_encdec_a5_predictions.pkl'),
        'LSTM': _load_arch('results/arch_baselines_A2/predictions.pkl', 'lstm'),
        'N-BEATS': _load_arch('results/arch_baselines_A2/predictions.pkl', 'nbeats'),
        'GRU': _load_arch(_extra, 'gru'),
        'RF': _load_arch(_extra, 'rf'),
        'XGBoost': _load_arch(_extra, 'xgboost'),
    }

    rows = []
    for B in ['A5', 'LSTM', 'N-BEATS', 'GRU', 'RF', 'XGBoost']:
        for tk in TICKERS:
            pa, ta = cfg['DA-LLM'][tk]
            pb, tb = cfg[B][tk]
            idx = pa.index.intersection(pb.index)
            ea = (pa.loc[idx] - ta.loc[idx]).values
            eb = (pb.loc[idx] - tb.loc[idx]).values
            stat, p = diebold_mariano_test(ea, eb, h=1)
            rows.append({
                'comparison': f'DA-LLM vs {B}',
                'ticker': tk, 'n': len(idx),
                'dm_stat': round(float(stat), 3), 'p_value': round(float(p), 4),
                'da_llm_lower_mse_sig': p < 0.05,
            })

    df = pd.DataFrame(rows)
    os.makedirs('results/aggregated', exist_ok=True)
    out = 'results/aggregated/dm_test.csv'
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved -> {out}')
    return df


if __name__ == '__main__':
    main()
