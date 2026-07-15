"""
Three-way trading comparison: DA-LLM vs Static (A4) vs Paper Hardcoded (A5).

Same container (same model, data, 70/30 split, 10 seeds ensemble).
Trading rule: position = sign(pred_close[t+1] - actual_close[t]), TC=10bps.

Usage:
    python scripts/trading_three_way.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import empyrical as emp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICKERS = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']
TC_BPS = 10
TRADING_DAYS = 252

CONFIGS = {
    'DA-LLM':            'results/polarity_directional/predictions.pkl',
    'A5 (paper static)': 'results/sweep_backup/transformer_encdec_a5_predictions.pkl',
}


def load_ensemble(pkl_path):
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    out = {}
    for (fold_idx, ticker), results in raw.items():
        preds = np.array([r.test_preds for r in results]).mean(axis=0)
        actual = results[0].test_targets
        dates = pd.to_datetime(results[0].test_dates)
        out[ticker] = (preds, actual, dates)
    return out


def simulate(pred_close, actual_close, tc_bps=TC_BPS):
    pred = np.asarray(pred_close, dtype=float)
    actual = np.asarray(actual_close, dtype=float)
    realized = np.diff(actual) / actual[:-1]
    pred_delta = pred[1:] - actual[:-1]
    pos = np.sign(pred_delta)
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - pos_prev)
    return pos * realized - (tc_bps / 10_000.0) * turnover


def metrics(daily_ret):
    # Trading metrics via the empyrical library (Rf=0, 252-day annualization).
    r = pd.Series(daily_ret).dropna()
    if len(r) < 2 or r.std() < 1e-10:
        return dict(sharpe=0.0, cum_ret_pct=0.0, max_dd_pct=0.0)
    sharpe = float(emp.sharpe_ratio(r, annualization=TRADING_DAYS))
    cum_ret = float(emp.cum_returns_final(r)) * 100
    max_dd = float(emp.max_drawdown(r)) * 100
    return dict(sharpe=round(sharpe, 3),
                cum_ret_pct=round(cum_ret, 1),
                max_dd_pct=round(max_dd, 1))


def main():
    rows = []
    returns_store = {}

    for label, path in CONFIGS.items():
        if not os.path.exists(path):
            print(f'Missing: {path} - skip {label}')
            continue
        preds = load_ensemble(path)
        for ticker in TICKERS:
            if ticker not in preds:
                continue
            pred_close, actual_close, dates = preds[ticker]
            ret = simulate(pred_close, actual_close)
            m = metrics(ret)
            rows.append({'config': label, 'ticker': ticker, **m})
            returns_store[(label, ticker)] = pd.Series(ret, index=dates[1:])

    df = pd.DataFrame(rows)

    # Sharpe pivot
    sharpe = df.pivot(index='config', columns='ticker', values='sharpe')
    sharpe['mean'] = sharpe.mean(axis=1)
    sharpe = sharpe.reindex(list(CONFIGS.keys()))

    # CumRet pivot
    cumret = df.pivot(index='config', columns='ticker', values='cum_ret_pct')
    cumret['mean'] = cumret.mean(axis=1)
    cumret = cumret.reindex(list(CONFIGS.keys()))

    # MaxDD pivot
    maxdd = df.pivot(index='config', columns='ticker', values='max_dd_pct')
    maxdd['mean'] = maxdd.mean(axis=1)
    maxdd = maxdd.reindex(list(CONFIGS.keys()))

    print('\n=== Sharpe Ratio (annualised, TC=10bps) ===')
    print(sharpe.round(3).to_string())

    print('\n=== Cumulative Return % ===')
    print(cumret.round(1).to_string())

    print('\n=== Max Drawdown % ===')
    print(maxdd.round(1).to_string())

    os.makedirs('results/aggregated', exist_ok=True)
    df.to_csv('results/aggregated/trading_three_way.csv', index=False)
    sharpe.to_csv('results/aggregated/trading_three_way_sharpe.csv')
    print('\nSaved to results/aggregated/trading_three_way*.csv')

    return df, sharpe, cumret, maxdd


if __name__ == '__main__':
    main()
