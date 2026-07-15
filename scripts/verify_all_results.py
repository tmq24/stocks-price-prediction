"""
Verify all ablation results: MSE from aggregated_results.csv + trading from sweep_backup pkls.
No model training - reads existing files only.

Usage:
    python scripts/verify_all_results.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICKERS = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']
TC_BPS = 10

MSE_DIRS = {
    'DA-LLM':   'results/DALLM_s',
    'A1 no_regime':   'results/A1_s',
    'A2 no_feedback': 'results/A2_s',
    'A3 no_filter':   'results/A3_s',
    'A4 static':      'results/A4_s',
    'A5 paper':       'results/A5_s',
}

PRED_PKLS = {
    'DA-LLM':         'results/sweep_backup/transformer_encdec_dallm_predictions.pkl',
    'A1 no_regime':   'results/sweep_backup/transformer_encdec_a1_predictions.pkl',
    'A2 no_feedback': 'results/sweep_backup/transformer_encdec_a2_predictions.pkl',
    'A3 no_filter':   'results/sweep_backup/transformer_encdec_a3_predictions.pkl',
    'A4 static':      'results/sweep_backup/transformer_encdec_a4_predictions.pkl',
    'A5 paper':       'results/sweep_backup/transformer_encdec_a5_predictions.pkl',
    'A8 regen_only':  'results/sweep_backup/transformer_encdec_a8_predictions.pkl',
    'A9 static_nof':  'results/sweep_backup/transformer_encdec_a9_predictions.pkl',
    'A10 c2_only':    'results/sweep_backup/transformer_encdec_a10_predictions.pkl',
    'A11 c3_only':    'results/sweep_backup/transformer_encdec_a11_predictions.pkl',
    'A12 c1_only':    'results/sweep_backup/transformer_encdec_a12_predictions.pkl',
}


# ---------------------------------------------------------------------------
# MSE table
# ---------------------------------------------------------------------------

def load_mse_table():
    rows = []
    for label, d in MSE_DIRS.items():
        path = os.path.join(d, 'aggregated_results.csv')
        if not os.path.exists(path):
            print(f'Missing: {path}')
            continue
        df = pd.read_csv(path)
        df = df[df.model_type == 'transformer_encdec']
        for _, row in df.iterrows():
            rows.append({'config': label, 'ticker': row['ticker'],
                         'mse_scaled': row['mse_scaled_mean']})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='config', columns='ticker', values='mse_scaled')
    pivot['mean'] = pivot.mean(axis=1)
    return pivot.reindex([k for k in MSE_DIRS if k in pivot.index])


# ---------------------------------------------------------------------------
# Trading metrics
# ---------------------------------------------------------------------------

def simulate(pred_close, actual_close):
    pred = np.asarray(pred_close, dtype=float)
    actual = np.asarray(actual_close, dtype=float)
    realized = np.diff(actual) / actual[:-1]
    pos = np.sign(pred[1:] - actual[:-1])
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    ret = pos * realized - (TC_BPS / 10_000.0) * np.abs(pos - pos_prev)
    return ret


def trading_metrics(ret):
    r = pd.Series(ret).dropna()
    if len(r) < 2 or r.std() < 1e-10:
        return dict(sharpe=0.0, cum_ret=0.0, max_dd=0.0)
    sharpe = float(r.mean() / r.std() * np.sqrt(252))
    cum = float((1 + r).prod() - 1) * 100
    wealth = (1 + r).cumprod()
    max_dd = float(((wealth - wealth.cummax()) / wealth.cummax()).min()) * 100
    return dict(sharpe=round(sharpe, 3), cum_ret=round(cum, 1), max_dd=round(max_dd, 1))


def load_trading_table():
    sharpe_rows, cumret_rows, maxdd_rows = [], [], []
    for label, path in PRED_PKLS.items():
        if not os.path.exists(path):
            print(f'Missing: {path}')
            continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for ticker in TICKERS:
            results = d.get((0, ticker))
            if results is None:
                continue
            preds = np.array([r.test_preds for r in results]).mean(axis=0)
            actual = results[0].test_targets
            ret = simulate(preds, actual)
            m = trading_metrics(ret)
            sharpe_rows.append({'config': label, 'ticker': ticker, 'sharpe': m['sharpe']})
            cumret_rows.append({'config': label, 'ticker': ticker, 'cum_ret': m['cum_ret']})
            maxdd_rows.append({'config': label, 'ticker': ticker, 'max_dd': m['max_dd']})

    def pivot(rows, val):
        df = pd.DataFrame(rows)
        p = df.pivot(index='config', columns='ticker', values=val)
        p['mean'] = p.mean(axis=1)
        return p.reindex([k for k in PRED_PKLS if k in p.index])

    return pivot(sharpe_rows, 'sharpe'), pivot(cumret_rows, 'cum_ret'), pivot(maxdd_rows, 'max_dd')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('MSE (scaled, mean over 10 seeds)')
    print('=' * 60)
    mse = load_mse_table()
    print(mse.map(lambda x: f'{x:.6f}' if pd.notna(x) else '').to_string())

    print('\n' + '=' * 60)
    print('Trading Sharpe (TC=10bps, sign rule, ensemble 10 seeds)')
    print('=' * 60)
    sharpe, cumret, maxdd = load_trading_table()
    print(sharpe.round(3).to_string())

    print('\n' + '=' * 60)
    print('Trading CumRet %')
    print('=' * 60)
    print(cumret.round(1).to_string())

    print('\n' + '=' * 60)
    print('Trading MaxDD %')
    print('=' * 60)
    print(maxdd.round(1).to_string())

    os.makedirs('results/aggregated', exist_ok=True)
    mse.to_csv('results/aggregated/verify_mse.csv')
    sharpe.to_csv('results/aggregated/verify_sharpe.csv')
    print('\nSaved to results/aggregated/verify_mse.csv and verify_sharpe.csv')


if __name__ == '__main__':
    main()
