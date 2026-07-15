"""
Extra architecture baselines on the A2 (C1+C3, polarity) alpha features —
same alphas as the DA-LLM transformer, only the predictor differs.

Adds three models to the LSTM / N-BEATS comparison:
  - GRU      : recurrent net, trained through the same torch loop as LSTM
               (registered in MODEL_REGISTRY as 'gru').
  - RandomForest, XGBoost : tree ensembles trained on the *flattened* window
               (window_size * n_features features), since they cannot consume
               a sequence. They reuse the exact same prepare_fold_data slicing,
               scaling and 70/30 split as the torch models.

All five models therefore see identical A2 alphas, the identical train/test
split and the identical target scaling, so MSE and the trading metrics are
directly comparable. Reuses cached A2 alphas (data/alphas/A2_arch) -> NO LLM
calls. 10-seed ensemble (mean prediction) like the main results.

Output: results/arch_baselines_extra/
  - predictions.pkl   {(fold_idx, ticker, model_type): [FoldResult, ...seeds]}
  - fold_results.csv  per (ticker, model, seed) MSE/IC/DA
  - summary.csv       per-model mean MSE/Sharpe/CumRet/MaxDD (arch_compare format)
  - summary_per_ticker.csv

Run (NOT executed automatically):
    python experiments/paper_sentiment_swap/train_a2_extra_baselines.py
"""
import copy
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO, '.env'))
except ImportError:
    pass
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('extra_baselines')

TICKERS = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']
HORIZON = 1
OUT = os.path.join(_REPO, 'results', 'arch_baselines_extra')

TORCH_MODELS = ['gru']           # go through the standard torch fold runner
TREE_MODELS = ['rf', 'xgboost']  # flattened-window sklearn / xgboost path

# Trading simulation constants — mirror scripts/trading_three_way.py exactly so
# the numbers are comparable with arch_compare_A2.csv.
TC_BPS = 10
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Trading metrics (identical formulas to scripts/trading_three_way.py)
# ---------------------------------------------------------------------------
def simulate(pred_close, actual_close, tc_bps=TC_BPS):
    pred = np.asarray(pred_close, dtype=float)
    actual = np.asarray(actual_close, dtype=float)
    realized = np.diff(actual) / actual[:-1]
    pred_delta = pred[1:] - actual[:-1]
    pos = np.sign(pred_delta)
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - pos_prev)
    return pos * realized - (tc_bps / 10_000.0) * turnover


def trading_metrics(daily_ret):
    r = pd.Series(daily_ret).dropna()
    if len(r) < 2 or r.std() < 1e-10:
        return dict(sharpe=0.0, cum_ret_pct=0.0, max_dd_pct=0.0)
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))
    cum_ret = float((1 + r).prod() - 1) * 100
    wealth = (1 + r).cumprod()
    max_dd = float(((wealth - wealth.cummax()) / wealth.cummax()).min()) * 100
    return dict(sharpe=round(sharpe, 3),
                cum_ret_pct=round(cum_ret, 1),
                max_dd_pct=round(max_dd, 1))


# ---------------------------------------------------------------------------
# Tree-model fold runner (flattened window) — mirrors src/walkforward/fold_runner
# ---------------------------------------------------------------------------
def _make_tree_model(model_type, seed):
    if model_type == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300, max_depth=30,
            min_samples_split=10, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        )
    if model_type == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            objective='reg:squarederror',
            random_state=seed, n_jobs=-1,
        )
    raise ValueError(f"Unknown tree model: {model_type!r}")


def run_fold_tree(fold, ticker, model_type, seed, df_enriched, feature_cols, config):
    """One (fold, ticker, tree-model, seed). Returns a FoldResult, same shape as
    the torch path, so it flows through the same trading/MSE aggregation."""
    import scipy.stats as sp
    from src.utils.data_utils import prepare_fold_data
    from src.walkforward.fold_runner import FoldResult

    window_size = config['data'].get('window_size', 5)
    target_col = config['data'].get('target_col', 'close')

    np.random.seed(seed)

    fold_data = prepare_fold_data(
        df=df_enriched,
        train_start=fold.train_start, train_end=fold.train_end,
        val_start=fold.val_start, val_end=fold.val_end,
        test_start=fold.test_start, test_end=fold.test_end,
        feature_cols=feature_cols,
        window_size=window_size, horizon=HORIZON, target_col=target_col,
    )

    X_train, y_train = fold_data['X_train'], fold_data['y_train']
    X_val, y_val = fold_data['X_val'], fold_data['y_val']
    X_test = fold_data['X_test']
    y_test_s = fold_data['y_test']
    y_test_raw = fold_data['y_test_raw']
    test_dates = fold_data['test_dates']
    target_scaler = fold_data['target_scaler']

    if len(X_test) == 0:
        return FoldResult(
            fold_idx=fold.fold_idx, ticker=ticker, model_type=model_type,
            horizon=HORIZON, seed=seed, test_dates=np.array([]),
            test_preds=np.array([]), test_targets=np.array([]),
        )

    # Flatten the window dimension: (n, window, n_features) -> (n, window*n_features).
    # Trees have no sequence prior, so concatenating the lagged steps is the
    # standard way to expose the same information the recurrent nets receive.
    def _flat(X):
        return X.reshape(X.shape[0], -1)

    # Fit on train+val combined (matches the scaler fit range in prepare_fold_data).
    if len(X_val) > 0:
        X_fit = np.concatenate([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
    else:
        X_fit, y_fit = X_train, y_train

    model = _make_tree_model(model_type, seed)
    model.fit(_flat(X_fit), y_fit)

    preds_scaled = model.predict(_flat(X_test)).astype(np.float32)
    preds_raw = target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()

    targets_raw = y_test_raw
    mse_scaled = float(np.mean((preds_scaled - y_test_s) ** 2))
    mse = float(np.mean((preds_raw - targets_raw) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds_raw - targets_raw)))
    da = (float(np.mean(np.sign(np.diff(preds_raw)) == np.sign(np.diff(targets_raw))))
          if len(preds_raw) >= 2 else 0.0)

    def _safe_corr(a, b):
        if len(a) < 2 or a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    ic = _safe_corr(preds_raw, targets_raw)
    rank_ic = float(sp.spearmanr(preds_raw, targets_raw).correlation) if len(preds_raw) >= 2 else 0.0

    return FoldResult(
        fold_idx=fold.fold_idx, ticker=ticker, model_type=model_type,
        horizon=HORIZON, seed=seed,
        test_dates=test_dates, test_preds=preds_raw, test_targets=targets_raw,
        mse=mse, mse_scaled=mse_scaled, rmse=rmse, mae=mae,
        directional_accuracy=da, pearson_ic=ic, spearman_rank_ic=rank_ic,
        n_train_sequences=len(X_fit),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = copy.deepcopy(yaml.safe_load(open(os.path.join(_REPO, 'config.yaml'))))
    cfg['data']['alphas_dir'] = 'data/alphas/A2_arch'   # reuse cached A2 alphas
    cfg['data']['tickers'] = TICKERS
    cfg['data']['horizons'] = [HORIZON]
    cfg['use_c2_feedback'] = False                       # A2 = C2 off
    # Only GRU needs a torch model config entry; trees are handled separately.
    cfg['models'] = {k: v for k, v in cfg['models'].items() if k in TORCH_MODELS}
    os.makedirs(OUT, exist_ok=True)

    seeds = cfg.get('training', {}).get('seeds', [42])

    from src.walkforward.wf_orchestrator import WalkForwardOrchestrator
    from src.walkforward.fold_runner import run_fold

    orch = WalkForwardOrchestrator(cfg, logger_=logging.getLogger('arch'))

    # Single 70/30 fold + enriched A2 dataframes (cached alphas, no LLM call).
    folds = orch._generate_single_fold(TICKERS)
    if not folds:
        raise RuntimeError("Could not generate the 70/30 fold — check price data.")
    fold = folds[0]

    enriched, feat_cols = {}, {}
    for ticker in TICKERS:
        df_e, cols = orch.prepare_enriched_dataframe(
            ticker=ticker, folds=folds,
            use_regime=orch._use_regime, use_sentiment=orch._use_sentiment,
            with_narrative=orch._with_narrative, use_c3_filter=orch._use_c3_filter,
        )
        enriched[ticker] = df_e
        feat_cols[ticker] = cols
        logger.info(f"[{ticker}] enriched A2 dataframe ready ({len(df_e)} rows).")

    # ------------------------------------------------------------------
    # Train every (model, ticker, seed)
    # ------------------------------------------------------------------
    results = []
    for ticker in TICKERS:
        for seed in seeds:
            for m in TORCH_MODELS:
                results.append(run_fold(
                    fold=fold, ticker=ticker, model_type=m, horizon=HORIZON,
                    seed=seed, df_enriched=enriched[ticker],
                    feature_cols=feat_cols[ticker], config=cfg,
                ))
            for m in TREE_MODELS:
                results.append(run_fold_tree(
                    fold=fold, ticker=ticker, model_type=m, seed=seed,
                    df_enriched=enriched[ticker], feature_cols=feat_cols[ticker],
                    config=cfg,
                ))
        logger.info(f"[{ticker}] done — all models × {len(seeds)} seeds.")

    # ------------------------------------------------------------------
    # Persist raw predictions + per-(ticker, model, seed) fold table
    # ------------------------------------------------------------------
    preds_store = defaultdict(list)
    for r in results:
        preds_store[(r.fold_idx, r.ticker, r.model_type)].append(r)
    with open(os.path.join(OUT, 'predictions.pkl'), 'wb') as f:
        pickle.dump(dict(preds_store), f)

    fold_rows = [dict(ticker=r.ticker, model=r.model_type, seed=r.seed,
                      mse=r.mse, mse_scaled=r.mse_scaled,
                      ic=r.pearson_ic, da=r.directional_accuracy)
                 for r in results if len(r.test_preds) > 0]
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, 'fold_results.csv'), index=False)

    # ------------------------------------------------------------------
    # Ensemble (mean over seeds) -> MSE + trading metrics per (model, ticker)
    # ------------------------------------------------------------------
    per_ticker_rows = []
    for (fold_idx, ticker, model_type), rs in preds_store.items():
        rs = [r for r in rs if len(r.test_preds) > 0]
        if not rs:
            continue
        pred_close = np.array([r.test_preds for r in rs]).mean(axis=0)
        actual_close = rs[0].test_targets
        # MSE in MinMax-scaled space (mean over seeds) — comparable to the paper's
        # Table 7 / arch_compare_A2.csv. Raw-price MSE is not comparable across
        # tickers (different price scales), so we report the scaled metric.
        mse = float(np.mean([r.mse_scaled for r in rs]))
        tm = trading_metrics(simulate(pred_close, actual_close))
        per_ticker_rows.append(dict(
            model=model_type, ticker=ticker, mse=mse,
            sharpe=tm['sharpe'], cum_ret_pct=tm['cum_ret_pct'],
            max_dd_pct=tm['max_dd_pct'],
        ))

    per_ticker = pd.DataFrame(per_ticker_rows)
    per_ticker.to_csv(os.path.join(OUT, 'summary_per_ticker.csv'), index=False)

    # Per-model means across tickers — same columns as arch_compare_A2.csv.
    summary = (per_ticker.groupby('model')
               .agg(MSE=('mse', 'mean'), Sharpe=('sharpe', 'mean'),
                    CumRet=('cum_ret_pct', 'mean'), MaxDD=('max_dd_pct', 'mean'))
               .reset_index())
    summary.to_csv(os.path.join(OUT, 'summary.csv'), index=False)

    print('\n=== Extra A2 baselines (mean over 5 tickers) ===')
    print(summary.to_string(index=False))
    print(f'\nwritten -> {OUT}')


if __name__ == '__main__':
    main()
