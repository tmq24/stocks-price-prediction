"""
Atomic experiment unit: run one (fold, ticker, model_type, horizon, seed).
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader

from ..utils.data_utils import prepare_fold_data, StockDataset
from ..utils.train import (
    create_model,
    train_single_model,
    get_predictions,
    save_model,
    save_training_history,
)
from ..walkforward.fold_generator import Fold

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_idx: int
    ticker: str
    model_type: str
    horizon: int
    seed: int

    # Raw predictions / targets in log_return space (unscaled)
    test_dates: np.ndarray
    test_preds: np.ndarray       # inverse-transformed log_return predictions
    test_targets: np.ndarray     # actual log_returns

    # Metrics
    mse: float = 0.0
    mse_scaled: float = 0.0   # MSE in MinMax-scaled space (comparable to paper Table 6)
    rmse: float = 0.0
    mae: float = 0.0
    directional_accuracy: float = 0.0
    pearson_ic: float = 0.0
    spearman_rank_ic: float = 0.0

    # Training diagnostics
    training_history: Dict = field(default_factory=dict)
    n_train_sequences: int = 0
    n_dropped_alpha_rows: int = 0


def run_fold(
    fold: Fold,
    ticker: str,
    model_type: str,
    horizon: int,
    seed: int,
    df_enriched,
    feature_cols: List[str],
    config: dict,
    save_dir: Optional[str] = None,
) -> FoldResult:
    """
    Execute one walk-forward fold.

    Steps:
    1. Slice df_enriched by fold boundaries
    2. prepare_sequences on each split
    3. Fit StandardScaler on train rows only to transform val/test
    4. Fit target scaler on train y only
    5. Build DataLoaders
    6. create_model with config hyperparameters
    7. train_single_model (MSE loss, given seed)
    8. get_predictions on test split
    9. Inverse-transform predictions to compute metrics
    """
    import scipy.stats as sp
    import torch

    training_cfg = config.get('training', {})
    model_cfg = config.get('models', {}).get(model_type, {})

    batch_size = training_cfg.get('batch_size', 32)
    num_epochs = training_cfg.get('num_epochs', 100)
    lr = training_cfg.get('lr', 5e-4)
    patience = training_cfg.get('patience', 20)
    window_size = config.get('data', {}).get('window_size', 20)
    target_col = config.get('data', {}).get('target_col', 'close')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1. Prepare fold data (slicing, sequencing, scaling)
    # ------------------------------------------------------------------
    try:
        fold_data = prepare_fold_data(
            df=df_enriched,
            train_start=fold.train_start,
            train_end=fold.train_end,
            val_start=fold.val_start,
            val_end=fold.val_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            feature_cols=feature_cols,
            window_size=window_size,
            horizon=horizon,
            target_col=target_col,
        )
    except ValueError as e:
        logger.error(f"[{ticker}] fold {fold.fold_idx} data prep failed: {e}")
        return FoldResult(
            fold_idx=fold.fold_idx, ticker=ticker,
            model_type=model_type, horizon=horizon, seed=seed,
            test_dates=np.array([]), test_preds=np.array([]),
            test_targets=np.array([]),
        )

    X_train = fold_data['X_train']
    y_train = fold_data['y_train']
    X_val = fold_data['X_val']
    y_val = fold_data['y_val']
    X_test = fold_data['X_test']
    y_test_s = fold_data['y_test']
    y_test_raw = fold_data['y_test_raw']
    test_dates = fold_data['test_dates']
    target_scaler = fold_data['target_scaler']
    n_features = fold_data['n_features']

    if len(X_test) == 0:
        logger.warning(f"[{ticker}] fold {fold.fold_idx} has no test sequences.")
        return FoldResult(
            fold_idx=fold.fold_idx, ticker=ticker,
            model_type=model_type, horizon=horizon, seed=seed,
            test_dates=np.array([]), test_preds=np.array([]),
            test_targets=np.array([]),
        )

    # ------------------------------------------------------------------
    # 2. DataLoaders
    # ------------------------------------------------------------------
    train_ds = StockDataset(X_train, y_train)
    test_ds = StockDataset(X_test, y_test_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Paper §3: no validation set. If val is empty, skip val loop entirely so
    # the trained model is the final-epoch state (not best-on-val).
    if len(X_val) > 0:
        val_ds = StockDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = create_model(
        model_type=model_type,
        input_dim=n_features,
        window_size=window_size,
        horizon=horizon,
        **model_cfg,
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    model, history = train_single_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        patience=patience,
        seed=seed,
        verbose=False,
    )

    # Persist state_dict + per-epoch history when caller opts in
    if save_dir is not None:
        ticker_dir = os.path.join(save_dir, ticker)
        pth_path = os.path.join(ticker_dir, f"seed_{seed}.pth")
        hist_path = os.path.join(ticker_dir, f"seed_{seed}_history.json")
        save_model(model, pth_path)
        save_training_history(history, hist_path)
        logger.info(f"[{ticker}] seed={seed} saved -> {pth_path}")

    # ------------------------------------------------------------------
    # 5. Predict and inverse-transform
    # ------------------------------------------------------------------
    preds_scaled, _ = get_predictions(model, test_loader)
    preds_raw = target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()

    # ------------------------------------------------------------------
    # 6. Metrics
    # ------------------------------------------------------------------
    targets_raw = y_test_raw
    mse_scaled = float(np.mean((preds_scaled - y_test_s) ** 2))

    def _safe_corr(a, b):
        if len(a) < 2 or a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    mse = float(np.mean((preds_raw - targets_raw) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds_raw - targets_raw)))
    da = (float(np.mean(np.sign(np.diff(preds_raw)) == np.sign(np.diff(targets_raw))))
          if len(preds_raw) >= 2 else 0.0)
    ic = _safe_corr(preds_raw, targets_raw)
    rank_ic = float(sp.spearmanr(preds_raw, targets_raw).correlation) if len(preds_raw) >= 2 else 0.0

    logger.debug(
        f"[{ticker}] fold={fold.fold_idx} {model_type} h={horizon} "
        f"seed={seed} | MSE={mse:.6f} DA={da:.3f} IC={ic:.3f}"
    )

    return FoldResult(
        fold_idx=fold.fold_idx,
        ticker=ticker,
        model_type=model_type,
        horizon=horizon,
        seed=seed,
        test_dates=test_dates,
        test_preds=preds_raw,
        test_targets=targets_raw,
        mse=mse,
        mse_scaled=mse_scaled,
        rmse=rmse,
        mae=mae,
        directional_accuracy=da,
        pearson_ic=ic,
        spearman_rank_ic=rank_ic,
        training_history=history,
        n_train_sequences=len(X_train),
    )
