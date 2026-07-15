import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PRICE_DIR = os.path.join(DATA_DIR, "price")

from ..features.technical import INDICATOR_COLS, compute_all_indicators

ALPHA_COLS = [f'alpha_{i}' for i in range(1, 6)]
TEMPORAL_COLS = ['tm_dow_sin', 'tm_dow_cos', 'tm_month_sin', 'tm_month_cos']

# DA-LLM encoder-decoder input (10 features total):
#   Encoder: 5 alpha values per timestep (alpha_1..alpha_5)
#   Decoder: close price history (1 feature, close_feat = today's close)
#   Temporal (added to both via TemporalEmbedding): 4 sin/cos features (dow, month)
METHOD_M_COLS = ALPHA_COLS + ['close_feat'] + TEMPORAL_COLS


def load_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load raw OHLCV CSV for a ticker, compute all technical indicators,
    and optionally filter to [start_date, end_date] (inclusive)."""
    file_path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    if compute_all_indicators is not None:
        df = compute_all_indicators(df)

    if start_date is not None:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def prepare_sequences(df: pd.DataFrame,
                      window_size: int,
                      horizon: int,
                      return_dates: bool = False,
                      feature_cols: Optional[List[str]] = None,
                      target_col: str = 'close') -> Tuple:
    """
    Prepare sequences for prediction.

    Args:
        feature_cols: Columns to use as features. Defaults to METHOD_M_COLS.
        target_col:   Column whose future value is the prediction target.
                      'close' to raw close price at t+horizon (paper §3 Algorithm 1)
                      'log_return' to log return at t+horizon (legacy thesis pipeline)

    Target: target_col shifted forward by horizon steps (value at t+horizon).
    Rows where the target is NaN are dropped (end of series).
    Rows where any feature is NaN are also dropped (start of series / warmup).
    """
    df = df.sort_values('date').reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [f for f in METHOD_M_COLS if f in df.columns]
    else:
        feature_cols = [f for f in feature_cols if f in df.columns]

    df = df.copy()

    if target_col != 'close':
        raise ValueError(f"Only target_col='close' is supported (got {target_col!r}).")
    df['_target'] = df['close'].shift(-horizon)

    df = df.dropna(subset=['_target'] + feature_cols).reset_index(drop=True)

    X, y, dates = [], [], []

    for i in range(len(df) - window_size + 1):
        window = df[feature_cols].iloc[i:i + window_size]
        if window.isna().any().any():
            continue
        X.append(window.values)
        y.append(df['_target'].iloc[i + window_size - 1])
        dates.append(df['date'].iloc[i + window_size - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    if return_dates:
        return X, y, np.array(dates), feature_cols
    return X, y, feature_cols


def prepare_fold_data(
    df: pd.DataFrame,
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
    feature_cols: List[str],
    window_size: int,
    horizon: int,
    target_col: str = 'close',
) -> Dict:
    """
    Slice df by row-index ranges, build sequences, fit MinMaxScaler on
    training rows only, and apply to val/test.

    Paper §3 Algorithm 1: scalers are fit on training set only and applied
    (transform only) to the test set.  MSE before inverse transform is what
    the paper reports (Table 6).

    Returns a dict with keys:
        X_train, y_train, X_val, y_val,
        X_test, y_test, test_dates, y_test_raw,
        feat_scaler, target_scaler,
        feature_cols, n_features
    """
    train_df = df.iloc[train_start:train_end].copy()
    val_df   = df.iloc[val_start:val_end].copy()
    test_df  = df.iloc[test_start:test_end].copy()

    X_train, y_train, _ = prepare_sequences(
        train_df, window_size, horizon, feature_cols=feature_cols, target_col=target_col
    )
    X_val, y_val, _ = prepare_sequences(
        val_df, window_size, horizon, feature_cols=feature_cols, target_col=target_col
    )
    X_test, y_test, test_dates, _ = prepare_sequences(
        test_df, window_size, horizon, return_dates=True,
        feature_cols=feature_cols, target_col=target_col
    )

    if len(X_train) == 0:
        raise ValueError(
            f"No training sequences in rows [{train_start},{train_end}). "
            "Check fold boundaries and window_size."
        )

    n_train, window, n_features = X_train.shape

    # Fit scalers on train+val combined so the test period (paper's 30%) stays
    # within the scaler range - matching paper §3 Algorithm 1 where "training"
    # covers the full 70% window that precedes the held-out test set.
    X_for_scale = np.concatenate([X_train, X_val]) if len(X_val) > 0 else X_train
    y_for_scale = np.concatenate([y_train, y_val]) if len(X_val) > 0 else y_train

    feat_scaler = MinMaxScaler()
    feat_scaler.fit(X_for_scale.reshape(-1, n_features))
    X_train_s = feat_scaler.transform(
        X_train.reshape(-1, n_features)
    ).reshape(n_train, window, n_features).astype(np.float32)

    def _scale_X(X):
        n = X.shape[0]
        return feat_scaler.transform(
            X.reshape(-1, n_features)
        ).reshape(n, window, n_features).astype(np.float32)

    X_val_s  = _scale_X(X_val)  if len(X_val)  > 0 else X_val.astype(np.float32)
    X_test_s = _scale_X(X_test) if len(X_test) > 0 else X_test.astype(np.float32)

    target_scaler = MinMaxScaler()
    target_scaler.fit(y_for_scale.reshape(-1, 1))
    y_train_s = target_scaler.transform(
        y_train.reshape(-1, 1)
    ).flatten().astype(np.float32)
    y_val_s = (
        target_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
        if len(y_val) > 0 else y_val.astype(np.float32)
    )
    y_test_s = (
        target_scaler.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)
        if len(y_test) > 0 else y_test.astype(np.float32)
    )

    return {
        'X_train': X_train_s,
        'y_train': y_train_s,
        'X_val':   X_val_s,
        'y_val':   y_val_s,
        'X_test':  X_test_s,
        'y_test':  y_test_s,
        'y_test_raw': y_test,           # unscaled targets for inverse-transform metrics
        'test_dates': np.array(test_dates),
        'feat_scaler':   feat_scaler,
        'target_scaler': target_scaler,
        'feature_cols':  feature_cols,
        'n_features':    n_features,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StockDataset(Dataset):
    """PyTorch Dataset for stock price prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

