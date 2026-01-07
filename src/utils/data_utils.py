import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Technical features (from data.ipynb)
FEATURE_COLS = [
    # Returns
    'return_1d', 'return_5d', 'log_return',
    # Trend/Momentum
    'SMA_10', 'SMA_20', 'EMA_10', 'price_dev_SMA_20',
    'momentum_5', 'momentum_10',
    # RSI, MACD
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
    # Bollinger Bands
    'BB_upper', 'BB_lower', 'BB_middle', 'BB_width',
    # Volatility
    'rolling_std_5', 'rolling_std_20', 'ATR_14',
    # Volume
    'Volume_MA_5', 'Volume_ratio', 'OBV'
]


def load_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load processed stock data for a ticker."""
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


def split_by_date(df: pd.DataFrame,
                  train_end: str = '2021-12-31',
                  val_end: str = '2023-12-31') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by date boundaries."""
    df = df.sort_values('date').reset_index(drop=True)
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_df = df[df['date'] <= train_end_dt]
    val_df = df[(df['date'] > train_end_dt) & (df['date'] <= val_end_dt)]
    test_df = df[df['date'] > val_end_dt]
    
    return train_df, val_df, test_df

def prepare_sequences(df: pd.DataFrame,
                      window_size: int,
                      horizon: int,
                      return_dates: bool = False) -> Tuple:
    """
    Prepare sequences for prediction.
    Target: Log Close Price at t+horizon.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    feature_cols = [f for f in FEATURE_COLS if f in df.columns]
    
    # Target: Log Close Price at t+horizon
    df = df.copy()
    df['target'] = np.log(df['close'].shift(-horizon))
    df = df.dropna(subset=['target']).reset_index(drop=True)
    
    X, y, dates = [], [], []
    
    for i in range(len(df) - window_size + 1):
        X.append(df[feature_cols].iloc[i:i+window_size].values)
        y.append(df['target'].iloc[i+window_size-1])
        dates.append(df['date'].iloc[i+window_size-1])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    if return_dates:
        return X, y, np.array(dates), feature_cols
    return X, y, feature_cols


# DATASET CLASS
class StockDataset(Dataset):
    """PyTorch Dataset for stock price prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_benchmark_data(tickers: List[str],
                           window_size: int,
                           horizon: int,
                           train_end: str = '2021-12-31',
                           val_end: str = '2023-12-31') -> Dict:
    """
    Prepare data for benchmark.
    - Combines train/val across tickers
    - Keeps test separate per ticker
    - Scales features and target with StandardScaler
    """
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    test_per_ticker = {}
    feature_cols = None
    
    for ticker in tickers:
        df = load_stock_data(ticker)
        if df is None:
            print(f"Warning: No data for {ticker}")
            continue
        
        # Split by date
        train_df, val_df, test_df = split_by_date(df, train_end, val_end)
        
        # Train
        if len(train_df) >= window_size:
            X, y, cols = prepare_sequences(train_df, window_size, horizon)
            if len(X) > 0:
                all_X_train.append(X)
                all_y_train.append(y)
                feature_cols = cols
        
        # Val
        if len(val_df) >= window_size:
            X, y, _ = prepare_sequences(val_df, window_size, horizon)
            if len(X) > 0:
                all_X_val.append(X)
                all_y_val.append(y)
        
        # Test
        if len(test_df) >= window_size:
            X, y, dates, _ = prepare_sequences(test_df, window_size, horizon, return_dates=True)
            if len(X) > 0:
                test_per_ticker[ticker] = (X, y, dates)
    
    if not all_X_train:
        raise ValueError("No training data available")
    
    # Concatenate
    X_train = np.concatenate(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_val = np.concatenate(all_X_val) if all_X_val else np.array([]).reshape(0, window_size, len(feature_cols))
    y_val = np.concatenate(all_y_val) if all_y_val else np.array([])
    
    n_train, window, n_features = X_train.shape
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, window, n_features)
    
    if len(X_val) > 0:
        n_val = X_val.shape[0]
        X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, window, n_features)
    else:
        X_val_scaled = X_val
    
    # Scale target (Log Price)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten() if len(y_val) > 0 else y_val
    
    # Scale test data
    test_per_ticker_scaled = {}
    for ticker, (X_test, y_test, dates_test) in test_per_ticker.items():
        n_test = X_test.shape[0]
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, window, n_features)
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        test_per_ticker_scaled[ticker] = (
            X_test_scaled.astype(np.float32),
            y_test_scaled.astype(np.float32),
            dates_test,
            y_test.astype(np.float32)
        )
    
    return {
        'X_train': X_train_scaled.astype(np.float32),
        'X_val': X_val_scaled.astype(np.float32),
        'y_train': y_train_scaled.astype(np.float32),
        'y_val': y_val_scaled.astype(np.float32),
        'test_per_ticker': test_per_ticker_scaled,
        'feature_cols': feature_cols,
        'n_features': n_features,
        'scaler': scaler,
        'target_scaler': target_scaler
    }



