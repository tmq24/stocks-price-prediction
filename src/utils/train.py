import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..models.transformer_variants import create_transformer_model
from ..models.lstm import create_lstm_model
from ..models.gru import create_gru_model
from ..models.nbeats import create_nbeats_model


def save_model(model: nn.Module, path: str) -> None:
    """Save model state_dict to disk. Creates parent directories if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model_type: str,
               path: str,
               input_dim: int,
               window_size: int,
               horizon: int = 1,
               map_location: Optional[str] = None,
               **model_kwargs) -> nn.Module:
    """Re-create model architecture and load state_dict from disk."""
    model = create_model(
        model_type=model_type,
        input_dim=input_dim,
        window_size=window_size,
        horizon=horizon,
        **model_kwargs,
    )
    state_dict = torch.load(path, map_location=map_location or 'cpu', weights_only=True)
    model.load_state_dict(state_dict)
    return model


def save_training_history(history: Dict, path: str) -> None:
    """Persist per-epoch training metrics as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {k: [float(x) for x in v] for k, v in history.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def _transformer_factory(variant: str):
    return lambda input_dim, window_size, horizon, **kw: create_transformer_model(variant, input_dim, horizon, **kw)


MODEL_REGISTRY = {
    'transformer_encdec':  _transformer_factory('transformer_encdec'),
    'lstm':   lambda input_dim, window_size, horizon, **kw: create_lstm_model(input_dim, horizon, **kw),
    'gru':    lambda input_dim, window_size, horizon, **kw: create_gru_model(input_dim, horizon, **kw),
    'nbeats': create_nbeats_model,
}


def create_model(model_type: str,
                 input_dim: int,
                 window_size: int,
                 horizon: int = 1,
                 **kwargs) -> nn.Module:
    kwargs.pop('target_scaler', None)
    factory = MODEL_REGISTRY.get(model_type)
    if factory is None:
        raise ValueError(f"Unknown model type: {model_type!r}. Known: {list(MODEL_REGISTRY)}")
    return factory(input_dim, window_size, horizon, **kwargs)


# Metrics
def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - targets)))


def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def compute_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Pearson correlation."""
    if len(predictions) < 2:
        return 0.0
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()
    if pred_flat.std() < 1e-8 or tgt_flat.std() < 1e-8:
        return 0.0
    corr = np.corrcoef(pred_flat, tgt_flat)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_directional_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of samples where sign(pred) == sign(target)."""
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(np.sign(predictions) == np.sign(targets)))


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        'mae': compute_mae(predictions, targets),
        'mse': compute_mse(predictions, targets),
        'rmse': compute_rmse(predictions, targets),
        'directional_accuracy': compute_directional_accuracy(predictions, targets),
        'pearson_ic': compute_correlation(predictions, targets),
    }


def get_predictions(model: nn.Module,
                    loader: DataLoader,
                    device: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.
    Returns (predictions, targets) as numpy arrays.
    """
    if device == 'auto':
        device = _auto_device()
    model = model.to(device)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            if out.dim() > 1:
                out = out.squeeze(-1)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def train_single_model(model: nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       num_epochs: int = 100,
                       lr: float = 1e-3,
                       patience: int = 25,
                       device: str = 'auto',
                       verbose: bool = True,
                       seed: Optional[int] = None) -> Tuple[nn.Module, Dict]:
    """Train one model with MSE loss (paper-faithful)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device == 'auto':
        device = _auto_device()

    model = model.to(device)

    criterion = nn.MSELoss()

    # Paper §3.3: Adam with fixed lr=5e-5 over 50 epochs, no weight decay,
    # no LR scheduler. Val is used only for model selection (best checkpoint).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_corr': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        if val_loader is None:
            history['val_loss'].append(float('inf'))
            history['val_mae'].append(float('inf'))
            history['val_corr'].append(0.0)
            if verbose:
                print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} | no val")
            continue

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)

                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                val_preds.append(outputs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_mae = compute_mae(val_preds, val_targets)
        val_corr = compute_correlation(val_preds, val_targets)

        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr)

        if verbose:
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_loss:.4f} | "
                  f"V.Loss: {val_loss:.4f} | "
                  f"MAE: {val_mae:.4f} | "
                  f"Corr: {val_corr:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Paper §3.3: train full num_epochs (no early stopping). Val is used only
    # to select the best checkpoint at the end.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
