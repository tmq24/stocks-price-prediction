import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import prepare_data_for_model, FinancialDataset


def inverse_scale_close(values, y_scaler):
    """
    Inverse scale adjClose (1D array)
    """
    values = values.reshape(-1, 1)
    return y_scaler.inverse_transform(values).flatten()


def evaluate_model(
    model,
    test_df_normalized,
    y_scaler,
    window_size=5,
    batch_size=32,
    show_samples=5
):
    # === 1. Prepare data ===
    src_num, src_tm, tgt_num, tgt_tm, labels = prepare_data_for_model(
        test_df_normalized, window_size
    )

    test_dataset = FinancialDataset(src_num, src_tm, tgt_num, tgt_tm, labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, trues = [], []

    # === 2. Predict (SCALE SPACE) ===
    with torch.no_grad():
        for src_num, src_tm, tgt_num, tgt_tm, labels in test_loader:
            src_num, src_tm, tgt_num, tgt_tm = (
                src_num.to(device), src_tm.to(device),
                tgt_num.to(device), tgt_tm.to(device)
            )

            output = model(src_num, src_tm, tgt_num, tgt_tm)

            preds.append(output.cpu().numpy())
            trues.append(labels.numpy())

    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()

    # === 3. INVERSE SCALE → GIÁ GỐC ===
    preds_inv = inverse_scale_close(preds, y_scaler)
    trues_inv = inverse_scale_close(trues, y_scaler)

    # === 4. METRICS (GIÁ GỐC) ===
    mse = mean_squared_error(trues_inv, preds_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues_inv, preds_inv)

    print("\n========== ĐÁNH GIÁ DỰ ĐOÁN GIÁ adjClose ==========")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    # === 5. TRUE vs PRED ===
    print("\n--- True vs Pred (GIÁ GỐC) ---")
    for i in range(min(show_samples, len(preds_inv))):
        print(
            f"[{i}] "
            f"True: {trues_inv[i]:.2f} | "
            f"Pred: {preds_inv[i]:.2f} | "
            f"Error: {preds_inv[i] - trues_inv[i]:+.2f}"
        )

    return mse, rmse, mae, preds_inv, trues_inv