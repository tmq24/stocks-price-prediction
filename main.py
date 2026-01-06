import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from features import add_features
from alpha_generator import GeminiAlphaGenerator, calculate_generated_alphas_dynamic
from scaling import full_scale_pipeline
from model import FinancialTransformer, FinancialDataset, prepare_data_for_model, train_model
from evaluate import evaluate_model
from plot import plot_predictions_real
from torch.utils.data import DataLoader
def main():
    # ============================================================
    # 1. Đọc dữ liệu & Chia Hybrid Train/Test
    # ============================================================
    df = pd.read_csv("https://drive.google.com/uc?export=download&id=11jFbimgU65UG6F-QPzfoSoDR8b1oPrUh")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Hybrid split: mỗi công ty train trước 2023, test sau 2023
    train_list, test_list = [], []
    for ticker in df['ticker'].unique():
        sub = df[df['ticker'] == ticker]
        train = sub[sub['date'] < '2023-01-01']
        test = sub[sub['date'] >= '2023-01-01']
        train_list.append(train)
        test_list.append(test)
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    print(f"Hybrid split done: {df['ticker'].nunique()} companies")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # ============================================================
    # 2. Tạo đặc trưng kỹ thuật
    # ============================================================
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    print("Features added:", train_df.columns.tolist())

    # ============================================================
    # 3. Tạo và áp dụng Alpha bằng Gemini
    # ============================================================
    gemini_gen = GeminiAlphaGenerator(os.getenv('GEMINI_API_KEY'))
    features = list(train_df.columns)
    alphas = gemini_gen.generate_alpha_formulas(features, sample_df=train_df)

    if alphas:
        print("\nÁp dụng các Alpha lên dữ liệu...")
        train_final = train_df.groupby('ticker').apply(lambda g: calculate_generated_alphas_dynamic(g, alphas)).reset_index(drop=True)
        test_final = test_df.groupby('ticker').apply(lambda g: calculate_generated_alphas_dynamic(g, alphas)).reset_index(drop=True)

        train_final.fillna(train_final.mean(numeric_only=True), inplace=True)
        test_final.fillna(test_final.mean(numeric_only=True), inplace=True)

        train_df = train_final
        test_df = test_final

        print("\n✓ HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print("DataFrame giờ có các Alpha mới từ Gemini dựa trên technical indicators.")
    else:
        print("✗ Không tạo được Alpha từ Gemini.")
        return  # Thoát nếu không có alpha

    # ============================================================
    # 4. Scaling pipeline
    # ============================================================
    df_train_normalized, df_test_normalized, feature_cols, x_scaler, y_scaler = full_scale_pipeline(train_df, test_df)

    # ============================================================
    # 5. Chuẩn bị dữ liệu cho model
    # ============================================================
    print("=" * 70)
    print("FINANCIAL TRANSFORMER - TRAINING PIPELINE")
    print("=" * 70)

    print("\n📂 Bước 1: Chuẩn bị dữ liệu...")
    window_size = 5
    src_num_train, src_tm_train, tgt_num_train, tgt_tm_train, labels_train = prepare_data_for_model(df_train_normalized, window_size)
    src_num_test, src_tm_test, tgt_num_test, tgt_tm_test, labels_test = prepare_data_for_model(df_test_normalized, window_size)

    print(f"✓ Training samples: {len(labels_train):,}")
    print(f"✓ Test samples: {len(labels_test):,}")
    print(f"✓ Window size: {window_size} days")

    # ============================================================
    # 6. Tạo DataLoader
    # ============================================================
    print("\n📦 Bước 2: Tạo DataLoader...")
    batch_size = 32
    train_dataset = FinancialDataset(src_num_train, src_tm_train, tgt_num_train, tgt_tm_train, labels_train)
    test_dataset = FinancialDataset(src_num_test, src_tm_test, tgt_num_test, tgt_tm_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # ============================================================
    # 7. Khởi tạo model
    # ============================================================
    print("\n🧠 Bước 3: Khởi tạo Model...")
    model = FinancialTransformer(
        d_model=512,
        window_size=window_size,
        num_enc_features=5,
        num_dec_features=1,
        nhead=8,
        num_layers=2,
        dropout=0.1
    )
    print(f"✓ Model architecture: Transformer Encoder-Decoder")
    print(f"✓ d_model=512, nhead=8, num_layers=2")

    # ============================================================
    # 8. Huấn luyện model
    # ============================================================
    print("\n🔥 Bước 4: Bắt đầu Training với Validation...")
    trained_model = train_model(
        model,
        train_loader,
        test_loader=test_loader,
        num_epochs=50,
        learning_rate=1e-4
    )
    print("✅ PIPELINE HOÀN THÀNH!")

    # ============================================================
    # 9. Đánh giá model
    # ============================================================
    mse, rmse, mae, preds, trues = evaluate_model(
        model=trained_model,
        test_df_normalized=df_test_normalized,
        y_scaler=y_scaler,
        window_size=window_size,
        batch_size=batch_size,
        show_samples=30
    )

    # ============================================================
    # 10. Vẽ biểu đồ
    # ============================================================
    plot_predictions_real(
        preds=preds,
        trues=trues,
        df_test_raw=test_df,
        window_size=window_size
    )

if __name__ == "__main__":
    main()