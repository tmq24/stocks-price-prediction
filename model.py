import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

# ====================================================================
# PHẦN 1: CẤU TRÚC MÔ HÌNH TRANSFORMER (Theo sơ đồ)
# ====================================================================

class FinancialTransformer(nn.Module):
    def __init__(self,
                 d_model=512,          # Kích thước vector nhúng
                 window_size=5,        # Kích thước cửa sổ (SỬA: 20 → 5)
                 num_enc_features=5,   # Encoder: 5 formulaic alphas
                 num_dec_features=1,   # Decoder: Close price
                 nhead=8,              # Số lượng Head
                 num_layers=2,         # Số lớp Encoder/Decoder
                 dropout=0.1):
        super(FinancialTransformer, self).__init__()

        # --- Embedding Layer Khai báo ---

        # 1D Conv Embedding (Numeric features -> d_model)
        self.enc_conv_embedding = nn.Conv1d(num_enc_features, d_model, kernel_size=1)
        self.dec_conv_embedding = nn.Conv1d(num_dec_features, d_model, kernel_size=1)

        # Temporal Embedding (Day, Week, Month) - Giả định kích thước từ điển
        # Cần đảm bảo các giá trị đầu vào time index không vượt quá kích thước này
        self.day_emb = nn.Embedding(32, d_model)   # 31 ngày + 1
        self.week_emb = nn.Embedding(54, d_model)  # 53 tuần + 1
        self.month_emb = nn.Embedding(13, d_model) # 12 tháng + 1

        # Position Embedding (Learnable) - TÁCH RIÊNG CHO ENCODER VÀ DECODER
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model))
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model))

        # Layer Normalization
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

        # --- Transformer Core ---

        # Sử dụng nn.Transformer với batch_first=True
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )

        # --- Output Layer ---
        self.output_linear = nn.Linear(d_model, 1)
        self.output_dropout = nn.Dropout(dropout)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số theo Xavier/He initialization"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, src_numeric, src_time, tgt_numeric, tgt_time):

        # 1. ENCODER EMBEDDING (⊕ 1D Conv + Temporal + Position)
        # 1D Conv: (B, L, C) -> (B, C, L) -> Conv1D -> (B, d_model, L) -> (B, L, d_model)
        enc_conv_out = self.enc_conv_embedding(src_numeric.transpose(1, 2)).transpose(1, 2)

        # Temporal: (B, L, 3) -> (B, L, d_model)
        enc_temp_out = (self.day_emb(src_time[:, :, 0]) +
                        self.week_emb(src_time[:, :, 1]) +
                        self.month_emb(src_time[:, :, 2]))

        # Hợp nhất: (B, L, d_model)
        encoder_input = enc_conv_out + enc_temp_out + self.enc_pos_embedding
        encoder_input = self.enc_norm(encoder_input)  # Normalize

        # 2. DECODER EMBEDDING (⊕ 1D Conv + Temporal + Position)
        dec_conv_out = self.dec_conv_embedding(tgt_numeric.transpose(1, 2)).transpose(1, 2)

        dec_temp_out = (self.day_emb(tgt_time[:, :, 0]) +
                        self.week_emb(tgt_time[:, :, 1]) +
                        self.month_emb(tgt_time[:, :, 2]))

        # Hợp nhất: (B, L, d_model)
        decoder_input = dec_conv_out + dec_temp_out + self.dec_pos_embedding
        decoder_input = self.dec_norm(decoder_input)  # Normalize

        # 3. TẠO CAUSAL MASK CHO DECODER (NGĂN DATA LEAKAGE)
        tgt_len = decoder_input.size(1)
        device = decoder_input.device
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # 4. TRANSFORMER PASS
        transformer_out = self.transformer(
            src=encoder_input,
            tgt=decoder_input,
            tgt_mask=tgt_mask  # ← CRITICAL FIX
        )

        # 5. OUTPUT PREDICTION
        # Dự đoán từ token cuối cùng của Decoder
        last_step_output = transformer_out[:, -1, :]
        last_step_output = self.output_dropout(last_step_output)
        prediction = self.output_linear(last_step_output)

        return prediction


# ====================================================================
# PHẦN 2: PREPROCESSING VÀ DATASET (Áp dụng cho train_df của bạn)
# ====================================================================

class FinancialDataset(Dataset):
    def __init__(self, src_num, src_tm, tgt_num, tgt_tm, labels):
        self.src_num = src_num
        self.src_tm = src_tm
        self.tgt_num = tgt_num
        self.tgt_tm = tgt_tm
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.src_num[idx], self.src_tm[idx],
                self.tgt_num[idx], self.tgt_tm[idx],
                self.labels[idx])

# Hàm xử lý DataFrame đã cập nhật từ yêu cầu trước
def prepare_data_for_model(df, window_size=5):  # SỬA: 20 → 5

    ALPHA_COLS = [f'Alpha_{i}' for i in range(1, 6)]
    CLOSE_COL = 'adjClose'
    TICKER_COL = 'ticker'

    df = df.copy()  # Tránh modify df gốc
    df['date'] = pd.to_datetime(df['date'])
    # Explicitly cast to int và clip để tránh index out of range
    df['day_idx'] = (df['date'].dt.day - 1).astype(int).clip(0, 30)
    df['week_idx'] = (df['date'].dt.isocalendar().week - 1).astype(int).clip(0, 52)
    df['month_idx'] = (df['date'].dt.month - 1).astype(int).clip(0, 11)
    TIME_COLS = ['day_idx', 'week_idx', 'month_idx']

    all_src_numeric, all_src_time = [], []
    all_tgt_numeric, all_tgt_time = [], []
    all_labels = []

    grouped = df.groupby(TICKER_COL)

    for ticker, group in grouped:
        group = group.sort_values('date').reset_index(drop=True)

        alphas = group[ALPHA_COLS].values
        time_feats = group[TIME_COLS].values
        close_prices = group[[CLOSE_COL]].values

        num_samples = len(group) - window_size

        if num_samples <= 0:
            continue

        for i in range(num_samples):
            window_slice = slice(i, i + window_size)
            label_index = i + window_size

            # Encoder Data
            all_src_numeric.append(alphas[window_slice])
            all_src_time.append(time_feats[window_slice])

            # Decoder Data
            all_tgt_numeric.append(close_prices[window_slice])
            all_tgt_time.append(time_feats[window_slice])

            # Label
            all_labels.append(close_prices[label_index])

    src_num = torch.tensor(np.array(all_src_numeric), dtype=torch.float32)
    src_tm = torch.tensor(np.array(all_src_time), dtype=torch.long)
    tgt_num = torch.tensor(np.array(all_tgt_numeric), dtype=torch.float32)
    tgt_tm = torch.tensor(np.array(all_tgt_time), dtype=torch.long)
    labels = torch.tensor(np.array(all_labels), dtype=torch.float32)

    return src_num, src_tm, tgt_num, tgt_tm, labels


# ====================================================================
# PHẦN 3: VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) CƠ BẢN + VALIDATION
# ====================================================================

def train_model(model, train_loader, test_loader=None, num_epochs=50, learning_rate=1e-4):
    """
    Training loop với validation và learning rate scheduler

    Args:
        model: FinancialTransformer model
        train_loader: DataLoader cho training data
        test_loader: DataLoader cho test data (dùng làm validation)
        num_epochs: Số epochs (SỬA: 100 → 50)
        learning_rate: Learning rate ban đầu
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler (Cosine Annealing - đơn giản hơn OneCycleLR)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate*0.1)

    # Thiết lập thiết bị (CPU hoặc GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    print(f"Optimizer: AdamW với weight_decay=0.01")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Số parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        total_loss = 0

        for src_num, src_tm, tgt_num, tgt_tm, labels in train_loader:
            # Chuyển dữ liệu lên thiết bị
            src_num, src_tm, tgt_num, tgt_tm, labels = (
                src_num.to(device), src_tm.to(device),
                tgt_num.to(device), tgt_tm.to(device),
                labels.to(device)
            )

            # 1. Reset gradient
            optimizer.zero_grad()

            # 2. Forward pass
            predictions = model(src_num, src_tm, tgt_num, tgt_tm)

            # 3. Tính Loss
            loss = criterion(predictions, labels)

            # 4. Backward pass
            loss.backward()

            # 5. Gradient clipping (ngăn exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 6. Cập nhật tham số
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # === VALIDATION ===
        if test_loader is not None:
            model.eval()
            total_test_loss = 0

            with torch.no_grad():
                for src_num, src_tm, tgt_num, tgt_tm, labels in test_loader:
                    src_num, src_tm, tgt_num, tgt_tm, labels = (
                        src_num.to(device), src_tm.to(device),
                        tgt_num.to(device), tgt_tm.to(device),
                        labels.to(device)
                    )

                    predictions = model(src_num, src_tm, tgt_num, tgt_tm)
                    loss = criterion(predictions, labels)
                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)

            # Lưu best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), 'best_model.pth')
                marker = " ⭐"
            else:
                marker = ""

            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss: {avg_train_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Update learning rate
        scheduler.step()

    print("=" * 70)
    print("Huấn luyện hoàn tất!")
    if test_loader is not None:
        print(f"Best test loss: {best_test_loss:.4f}")
        print(f"Best test RMSE: {np.sqrt(best_test_loss):.4f}")
        print("Best model đã được lưu tại: best_model.pth")

    return model