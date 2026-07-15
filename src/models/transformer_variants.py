import math
import torch
import torch.nn as nn
from typing import Dict, List


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding - kept for legacy variants."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding via an embedding table."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Paper model: TransformerEncDec (Chen & Kawashima 2025, Figure 4, Table 4)
# ---------------------------------------------------------------------------

class TransformerEncDec(nn.Module):
    """
    Encoder-Decoder Transformer matching the paper architecture (§3.3, Table 4).

    Encoder: 5 alpha features -> 1D Conv Embedding -> Position Embedding -> 3-layer encoder
    Decoder: 1 close-price feature -> 1D Conv Embedding -> Position Embedding -> 2-layer decoder
    Output:  decoder's last token -> Linear -> scalar (next-day close price)

    Paper hyperparameters (Table 4):
        encoder_layers=3, decoder_layers=2, d_model=512, dim_feedforward=512,
        dropout=0.0, activation=GELU, window=5

    Temporal Embedding (paper §3.3): cyclic sin/cos features for day-of-week
    and month-of-year are projected to d_model via a small Linear and added to
    both encoder and decoder pre-attention representations.

    Input convention (see data_utils.METHOD_M_COLS):
        X: (batch, window, 10)  where
            X[:, :, :5]   = 5 alpha features -> encoder
            X[:, :, 5:6] = 1 close_feat col -> decoder
            X[:, :, 6:10] = 4 temporal sin/cos features -> both
    """

    N_ALPHA = 5    # encoder input channels
    N_CLOSE = 1    # decoder input channels
    N_TEMPORAL = 4 # temporal (dow_sin, dow_cos, month_sin, month_cos)

    def __init__(
        self,
        input_dim: int = 10,      # total cols in X (ignored; split hardcoded above)
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        horizon: int = 1,
        use_skip: bool = True,    # if False, pred = output_head(out); no close[t] residual
    ):
        super().__init__()

        self.model_name = 'transformer_encdec'
        self.d_model = d_model
        self.horizon = horizon
        self.use_skip = use_skip

        # 1D Conv Embedding for encoder (alpha features)
        # Conv1d: (batch, in_channels, seq_len) to (batch, d_model, seq_len)
        self.enc_conv = nn.Conv1d(
            in_channels=self.N_ALPHA,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
        )

        # 1D Conv Embedding for decoder (close price)
        self.dec_conv = nn.Conv1d(
            in_channels=self.N_CLOSE,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
        )

        # Temporal Embedding (paper §3.3): linear projection of cyclic sin/cos
        # date features to d_model, added to both encoder and decoder.
        self.temporal_proj = nn.Linear(self.N_TEMPORAL, d_model)

        # Learnable position encodings shared by encoder and decoder
        self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout)
        self.pos_dec = LearnablePositionalEncoding(d_model, dropout=dropout)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # Decoder (cross-attention with encoder memory)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Output projection: last decoder token to scalar (residual)
        # Skip connection: final pred = output_head(...) + close_feat[t]
        # Output head is initialized near-zero so the model starts as naive copy.
        self.output_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.use_skip:
            # Near-zero init so the skip-connection dominates at start
            nn.init.uniform_(self.output_head.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.output_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 10) - alphas / close_feat / temporal
        Returns:
            (batch, 1) - predicted next-day close price (in MinMax-scaled space)
        """
        # Split encoder, decoder, and temporal inputs
        x_enc  = x[:, :, :self.N_ALPHA]                                                 # (B, S, 5)
        x_dec  = x[:, :, self.N_ALPHA:self.N_ALPHA + self.N_CLOSE]                      # (B, S, 1)
        x_time = x[:, :, self.N_ALPHA + self.N_CLOSE:                                   # (B, S, 4)
                   self.N_ALPHA + self.N_CLOSE + self.N_TEMPORAL]

        # Conv embedding: Conv1d expects (batch, channels, seq_len)
        e = self.enc_conv(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # (B, S, d_model)
        d = self.dec_conv(x_dec.permute(0, 2, 1)).permute(0, 2, 1)  # (B, S, d_model)

        # Temporal embedding (added to both sides as in paper §3.3)
        t = self.temporal_proj(x_time)                              # (B, S, d_model)
        e = e + t
        d = d + t

        # Add positional encoding
        e = self.pos_enc(e)
        d = self.pos_dec(d)

        # Encode
        memory = self.encoder(e)   # (batch, seq, d_model)

        # Decode (cross-attention against encoder memory)
        out = self.decoder(d, memory)   # (batch, seq, d_model)

        if self.use_skip:
            close_t = x_dec[:, -1, :]                        # (B, 1) - scaled close[t]
            delta = self.output_head(out[:, -1, :])          # (B, 1) - learned correction
            pred = close_t + delta
        else:
            pred = self.output_head(out[:, -1, :])           # no skip - model must predict from scratch
        return pred


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_transformer_model(model_type: str, input_dim: int, horizon: int = 1, **kwargs) -> nn.Module:
    """Create a transformer variant by name."""
    if model_type == 'transformer_encdec':
        default = {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 2,
            'dim_feedforward': 512,
            'dropout': 0.0,
            'use_skip': True,
        }
        default.update(kwargs)
        return TransformerEncDec(
            input_dim=input_dim,
            d_model=default['d_model'],
            nhead=default['nhead'],
            num_encoder_layers=default['num_encoder_layers'],
            num_decoder_layers=default['num_decoder_layers'],
            dim_feedforward=default['dim_feedforward'],
            dropout=default['dropout'],
            use_skip=default['use_skip'],
            horizon=horizon,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
