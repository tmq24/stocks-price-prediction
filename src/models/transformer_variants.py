import math
import torch
import torch.nn as nn
from typing import Dict, List


class PositionalEncoding(nn.Module):
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


class TransformerEncoderOnly(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 horizon: int = 1):
        super().__init__()
        
        self.model_name = 'transformer_encoder'
        self.input_dim = input_dim
        self.d_model = d_model
        self.horizon = horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head with more capacity
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        # Initialize output layer with small weights
        for m in self.output_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input mean for residual
        input_mean = x.mean(dim=(1, 2), keepdim=True).squeeze(-1)
        
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Encoder forward
        enc_out = self.encoder(x)  # (batch, seq_len, d_model)
        
        # Use last token as feature
        features = enc_out[:, -1, :]  # (batch, d_model)
        
        # Predict price
        output = self.output_head(features)  # (batch, 1)
        
        return output


class TransformerDecoderOnly(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 horizon: int = 1):
        super().__init__()
        
        self.model_name = 'transformer_decoder'
        self.input_dim = input_dim
        self.d_model = d_model
        self.horizon = horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer decoder (used as decoder-only with causal mask)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Output head with more capacity
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        for m in self.output_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # Decoder forward with causal mask
        dec_out = self.decoder(x, mask=causal_mask)  # (batch, seq_len, d_model)
        
        # Use last token as feature
        features = dec_out[:, -1, :]  # (batch, d_model)
        
        # Predict price
        output = self.output_head(features)  # (batch, 1)
        
        return output


class VanillaTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 horizon: int = 1):
        super().__init__()
        
        self.model_name = 'vanilla_transformer'
        self.input_dim = input_dim
        self.d_model = d_model
        self.horizon = horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Learnable query token for decoder
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # Output head with more capacity
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        for m in self.output_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, 1) - predicted price
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Prepare decoder input (learnable query token)
        tgt = self.query_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        
        # Transformer forward
        out = self.transformer(x, tgt)  # (batch, 1, d_model)
        
        # Output
        features = out.squeeze(1)  # (batch, d_model)
        output = self.output_head(features)  # (batch, 1)
        
        return output


def create_transformer_model(model_type: str, input_dim: int, horizon: int = 1, **kwargs) -> nn.Module:
    """
    Factory function to create transformer variants.
    
    Args:
        model_type: 'transformer_encoder', 'transformer_decoder', or 'vanilla_transformer'
        input_dim: Number of input features
        horizon: Prediction horizon (for reference, actual horizon handled by data)
        **kwargs: Additional model parameters
    
    Returns:
        Model instance
    """
    default_config = {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.05,
    }
    default_config.update(kwargs)
    
    if model_type == 'transformer_encoder':
        return TransformerEncoderOnly(
            input_dim=input_dim,
            d_model=default_config['d_model'],
            nhead=default_config['nhead'],
            num_layers=default_config['num_layers'],
            dim_feedforward=default_config['dim_feedforward'],
            dropout=default_config['dropout'],
            horizon=horizon
        )
    elif model_type == 'transformer_decoder':
        return TransformerDecoderOnly(
            input_dim=input_dim,
            d_model=default_config['d_model'],
            nhead=default_config['nhead'],
            num_layers=default_config['num_layers'],
            dim_feedforward=default_config['dim_feedforward'],
            dropout=default_config['dropout'],
            horizon=horizon
        )
    elif model_type == 'vanilla_transformer':
        return VanillaTransformer(
            input_dim=input_dim,
            d_model=default_config['d_model'],
            nhead=default_config['nhead'],
            num_encoder_layers=default_config.get('num_encoder_layers', default_config['num_layers']),
            num_decoder_layers=default_config.get('num_decoder_layers', default_config['num_layers']),
            dim_feedforward=default_config['dim_feedforward'],
            dropout=default_config['dropout'],
            horizon=horizon
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

