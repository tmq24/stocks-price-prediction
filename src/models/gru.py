import torch
import torch.nn as nn
from typing import Optional


class GRUModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 horizon: int = 1):
        super().__init__()

        self.model_name = 'gru'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.output_head = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor,
                h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            h0: Optional initial hidden state
        Returns:
            output: (batch, 1) - predicted price
        """
        batch_size = x.shape[0]

        if h0 is None:
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(self.num_layers * num_directions, batch_size,
                             self.hidden_dim, device=x.device)

        gru_out, hn = self.gru(x, h0)
        features = gru_out[:, -1, :]  # (batch, hidden_dim)

        output = self.output_head(features)  # (batch, 1)

        return output


def create_gru_model(input_dim: int, horizon: int = 1, **kwargs) -> GRUModel:
    default_config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': False,
    }
    default_config.update(kwargs)

    return GRUModel(
        input_dim=input_dim,
        hidden_dim=default_config['hidden_dim'],
        num_layers=default_config['num_layers'],
        dropout=default_config['dropout'],
        bidirectional=default_config['bidirectional'],
        horizon=horizon
    )
