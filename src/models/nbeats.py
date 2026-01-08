import torch
import torch.nn as nn
import numpy as np


class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, block_type: str = 'generic', 
                 n_layers: int = 4, n_neurons: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.block_type = block_type
        
        # FC Stack
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_size if i == 0 else n_neurons, n_neurons))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Theta projection
        layers.append(nn.Linear(n_neurons, theta_size))
        self.fc_stack = nn.Sequential(*layers)
        
        # Basis function parameters
        if block_type == 'generic':
            self.backcast_linear = nn.Linear(theta_size, input_size)
            self.forecast_linear = nn.Linear(theta_size, 1)
            
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch_size, input_size)
        Returns:
            backcast: (batch_size, input_size)
            forecast: (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Compute theta
        theta = self.fc_stack(x)  # (batch_size, theta_size)
        
        if self.block_type == 'trend':
            # Trend Basis: Polynomial
            t_backcast = torch.linspace(0, 1, self.input_size, device=x.device).unsqueeze(0)
            t_forecast = torch.tensor([1 + 1/self.input_size], device=x.device).unsqueeze(0)
            
            degree = self.theta_size
            backcast = torch.zeros((batch_size, self.input_size), device=x.device)
            forecast = torch.zeros((batch_size, 1), device=x.device)
            
            for p in range(degree):
                backcast += theta[:, p:p+1] * (t_backcast ** p)
                forecast += theta[:, p:p+1] * (t_forecast ** p)
                
        elif self.block_type == 'seasonality':
            # Seasonality Basis: Fourier
            t_backcast = torch.arange(self.input_size, device=x.device).float().unsqueeze(0)
            t_forecast = torch.tensor([self.input_size], device=x.device).float().unsqueeze(0)
            
            num_harmonics = self.theta_size // 2
            backcast = torch.zeros((batch_size, self.input_size), device=x.device)
            forecast = torch.zeros((batch_size, 1), device=x.device)
            
            P = float(self.input_size)
            
            for i in range(num_harmonics):
                weight_cos = theta[:, i:i+1]
                weight_sin = theta[:, i+num_harmonics:i+num_harmonics+1]
                
                freq = 2 * np.pi * (i + 1) / P
                
                backcast += weight_cos * torch.cos(freq * t_backcast) + weight_sin * torch.sin(freq * t_backcast)
                forecast += weight_cos * torch.cos(freq * t_forecast) + weight_sin * torch.sin(freq * t_forecast)
                
        else:  # Generic
            backcast = self.backcast_linear(theta)
            forecast = self.forecast_linear(theta)
            
        return backcast, forecast


class NBeatsStack(nn.Module):
    """
    Stack of N-BEATS blocks with residual connection.
    """
    def __init__(self, input_size: int, n_blocks: int = 3, n_layers: int = 4, 
                 n_neurons: int = 256, dropout: float = 0.1, theta_size: int = 8, 
                 block_type: str = 'generic'):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                theta_size=theta_size,
                block_type=block_type,
                n_layers=n_layers,
                n_neurons=n_neurons,
                dropout=dropout
            )
            for _ in range(n_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch_size, input_size)
        Returns:
            backcast: (batch_size, input_size)
            forecast: (batch_size, 1)
        """
        stack_forecast = 0
        stack_backcast = x
        
        for block in self.blocks:
            backcast, forecast = block(stack_backcast)
            stack_backcast = stack_backcast - backcast
            stack_forecast = stack_forecast + forecast
        
        return stack_backcast, stack_forecast


class NBeatsModel(nn.Module):
    """
    N-BEATS model for price prediction.
    
    Architecture:
    - Trend Stack -> Seasonality Stack -> Generic Stack
    - Single output for predicted price
    """
    def __init__(self, 
                 input_size: int,
                 horizon: int = 1,
                 # Trend stack params
                 trend_blocks: int = 2,
                 trend_layers: int = 4,
                 trend_neurons: int = 64,
                 trend_degree: int = 3,
                 # Seasonality stack params
                 seasonality_blocks: int = 2,
                 seasonality_layers: int = 4,
                 seasonality_neurons: int = 64,
                 seasonality_harmonics: int = 5,
                 # Generic stack params
                 generic_blocks: int = 2,
                 generic_layers: int = 4,
                 generic_neurons: int = 64,
                 generic_theta: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_name = 'nbeats'
        self.input_size = input_size
        self.horizon = horizon
        
        self.stacks = nn.ModuleList()
        
        # Trend Stack
        if trend_blocks > 0:
            self.stacks.append(NBeatsStack(
                input_size=input_size,
                n_blocks=trend_blocks,
                n_layers=trend_layers,
                n_neurons=trend_neurons,
                dropout=dropout,
                theta_size=trend_degree + 1,
                block_type='trend'
            ))
            
        # Seasonality Stack
        if seasonality_blocks > 0:
            self.stacks.append(NBeatsStack(
                input_size=input_size,
                n_blocks=seasonality_blocks,
                n_layers=seasonality_layers,
                n_neurons=seasonality_neurons,
                dropout=dropout,
                theta_size=seasonality_harmonics * 2,
                block_type='seasonality'
            ))
            
        # Generic Stack
        if generic_blocks > 0:
            self.stacks.append(NBeatsStack(
                input_size=input_size,
                n_blocks=generic_blocks,
                n_layers=generic_layers,
                n_neurons=generic_neurons,
                dropout=dropout,
                theta_size=generic_theta,
                block_type='generic'
            ))
        
        # Output projection
        self.output_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, n_features) or (batch, input_size)
        Returns:
            output: (batch, 1) - predicted price
        """
        # Flatten if needed
        if len(x.shape) == 3:
            batch_size, window_size, n_features = x.shape
            x = x.reshape(batch_size, window_size * n_features)
            
        # N-BEATS forward
        forecast = 0
        backcast = x
        
        for stack in self.stacks:
            backcast, stack_forecast = stack(backcast)
            forecast = forecast + stack_forecast
        
        # Output
        output = self.output_head(forecast)
        
        return output


def create_nbeats_model(input_dim: int, window_size: int, horizon: int = 1, **kwargs) -> NBeatsModel:
    """
    Factory function to create N-BEATS model.
    
    Args:
        input_dim: Number of input features
        window_size: Input window size
        horizon: Prediction horizon (for reference)
        **kwargs: Additional model parameters
    
    Returns:
        NBeatsModel instance
    """
    default_config = {
        'trend_blocks': 2,
        'trend_layers': 4,
        'trend_neurons': 64,
        'trend_degree': 3,
        'seasonality_blocks': 2,
        'seasonality_layers': 4,
        'seasonality_neurons': 64,
        'seasonality_harmonics': 5,
        'generic_blocks': 2,
        'generic_layers': 4,
        'generic_neurons': 64,
        'generic_theta': 8,
        'dropout': 0.1,
    }
    default_config.update(kwargs)
    
    input_size = window_size * input_dim
    
    return NBeatsModel(
        input_size=input_size,
        horizon=horizon,
        **default_config
    )
