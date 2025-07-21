import torch
import torch.nn as nn


class LSTMBikePredictor(nn.Module):
    """Multi-step bike demand predictor using LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # Use last layer's hidden state
        out = h_n[-1]
        out = self.fc(out)
        return out