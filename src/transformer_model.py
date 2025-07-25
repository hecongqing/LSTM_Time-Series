import torch
import torch.nn as nn
import numpy as np


class RMSLELoss(nn.Module):
    """Root Mean Squared Logarithmic Error Loss"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.LeakyReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-6)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        
        # Feed-forward
        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.ln2(x)
        
        return x


class TransformerBikePredictor(nn.Module):
    """Transformer-based bike demand predictor for time series forecasting"""
    
    def __init__(
        self,
        numeric_features: int,
        categorical_features_dims: list,
        static_features_dims: list,
        embed_size: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        output_dim: int = 6,
        dropout: float = 0.1,
    ):
        super(TransformerBikePredictor, self).__init__()
        
        self.embed_size = embed_size
        self.numeric_features = numeric_features
        
        # Embedding layers for categorical time-varying features
        self.embedding_cov = nn.ModuleList([
            nn.Embedding(dim, embed_size - numeric_features) 
            for dim in categorical_features_dims
        ])
        
        # Embedding layers for static categorical features
        self.embedding_static = nn.ModuleList([
            nn.Embedding(dim, embed_size - numeric_features) 
            for dim in static_features_dims
        ])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout) 
            for _ in range(num_blocks)
        ])
        
        # Forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size * 4),
            nn.LeakyReLU(),
            nn.Linear(embed_size * 4, output_dim),
            nn.ReLU()  # Ensure non-negative predictions
        )
    
    def forward(self, x_numeric, x_category=None, x_static=None):
        """
        Args:
            x_numeric: (batch_size, seq_len, numeric_features)
            x_category: (batch_size, seq_len, categorical_features) - optional
            x_static: (batch_size, static_features) - optional
        """
        batch_size, seq_len, _ = x_numeric.shape
        
        # Start with numeric features
        x = x_numeric
        
        # Add categorical time-varying embeddings if provided
        if x_category is not None and len(self.embedding_cov) > 0:
            categorical_embeddings = []
            for i, embed_layer in enumerate(self.embedding_cov):
                if i < x_category.size(2):
                    categorical_embeddings.append(embed_layer(x_category[:, :, i]))
            if categorical_embeddings:
                categorical_cov_embeddings = torch.stack(categorical_embeddings).mean(dim=0)
                # Combine numeric and categorical features
                x = torch.cat((x_numeric, categorical_cov_embeddings), dim=-1)
        
        # Add static embeddings if provided
        if x_static is not None and len(self.embedding_static) > 0:
            static_embeddings = []
            for i, embed_layer in enumerate(self.embedding_static):
                if i < x_static.size(1):
                    static_embeddings.append(embed_layer(x_static[:, i]))
            if static_embeddings:
                categorical_static_embeddings = torch.stack(static_embeddings).mean(dim=0).unsqueeze(1)
                
                # Broadcast static embeddings across time dimension
                categorical_static_embeddings = categorical_static_embeddings.repeat(1, seq_len, 1)
                
                # If we have both categorical covariates and static features
                if x_category is not None and len(self.embedding_cov) > 0 and categorical_embeddings:
                    # Average the categorical embeddings
                    embed_out = (categorical_cov_embeddings + categorical_static_embeddings) / 2
                    x = torch.cat((x_numeric, embed_out), dim=-1)
                else:
                    x = torch.cat((x_numeric, categorical_static_embeddings), dim=-1)
        
        # Ensure the final dimension matches embed_size
        if x.shape[-1] != self.embed_size:
            # Pad or project to match embed_size
            diff = self.embed_size - x.shape[-1]
            if diff > 0:
                padding = torch.zeros(batch_size, seq_len, diff, device=x.device)
                x = torch.cat((x, padding), dim=-1)
            elif diff < 0:
                x = x[:, :, :self.embed_size]
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Aggregate over time dimension (mean pooling)
        x = x.mean(dim=1)
        
        # Generate final predictions
        x = self.forecast_head(x)
        
        return x


class SimplifiedTransformerBikePredictor(nn.Module):
    """Simplified Transformer for bike demand prediction with only numeric features"""
    
    def __init__(
        self,
        input_dim: int,
        embed_size: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        output_dim: int = 6,
        dropout: float = 0.1,
    ):
        super(SimplifiedTransformerBikePredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, embed_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout) 
            for _ in range(num_blocks)
        ])
        
        # Forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size * 4),
            nn.LeakyReLU(),
            nn.Linear(embed_size * 4, output_dim),
            nn.ReLU()  # Ensure non-negative predictions
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        """
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Aggregate over time dimension (mean pooling)
        x = x.mean(dim=1)
        
        # Generate final predictions
        x = self.forecast_head(x)
        
        return x