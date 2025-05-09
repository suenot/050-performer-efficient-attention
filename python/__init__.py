"""
Performer: Efficient Attention with FAVOR+

This module provides a PyTorch implementation of the Performer architecture
with FAVOR+ (Fast Attention Via positive Orthogonal Random features) mechanism.

Key components:
- FAVORPlusAttention: Linear complexity attention mechanism
- PerformerEncoder: Transformer encoder with FAVOR+ attention
- PerformerForecaster: Complete model for financial time series prediction

Example:
    from performer import PerformerForecaster, PerformerConfig

    config = PerformerConfig(
        d_model=128,
        num_heads=8,
        num_layers=4,
        num_features=64
    )
    model = PerformerForecaster(config)

    # Input: [batch, seq_len, features]
    x = torch.randn(2, 512, 6)
    output = model(x)
"""

from .model import (
    FAVORPlusAttention,
    PerformerEncoder,
    PerformerEncoderLayer,
    PerformerForecaster,
    PerformerConfig,
)

from .data import (
    PerformerDataset,
    prepare_performer_data,
    load_bybit_data,
    compute_features,
)

from .strategy import (
    PerformerStrategy,
    backtest_performer_strategy,
    generate_signals,
)

__all__ = [
    # Model
    "FAVORPlusAttention",
    "PerformerEncoder",
    "PerformerEncoderLayer",
    "PerformerForecaster",
    "PerformerConfig",
    # Data
    "PerformerDataset",
    "prepare_performer_data",
    "load_bybit_data",
    "compute_features",
    # Strategy
    "PerformerStrategy",
    "backtest_performer_strategy",
    "generate_signals",
]

__version__ = "0.1.0"
