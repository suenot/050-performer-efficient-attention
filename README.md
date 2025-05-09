# Chapter 52: Performer — Efficient Attention with FAVOR+

This chapter explores **Performer**, a Transformer architecture that achieves linear time and space complexity through the FAVOR+ (Fast Attention Via positive Orthogonal Random features) mechanism. Unlike standard Transformers with O(L²) attention complexity, Performer scales linearly O(L), making it ideal for processing long financial time series.

<p align="center">
<img src="https://i.imgur.com/8KvZmWJ.png" width="70%">
</p>

## Contents

1. [Introduction to Performer](#introduction-to-performer)
    * [The Attention Bottleneck](#the-attention-bottleneck)
    * [Key Advantages](#key-advantages)
    * [Comparison with Other Efficient Attention Methods](#comparison-with-other-efficient-attention-methods)
2. [FAVOR+ Mechanism](#favor-mechanism)
    * [The Kernel Trick](#the-kernel-trick)
    * [Random Fourier Features](#random-fourier-features)
    * [Positive Random Features](#positive-random-features)
    * [Orthogonal Random Features](#orthogonal-random-features)
3. [Mathematical Foundation](#mathematical-foundation)
    * [Standard Attention Formulation](#standard-attention-formulation)
    * [Kernel Reformulation](#kernel-reformulation)
    * [Feature Map Approximation](#feature-map-approximation)
    * [Complexity Analysis](#complexity-analysis)
4. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Performer Architecture](#02-performer-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Financial Time Series Prediction](#04-financial-time-series-prediction)
    * [05: Backtesting Strategy](#05-backtesting-strategy)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Best Practices](#best-practices)
8. [Resources](#resources)

## Introduction to Performer

Performer is a Transformer architecture introduced by Google Research in 2020 that addresses the fundamental quadratic complexity limitation of standard Transformers. By using FAVOR+ (Fast Attention Via positive Orthogonal Random features), Performers approximate softmax attention with provable accuracy while maintaining linear complexity.

### The Attention Bottleneck

Standard Transformer attention has O(L²) complexity, where L is sequence length:

```
Standard Attention:
┌─────────────────────────────────────────────────────┐
│  Attention(Q, K, V) = softmax(QK^T / √d) · V        │
│                                                      │
│  A = softmax(QK^T / √d)   ← This matrix is L × L    │
│                                                      │
│  For L = 1000:  A has 1,000,000 elements            │
│  For L = 10000: A has 100,000,000 elements          │
└─────────────────────────────────────────────────────┘
```

This becomes prohibitive for long financial time series:
- **Tick data**: Millions of data points per day
- **Multi-asset portfolios**: Long lookback windows needed
- **High-frequency trading**: Real-time processing requirements

### Key Advantages

1. **Linear Complexity O(L)**
   - Scales to arbitrary sequence lengths
   - Memory-efficient for long time series
   - Enables processing of tick-level data

2. **Provable Approximation Guarantees**
   - Unbiased estimation of attention matrix
   - Uniform convergence properties
   - Low estimation variance

3. **Drop-in Replacement**
   - Compatible with existing Transformer architectures
   - Can be used with pre-trained models
   - Same API as standard attention

4. **Kernel Flexibility**
   - Not limited to softmax attention
   - Can use other kernel functions
   - Enables novel attention mechanisms

### Comparison with Other Efficient Attention Methods

| Method | Complexity | Exact/Approx | Strengths | Weaknesses |
|--------|------------|--------------|-----------|------------|
| **Performer** | O(L) | Approx | Provable bounds, general kernels | Random feature variance |
| Linformer | O(L) | Approx | Simple projection | Fixed sequence length |
| BigBird | O(L) | Exact (sparse) | Maintains exact attention | Hand-crafted sparsity |
| Reformer | O(L·log(L)) | Exact (LSH) | Reversible layers | Complex implementation |
| Flash Attention | O(L²) | Exact | IO-aware, fast | Still quadratic |
| Longformer | O(L) | Exact (sparse) | Sliding window | Limited global attention |

## FAVOR+ Mechanism

FAVOR+ (Fast Attention Via positive Orthogonal Random features) is the core innovation that enables linear attention in Performers.

### The Kernel Trick

The key insight is that softmax attention can be viewed as a kernel function:

```python
# Standard attention computes:
# A[i,j] = exp(q_i · k_j / √d) / Σ_l exp(q_i · k_l / √d)

# This is equivalent to a softmax kernel:
# K_SM(x, y) = exp(x · y)

# If we can approximate this kernel with feature maps φ:
# K_SM(x, y) ≈ φ(x)^T · φ(y)

# Then attention becomes:
# Attention ≈ D^(-1) · φ(Q) · (φ(K)^T · V)
#                          ↑
#              This can be computed in O(L·d²) instead of O(L²·d)
```

### Random Fourier Features

The approximation uses random Fourier features based on Bochner's theorem:

```
For shift-invariant kernels K(x, y) = K(x - y):

K(x, y) = E_ω[exp(iω^T(x - y))]
        = E_ω[cos(ω^T(x - y))] + i·E_ω[sin(ω^T(x - y))]

Feature map:
z(x) = [cos(ω_1^T x), sin(ω_1^T x), ..., cos(ω_m^T x), sin(ω_m^T x)]

Where ω_i ~ p(ω) (spectral density of kernel)
```

### Positive Random Features

Standard random Fourier features can produce negative values, leading to training instabilities. FAVOR+ uses **positive random features**:

```python
# Standard feature map (can be negative):
z_sin_cos(x) = exp(||x||²/2) · [cos(ω^T x), sin(ω^T x)]

# Positive feature map (always positive):
z_positive(x) = exp(-||x||²/2) · exp(ω^T x)

# Why positive matters:
# - Attention weights should be non-negative
# - Negative approximations cause instability
# - Positive features maintain softmax-like behavior
```

### Orthogonal Random Features

FAVOR+ further improves accuracy using orthogonal random features:

```python
# IID random features:
Ω_iid = [ω_1, ω_2, ..., ω_m]  where ω_i ~ N(0, I_d)

# Orthogonal random features:
Ω_orth = Q · S  where Q is orthonormal, S is diagonal scaling

# Benefits of orthogonality:
# - Lower variance in kernel approximation
# - Better coverage of feature space
# - Maintains unbiasedness
```

## Mathematical Foundation

### Standard Attention Formulation

Given queries Q, keys K, values V ∈ ℝ^(L×d):

```
Attention(Q, K, V) = softmax(QK^T / √d) · V

Where softmax is applied row-wise:
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

The attention matrix A ∈ ℝ^(L×L):
```
A = softmax(QK^T / √d)
A_ij = exp(q_i · k_j / √d) / Σ_l exp(q_i · k_l / √d)
```

### Kernel Reformulation

Express attention using kernel notation:
```
A = D^(-1) · exp(QK^T / √d)

Where D = diag(exp(QK^T / √d) · 1_L) is the normalization
```

The softmax kernel:
```
K_SM(q, k) = exp(q · k)

Can be decomposed using:
exp(q · k) = exp(||q||²/2) · exp(-||q-k||²/2) · exp(||k||²/2)
                                ↑
                        Gaussian kernel
```

### Feature Map Approximation

The FAVOR+ feature map φ: ℝ^d → ℝ^m:

```python
def favor_plus_feature_map(x, omega, scale=True):
    """
    FAVOR+ positive orthogonal random feature map.

    Args:
        x: Input vectors [batch, length, d]
        omega: Random features [m, d] (orthogonalized)
        scale: Whether to apply d^(-1/4) scaling

    Returns:
        Feature vectors [batch, length, m]
    """
    if scale:
        x = x / (x.shape[-1] ** 0.25)  # d^(-1/4) scaling

    # Project onto random features
    x_omega = x @ omega.T  # [batch, length, m]

    # Positive feature map: exp(-||x||²/2) · exp(ω^T x)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    phi = torch.exp(-x_norm_sq / 2 + x_omega)

    return phi / math.sqrt(omega.shape[0])  # Normalize by √m
```

### Complexity Analysis

**Standard Attention:**
```
1. Compute QK^T:           O(L² · d)
2. Apply softmax:          O(L²)
3. Multiply by V:          O(L² · d)
Total:                     O(L² · d)
Memory for A:              O(L²)
```

**Performer FAVOR+:**
```
1. Compute φ(Q), φ(K):     O(L · m · d)
2. Compute φ(K)^T · V:     O(L · m · d)  ← Key insight!
3. Compute φ(Q) · result:  O(L · m · d)
4. Normalize:              O(L · m)
Total:                     O(L · m · d)
Memory:                    O(L · m + m · d)
```

When m << L (typically m ~ O(d·log(d))):
- Time: O(L · d² · log(d)) vs O(L² · d)
- Memory: O(L · d · log(d)) vs O(L²)

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

def prepare_performer_data(
    symbols: List[str],
    lookback: int = 512,  # Performer can handle long sequences
    horizon: int = 24,
    features: List[str] = ['log_return', 'volume_change', 'volatility', 'rsi']
) -> Dict:
    """
    Prepare data for Performer training.

    Performer's linear complexity allows longer lookback windows
    compared to standard Transformers.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Number of historical time steps (can be large!)
        horizon: Prediction horizon
        features: Features to compute

    Returns:
        Dictionary with X (features), y (targets), and metadata
    """
    all_data = []

    for symbol in symbols:
        # Load data from Bybit or other source
        df = load_bybit_data(symbol)

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['log_return'].rolling(20).std()
        df['rsi'] = compute_rsi(df['close'], 14)

        # Additional features for long-term patterns
        df['ma_50'] = df['close'].rolling(50).mean() / df['close']
        df['ma_200'] = df['close'].rolling(200).mean() / df['close']

        all_data.append(df[features + ['log_return']].dropna())

    # Align all dataframes
    aligned_data = pd.concat(all_data, axis=1, keys=symbols)
    aligned_data = aligned_data.dropna()

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(aligned_data) - horizon):
        X.append(aligned_data.iloc[i-lookback:i].values)
        # Predict next horizon returns for first symbol
        y.append(aligned_data[symbols[0]]['log_return'].iloc[i:i+horizon].values)

    return {
        'X': np.array(X),
        'y': np.array(y),
        'symbols': symbols,
        'features': features,
        'lookback': lookback,
        'horizon': horizon
    }


class PerformerDataset(Dataset):
    """PyTorch Dataset for Performer training."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### 02: Performer Architecture

See [python/model.py](python/model.py) for complete implementation.

```python
# Core FAVOR+ attention implementation

class FAVORPlusAttention(nn.Module):
    """
    FAVOR+ attention mechanism for linear complexity.

    Key features:
    - Positive random features for stability
    - Orthogonal features for lower variance
    - Linear O(L) complexity
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_features: int = None,
        use_orthogonal: bool = True,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Default: m = d * log(d) features
        self.num_features = num_features or int(self.head_dim * np.log(self.head_dim + 1))

        self.use_orthogonal = use_orthogonal
        self.epsilon = epsilon

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Random features (re-sampled or fixed)
        self.register_buffer(
            'random_features',
            self._create_random_features()
        )

    def _create_random_features(self) -> torch.Tensor:
        """Create (orthogonal) random feature matrix."""
        if self.use_orthogonal:
            # Orthogonal random features via QR decomposition
            random_matrix = torch.randn(self.num_features, self.head_dim)
            q, _ = torch.linalg.qr(random_matrix.T)
            return q.T * math.sqrt(self.head_dim)
        else:
            # IID random features
            return torch.randn(self.num_features, self.head_dim)

    def _positive_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positive random feature map: φ(x) = exp(-||x||²/2) * exp(ω^T x)

        Args:
            x: Input [batch, heads, length, head_dim]

        Returns:
            Features [batch, heads, length, num_features]
        """
        # Scaling factor d^(-1/4)
        x = x / (self.head_dim ** 0.25)

        # Project onto random features
        x_omega = torch.einsum('bhld,md->bhlm', x, self.random_features)

        # Compute ||x||² / 2
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2

        # Positive feature map
        phi = torch.exp(x_omega - x_norm_sq)

        # Normalize by sqrt(m)
        return phi / math.sqrt(self.num_features)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with FAVOR+ attention.

        Args:
            x: Input [batch, length, d_model]
            causal: Whether to use causal (autoregressive) attention

        Returns:
            Output [batch, length, d_model]
        """
        batch, length, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, length, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply positive feature map
        q_prime = self._positive_feature_map(q)  # [batch, heads, length, features]
        k_prime = self._positive_feature_map(k)

        if causal:
            # Causal attention using prefix sums
            output = self._causal_attention(q_prime, k_prime, v)
        else:
            # Bidirectional attention
            output = self._bidirectional_attention(q_prime, k_prime, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, length, self.d_model)
        return self.out_proj(output)

    def _bidirectional_attention(
        self,
        q_prime: torch.Tensor,
        k_prime: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Bidirectional FAVOR+ attention.

        Instead of computing L×L attention matrix:
        1. Compute K'^T V: [batch, heads, features, head_dim]
        2. Compute Q' @ (K'^T V): [batch, heads, length, head_dim]
        3. Normalize by Q' @ K'^T @ 1
        """
        # K'^T @ V: aggregate values weighted by key features
        kv = torch.einsum('bhlm,bhld->bhmd', k_prime, v)

        # Q' @ (K'^T @ V): query the aggregated representation
        qkv = torch.einsum('bhlm,bhmd->bhld', q_prime, kv)

        # Normalization: Q' @ K'^T @ 1 = Q' @ (sum of K' over length)
        k_sum = k_prime.sum(dim=2)  # [batch, heads, features]
        normalizer = torch.einsum('bhlm,bhm->bhl', q_prime, k_sum)
        normalizer = normalizer.unsqueeze(-1) + self.epsilon

        return qkv / normalizer

    def _causal_attention(
        self,
        q_prime: torch.Tensor,
        k_prime: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Causal FAVOR+ attention using prefix sums.

        For autoregressive models, we need:
        output[t] = Σ_{s≤t} attention(q[t], k[s]) * v[s]

        This is computed efficiently using cumulative sums.
        """
        batch, heads, length, head_dim = v.shape
        features = q_prime.shape[-1]

        # Initialize prefix sums
        kv_prefix = torch.zeros(batch, heads, features, head_dim, device=v.device)
        k_prefix = torch.zeros(batch, heads, features, device=v.device)

        outputs = []

        for t in range(length):
            # Update prefix sums
            k_t = k_prime[:, :, t, :]  # [batch, heads, features]
            v_t = v[:, :, t, :]        # [batch, heads, head_dim]

            kv_prefix = kv_prefix + torch.einsum('bhm,bhd->bhmd', k_t, v_t)
            k_prefix = k_prefix + k_t

            # Compute output for position t
            q_t = q_prime[:, :, t, :]  # [batch, heads, features]

            qkv_t = torch.einsum('bhm,bhmd->bhd', q_t, kv_prefix)
            normalizer = torch.einsum('bhm,bhm->bh', q_t, k_prefix).unsqueeze(-1)
            normalizer = normalizer + self.epsilon

            outputs.append(qkv_t / normalizer)

        return torch.stack(outputs, dim=2)
```

### 03: Model Training

```python
# python/03_train_model.py

import torch
import torch.nn as nn
from performer import PerformerForecaster

# Model configuration for financial time series
config = {
    'd_model': 128,
    'num_heads': 8,
    'num_layers': 4,
    'd_ff': 512,
    'num_features': 64,      # Random features dimension
    'use_orthogonal': True,  # Use orthogonal random features
    'dropout': 0.1,
    'max_seq_len': 1024,     # Can handle long sequences!
    'prediction_horizon': 24
}

# Initialize model
model = PerformerForecaster(**config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Loss function: MSE for returns prediction
criterion = nn.MSELoss()

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_x)

        # Compute loss
        loss = criterion(predictions, batch_y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    # Validation
    val_loss = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
```

### 04: Financial Time Series Prediction

```python
# python/04_prediction.py

def predict_returns(
    model: PerformerForecaster,
    data: torch.Tensor,
    horizon: int = 24
) -> dict:
    """
    Predict future returns using Performer model.

    Args:
        model: Trained Performer model
        data: Input sequence [batch, seq_len, features]
        horizon: Prediction horizon

    Returns:
        Dictionary with predictions and confidence intervals
    """
    model.eval()

    with torch.no_grad():
        # Get predictions
        predictions = model(data)  # [batch, horizon]

        # Monte Carlo dropout for uncertainty estimation
        model.train()  # Enable dropout
        mc_predictions = []

        for _ in range(100):
            with torch.no_grad():
                mc_pred = model(data)
                mc_predictions.append(mc_pred)

        mc_predictions = torch.stack(mc_predictions)

        # Compute statistics
        mean_pred = mc_predictions.mean(dim=0)
        std_pred = mc_predictions.std(dim=0)

        # Confidence intervals (95%)
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred

    model.eval()

    return {
        'predictions': predictions.numpy(),
        'mean': mean_pred.numpy(),
        'std': std_pred.numpy(),
        'lower_95': lower_bound.numpy(),
        'upper_95': upper_bound.numpy()
    }


def visualize_predictions(predictions: dict, actual: np.ndarray, symbol: str):
    """Visualize predictions with confidence intervals."""
    import matplotlib.pyplot as plt

    horizon = len(predictions['mean'][0])
    x = np.arange(horizon)

    plt.figure(figsize=(12, 6))

    # Plot predictions
    plt.plot(x, predictions['mean'][0], 'b-', label='Predicted', linewidth=2)
    plt.fill_between(
        x,
        predictions['lower_95'][0],
        predictions['upper_95'][0],
        alpha=0.3,
        color='blue',
        label='95% CI'
    )

    # Plot actual
    plt.plot(x, actual[0], 'r--', label='Actual', linewidth=2)

    plt.xlabel('Time Steps')
    plt.ylabel('Log Returns')
    plt.title(f'{symbol} Return Predictions with Performer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 05: Backtesting Strategy

```python
# python/05_backtest.py

def backtest_performer_strategy(
    model: PerformerForecaster,
    test_data: DataLoader,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    position_sizing: str = 'confidence'  # 'fixed', 'confidence', 'kelly'
) -> dict:
    """
    Backtest Performer-based trading strategy.

    Args:
        model: Trained Performer model
        test_data: Test DataLoader
        initial_capital: Starting capital
        transaction_cost: Trading fee as fraction
        position_sizing: Method for determining position size

    Returns:
        Dictionary with backtest results
    """
    model.eval()

    capital = initial_capital
    position = 0.0
    positions_history = []
    returns_history = []
    capital_history = [capital]

    for batch_x, batch_y in test_data:
        with torch.no_grad():
            # Get predictions and uncertainty
            predictions = model(batch_x)

            # Use first prediction for trading signal
            pred_return = predictions[0, 0].item()
            actual_return = batch_y[0, 0].item()

            # Generate signal
            if pred_return > 0.001:  # Bullish threshold
                target_position = 1.0
            elif pred_return < -0.001:  # Bearish threshold
                target_position = -1.0
            else:
                target_position = 0.0

            # Apply position sizing based on confidence
            if position_sizing == 'confidence':
                # Scale by inverse of uncertainty
                confidence = 1.0 / (abs(pred_return) + 0.001)
                target_position *= min(1.0, confidence)

            # Calculate trading costs
            position_change = abs(target_position - position)
            costs = position_change * transaction_cost * capital

            # Update position
            position = target_position

            # Calculate PnL
            pnl = position * actual_return * capital - costs
            capital += pnl

            # Record history
            positions_history.append(position)
            returns_history.append(pnl / capital_history[-1])
            capital_history.append(capital)

    # Calculate metrics
    returns = np.array(returns_history)

    results = {
        'total_return': (capital - initial_capital) / initial_capital,
        'sharpe_ratio': np.sqrt(252) * returns.mean() / (returns.std() + 1e-8),
        'sortino_ratio': calculate_sortino(returns),
        'max_drawdown': calculate_max_drawdown(capital_history),
        'win_rate': (returns > 0).sum() / len(returns),
        'profit_factor': abs(returns[returns > 0].sum()) / (abs(returns[returns < 0].sum()) + 1e-8),
        'capital_history': capital_history,
        'returns_history': returns_history,
        'positions_history': positions_history
    }

    return results


def calculate_max_drawdown(capital_history: list) -> float:
    """Calculate maximum drawdown."""
    peak = capital_history[0]
    max_dd = 0

    for capital in capital_history:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        max_dd = max(max_dd, drawdown)

    return max_dd


def calculate_sortino(returns: np.ndarray, target: float = 0) -> float:
    """Calculate Sortino ratio."""
    excess = returns - target
    downside = returns[returns < target]
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-8
    return np.sqrt(252) * excess.mean() / downside_std
```

## Rust Implementation

See [rust_performer](rust_performer/) for complete Rust implementation using Bybit data.

```
rust_performer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client for Bybit
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset for training
│   ├── model/              # Performer architecture
│   │   ├── mod.rs
│   │   ├── config.rs       # Model configuration
│   │   ├── favor.rs        # FAVOR+ attention mechanism
│   │   ├── embedding.rs    # Positional embedding
│   │   └── performer.rs    # Complete model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download Bybit data
    ├── train.rs            # Train model
    └── backtest.rs         # Run backtest
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust_performer

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Train model
cargo run --example train -- --epochs 100 --batch-size 32 --seq-len 512

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── model.py                # Performer model implementation
├── data.py                 # Data loading and preprocessing
├── strategy.py             # Trading strategy and backtesting
├── example_usage.py        # Complete example
├── requirements.txt        # Dependencies
└── __init__.py            # Package initialization
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python example_usage.py

# Or use as library
python -c "from model import PerformerForecaster; print('Ready!')"
```

## Best Practices

### When to Use Performer

**Ideal use cases:**
- Long sequence modeling (tick data, order book)
- Memory-constrained environments
- Real-time inference requirements
- Multi-horizon forecasting

**Consider alternatives for:**
- Very short sequences (L < 100) - standard attention may be faster
- Tasks requiring exact attention patterns
- When interpretability of attention weights is crucial

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `num_features` | d·log(d) | Balance accuracy vs speed |
| `use_orthogonal` | True | Better accuracy, minimal overhead |
| `num_layers` | 4-6 | Deeper than standard for long sequences |
| `d_model` | 128-256 | Standard Transformer sizes work |
| `dropout` | 0.1-0.2 | Regularization for stability |

### Common Pitfalls

1. **Feature dimension too small**: Use at least d·log(d) features
2. **Not using orthogonal features**: Leads to higher variance
3. **Forgetting epsilon in normalization**: Causes division by zero
4. **Using causal attention for bidirectional tasks**: Unnecessary complexity

### Memory Comparison

For sequence length L = 4096, d = 256:

| Method | Attention Memory | Total Memory |
|--------|------------------|--------------|
| Standard | 64 MB | ~100 MB |
| Performer | 4 MB | ~40 MB |
| Savings | 16x | 2.5x |

## Resources

### Papers

- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — Original FAVOR+ paper (2020)
- [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) — Random Fourier features foundation
- [Orthogonal Random Features](https://arxiv.org/abs/1610.09072) — Orthogonal feature improvements
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) — Theoretical insights

### Implementations

- [Google Research Performer](https://github.com/google-research/google-research/tree/master/performer) — Official implementation
- [Hugging Face Performer](https://huggingface.co/docs/transformers/model_doc/performer) — HuggingFace integration
- [Phil Wang's Performer PyTorch](https://github.com/lucidrains/performer-pytorch) — Clean PyTorch implementation

### Related Chapters

- [Chapter 51: Linformer Long Sequences](../51_linformer_long_sequences) — Linear projection approach
- [Chapter 53: BigBird Sparse Attention](../53_bigbird_sparse_attention) — Sparse attention patterns
- [Chapter 54: Reformer LSH Attention](../54_reformer_lsh_attention) — Locality-sensitive hashing
- [Chapter 58: Flash Attention Trading](../58_flash_attention_trading) — IO-aware exact attention

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Transformer architecture fundamentals
- Linear algebra (matrix decomposition, kernel methods)
- Probability theory (random features, concentration bounds)
- PyTorch/Rust ML programming
