"""
Performer Model Implementation in PyTorch

Provides:
- PerformerConfig: Model configuration
- FAVORPlusAttention: FAVOR+ linear attention mechanism
- PerformerEncoder: Transformer encoder with FAVOR+
- PerformerForecaster: Complete forecasting model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class OutputType(Enum):
    """Type of model output"""
    REGRESSION = "regression"
    DIRECTION = "direction"
    PORTFOLIO = "portfolio"
    QUANTILE = "quantile"


@dataclass
class PerformerConfig:
    """
    Configuration for Performer model

    Example:
        config = PerformerConfig(
            d_model=128,
            num_heads=8,
            num_layers=4
        )
    """
    # Architecture
    input_features: int = 6
    d_model: int = 128
    num_heads: int = 8
    d_ff: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048

    # FAVOR+ attention
    num_features: int = None  # If None, uses d_head * log(d_head)
    use_orthogonal: bool = True
    redraw_features: bool = False  # Redraw random features each forward pass
    epsilon: float = 1e-6

    # Output
    output_type: OutputType = OutputType.REGRESSION
    prediction_horizon: int = 24
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Embedding
    use_positional_encoding: bool = True
    kernel_size: int = 3

    def validate(self):
        """Validate configuration"""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.kernel_size % 2 == 1, "kernel_size must be odd"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def computed_num_features(self) -> int:
        """Compute number of random features if not specified"""
        if self.num_features is not None:
            return self.num_features
        # Default: d_head * log(d_head)
        return max(16, int(self.head_dim * math.log(self.head_dim + 1)))


class FAVORPlusAttention(nn.Module):
    """
    FAVOR+ (Fast Attention Via positive Orthogonal Random features)

    Achieves linear O(L) complexity instead of quadratic O(L²)
    by approximating softmax attention with random feature maps.

    Key innovations:
    - Positive random features for numerical stability
    - Orthogonal features for lower variance
    - Unbiased approximation with theoretical guarantees

    Example:
        attn = FAVORPlusAttention(d_model=128, num_heads=8, num_features=64)
        output = attn(x)  # x: [batch, length, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_features: int = None,
        use_orthogonal: bool = True,
        redraw_features: bool = False,
        epsilon: float = 1e-6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_orthogonal = use_orthogonal
        self.redraw_features = redraw_features
        self.epsilon = epsilon

        # Number of random features
        if num_features is None:
            self.num_features = max(16, int(self.head_dim * math.log(self.head_dim + 1)))
        else:
            self.num_features = num_features

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Register random features buffer
        self.register_buffer(
            'random_features',
            self._create_random_features()
        )

    def _create_random_features(self) -> torch.Tensor:
        """
        Create (orthogonal) random feature matrix.

        For orthogonal features, we use Gram-Schmidt orthogonalization
        which provides lower variance than IID random features.
        """
        if self.use_orthogonal:
            # Create orthogonal random features via QR decomposition
            # Sample more rows than needed, then orthogonalize
            num_blocks = math.ceil(self.num_features / self.head_dim)
            random_matrix = torch.zeros(num_blocks * self.head_dim, self.head_dim)

            for i in range(num_blocks):
                block = torch.randn(self.head_dim, self.head_dim)
                q, _ = torch.linalg.qr(block)
                random_matrix[i * self.head_dim:(i + 1) * self.head_dim] = q

            # Scale by sqrt(d) and truncate to desired number of features
            random_matrix = random_matrix[:self.num_features] * math.sqrt(self.head_dim)
            return random_matrix
        else:
            # IID random features from N(0, 1)
            return torch.randn(self.num_features, self.head_dim)

    def _positive_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positive random feature map: phi(x) = exp(-||x||²/2) * exp(omega^T x)

        This ensures all feature values are positive, which is crucial for:
        1. Numerical stability during training
        2. Proper approximation of softmax attention (which is always positive)
        3. Preventing negative attention weights

        Args:
            x: Input tensor [batch, heads, length, head_dim]

        Returns:
            Feature tensor [batch, heads, length, num_features]
        """
        # Scaling factor d^(-1/4) for better numerical stability
        scaling = self.head_dim ** (-0.25)
        x_scaled = x * scaling

        # Project onto random features: omega^T x
        # x_scaled: [batch, heads, length, head_dim]
        # random_features: [num_features, head_dim]
        x_omega = torch.einsum('bhld,md->bhlm', x_scaled, self.random_features)

        # Compute ||x||² / 2
        x_norm_sq = (x_scaled ** 2).sum(dim=-1, keepdim=True) / 2

        # Positive feature map: exp(-||x||²/2 + omega^T x)
        # This is equivalent to: exp(-||x||²/2) * exp(omega^T x)
        phi = torch.exp(x_omega - x_norm_sq)

        # Normalize by sqrt(num_features) for proper approximation
        return phi / math.sqrt(self.num_features)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with FAVOR+ attention.

        The key insight is that instead of computing the full L×L attention matrix:
        Attention = softmax(QK^T) V

        We compute:
        Attention ≈ D^(-1) phi(Q) (phi(K)^T V)

        Where phi is the positive random feature map and D is normalization.
        This changes complexity from O(L²d) to O(Lmd) where m << L.

        Args:
            x: Input tensor [batch, length, d_model]
            causal: Whether to use causal (autoregressive) attention
            return_attention: Whether to return approximated attention weights

        Returns:
            output: [batch, length, d_model]
            attention: Optional approximated attention weights (expensive to compute)
        """
        batch, length, _ = x.shape

        # Optionally redraw random features (for ensemble-like behavior)
        if self.redraw_features and self.training:
            self.random_features.copy_(self._create_random_features())

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
        output = self.out_proj(output)
        output = self.dropout(output)

        # Optionally compute approximated attention weights (expensive!)
        attn_weights = None
        if return_attention:
            # A ≈ phi(Q) @ phi(K)^T (normalized)
            attn_weights = torch.einsum('bhlm,bhkm->bhlk', q_prime, k_prime)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + self.epsilon)

        return output, attn_weights

    def _bidirectional_attention(
        self,
        q_prime: torch.Tensor,
        k_prime: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Bidirectional FAVOR+ attention.

        Instead of computing the full L×L attention matrix:
        1. Compute K'^T V: [batch, heads, features, head_dim]
           This aggregates values weighted by key features
        2. Compute Q' @ (K'^T V): [batch, heads, length, head_dim]
           This queries the aggregated representation
        3. Normalize by Q' @ K'^T @ 1 = Q' @ (sum of K' over length)
        """
        # K'^T @ V: aggregate values weighted by key features
        # k_prime: [batch, heads, length, features]
        # v: [batch, heads, length, head_dim]
        kv = torch.einsum('bhlm,bhld->bhmd', k_prime, v)

        # Q' @ (K'^T @ V): query the aggregated representation
        # q_prime: [batch, heads, length, features]
        # kv: [batch, heads, features, head_dim]
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
        output[t] = sum_{s<=t} attention(q[t], k[s]) * v[s]

        This is computed efficiently using cumulative sums over time.
        """
        batch, heads, length, head_dim = v.shape
        features = q_prime.shape[-1]
        device = v.device
        dtype = v.dtype

        # Initialize prefix sums
        kv_prefix = torch.zeros(batch, heads, features, head_dim, device=device, dtype=dtype)
        k_prefix = torch.zeros(batch, heads, features, device=device, dtype=dtype)

        outputs = []

        for t in range(length):
            # Update prefix sums with current position
            k_t = k_prime[:, :, t, :]  # [batch, heads, features]
            v_t = v[:, :, t, :]        # [batch, heads, head_dim]

            # KV prefix: sum of k_s * v_s^T for s <= t
            kv_prefix = kv_prefix + torch.einsum('bhm,bhd->bhmd', k_t, v_t)
            # K prefix: sum of k_s for s <= t
            k_prefix = k_prefix + k_t

            # Compute output for position t
            q_t = q_prime[:, :, t, :]  # [batch, heads, features]

            # Q' @ (cumulative K' V)
            qkv_t = torch.einsum('bhm,bhmd->bhd', q_t, kv_prefix)

            # Normalization
            normalizer = torch.einsum('bhm,bhm->bh', q_t, k_prefix).unsqueeze(-1)
            normalizer = normalizer + self.epsilon

            outputs.append(qkv_t / normalizer)

        return torch.stack(outputs, dim=2)


class PerformerEncoderLayer(nn.Module):
    """Single Performer encoder layer with FAVOR+ attention"""

    def __init__(self, config: PerformerConfig):
        super().__init__()

        self.attention = FAVORPlusAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_features=config.computed_num_features,
            use_orthogonal=config.use_orthogonal,
            redraw_features=config.redraw_features,
            epsilon=config.epsilon,
            dropout=config.dropout
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with pre-norm architecture.

        Args:
            x: Input [batch, length, d_model]
            causal: Use causal attention
            return_attention: Return attention weights

        Returns:
            output: [batch, length, d_model]
            attention: Optional attention weights
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(
            self.norm1(x), causal=causal, return_attention=return_attention
        )
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token embedding using 1D convolution"""

    def __init__(self, input_features: int, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input features.

        Args:
            x: [batch, seq_len, features]

        Returns:
            [batch, seq_len, d_model]
        """
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.conv(x)       # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        return self.activation(x)


class PerformerEncoder(nn.Module):
    """Performer encoder stack"""

    def __init__(self, config: PerformerConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = TokenEmbedding(
            config.input_features, config.d_model, config.kernel_size
        )

        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        ) if config.use_positional_encoding else None

        self.layers = nn.ModuleList([
            PerformerEncoderLayer(config) for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through encoder.

        Args:
            x: Input [batch, seq_len, features]
            causal: Use causal attention
            return_attention: Return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention_dict: Dictionary of attention weights per layer
        """
        # Embed tokens
        x = self.token_embedding(x)

        # Add positional encoding
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        # Pass through encoder layers
        attention_dict = {}
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, causal=causal, return_attention=return_attention)
            if attn is not None:
                attention_dict[f'layer_{i}'] = attn

        # Final normalization
        x = self.final_norm(x)

        return x, attention_dict


class PerformerForecaster(nn.Module):
    """
    Performer-based forecasting model for financial time series.

    Example:
        config = PerformerConfig(
            d_model=128,
            num_heads=8,
            num_layers=4,
            prediction_horizon=24
        )
        model = PerformerForecaster(config)

        x = torch.randn(2, 512, 6)  # [batch, seq_len, features]
        output = model(x)
        print(output['predictions'].shape)  # [2, 24]
    """

    def __init__(self, config: PerformerConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = PerformerEncoder(config)

        # Output head
        self.output_head = self._build_output_head(config)

    def _build_output_head(self, config: PerformerConfig) -> nn.Module:
        """Build output projection layer"""
        if config.output_type == OutputType.QUANTILE:
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, config.prediction_horizon * len(config.quantiles))
            )
        elif config.output_type == OutputType.DIRECTION:
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, config.prediction_horizon * 3)
            )
        elif config.output_type == OutputType.PORTFOLIO:
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, config.prediction_horizon)
            )
        else:  # REGRESSION
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, config.prediction_horizon)
            )

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, features]
            causal: Use causal attention
            return_attention: Return attention weights

        Returns:
            Dictionary with predictions and optional attention weights
        """
        batch = x.shape[0]

        # Encode sequence
        encoded, attention_dict = self.encoder(x, causal=causal, return_attention=return_attention)

        # Use last position for prediction (or could use mean pooling)
        last_hidden = encoded[:, -1, :]  # [batch, d_model]

        # Generate predictions
        predictions = self._compute_output(last_hidden)

        result = {
            'predictions': predictions,
            'encoded': encoded,
            'attention_weights': attention_dict if return_attention else None
        }

        # Add confidence for quantile predictions
        if self.config.output_type == OutputType.QUANTILE:
            result['confidence'] = self._compute_confidence(predictions)

        return result

    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """Compute final predictions"""
        batch = x.shape[0]
        raw_output = self.output_head(x)

        if self.config.output_type == OutputType.PORTFOLIO:
            # Portfolio weights via softmax (sum to 1)
            return F.softmax(raw_output, dim=-1)
        elif self.config.output_type == OutputType.DIRECTION:
            # Per-timestep direction classification
            return raw_output.view(batch, self.config.prediction_horizon, 3)
        elif self.config.output_type == OutputType.QUANTILE:
            # Reshape to [batch, horizon, num_quantiles]
            return raw_output.view(batch, self.config.prediction_horizon, len(self.config.quantiles))
        else:  # REGRESSION
            return raw_output

    def _compute_confidence(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute confidence from quantile predictions"""
        # Confidence is inverse of prediction interval width
        # predictions: [batch, horizon, num_quantiles]
        interval_width = (predictions[:, :, -1] - predictions[:, :, 0]).abs()
        confidence = 1.0 / (1.0 + interval_width)
        return confidence


# Convenience function to create model from config dict
def create_performer(config_dict: dict = None) -> PerformerForecaster:
    """Create Performer model from configuration dictionary"""
    if config_dict is None:
        config_dict = {}

    config = PerformerConfig(**config_dict)
    return PerformerForecaster(config)


if __name__ == "__main__":
    # Test the model
    print("Testing Performer model...")

    config = PerformerConfig(
        input_features=6,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=512,
        prediction_horizon=24
    )

    model = PerformerForecaster(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with long sequence
    x = torch.randn(2, 256, 6)
    output = model(x, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Encoded shape: {output['encoded'].shape}")
    print(f"Attention weights available: {output['attention_weights'] is not None}")

    # Test causal attention
    output_causal = model(x, causal=True)
    print(f"Causal predictions shape: {output_causal['predictions'].shape}")

    # Test different output types
    for output_type in OutputType:
        config.output_type = output_type
        model = PerformerForecaster(config)
        output = model(x)
        print(f"{output_type.value}: predictions shape = {output['predictions'].shape}")

    print("\nAll tests passed!")
