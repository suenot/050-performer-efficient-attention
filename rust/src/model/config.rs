//! Model configuration

use serde::{Deserialize, Serialize};

/// Configuration for the Performer model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformerConfig {
    /// Input feature dimension
    pub input_dim: usize,

    /// Model dimension (hidden size)
    pub d_model: usize,

    /// Number of attention heads
    pub n_heads: usize,

    /// Number of encoder layers
    pub n_layers: usize,

    /// Feed-forward hidden dimension
    pub d_ff: usize,

    /// Number of random features for FAVOR+
    pub num_features: usize,

    /// Dropout rate
    pub dropout: f64,

    /// Whether to use orthogonal random features
    pub use_orthogonal: bool,

    /// Whether to redraw random features during training
    pub redraw_features: bool,

    /// Feature redrawing interval (in forward passes)
    pub redraw_interval: usize,

    /// Whether to use causal masking
    pub causal: bool,

    /// Output dimension (prediction size)
    pub output_dim: usize,

    /// Sequence length
    pub seq_len: usize,

    /// Learning rate for training
    pub learning_rate: f64,
}

impl Default for PerformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            d_model: 64,
            n_heads: 4,
            n_layers: 3,
            d_ff: 256,
            num_features: 256,
            dropout: 0.1,
            use_orthogonal: true,
            redraw_features: true,
            redraw_interval: 1000,
            causal: true,
            output_dim: 1,
            seq_len: 168,
            learning_rate: 0.001,
        }
    }
}

impl PerformerConfig {
    /// Create configuration for small model
    pub fn small() -> Self {
        Self {
            d_model: 32,
            n_heads: 2,
            n_layers: 2,
            d_ff: 128,
            num_features: 64,
            ..Default::default()
        }
    }

    /// Create configuration for medium model
    pub fn medium() -> Self {
        Self {
            d_model: 128,
            n_heads: 8,
            n_layers: 4,
            d_ff: 512,
            num_features: 256,
            ..Default::default()
        }
    }

    /// Create configuration for large model
    pub fn large() -> Self {
        Self {
            d_model: 256,
            n_heads: 16,
            n_layers: 6,
            d_ff: 1024,
            num_features: 512,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }

        if self.num_features == 0 {
            return Err("num_features must be positive".to_string());
        }

        if self.n_layers == 0 {
            return Err("n_layers must be positive".to_string());
        }

        Ok(())
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PerformerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_small_config() {
        let config = PerformerConfig::small();
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 16);
    }

    #[test]
    fn test_invalid_config() {
        let config = PerformerConfig {
            d_model: 63,
            n_heads: 4,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
