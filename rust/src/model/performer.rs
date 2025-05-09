//! Performer model implementation

use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;

use crate::data::Dataset;
use crate::model::{FAVORPlusAttention, PerformerConfig, TokenEmbedding};
use crate::model::embedding::PositionalEncoding;

/// Performer Encoder Layer
#[derive(Debug)]
pub struct PerformerEncoderLayer {
    /// FAVOR+ self-attention
    attention: FAVORPlusAttention,
    /// Feed-forward network: first linear
    ff_linear1: Array2<f64>,
    /// Feed-forward network: second linear
    ff_linear2: Array2<f64>,
    /// Layer norm 1 parameters (gamma, beta)
    ln1_gamma: Array1<f64>,
    ln1_beta: Array1<f64>,
    /// Layer norm 2 parameters
    ln2_gamma: Array1<f64>,
    ln2_beta: Array1<f64>,
    /// Model dimension
    d_model: usize,
    /// Feed-forward dimension
    d_ff: usize,
    /// Dropout rate
    dropout: f64,
}

impl PerformerEncoderLayer {
    /// Create new encoder layer
    pub fn new(config: &PerformerConfig) -> Self {
        let d_model = config.d_model;
        let d_ff = config.d_ff;

        // Xavier initialization
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        let ff_linear1 = Array2::from_shape_fn((d_model, d_ff), |_| rand_normal() * scale1);
        let ff_linear2 = Array2::from_shape_fn((d_ff, d_model), |_| rand_normal() * scale2);

        // Layer norm initialized to identity
        let ln1_gamma = Array1::ones(d_model);
        let ln1_beta = Array1::zeros(d_model);
        let ln2_gamma = Array1::ones(d_model);
        let ln2_beta = Array1::zeros(d_model);

        Self {
            attention: FAVORPlusAttention::new(config),
            ff_linear1,
            ff_linear2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            d_model,
            d_ff,
            dropout: config.dropout,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();

        // Self-attention with residual connection
        let attn_output = self.attention.forward(x);
        let mut x1 = x.clone();
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    x1[[b, t, d]] += attn_output[[b, t, d]];
                }
            }
        }

        // Layer norm 1
        let x1_norm = self.layer_norm(&x1, &self.ln1_gamma, &self.ln1_beta);

        // Feed-forward with residual
        let ff_output = self.feed_forward(&x1_norm);
        let mut x2 = x1_norm.clone();
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    x2[[b, t, d]] += ff_output[[b, t, d]];
                }
            }
        }

        // Layer norm 2
        self.layer_norm(&x2, &self.ln2_gamma, &self.ln2_beta)
    }

    /// Feed-forward network
    fn feed_forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();

        // First linear + GELU activation
        let mut hidden = Array3::zeros((batch_size, seq_len, self.d_ff));
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_out in 0..self.d_ff {
                    let mut sum = 0.0;
                    for d_in in 0..self.d_model {
                        sum += x[[b, t, d_in]] * self.ff_linear1[[d_in, d_out]];
                    }
                    // GELU activation
                    hidden[[b, t, d_out]] = gelu(sum);
                }
            }
        }

        // Second linear
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_out in 0..self.d_model {
                    let mut sum = 0.0;
                    for d_in in 0..self.d_ff {
                        sum += hidden[[b, t, d_in]] * self.ff_linear2[[d_in, d_out]];
                    }
                    output[[b, t, d_out]] = sum;
                }
            }
        }

        output
    }

    /// Layer normalization
    fn layer_norm(&self, x: &Array3<f64>, gamma: &Array1<f64>, beta: &Array1<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = x.clone();

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Compute mean and variance
                let mean: f64 = (0..d_model).map(|d| x[[b, t, d]]).sum::<f64>() / d_model as f64;
                let variance: f64 = (0..d_model)
                    .map(|d| (x[[b, t, d]] - mean).powi(2))
                    .sum::<f64>() / d_model as f64;
                let std = (variance + 1e-6).sqrt();

                // Normalize and scale
                for d in 0..d_model {
                    output[[b, t, d]] = (x[[b, t, d]] - mean) / std * gamma[d] + beta[d];
                }
            }
        }

        output
    }
}

/// Performer Encoder (stack of encoder layers)
#[derive(Debug)]
pub struct PerformerEncoder {
    /// Token embedding
    embedding: TokenEmbedding,
    /// Positional encoding
    pos_encoding: PositionalEncoding,
    /// Encoder layers
    layers: Vec<PerformerEncoderLayer>,
    /// Configuration
    config: PerformerConfig,
}

impl PerformerEncoder {
    /// Create new encoder
    pub fn new(config: PerformerConfig) -> Self {
        let embedding = TokenEmbedding::new(config.input_dim, config.d_model);
        let pos_encoding = PositionalEncoding::new(config.d_model, config.seq_len + 100);

        let layers = (0..config.n_layers)
            .map(|_| PerformerEncoderLayer::new(&config))
            .collect();

        Self {
            embedding,
            pos_encoding,
            layers,
            config,
        }
    }

    /// Forward pass through encoder
    pub fn forward(&mut self, x: &Array3<f64>) -> Array3<f64> {
        // Token embedding
        let mut hidden = self.embedding.forward(x);

        // Add positional encoding
        hidden = self.pos_encoding.forward(&hidden);

        // Pass through encoder layers
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden);
        }

        hidden
    }

    /// Get encoder output dimension
    pub fn output_dim(&self) -> usize {
        self.config.d_model
    }
}

/// Complete Performer model for forecasting
#[derive(Debug)]
pub struct PerformerModel {
    /// Encoder
    encoder: PerformerEncoder,
    /// Output projection
    output_projection: Array2<f64>,
    /// Configuration
    config: PerformerConfig,
}

impl PerformerModel {
    /// Create new model
    pub fn new(config: PerformerConfig) -> Self {
        config.validate().expect("Invalid configuration");

        let encoder = PerformerEncoder::new(config.clone());

        // Output projection: [d_model, output_dim]
        let scale = (2.0 / (config.d_model + config.output_dim) as f64).sqrt();
        let output_projection =
            Array2::from_shape_fn((config.d_model, config.output_dim), |_| rand_normal() * scale);

        Self {
            encoder,
            output_projection,
            config,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array3<f64>) -> Array2<f64> {
        let (batch_size, seq_len, _) = x.dim();

        // Encode
        let encoded = self.encoder.forward(x);

        // Use last time step for prediction
        let mut output = Array2::zeros((batch_size, self.config.output_dim));

        for b in 0..batch_size {
            for d_out in 0..self.config.output_dim {
                let mut sum = 0.0;
                for d_in in 0..self.config.d_model {
                    sum += encoded[[b, seq_len - 1, d_in]] * self.output_projection[[d_in, d_out]];
                }
                output[[b, d_out]] = sum;
            }
        }

        output
    }

    /// Make predictions on a dataset
    pub fn predict(&mut self, dataset: &Dataset) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(dataset.len());

        for (batch_input, _) in dataset.iter_batches(32) {
            let output = self.forward(&batch_input);
            for b in 0..output.dim().0 {
                predictions.push(output[[b, 0]]);
            }
        }

        predictions.truncate(dataset.len());
        predictions
    }

    /// Get model configuration
    pub fn config(&self) -> &PerformerConfig {
        &self.config
    }

    /// Compute loss (MSE)
    pub fn compute_loss(&mut self, x: &Array3<f64>, targets: &[f64]) -> f64 {
        let predictions = self.forward(x);
        let batch_size = predictions.dim().0;

        let mse: f64 = (0..batch_size)
            .map(|b| (predictions[[b, 0]] - targets[b]).powi(2))
            .sum::<f64>() / batch_size as f64;

        mse
    }
}

/// GELU activation function
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0_f64 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Generate random number from standard normal distribution
fn rand_normal() -> f64 {
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataLoader, Features};
    use crate::api::Kline;

    fn create_test_input(batch: usize, seq_len: usize, d_input: usize) -> Array3<f64> {
        Array3::from_shape_fn((batch, seq_len, d_input), |_| rand_normal())
    }

    #[test]
    fn test_encoder_layer() {
        let config = PerformerConfig::small();
        let mut layer = PerformerEncoderLayer::new(&config);

        let x = create_test_input(2, 32, config.d_model);
        let output = layer.forward(&x);

        assert_eq!(output.dim(), (2, 32, config.d_model));
    }

    #[test]
    fn test_encoder() {
        let config = PerformerConfig {
            input_dim: 8,
            seq_len: 64,
            ..PerformerConfig::small()
        };

        let mut encoder = PerformerEncoder::new(config.clone());
        let x = create_test_input(2, 64, 8);

        let output = encoder.forward(&x);

        assert_eq!(output.dim(), (2, 64, config.d_model));
    }

    #[test]
    fn test_performer_model() {
        let config = PerformerConfig {
            input_dim: 8,
            seq_len: 32,
            output_dim: 1,
            ..PerformerConfig::small()
        };

        let mut model = PerformerModel::new(config);
        let x = create_test_input(4, 32, 8);

        let output = model.forward(&x);

        assert_eq!(output.dim(), (4, 1));
    }

    #[test]
    fn test_gelu() {
        // GELU(0) should be 0
        assert!((gelu(0.0) - 0.0).abs() < 0.01);

        // GELU is monotonically increasing for x > 0
        assert!(gelu(1.0) > gelu(0.5));
        assert!(gelu(2.0) > gelu(1.0));
    }

    #[test]
    fn test_model_predict() {
        let config = PerformerConfig {
            input_dim: 8,
            seq_len: 32,
            output_dim: 1,
            ..PerformerConfig::small()
        };

        let mut model = PerformerModel::new(config);

        // Create simple dataset
        let klines: Vec<Kline> = (0..200)
            .map(|i| Kline {
                timestamp: i as u64 * 3600000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect();

        let loader = DataLoader::with_config(32, 24, true);
        let dataset = loader.prepare_dataset(&klines).unwrap();

        let predictions = model.predict(&dataset);

        assert_eq!(predictions.len(), dataset.len());
    }

    #[test]
    fn test_compute_loss() {
        let config = PerformerConfig {
            input_dim: 8,
            seq_len: 16,
            output_dim: 1,
            ..PerformerConfig::small()
        };

        let mut model = PerformerModel::new(config);
        let x = create_test_input(4, 16, 8);
        let targets = vec![0.1, -0.2, 0.05, -0.1];

        let loss = model.compute_loss(&x, &targets);

        // Loss should be non-negative
        assert!(loss >= 0.0);
    }
}
