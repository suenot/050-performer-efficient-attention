//! Token embedding and positional encoding

use ndarray::{Array2, Array3};
use std::f64::consts::PI;

/// Token embedding layer
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    /// Projection matrix [input_dim, d_model]
    projection: Array2<f64>,
    /// Input dimension
    input_dim: usize,
    /// Model dimension
    d_model: usize,
}

impl TokenEmbedding {
    /// Create new token embedding
    pub fn new(input_dim: usize, d_model: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (input_dim + d_model) as f64).sqrt();
        let projection = Array2::from_shape_fn((input_dim, d_model), |_| rand_normal() * scale);

        Self {
            projection,
            input_dim,
            d_model,
        }
    }

    /// Forward pass: project input to model dimension
    /// Input: [batch, seq_len, input_dim]
    /// Output: [batch, seq_len, d_model]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_out in 0..self.d_model {
                    let mut sum = 0.0;
                    for d_in in 0..self.input_dim {
                        sum += x[[b, t, d_in]] * self.projection[[d_in, d_out]];
                    }
                    output[[b, t, d_out]] = sum;
                }
            }
        }

        output
    }

    /// Get projection matrix (for gradient updates)
    pub fn projection(&self) -> &Array2<f64> {
        &self.projection
    }

    /// Set projection matrix (for loading weights)
    pub fn set_projection(&mut self, projection: Array2<f64>) {
        self.projection = projection;
    }
}

/// Sinusoidal positional encoding
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Precomputed positional encodings [max_len, d_model]
    encodings: Array2<f64>,
    /// Model dimension
    d_model: usize,
    /// Maximum sequence length
    max_len: usize,
}

impl PositionalEncoding {
    /// Create positional encoding
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut encodings = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
                encodings[[pos, 2 * i]] = angle.sin();
                encodings[[pos, 2 * i + 1]] = angle.cos();
            }
        }

        Self {
            encodings,
            d_model,
            max_len,
        }
    }

    /// Add positional encoding to input
    /// Input: [batch, seq_len, d_model]
    /// Output: [batch, seq_len, d_model]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();
        let mut output = x.clone();

        let actual_len = seq_len.min(self.max_len);

        for b in 0..batch_size {
            for t in 0..actual_len {
                for d in 0..self.d_model {
                    output[[b, t, d]] += self.encodings[[t, d]];
                }
            }
        }

        output
    }

    /// Get encoding for a specific position
    pub fn get_encoding(&self, pos: usize) -> Option<ndarray::ArrayView1<f64>> {
        if pos < self.max_len {
            Some(self.encodings.row(pos))
        } else {
            None
        }
    }
}

/// Learnable positional embedding
#[derive(Debug, Clone)]
pub struct LearnablePositionalEmbedding {
    /// Embedding matrix [max_len, d_model]
    embeddings: Array2<f64>,
    /// Model dimension
    d_model: usize,
    /// Maximum sequence length
    max_len: usize,
}

impl LearnablePositionalEmbedding {
    /// Create learnable positional embedding
    pub fn new(d_model: usize, max_len: usize) -> Self {
        // Initialize from normal distribution
        let scale = 0.02;
        let embeddings = Array2::from_shape_fn((max_len, d_model), |_| rand_normal() * scale);

        Self {
            embeddings,
            d_model,
            max_len,
        }
    }

    /// Add positional embedding to input
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();
        let mut output = x.clone();

        let actual_len = seq_len.min(self.max_len);

        for b in 0..batch_size {
            for t in 0..actual_len {
                for d in 0..self.d_model {
                    output[[b, t, d]] += self.embeddings[[t, d]];
                }
            }
        }

        output
    }

    /// Get embeddings matrix (for gradient updates)
    pub fn embeddings(&self) -> &Array2<f64> {
        &self.embeddings
    }

    /// Set embeddings matrix (for loading weights)
    pub fn set_embeddings(&mut self, embeddings: Array2<f64>) {
        self.embeddings = embeddings;
    }
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

    #[test]
    fn test_token_embedding() {
        let embedding = TokenEmbedding::new(8, 64);
        let x = Array3::from_shape_fn((2, 10, 8), |_| rand_normal());

        let output = embedding.forward(&x);

        assert_eq!(output.dim(), (2, 10, 64));
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(64, 512);
        let x = Array3::zeros((2, 100, 64));

        let output = pe.forward(&x);

        assert_eq!(output.dim(), (2, 100, 64));

        // Check that positional encodings are non-zero
        let pos_0 = pe.get_encoding(0).unwrap();
        let pos_1 = pe.get_encoding(1).unwrap();

        // Different positions should have different encodings
        let diff: f64 = pos_0.iter().zip(pos_1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn test_learnable_positional_embedding() {
        let pe = LearnablePositionalEmbedding::new(64, 512);
        let x = Array3::zeros((2, 100, 64));

        let output = pe.forward(&x);

        assert_eq!(output.dim(), (2, 100, 64));
    }

    #[test]
    fn test_positional_encoding_values() {
        let pe = PositionalEncoding::new(4, 10);

        // Position 0 should have sin(0), cos(0), sin(0), cos(0) = 0, 1, 0, 1
        let pos_0 = pe.get_encoding(0).unwrap();
        assert!((pos_0[0] - 0.0).abs() < 0.01);
        assert!((pos_0[1] - 1.0).abs() < 0.01);
    }
}
