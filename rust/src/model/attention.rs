//! FAVOR+ Attention Mechanism
//!
//! Fast Attention Via positive Orthogonal Random features
//!
//! Key innovations:
//! 1. Positive random features to ensure non-negative attention weights
//! 2. Orthogonal random features for lower variance
//! 3. Linear O(L) complexity instead of O(L^2)

use ndarray::{Array2, Array3, Array4};
use std::f64::consts::PI;

use crate::model::PerformerConfig;

/// FAVOR+ Attention mechanism
///
/// Approximates softmax attention using random feature maps:
/// softmax(QK^T / sqrt(d)) V ≈ φ(Q) (φ(K)^T V)
///
/// where φ(x) = exp(-||x||^2 / 2) * [exp(ω_1 · x), ..., exp(ω_m · x)]
#[derive(Debug, Clone)]
pub struct FAVORPlusAttention {
    /// Query projection [d_model, d_model]
    w_q: Array2<f64>,
    /// Key projection [d_model, d_model]
    w_k: Array2<f64>,
    /// Value projection [d_model, d_model]
    w_v: Array2<f64>,
    /// Output projection [d_model, d_model]
    w_o: Array2<f64>,
    /// Random features matrix [num_features, head_dim]
    random_features: Array2<f64>,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Number of random features
    num_features: usize,
    /// Model dimension
    d_model: usize,
    /// Whether to use causal masking
    causal: bool,
    /// Whether to use orthogonal features
    use_orthogonal: bool,
    /// Forward pass counter for redrawing
    forward_count: usize,
    /// Redraw interval
    redraw_interval: usize,
    /// Whether to redraw features
    redraw_features: bool,
}

impl FAVORPlusAttention {
    /// Create new FAVOR+ attention layer
    pub fn new(config: &PerformerConfig) -> Self {
        let d_model = config.d_model;
        let num_heads = config.n_heads;
        let head_dim = d_model / num_heads;
        let num_features = config.num_features;

        // Xavier initialization for projections
        let scale = (2.0 / (d_model * 2) as f64).sqrt();
        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale);
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale);

        // Generate random features
        let random_features = if config.use_orthogonal {
            generate_orthogonal_random_features(num_features, head_dim)
        } else {
            generate_random_features(num_features, head_dim)
        };

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            random_features,
            num_heads,
            head_dim,
            num_features,
            d_model,
            causal: config.causal,
            use_orthogonal: config.use_orthogonal,
            forward_count: 0,
            redraw_interval: config.redraw_interval,
            redraw_features: config.redraw_features,
        }
    }

    /// Forward pass with FAVOR+ attention
    ///
    /// Input: [batch, seq_len, d_model]
    /// Output: [batch, seq_len, d_model]
    pub fn forward(&mut self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();

        // Optionally redraw random features
        if self.redraw_features && self.forward_count % self.redraw_interval == 0 {
            self.random_features = if self.use_orthogonal {
                generate_orthogonal_random_features(self.num_features, self.head_dim)
            } else {
                generate_random_features(self.num_features, self.head_dim)
            };
        }
        self.forward_count += 1;

        // Linear projections: Q, K, V
        let q = self.linear_transform(x, &self.w_q);
        let k = self.linear_transform(x, &self.w_k);
        let v = self.linear_transform(x, &self.w_v);

        // Reshape for multi-head attention
        // [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
        let q_heads = self.split_heads(&q);
        let k_heads = self.split_heads(&k);
        let v_heads = self.split_heads(&v);

        // Apply FAVOR+ attention for each head
        let mut output_heads = Array4::zeros((batch_size, self.num_heads, seq_len, self.head_dim));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                // Extract Q, K, V for this head
                let q_h = extract_head(&q_heads, b, h);
                let k_h = extract_head(&k_heads, b, h);
                let v_h = extract_head(&v_heads, b, h);

                // Apply positive feature map
                let q_prime = self.positive_feature_map(&q_h);
                let k_prime = self.positive_feature_map(&k_h);

                // FAVOR+ attention
                let attn_output = if self.causal {
                    self.causal_attention(&q_prime, &k_prime, &v_h)
                } else {
                    self.bidirectional_attention(&q_prime, &k_prime, &v_h)
                };

                // Store result
                for t in 0..seq_len {
                    for d in 0..self.head_dim {
                        output_heads[[b, h, t, d]] = attn_output[[t, d]];
                    }
                }
            }
        }

        // Concatenate heads and project
        let concat = self.concat_heads(&output_heads);
        self.linear_transform(&concat, &self.w_o)
    }

    /// Apply positive random feature map (FAVOR+)
    ///
    /// φ(x) = exp(-||x||^2 / 2) * [exp(ω_1 · x), ..., exp(ω_m · x)] / sqrt(m)
    fn positive_feature_map(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, _) = x.dim();
        let mut phi = Array2::zeros((seq_len, self.num_features));

        // Scaling factor
        let scale = (self.head_dim as f64).powf(-0.25);

        for t in 0..seq_len {
            // Compute ||x||^2 / 2
            let x_norm_sq: f64 = (0..self.head_dim)
                .map(|d| (x[[t, d]] * scale).powi(2))
                .sum::<f64>() / 2.0;

            // Compute exp(ω · x) for each random feature
            for m in 0..self.num_features {
                let dot_product: f64 = (0..self.head_dim)
                    .map(|d| self.random_features[[m, d]] * x[[t, d]] * scale)
                    .sum();

                // φ_m(x) = exp(-||x||^2/2) * exp(ω_m · x)
                phi[[t, m]] = (-x_norm_sq + dot_product).exp();
            }
        }

        // Normalize by sqrt(num_features)
        let normalizer = (self.num_features as f64).sqrt();
        phi.mapv(|v| v / normalizer)
    }

    /// Bidirectional FAVOR+ attention
    /// O = φ(Q) * (φ(K)^T V) / (φ(Q) * φ(K)^T 1)
    fn bidirectional_attention(
        &self,
        q_prime: &Array2<f64>,
        k_prime: &Array2<f64>,
        v: &Array2<f64>,
    ) -> Array2<f64> {
        let (seq_len, _) = q_prime.dim();
        let head_dim = v.dim().1;

        // Compute K^T V: [num_features, head_dim]
        let k_t_v = matmul_transpose(&k_prime.view(), &v.view());

        // Compute K^T 1: [num_features]
        let k_sum: Vec<f64> = (0..self.num_features)
            .map(|m| (0..seq_len).map(|t| k_prime[[t, m]]).sum())
            .collect();

        // Compute output
        let mut output = Array2::zeros((seq_len, head_dim));

        for t in 0..seq_len {
            // Numerator: Q' @ (K'^T V)
            let numerator: Vec<f64> = (0..head_dim)
                .map(|d| {
                    (0..self.num_features)
                        .map(|m| q_prime[[t, m]] * k_t_v[[m, d]])
                        .sum()
                })
                .collect();

            // Denominator: Q' @ (K'^T 1)
            let denominator: f64 = (0..self.num_features)
                .map(|m| q_prime[[t, m]] * k_sum[m])
                .sum();

            // Normalize
            let denom = denominator.max(1e-6);
            for d in 0..head_dim {
                output[[t, d]] = numerator[d] / denom;
            }
        }

        output
    }

    /// Causal FAVOR+ attention using prefix sums
    fn causal_attention(
        &self,
        q_prime: &Array2<f64>,
        k_prime: &Array2<f64>,
        v: &Array2<f64>,
    ) -> Array2<f64> {
        let (seq_len, _) = q_prime.dim();
        let head_dim = v.dim().1;

        let mut output: Array2<f64> = Array2::zeros((seq_len, head_dim));

        // Running sums for causal computation
        // S[t] = sum_{i<=t} φ(K_i)^T V_i
        // z[t] = sum_{i<=t} φ(K_i)
        let mut s: Array2<f64> = Array2::zeros((self.num_features, head_dim));
        let mut z = vec![0.0; self.num_features];

        for t in 0..seq_len {
            // Update running sums with current key-value
            for m in 0..self.num_features {
                for d in 0..head_dim {
                    s[[m, d]] += k_prime[[t, m]] * v[[t, d]];
                }
                z[m] += k_prime[[t, m]];
            }

            // Compute output: Q' @ S / (Q' @ z)
            let numerator: Vec<f64> = (0..head_dim)
                .map(|d| {
                    (0..self.num_features)
                        .map(|m| q_prime[[t, m]] * s[[m, d]])
                        .sum()
                })
                .collect();

            let denominator: f64 = (0..self.num_features)
                .map(|m| q_prime[[t, m]] * z[m])
                .sum();

            let denom = denominator.max(1e-6);
            for d in 0..head_dim {
                output[[t, d]] = numerator[d] / denom;
            }
        }

        output
    }

    /// Linear transformation
    fn linear_transform(&self, x: &Array3<f64>, w: &Array2<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_in) = x.dim();
        let d_out = w.dim().1;
        let mut output = Array3::zeros((batch_size, seq_len, d_out));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_o in 0..d_out {
                    let mut sum = 0.0;
                    for d_i in 0..d_in {
                        sum += x[[b, t, d_i]] * w[[d_i, d_o]];
                    }
                    output[[b, t, d_o]] = sum;
                }
            }
        }

        output
    }

    /// Split tensor into multiple heads
    /// [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
    fn split_heads(&self, x: &Array3<f64>) -> Array4<f64> {
        let (batch_size, seq_len, _) = x.dim();
        let mut output = Array4::zeros((batch_size, self.num_heads, seq_len, self.head_dim));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        output[[b, h, t, d]] = x[[b, t, h * self.head_dim + d]];
                    }
                }
            }
        }

        output
    }

    /// Concatenate heads back together
    /// [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
    fn concat_heads(&self, x: &Array4<f64>) -> Array3<f64> {
        let (batch_size, _, seq_len, _) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        output[[b, t, h * self.head_dim + d]] = x[[b, h, t, d]];
                    }
                }
            }
        }

        output
    }

    /// Get attention statistics (for debugging/analysis)
    pub fn get_stats(&self) -> AttentionStats {
        AttentionStats {
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            num_features: self.num_features,
            forward_count: self.forward_count,
            causal: self.causal,
        }
    }
}

/// Attention statistics for analysis
#[derive(Debug, Clone)]
pub struct AttentionStats {
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_features: usize,
    pub forward_count: usize,
    pub causal: bool,
}

/// Extract a single head from the 4D tensor
fn extract_head(x: &Array4<f64>, batch: usize, head: usize) -> Array2<f64> {
    let seq_len = x.dim().2;
    let head_dim = x.dim().3;
    let mut output = Array2::zeros((seq_len, head_dim));

    for t in 0..seq_len {
        for d in 0..head_dim {
            output[[t, d]] = x[[batch, head, t, d]];
        }
    }

    output
}

/// Matrix multiplication with transpose of first argument: A^T @ B
fn matmul_transpose(a: &ndarray::ArrayView2<f64>, b: &ndarray::ArrayView2<f64>) -> Array2<f64> {
    let m = a.dim().1;
    let n = b.dim().1;
    let k = a.dim().0;
    let mut output = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[[l, i]] * b[[l, j]];
            }
            output[[i, j]] = sum;
        }
    }

    output
}

/// Generate random features from normal distribution
fn generate_random_features(num_features: usize, dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((num_features, dim), |_| rand_normal())
}

/// Generate orthogonal random features using Gram-Schmidt
fn generate_orthogonal_random_features(num_features: usize, dim: usize) -> Array2<f64> {
    let mut features = Array2::zeros((num_features, dim));

    // Generate in blocks of 'dim' to maintain orthogonality
    let num_blocks = (num_features + dim - 1) / dim;

    for block in 0..num_blocks {
        let start = block * dim;
        let end = (start + dim).min(num_features);

        // Generate random matrix
        let mut block_features: Vec<Vec<f64>> = (start..end)
            .map(|_| (0..dim).map(|_| rand_normal()).collect())
            .collect();

        // Gram-Schmidt orthogonalization
        for i in 0..block_features.len() {
            // Subtract projections onto previous vectors
            for j in 0..i {
                let dot = dot_product(&block_features[i], &block_features[j]);
                let norm_sq = dot_product(&block_features[j], &block_features[j]);
                if norm_sq > 1e-10 {
                    let scale = dot / norm_sq;
                    for k in 0..dim {
                        block_features[i][k] -= scale * block_features[j][k];
                    }
                }
            }

            // Normalize
            let norm = dot_product(&block_features[i], &block_features[i]).sqrt();
            if norm > 1e-10 {
                for k in 0..dim {
                    block_features[i][k] /= norm;
                }
            }
        }

        // Copy to output, scaling by sqrt(dim) to maintain variance
        let scale = (dim as f64).sqrt();
        for (i, row) in block_features.iter().enumerate() {
            if start + i < num_features {
                for (j, &val) in row.iter().enumerate() {
                    features[[start + i, j]] = val * scale;
                }
            }
        }
    }

    features
}

/// Dot product of two vectors
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
    fn test_favor_plus_attention_shape() {
        let config = PerformerConfig {
            d_model: 32,
            n_heads: 4,
            num_features: 64,
            causal: false,
            ..Default::default()
        };

        let mut attn = FAVORPlusAttention::new(&config);
        let x = Array3::from_shape_fn((2, 16, 32), |_| rand_normal());

        let output = attn.forward(&x);

        assert_eq!(output.dim(), (2, 16, 32));
    }

    #[test]
    fn test_favor_plus_attention_causal() {
        let config = PerformerConfig {
            d_model: 16,
            n_heads: 2,
            num_features: 32,
            causal: true,
            ..Default::default()
        };

        let mut attn = FAVORPlusAttention::new(&config);
        let x = Array3::from_shape_fn((1, 10, 16), |_| rand_normal());

        let output = attn.forward(&x);

        assert_eq!(output.dim(), (1, 10, 16));
    }

    #[test]
    fn test_positive_feature_map() {
        let config = PerformerConfig::small();
        let attn = FAVORPlusAttention::new(&config);

        let x = Array2::from_shape_fn((5, config.head_dim()), |_| rand_normal());
        let phi = attn.positive_feature_map(&x);

        // All values should be positive (FAVOR+ property)
        for v in phi.iter() {
            assert!(*v >= 0.0, "FAVOR+ features should be non-negative");
        }
    }

    #[test]
    fn test_orthogonal_features() {
        let features = generate_orthogonal_random_features(16, 8);

        // Check orthogonality within block
        for i in 0..8 {
            for j in 0..8 {
                if i != j {
                    let dot: f64 = (0..8)
                        .map(|k| features[[i, k]] * features[[j, k]])
                        .sum();
                    assert!(
                        dot.abs() < 0.1,
                        "Vectors {} and {} should be orthogonal, got dot product {}",
                        i,
                        j,
                        dot
                    );
                }
            }
        }
    }

    #[test]
    fn test_feature_redraw() {
        let config = PerformerConfig {
            redraw_features: true,
            redraw_interval: 2,
            ..Default::default()
        };

        let mut attn = FAVORPlusAttention::new(&config);
        let x = Array3::from_shape_fn((1, 5, config.d_model), |_| rand_normal());

        let features_before = attn.random_features.clone();

        // First forward - no redraw
        attn.forward(&x);
        // Second forward - no redraw (count = 1)
        attn.forward(&x);
        // Third forward - redraw (count = 2)
        attn.forward(&x);

        // Features should have been redrawn
        let features_after = &attn.random_features;

        // At least some values should be different
        let diff: f64 = features_before
            .iter()
            .zip(features_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 0.1, "Features should have been redrawn");
    }

    #[test]
    fn test_attention_stats() {
        let config = PerformerConfig {
            d_model: 64,
            n_heads: 8,
            num_features: 128,
            causal: true,
            ..Default::default()
        };

        let attn = FAVORPlusAttention::new(&config);
        let stats = attn.get_stats();

        assert_eq!(stats.num_heads, 8);
        assert_eq!(stats.head_dim, 8);
        assert_eq!(stats.num_features, 128);
        assert!(stats.causal);
    }
}
