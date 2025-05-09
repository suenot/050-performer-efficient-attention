//! Trading signal generation

use serde::{Deserialize, Serialize};

/// Type of trading signal
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SignalType {
    /// Buy/long signal
    Long,
    /// Sell/short signal
    Short,
    /// Hold current position
    Hold,
    /// Close position
    Close,
}

/// Trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal type
    pub signal_type: SignalType,
    /// Predicted return
    pub predicted_return: f64,
    /// Confidence score [0, 1]
    pub confidence: f64,
    /// Recommended position size [0, 1]
    pub position_size: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl Signal {
    /// Create new signal
    pub fn new(
        signal_type: SignalType,
        predicted_return: f64,
        confidence: f64,
        position_size: f64,
        timestamp: u64,
    ) -> Self {
        Self {
            signal_type,
            predicted_return,
            confidence,
            position_size: position_size.clamp(0.0, 1.0),
            timestamp,
        }
    }

    /// Create hold signal
    pub fn hold(timestamp: u64) -> Self {
        Self {
            signal_type: SignalType::Hold,
            predicted_return: 0.0,
            confidence: 0.0,
            position_size: 0.0,
            timestamp,
        }
    }

    /// Check if signal is actionable (Long or Short)
    pub fn is_actionable(&self) -> bool {
        matches!(self.signal_type, SignalType::Long | SignalType::Short)
    }
}

/// Signal generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGeneratorConfig {
    /// Minimum predicted return to generate signal
    pub min_return_threshold: f64,
    /// Minimum confidence to generate signal
    pub min_confidence: f64,
    /// Base position size
    pub base_position_size: f64,
    /// Scale position by confidence
    pub scale_by_confidence: bool,
    /// Scale position by predicted return
    pub scale_by_return: bool,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            min_return_threshold: 0.001,
            min_confidence: 0.5,
            base_position_size: 0.1,
            scale_by_confidence: true,
            scale_by_return: false,
        }
    }
}

/// Signal generator from model predictions
pub struct SignalGenerator {
    config: SignalGeneratorConfig,
}

impl SignalGenerator {
    /// Create new signal generator with default config
    pub fn new() -> Self {
        Self {
            config: SignalGeneratorConfig::default(),
        }
    }

    /// Create signal generator with custom config
    pub fn with_config(config: SignalGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate signal from prediction
    pub fn generate(&self, predicted_return: f64, confidence: f64, timestamp: u64) -> Signal {
        // Check thresholds
        if confidence < self.config.min_confidence {
            return Signal::hold(timestamp);
        }

        if predicted_return.abs() < self.config.min_return_threshold {
            return Signal::hold(timestamp);
        }

        // Determine signal type
        let signal_type = if predicted_return > 0.0 {
            SignalType::Long
        } else {
            SignalType::Short
        };

        // Calculate position size
        let mut position_size = self.config.base_position_size;

        if self.config.scale_by_confidence {
            position_size *= confidence;
        }

        if self.config.scale_by_return {
            // Scale by magnitude of predicted return (capped)
            let return_scale = (predicted_return.abs() * 100.0).min(2.0);
            position_size *= return_scale;
        }

        Signal::new(
            signal_type,
            predicted_return,
            confidence,
            position_size,
            timestamp,
        )
    }

    /// Generate signals for multiple predictions
    pub fn generate_batch(
        &self,
        predictions: &[f64],
        confidences: &[f64],
        timestamps: &[u64],
    ) -> Vec<Signal> {
        predictions
            .iter()
            .zip(confidences.iter())
            .zip(timestamps.iter())
            .map(|((pred, conf), ts)| self.generate(*pred, *conf, *ts))
            .collect()
    }

    /// Generate signals with uniform confidence
    pub fn generate_batch_uniform_confidence(
        &self,
        predictions: &[f64],
        timestamps: &[u64],
        confidence: f64,
    ) -> Vec<Signal> {
        predictions
            .iter()
            .zip(timestamps.iter())
            .map(|(pred, ts)| self.generate(*pred, confidence, *ts))
            .collect()
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new(SignalType::Long, 0.02, 0.8, 0.15, 1234567890);

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!((signal.predicted_return - 0.02).abs() < 0.001);
        assert!(signal.is_actionable());
    }

    #[test]
    fn test_hold_signal() {
        let signal = Signal::hold(1234567890);

        assert_eq!(signal.signal_type, SignalType::Hold);
        assert!(!signal.is_actionable());
    }

    #[test]
    fn test_signal_generator_long() {
        let generator = SignalGenerator::new();
        let signal = generator.generate(0.015, 0.7, 1234567890);

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_signal_generator_short() {
        let generator = SignalGenerator::new();
        let signal = generator.generate(-0.02, 0.8, 1234567890);

        assert_eq!(signal.signal_type, SignalType::Short);
    }

    #[test]
    fn test_signal_generator_hold_low_confidence() {
        let generator = SignalGenerator::new();
        let signal = generator.generate(0.05, 0.3, 1234567890);

        assert_eq!(signal.signal_type, SignalType::Hold);
    }

    #[test]
    fn test_signal_generator_hold_small_return() {
        let generator = SignalGenerator::new();
        let signal = generator.generate(0.0005, 0.9, 1234567890);

        assert_eq!(signal.signal_type, SignalType::Hold);
    }

    #[test]
    fn test_generate_batch() {
        let generator = SignalGenerator::new();

        let predictions = vec![0.02, -0.015, 0.0003, 0.025];
        let confidences = vec![0.8, 0.7, 0.9, 0.6];
        let timestamps = vec![1, 2, 3, 4];

        let signals = generator.generate_batch(&predictions, &confidences, &timestamps);

        assert_eq!(signals.len(), 4);
        assert_eq!(signals[0].signal_type, SignalType::Long);
        assert_eq!(signals[1].signal_type, SignalType::Short);
        assert_eq!(signals[2].signal_type, SignalType::Hold); // Small return
        assert_eq!(signals[3].signal_type, SignalType::Long);
    }

    #[test]
    fn test_position_size_scaling() {
        let config = SignalGeneratorConfig {
            base_position_size: 0.2,
            scale_by_confidence: true,
            ..Default::default()
        };

        let generator = SignalGenerator::with_config(config);

        let signal_high_conf = generator.generate(0.02, 0.9, 0);
        let signal_low_conf = generator.generate(0.02, 0.6, 0);

        assert!(signal_high_conf.position_size > signal_low_conf.position_size);
    }
}
