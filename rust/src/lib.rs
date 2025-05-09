//! # Performer
//!
//! Efficient attention mechanism using FAVOR+ (Fast Attention Via positive Orthogonal
//! Random features) for financial time series prediction.
//!
//! ## Features
//!
//! - Linear O(L) complexity attention instead of O(L^2)
//! - Positive random features for numerical stability
//! - Orthogonal features for lower variance
//! - Bybit API integration for cryptocurrency data
//!
//! ## Modules
//!
//! - `api` - Client for Bybit API
//! - `data` - Data loading and preprocessing
//! - `model` - Performer architecture implementation
//! - `strategy` - Trading strategy and backtesting
//!
//! ## Example
//!
//! ```no_run
//! use performer::{BybitClient, DataLoader, PerformerModel, PerformerConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Fetch cryptocurrency data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "60", 1000).await.unwrap();
//!
//!     // Prepare dataset with builder pattern
//!     let loader = DataLoader::new()
//!         .seq_len(168)
//!         .target_horizon(24);
//!     let dataset = loader.prepare_dataset(&klines).unwrap();
//!
//!     // Create Performer model
//!     let config = PerformerConfig {
//!         d_model: 64,
//!         n_heads: 4,
//!         num_features: 256,
//!         use_orthogonal: true,
//!         ..Default::default()
//!     };
//!     let mut model = PerformerModel::new(config);
//!
//!     // Make prediction
//!     let predictions = model.predict(&dataset);
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-exports for convenience
pub use api::{BybitClient, BybitError, Kline, OrderBook, Ticker};
pub use data::{DataLoader, Dataset, Features};
pub use model::{
    FAVORPlusAttention, PerformerConfig, PerformerEncoder, PerformerModel,
    TokenEmbedding,
};
pub use strategy::{BacktestConfig, BacktestResult, Backtester, Signal, SignalGenerator, SignalGeneratorConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default settings
pub mod defaults {
    /// Hidden layer dimension
    pub const D_MODEL: usize = 64;

    /// Number of attention heads
    pub const N_HEADS: usize = 4;

    /// Number of random features for FAVOR+
    pub const NUM_FEATURES: usize = 256;

    /// Number of encoder layers
    pub const N_LAYERS: usize = 3;

    /// Dropout rate
    pub const DROPOUT: f64 = 0.1;

    /// Encoder context length
    pub const ENCODER_LENGTH: usize = 168; // 7 days of hourly data

    /// Prediction horizon
    pub const PREDICTION_LENGTH: usize = 24; // 24 hours

    /// Learning rate
    pub const LEARNING_RATE: f64 = 0.001;

    /// Batch size
    pub const BATCH_SIZE: usize = 32;

    /// Number of epochs
    pub const EPOCHS: usize = 100;

    /// Whether to use orthogonal random features
    pub const USE_ORTHOGONAL: bool = true;

    /// Whether to use causal masking
    pub const CAUSAL: bool = true;
}
