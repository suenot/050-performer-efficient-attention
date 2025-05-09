//! Performer model module
//!
//! Implements FAVOR+ attention mechanism for efficient attention computation.

mod attention;
mod config;
mod embedding;
mod performer;

pub use attention::FAVORPlusAttention;
pub use config::PerformerConfig;
pub use embedding::TokenEmbedding;
pub use performer::{PerformerEncoder, PerformerModel};
