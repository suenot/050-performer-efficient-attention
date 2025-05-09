//! Trading strategy and backtesting module

mod backtest;
mod signals;

pub use backtest::{BacktestConfig, BacktestResult, Backtester};
pub use signals::{Signal, SignalGenerator, SignalGeneratorConfig, SignalType};
