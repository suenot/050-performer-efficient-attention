//! Backtesting framework

use serde::{Deserialize, Serialize};
use crate::strategy::{Signal, SignalType};

/// Backtesting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission per trade (as fraction, e.g., 0.001 = 0.1%)
    pub commission: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
    /// Maximum leverage
    pub max_leverage: f64,
    /// Allow short positions
    pub allow_short: bool,
    /// Minimum position size (as fraction of capital)
    pub min_position_size: f64,
    /// Use stop loss
    pub use_stop_loss: bool,
    /// Stop loss level (as fraction)
    pub stop_loss_level: f64,
    /// Use take profit
    pub use_take_profit: bool,
    /// Take profit level (as fraction)
    pub take_profit_level: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission: 0.001,
            slippage: 0.0005,
            max_leverage: 1.0,
            allow_short: false,
            min_position_size: 0.01,
            use_stop_loss: true,
            stop_loss_level: 0.05,
            use_take_profit: true,
            take_profit_level: 0.10,
        }
    }
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry time index
    pub entry_time: usize,
    /// Exit time index
    pub exit_time: usize,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub position_size: f64,
    /// Direction (1 = long, -1 = short)
    pub direction: i32,
    /// Profit/Loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Exit reason
    pub exit_reason: String,
}

/// Backtesting results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// List of trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Total return
    pub total_return: f64,
    /// Annual return
    pub annual_return: f64,
    /// Volatility (annual)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Average profit per winning trade
    pub avg_profit: f64,
    /// Average loss per losing trade
    pub avg_loss: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade duration
    pub avg_trade_duration: f64,
}

impl BacktestResult {
    /// Create empty result
    pub fn empty(initial_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
            total_return: 0.0,
            annual_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            win_rate: 0.0,
            avg_profit: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            num_trades: 0,
            avg_trade_duration: 0.0,
        }
    }

    /// Print summary
    pub fn summary(&self) -> String {
        format!(
            r#"Backtest Results Summary
========================
Total Return: {:.2}%
Annual Return: {:.2}%
Volatility: {:.2}%
Sharpe Ratio: {:.2}
Sortino Ratio: {:.2}
Max Drawdown: {:.2}%
Calmar Ratio: {:.2}
Win Rate: {:.2}%
Profit Factor: {:.2}
Number of Trades: {}
Avg Trade Duration: {:.1} periods
"#,
            self.total_return * 100.0,
            self.annual_return * 100.0,
            self.volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.win_rate * 100.0,
            self.profit_factor,
            self.num_trades,
            self.avg_trade_duration,
        )
    }
}

/// Open position
#[derive(Debug, Clone)]
struct Position {
    entry_time: usize,
    entry_price: f64,
    size: f64,
    direction: i32,
}

/// Backtester
pub struct Backtester {
    config: BacktestConfig,
    position: Option<Position>,
    capital: f64,
    equity_curve: Vec<f64>,
    trades: Vec<Trade>,
}

impl Backtester {
    /// Create new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            capital: config.initial_capital,
            config,
            position: None,
            equity_curve: Vec::new(),
            trades: Vec::new(),
        }
    }

    /// Run backtest
    ///
    /// # Arguments
    /// * `prices` - Price array for each time step
    /// * `signals` - Trading signals for each time step
    /// * `periods_per_year` - Number of periods in a year (for annualization)
    pub fn run(
        &mut self,
        prices: &[f64],
        signals: &[Signal],
        periods_per_year: usize,
    ) -> BacktestResult {
        let n = prices.len().min(signals.len());

        if n == 0 {
            return BacktestResult::empty(self.config.initial_capital);
        }

        self.equity_curve.push(self.capital);

        for t in 0..n {
            let price = prices[t];
            let signal = &signals[t];

            // Check stop loss / take profit
            if t > 0 {
                self.check_exits(t, price);
            }

            // Process signal
            self.process_signal(t, price, signal);

            // Update equity
            let portfolio_value = self.calculate_portfolio_value(price);
            self.equity_curve.push(portfolio_value);
        }

        // Close any remaining position
        if self.position.is_some() {
            self.close_position(n - 1, prices[n - 1], "End of backtest");
        }

        // Calculate metrics
        self.calculate_metrics(periods_per_year)
    }

    /// Check and execute stop loss / take profit
    fn check_exits(&mut self, time: usize, price: f64) {
        if let Some(ref pos) = self.position {
            let return_pct = (price - pos.entry_price) / pos.entry_price * pos.direction as f64;

            if self.config.use_stop_loss && return_pct <= -self.config.stop_loss_level {
                self.close_position(time, price, "Stop Loss");
            } else if self.config.use_take_profit && return_pct >= self.config.take_profit_level {
                self.close_position(time, price, "Take Profit");
            }
        }
    }

    /// Process trading signal
    fn process_signal(&mut self, time: usize, price: f64, signal: &Signal) {
        match signal.signal_type {
            SignalType::Long => {
                if self.position.is_none() {
                    let position_value = self.capital * signal.position_size;
                    if position_value >= self.config.min_position_size * self.capital {
                        self.open_position(time, price, position_value, 1);
                    }
                }
            }
            SignalType::Short if self.config.allow_short => {
                if self.position.is_none() {
                    let position_value = self.capital * signal.position_size;
                    if position_value >= self.config.min_position_size * self.capital {
                        self.open_position(time, price, position_value, -1);
                    }
                }
            }
            SignalType::Close => {
                if self.position.is_some() {
                    self.close_position(time, price, "Signal");
                }
            }
            _ => {}
        }
    }

    /// Open a position
    fn open_position(&mut self, time: usize, price: f64, size: f64, direction: i32) {
        // Account for slippage
        let adjusted_price = price * (1.0 + self.config.slippage * direction as f64);

        // Account for commission
        let commission = size * self.config.commission;
        self.capital -= commission;

        self.position = Some(Position {
            entry_time: time,
            entry_price: adjusted_price,
            size,
            direction,
        });
    }

    /// Close a position
    fn close_position(&mut self, time: usize, price: f64, reason: &str) {
        if let Some(pos) = self.position.take() {
            // Account for slippage
            let adjusted_price = price * (1.0 - self.config.slippage * pos.direction as f64);

            // Calculate PnL
            let price_change = (adjusted_price - pos.entry_price) / pos.entry_price;
            let pnl = pos.size * price_change * pos.direction as f64;

            // Commission on exit
            let commission = pos.size * self.config.commission;

            self.capital += pos.size + pnl - commission;

            let trade = Trade {
                entry_time: pos.entry_time,
                exit_time: time,
                entry_price: pos.entry_price,
                exit_price: adjusted_price,
                position_size: pos.size,
                direction: pos.direction,
                pnl,
                return_pct: price_change * pos.direction as f64,
                exit_reason: reason.to_string(),
            };

            self.trades.push(trade);
        }
    }

    /// Calculate current portfolio value
    fn calculate_portfolio_value(&self, price: f64) -> f64 {
        let mut value = self.capital;

        if let Some(ref pos) = self.position {
            let price_change = (price - pos.entry_price) / pos.entry_price;
            let unrealized_pnl = pos.size * price_change * pos.direction as f64;
            value += pos.size + unrealized_pnl;
        }

        value
    }

    /// Calculate backtest metrics
    fn calculate_metrics(&self, periods_per_year: usize) -> BacktestResult {
        let initial = self.config.initial_capital;
        let final_value = *self.equity_curve.last().unwrap_or(&initial);

        // Returns
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let total_return = (final_value - initial) / initial;
        let num_periods = self.equity_curve.len() as f64;
        let annual_return =
            (1.0 + total_return).powf(periods_per_year as f64 / num_periods) - 1.0;

        // Volatility
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len().max(1) as f64;
        let volatility = variance.sqrt() * (periods_per_year as f64).sqrt();

        // Sharpe Ratio
        let sharpe_ratio = if volatility > 0.0 {
            annual_return / volatility
        } else {
            0.0
        };

        // Sortino Ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_variance.sqrt() * (periods_per_year as f64).sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            annual_return / downside_std
        } else {
            0.0
        };

        // Max Drawdown
        let mut max_drawdown = 0.0;
        let mut peak = self.equity_curve[0];
        for &value in &self.equity_curve {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calmar Ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let winning_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if !self.trades.is_empty() {
            winning_trades.len() as f64 / self.trades.len() as f64
        } else {
            0.0
        };

        let avg_profit = if !winning_trades.is_empty() {
            winning_trades.iter().map(|t| t.pnl).sum::<f64>() / winning_trades.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            losing_trades.iter().map(|t| t.pnl.abs()).sum::<f64>() / losing_trades.len() as f64
        } else {
            0.0
        };

        let total_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let total_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if total_loss > 0.0 {
            total_profit / total_loss
        } else if total_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_duration = if !self.trades.is_empty() {
            self.trades
                .iter()
                .map(|t| (t.exit_time - t.entry_time) as f64)
                .sum::<f64>()
                / self.trades.len() as f64
        } else {
            0.0
        };

        BacktestResult {
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            total_return,
            annual_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            avg_profit,
            avg_loss,
            profit_factor,
            num_trades: self.trades.len(),
            avg_trade_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prices(n: usize, trend: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let base = 100.0;
                let noise = ((i * 7) % 10) as f64 * 0.1 - 0.5;
                base * (1.0 + trend * i as f64 / 100.0 + noise / 100.0)
            })
            .collect()
    }

    fn create_test_signals(n: usize, long_intervals: bool) -> Vec<Signal> {
        (0..n)
            .map(|i| {
                if long_intervals && i % 20 == 0 {
                    Signal::new(SignalType::Long, 0.02, 0.8, 0.1, i as u64)
                } else if long_intervals && i % 20 == 10 {
                    Signal::new(SignalType::Close, 0.0, 0.0, 0.0, i as u64)
                } else {
                    Signal::hold(i as u64)
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_basic() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config);

        let prices = create_test_prices(100, 0.5);
        let signals = create_test_signals(100, true);

        let result = backtester.run(&prices, &signals, 252);

        assert!(result.num_trades > 0);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_backtest_no_signals() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config.clone());

        let prices = create_test_prices(100, 0.0);
        let signals = create_test_signals(100, false);

        let result = backtester.run(&prices, &signals, 252);

        assert_eq!(result.num_trades, 0);
        assert!((result.total_return).abs() < 0.01);
    }

    #[test]
    fn test_backtest_metrics() {
        let config = BacktestConfig::default();
        let mut backtester = Backtester::new(config);

        let prices = create_test_prices(200, 0.3);
        let signals = create_test_signals(200, true);

        let result = backtester.run(&prices, &signals, 252);

        assert!(result.volatility >= 0.0);
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }

    #[test]
    fn test_backtest_result_summary() {
        let result = BacktestResult {
            trades: vec![],
            equity_curve: vec![100000.0, 110000.0],
            total_return: 0.10,
            annual_return: 0.12,
            volatility: 0.15,
            sharpe_ratio: 0.80,
            sortino_ratio: 1.20,
            max_drawdown: 0.05,
            calmar_ratio: 2.40,
            win_rate: 0.60,
            avg_profit: 500.0,
            avg_loss: 300.0,
            profit_factor: 1.67,
            num_trades: 50,
            avg_trade_duration: 5.0,
        };

        let summary = result.summary();
        assert!(summary.contains("Total Return: 10.00%"));
        assert!(summary.contains("Sharpe Ratio: 0.80"));
    }
}
