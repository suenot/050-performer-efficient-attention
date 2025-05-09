//! Example: Backtest trading strategy with Performer model

use performer::{
    BacktestConfig, Backtester, DataLoader, PerformerConfig, PerformerModel,
    SignalGenerator,
};
use performer::api::Kline;
use performer::strategy::SignalGeneratorConfig;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Performer - Backtesting Example");
    println!("================================\n");

    // Generate synthetic data with trend
    println!("Generating synthetic price data with trend...");
    let klines = generate_synthetic_klines(800, 0.0002); // Slight upward trend

    // Prepare dataset
    let loader = DataLoader::new()
        .seq_len(64)
        .target_horizon(12)
        .normalize(true);

    let dataset = loader.prepare_dataset(&klines)?;
    println!("Prepared {} samples\n", dataset.len());

    // Split data
    let (train_dataset, test_dataset) = dataset.train_test_split(0.6);
    println!("Train: {} samples, Test: {} samples\n", train_dataset.len(), test_dataset.len());

    // Create model
    let config = PerformerConfig {
        input_dim: train_dataset.num_features,
        d_model: 32,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        num_features: 64,
        seq_len: train_dataset.seq_len,
        output_dim: 1,
        causal: true,
        use_orthogonal: true,
        ..Default::default()
    };

    let mut model = PerformerModel::new(config);

    // "Train" model (forward passes only for this demo)
    println!("Training model...");
    for _ in 0..5 {
        for (batch_inputs, batch_targets) in train_dataset.iter_batches(32) {
            let _ = model.compute_loss(&batch_inputs, &batch_targets);
        }
    }
    println!("Training complete.\n");

    // Make predictions on test set
    println!("Making predictions on test set...");
    let predictions = model.predict(&test_dataset);

    // Generate signals
    let signal_config = SignalGeneratorConfig {
        min_return_threshold: 0.001,
        min_confidence: 0.5,
        base_position_size: 0.1,
        scale_by_confidence: true,
        scale_by_return: false,
    };
    let signal_generator = SignalGenerator::with_config(signal_config);

    let signals = signal_generator.generate_batch_uniform_confidence(
        &predictions,
        &test_dataset.timestamps,
        0.7,
    );

    // Run backtest
    println!("Running backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: 0.001,
        slippage: 0.0005,
        max_leverage: 1.0,
        allow_short: false,
        min_position_size: 0.01,
        use_stop_loss: true,
        stop_loss_level: 0.03,
        use_take_profit: true,
        take_profit_level: 0.06,
    };

    let mut backtester = Backtester::new(backtest_config);
    let result = backtester.run(&test_dataset.close_prices, &signals, 252 * 24); // Hourly data

    // Print results
    println!("\n{}", result.summary());

    // Additional analysis
    println!("Trade Analysis:");
    println!("---------------");

    if !result.trades.is_empty() {
        let profitable_trades: Vec<_> = result.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<_> = result.trades.iter().filter(|t| t.pnl <= 0.0).collect();

        println!("  Profitable trades: {}", profitable_trades.len());
        println!("  Losing trades: {}", losing_trades.len());

        if !profitable_trades.is_empty() {
            let avg_profit_return: f64 = profitable_trades.iter().map(|t| t.return_pct).sum::<f64>()
                / profitable_trades.len() as f64;
            println!("  Avg profit return: {:.2}%", avg_profit_return * 100.0);
        }

        if !losing_trades.is_empty() {
            let avg_loss_return: f64 = losing_trades.iter().map(|t| t.return_pct).sum::<f64>()
                / losing_trades.len() as f64;
            println!("  Avg loss return: {:.2}%", avg_loss_return * 100.0);
        }

        // Exit reason breakdown
        println!("\nExit Reasons:");
        let mut stop_loss = 0;
        let mut take_profit = 0;
        let mut signal_exit = 0;
        let mut end_of_test = 0;

        for trade in &result.trades {
            match trade.exit_reason.as_str() {
                "Stop Loss" => stop_loss += 1,
                "Take Profit" => take_profit += 1,
                "Signal" => signal_exit += 1,
                _ => end_of_test += 1,
            }
        }

        println!("  Stop Loss: {}", stop_loss);
        println!("  Take Profit: {}", take_profit);
        println!("  Signal: {}", signal_exit);
        println!("  End of Backtest: {}", end_of_test);
    }

    // Show sample trades
    if !result.trades.is_empty() {
        println!("\nSample Trades (first 5):");
        println!("{:>8} {:>8} {:>10} {:>10} {:>10} {:>12}",
                 "Entry", "Exit", "Entry$", "Exit$", "PnL", "Reason");

        for trade in result.trades.iter().take(5) {
            println!(
                "{:>8} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>12}",
                trade.entry_time,
                trade.exit_time,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.exit_reason
            );
        }
    }

    // Equity curve stats
    println!("\nEquity Curve:");
    println!("  Start: ${:.2}", result.equity_curve.first().unwrap_or(&0.0));
    println!("  End: ${:.2}", result.equity_curve.last().unwrap_or(&0.0));
    println!("  Peak: ${:.2}", result.equity_curve.iter().cloned().fold(0.0, f64::max));
    println!("  Trough: ${:.2}", result.equity_curve.iter().cloned().fold(f64::INFINITY, f64::min));

    println!("\nBacktest complete!");

    Ok(())
}

/// Generate synthetic price data with configurable trend
fn generate_synthetic_klines(n: usize, trend: f64) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(n);
    let mut price = 100.0;

    for i in 0..n {
        let trend_component = trend * i as f64;
        let seasonality = 2.0 * (2.0 * PI * i as f64 / 120.0).sin();
        let noise = rand_normal() * 1.0;

        let change = trend_component + seasonality * 0.01 + noise * 0.01;
        price *= 1.0 + change;
        price = price.max(1.0); // Prevent negative prices

        let open = price;
        let close = price * (1.0 + rand_normal() * 0.004);
        let high = open.max(close) * (1.0 + rand_normal().abs() * 0.006);
        let low = open.min(close) * (1.0 - rand_normal().abs() * 0.006);
        let volume = 1000.0 + rand_normal().abs() * 200.0;

        klines.push(Kline {
            timestamp: i as u64 * 3600000,
            open,
            high,
            low,
            close,
            volume,
            turnover: close * volume,
        });

        price = close;
    }

    klines
}

fn rand_normal() -> f64 {
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
