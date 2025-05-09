//! Example: Make predictions with Performer model

use performer::{DataLoader, PerformerConfig, PerformerModel, SignalGenerator};
use performer::api::Kline;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Performer - Prediction Example");
    println!("==============================\n");

    // Generate synthetic data
    println!("Generating synthetic price data...");
    let klines = generate_synthetic_klines(500);

    // Prepare dataset
    let loader = DataLoader::new()
        .seq_len(64)
        .target_horizon(12)
        .normalize(true);

    let dataset = loader.prepare_dataset(&klines)?;
    println!("Prepared {} samples for prediction\n", dataset.len());

    // Create model
    let config = PerformerConfig {
        input_dim: dataset.num_features,
        d_model: 32,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        num_features: 64,
        seq_len: dataset.seq_len,
        output_dim: 1,
        causal: true,
        use_orthogonal: true,
        ..Default::default()
    };

    let mut model = PerformerModel::new(config);

    // Make predictions
    println!("Making predictions...");
    let predictions = model.predict(&dataset);
    println!("Generated {} predictions\n", predictions.len());

    // Prediction statistics
    let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let std_pred = (predictions
        .iter()
        .map(|p| (p - mean_pred).powi(2))
        .sum::<f64>() / predictions.len() as f64)
        .sqrt();
    let min_pred = predictions.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_pred = predictions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Prediction Statistics:");
    println!("  Mean: {:.6}", mean_pred);
    println!("  Std:  {:.6}", std_pred);
    println!("  Min:  {:.6}", min_pred);
    println!("  Max:  {:.6}", max_pred);

    // Generate trading signals
    println!("\nGenerating trading signals...");
    let signal_generator = SignalGenerator::new();

    // Use uniform confidence for demonstration
    let signals = signal_generator.generate_batch_uniform_confidence(
        &predictions,
        &dataset.timestamps,
        0.7,
    );

    // Count signal types
    let long_count = signals.iter().filter(|s| s.signal_type == performer::strategy::SignalType::Long).count();
    let short_count = signals.iter().filter(|s| s.signal_type == performer::strategy::SignalType::Short).count();
    let hold_count = signals.iter().filter(|s| s.signal_type == performer::strategy::SignalType::Hold).count();

    println!("\nSignal Distribution:");
    println!("  Long:  {} ({:.1}%)", long_count, long_count as f64 / signals.len() as f64 * 100.0);
    println!("  Short: {} ({:.1}%)", short_count, short_count as f64 / signals.len() as f64 * 100.0);
    println!("  Hold:  {} ({:.1}%)", hold_count, hold_count as f64 / signals.len() as f64 * 100.0);

    // Show recent signals
    println!("\nRecent Signals (last 10):");
    println!("{:>6} {:>12} {:>12} {:>10}", "Index", "Prediction", "Signal", "Position");
    for (i, signal) in signals.iter().rev().take(10).enumerate() {
        let idx = signals.len() - 1 - i;
        let signal_str = match signal.signal_type {
            performer::strategy::SignalType::Long => "LONG",
            performer::strategy::SignalType::Short => "SHORT",
            performer::strategy::SignalType::Hold => "HOLD",
            performer::strategy::SignalType::Close => "CLOSE",
        };
        println!(
            "{:>6} {:>12.6} {:>12} {:>10.2}%",
            idx,
            predictions[idx],
            signal_str,
            signal.position_size * 100.0
        );
    }

    println!("\nPrediction complete!");

    Ok(())
}

/// Generate synthetic price data
fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(n);
    let mut price = 100.0;

    for i in 0..n {
        let trend = 0.00005 * i as f64;
        let seasonality = 3.0 * (2.0 * PI * i as f64 / 80.0).sin();
        let noise = rand_normal() * 1.5;

        let change = trend + seasonality * 0.01 + noise * 0.01;
        price *= 1.0 + change;

        let open = price;
        let close = price * (1.0 + rand_normal() * 0.003);
        let high = open.max(close) * (1.0 + rand_normal().abs() * 0.008);
        let low = open.min(close) * (1.0 - rand_normal().abs() * 0.008);
        let volume = 1000.0 + rand_normal().abs() * 300.0;

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
