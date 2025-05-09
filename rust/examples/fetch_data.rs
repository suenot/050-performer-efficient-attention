//! Example: Fetch cryptocurrency data from Bybit

use performer::{BybitClient, DataLoader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("Performer - Fetch Data Example");
    println!("==============================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch BTCUSDT hourly data
    println!("Fetching BTCUSDT hourly data...");
    let btc_klines = client.get_klines("BTCUSDT", "60", 500).await?;
    println!("Fetched {} candles for BTCUSDT", btc_klines.len());

    // Show sample data
    if let Some(kline) = btc_klines.last() {
        println!("\nLatest candle:");
        println!("  Timestamp: {}", kline.timestamp);
        println!("  Open:  ${:.2}", kline.open);
        println!("  High:  ${:.2}", kline.high);
        println!("  Low:   ${:.2}", kline.low);
        println!("  Close: ${:.2}", kline.close);
        println!("  Volume: {:.2}", kline.volume);
    }

    // Fetch multiple symbols
    println!("\nFetching data for multiple symbols...");
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let multi_data = client.get_multi_klines(&symbols, "60", 200).await?;

    for (symbol, klines) in &multi_data {
        println!("  {}: {} candles", symbol, klines.len());
    }

    // Prepare dataset
    println!("\nPreparing dataset...");
    let loader = DataLoader::new()
        .seq_len(64)
        .target_horizon(24)
        .normalize(true);

    let dataset = loader.prepare_dataset(&btc_klines)?;
    println!("Dataset prepared:");
    println!("  Samples: {}", dataset.len());
    println!("  Sequence length: {}", dataset.seq_len);
    println!("  Features: {}", dataset.num_features);

    // Get current ticker
    println!("\nFetching current ticker...");
    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("BTCUSDT Ticker:");
    println!("  Last price: ${:.2}", ticker.last_price);
    if let Some(change) = ticker.price_change_24h_pct {
        println!("  24h change: {:.2}%", change * 100.0);
    }
    if let Some(volume) = ticker.volume_24h {
        println!("  24h volume: {:.2} BTC", volume);
    }

    println!("\nData fetch complete!");

    Ok(())
}
