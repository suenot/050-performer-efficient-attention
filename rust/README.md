# Performer Rust Implementation

Efficient attention mechanism using FAVOR+ (Fast Attention Via positive Orthogonal Random features) for financial time series prediction with Bybit data.

## Features

- FAVOR+ attention with O(L) complexity instead of O(L^2)
- Positive orthogonal random features for stable approximation
- Bybit API integration for cryptocurrency data
- Trading strategy with backtesting framework
- Performance metrics (Sharpe ratio, Sortino ratio, max drawdown)

## Building

```bash
cargo build --release
```

## Running Examples

### Fetch Data
```bash
cargo run --example fetch_data
```

### Train Model
```bash
cargo run --example train
```

### Make Predictions
```bash
cargo run --example predict
```

### Run Backtest
```bash
cargo run --example backtest
```

## Architecture

```
rust/
├── src/
│   ├── lib.rs          # Library root with exports
│   ├── api/            # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   └── types.rs
│   ├── data/           # Data loading and features
│   │   ├── mod.rs
│   │   ├── loader.rs
│   │   ├── features.rs
│   │   └── dataset.rs
│   ├── model/          # Performer model
│   │   ├── mod.rs
│   │   ├── config.rs
│   │   ├── attention.rs    # FAVOR+ attention
│   │   ├── embedding.rs
│   │   └── performer.rs
│   └── strategy/       # Trading strategy
│       ├── mod.rs
│       ├── signals.rs
│       └── backtest.rs
└── examples/
    ├── fetch_data.rs
    ├── train.rs
    ├── predict.rs
    └── backtest.rs
```

## Key Concepts

### FAVOR+ Attention

The Performer uses FAVOR+ to approximate softmax attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
                   ≈ φ(Q) (φ(K)^T V)
```

Where φ is the positive random feature map:

```
φ(x) = exp(-||x||^2 / 2) * [exp(ω_1 · x), ..., exp(ω_m · x)]
```

This decomposition allows computing attention in O(L) time instead of O(L^2).

### Orthogonal Random Features

We use orthogonal random features via Gram-Schmidt orthogonalization to reduce variance of the approximation.

## License

MIT
