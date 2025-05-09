"""
Example Usage of Performer for Financial Time Series Prediction

This script demonstrates:
1. Creating a Performer model
2. Training on synthetic data
3. Making predictions
4. Running a simple backtest
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import PerformerForecaster, PerformerConfig, OutputType
from data import PerformerDataset, compute_features
from strategy import (
    BacktestConfig,
    backtest_performer_strategy,
    generate_signals,
    calculate_sharpe_ratio
)


def create_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 6,
    seq_len: int = 256,
    horizon: int = 24
):
    """
    Create synthetic financial data for demonstration.

    In production, use real data from Bybit or other sources.
    """
    np.random.seed(42)

    # Simulate price process (geometric Brownian motion)
    returns = np.random.randn(n_samples) * 0.02  # 2% daily vol
    prices = 50000 * np.exp(np.cumsum(returns))

    # Create features
    features = np.zeros((n_samples, n_features))

    # Feature 1: Log returns
    features[1:, 0] = np.log(prices[1:] / prices[:-1])

    # Feature 2: Volatility (20-period rolling)
    for i in range(20, n_samples):
        features[i, 1] = np.std(features[i-20:i, 0])

    # Feature 3: Momentum (5-period)
    features[5:, 2] = np.log(prices[5:] / prices[:-5])

    # Feature 4: Volume (random but correlated with volatility)
    volume = 1000000 * (1 + 0.5 * np.abs(features[:, 0]) + 0.3 * np.random.randn(n_samples))
    features[:, 3] = volume / volume.mean()

    # Feature 5: RSI-like oscillator
    for i in range(14, n_samples):
        ups = features[i-14:i, 0][features[i-14:i, 0] > 0].sum()
        downs = -features[i-14:i, 0][features[i-14:i, 0] < 0].sum()
        if ups + downs > 0:
            features[i, 4] = ups / (ups + downs)
        else:
            features[i, 4] = 0.5

    # Feature 6: Moving average ratio
    ma_50 = np.convolve(prices, np.ones(50)/50, mode='same')
    features[:, 5] = prices / ma_50

    # Create targets (future returns)
    targets = np.zeros(n_samples)
    targets[:-horizon] = np.log(prices[horizon:] / prices[:-horizon])

    # Create sequences
    X_seq = []
    y_seq = []

    for i in range(seq_len + 50, n_samples - horizon):  # Skip initial NaN values
        X_seq.append(features[i-seq_len:i])
        y_seq.append(targets[i])

    X = np.array(X_seq, dtype=np.float32)
    y = np.array(y_seq, dtype=np.float32)

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def train_performer(
    model: PerformerForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = 'cpu'
):
    """Train the Performer model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_x)
            predictions = output['predictions']

            # Handle multi-horizon predictions
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predictions = predictions[:, 0]  # Use first step

            loss = criterion(predictions.squeeze(), batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                predictions = output['predictions']

                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    predictions = predictions[:, 0]

                loss = criterion(predictions.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_performer_model.pt')

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # Load best model
    model.load_state_dict(torch.load('best_performer_model.pt'))
    return model


def main():
    """Main example demonstrating Performer for trading."""
    print("=" * 60)
    print("Performer for Financial Time Series Prediction")
    print("=" * 60)

    # Configuration
    SEQ_LEN = 256  # Performer can handle long sequences!
    BATCH_SIZE = 32
    EPOCHS = 5  # Reduce for demo
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nUsing device: {DEVICE}")

    # 1. Create synthetic data
    print("\n1. Creating synthetic data...")
    X, y = create_synthetic_data(n_samples=5000, seq_len=SEQ_LEN)
    print(f"   Data shape: X={X.shape}, y={y.shape}")

    # Split data
    n_train = int(len(X) * 0.7)
    n_val = int(len(X) * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. Create Performer model
    print("\n2. Creating Performer model...")
    config = PerformerConfig(
        input_features=6,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_features=32,  # Random features for FAVOR+
        use_orthogonal=True,
        dropout=0.1,
        max_seq_len=SEQ_LEN * 2,
        prediction_horizon=1,
        output_type=OutputType.REGRESSION
    )

    model = PerformerForecaster(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    print(f"   Attention complexity: O(L) instead of O(L^2)")
    print(f"   Random features: {config.computed_num_features}")

    # 3. Train model
    print("\n3. Training model...")
    model = train_performer(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=1e-4, device=DEVICE
    )

    # 4. Evaluate predictions
    print("\n4. Evaluating predictions...")
    model.eval()

    all_preds = []
    all_actual = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x)
            predictions = output['predictions']

            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predictions = predictions[:, 0]

            all_preds.extend(predictions.squeeze().cpu().numpy())
            all_actual.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_actual = np.array(all_actual)

    # Correlation between predictions and actual
    correlation = np.corrcoef(all_preds, all_actual)[0, 1]
    print(f"   Prediction correlation: {correlation:.4f}")

    # Direction accuracy
    direction_correct = ((all_preds > 0) == (all_actual > 0)).mean()
    print(f"   Direction accuracy: {direction_correct:.2%}")

    # 5. Backtest simple strategy
    print("\n5. Running backtest...")
    backtest_config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        long_threshold=0.002,
        short_threshold=-0.002,
        allow_short=True
    )

    # Generate signals
    signals = generate_signals(all_preds, config=backtest_config)
    long_signals = sum(1 for s in signals if s.signal_type.value == 1)
    short_signals = sum(1 for s in signals if s.signal_type.value == -1)
    hold_signals = sum(1 for s in signals if s.signal_type.value == 0)
    print(f"   Signals: Long={long_signals}, Short={short_signals}, Hold={hold_signals}")

    # Simple backtest (without model)
    capital = backtest_config.initial_capital
    position = 0.0
    returns_list = []

    for i, signal in enumerate(signals):
        if i >= len(all_actual):
            break

        actual_ret = all_actual[i]

        # Update position based on signal
        new_position = signal.signal_type.value * backtest_config.max_position_size

        # Trading cost
        cost = abs(new_position - position) * backtest_config.commission * capital

        # PnL
        pnl = new_position * actual_ret * capital - cost
        capital += pnl
        position = new_position

        returns_list.append(pnl / (capital - pnl) if capital != pnl else 0)

    returns_array = np.array(returns_list)
    total_return = (capital - backtest_config.initial_capital) / backtest_config.initial_capital
    sharpe = calculate_sharpe_ratio(returns_array)

    print(f"\n   Backtest Results:")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Final Capital: ${capital:,.2f}")

    # 6. Test attention patterns (optional)
    print("\n6. Analyzing attention patterns...")
    with torch.no_grad():
        sample_x = torch.tensor(X_test[:1]).to(DEVICE)
        output = model(sample_x, return_attention=True)

        if output['attention_weights']:
            for layer_name, attn in output['attention_weights'].items():
                print(f"   {layer_name}: attention shape = {attn.shape}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    # Clean up
    import os
    if os.path.exists('best_performer_model.pt'):
        os.remove('best_performer_model.pt')


if __name__ == "__main__":
    main()
