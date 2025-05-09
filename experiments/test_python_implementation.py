#!/usr/bin/env python3
"""
Comprehensive test for the Python Performer implementation.

Tests:
1. FAVORPlusAttention module
2. PerformerEncoder
3. PerformerForecaster model
4. Data loading and feature computation
5. Signal generation and backtesting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pandas as pd
import torch

def test_favor_plus_attention():
    """Test FAVORPlusAttention module."""
    print("Testing FAVORPlusAttention...")

    from model import FAVORPlusAttention

    d_model = 64
    num_heads = 4
    num_features = 32
    batch_size = 8
    seq_len = 64

    attention = FAVORPlusAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_features=num_features,
        use_orthogonal=True,
        redraw_features=False,
        dropout=0.1
    )

    # Test forward pass - returns tuple (output, attention_weights)
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = attention(x)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    # Test with causal attention
    output_causal, _ = attention(x, causal=True)
    assert output_causal.shape == x.shape, f"Causal output shape mismatch"

    # Test with return_attention=True
    output_with_attn, attn_weights = attention(x, return_attention=True)
    assert output_with_attn.shape == x.shape, "Output with attention mismatch"
    assert attn_weights is not None, "Attention weights should be returned"

    print("  FAVORPlusAttention: PASSED")
    return True

def test_performer_encoder():
    """Test PerformerEncoder module with PerformerConfig."""
    print("Testing PerformerEncoder...")

    from model import PerformerEncoder, PerformerConfig

    config = PerformerConfig(
        input_features=6,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        num_features=32,
        use_orthogonal=True,
        dropout=0.1,
        max_seq_len=512
    )

    encoder = PerformerEncoder(config)

    # Test forward pass with raw features input
    batch_size = 4
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.input_features)
    output, attention_dict = encoder(x)

    expected_shape = (batch_size, seq_len, config.d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"

    print("  PerformerEncoder: PASSED")
    return True

def test_performer_forecaster():
    """Test PerformerForecaster model."""
    print("Testing PerformerForecaster...")

    from model import PerformerForecaster, PerformerConfig

    config = PerformerConfig(
        input_features=6,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        num_features=32,
        use_orthogonal=True,
        dropout=0.1,
        prediction_horizon=24
    )

    forecaster = PerformerForecaster(config)

    batch_size = 4
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.input_features)
    output = forecaster(x)

    # Output is a dictionary with 'predictions' key
    assert 'predictions' in output, "Output should contain 'predictions' key"
    predictions = output['predictions']

    expected_shape = (batch_size, config.prediction_horizon)
    assert predictions.shape == expected_shape, f"Expected shape {expected_shape}, got {predictions.shape}"
    assert not torch.isnan(predictions).any(), "Predictions contain NaN values"

    print("  PerformerForecaster: PASSED")
    return True

def test_data_loading():
    """Test data loading and feature computation."""
    print("Testing data loading...")

    from data import compute_features, PerformerDataset

    # Create synthetic price data
    np.random.seed(42)
    n_samples = 500
    prices = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.01))

    df = pd.DataFrame({
        'open': prices * (1 - np.random.rand(n_samples) * 0.001),
        'high': prices * (1 + np.random.rand(n_samples) * 0.002),
        'low': prices * (1 - np.random.rand(n_samples) * 0.002),
        'close': prices,
        'volume': np.random.rand(n_samples) * 1000000
    })
    df.index = pd.date_range('2024-01-01', periods=n_samples, freq='h')

    # Compute features
    features = compute_features(df)
    assert not features.empty, "Features DataFrame is empty"
    assert len(features.columns) >= 4, f"Expected at least 4 features, got {len(features.columns)}"

    # Create dataset
    X = features.dropna().values
    y = np.random.randn(len(X))

    dataset = PerformerDataset(X, y, seq_len=64)
    assert len(dataset) > 0, "Dataset is empty"

    x_sample, y_sample = dataset[0]
    assert x_sample.shape == (64, X.shape[1]), f"Sample shape mismatch: {x_sample.shape}"

    print("  Data loading: PASSED")
    return True

def test_signal_generation():
    """Test signal generation with BacktestConfig."""
    print("Testing signal generation...")

    from strategy import generate_signals, BacktestConfig

    # Generate mock predictions
    np.random.seed(42)
    predictions = np.random.randn(100) * 0.01

    # Generate signals with config
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        long_threshold=0.001,
        short_threshold=-0.001
    )

    signals = generate_signals(predictions, config=config)
    assert len(signals) == len(predictions), f"Signal length mismatch: {len(signals)} vs {len(predictions)}"

    # Check that signals are Signal objects
    from strategy import Signal
    assert all(isinstance(s, Signal) for s in signals), "Signals should be Signal objects"

    print("  Signal generation: PASSED")
    return True

def test_backtest_config():
    """Test BacktestConfig dataclass."""
    print("Testing BacktestConfig...")

    from strategy import BacktestConfig

    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    assert config.initial_capital == 100000
    assert config.commission == 0.001
    assert config.slippage == 0.0005

    print("  BacktestConfig: PASSED")
    return True

def test_end_to_end():
    """Test end-to-end pipeline."""
    print("Testing end-to-end pipeline...")

    from model import PerformerForecaster, PerformerConfig
    from data import compute_features, PerformerDataset
    from strategy import generate_signals, BacktestConfig
    from torch.utils.data import DataLoader

    # Create synthetic data
    np.random.seed(42)
    n_samples = 300
    prices = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.01))

    df = pd.DataFrame({
        'open': prices * (1 - np.random.rand(n_samples) * 0.001),
        'high': prices * (1 + np.random.rand(n_samples) * 0.002),
        'low': prices * (1 - np.random.rand(n_samples) * 0.002),
        'close': prices,
        'volume': np.random.rand(n_samples) * 1000000
    })
    df.index = pd.date_range('2024-01-01', periods=n_samples, freq='h')

    # Prepare data
    features = compute_features(df)
    clean_features = features.dropna()

    X = clean_features.values
    # Target: next period return (regression)
    y = np.log(df['close'].shift(-1) / df['close']).dropna().values[:len(X)]
    X = X[:len(y)]

    # Create dataset and dataloader
    dataset = PerformerDataset(X, y, seq_len=32)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Create model using config
    model_config = PerformerConfig(
        input_features=X.shape[1],
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_features=16,
        num_layers=1,
        dropout=0.0,
        prediction_horizon=1
    )

    model = PerformerForecaster(model_config)
    model.eval()

    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            output = model(batch_x)
            pred = output['predictions'].squeeze(-1)  # [batch, 1] -> [batch]
            predictions.extend(pred.numpy())

    predictions = np.array(predictions)
    assert len(predictions) > 0, "No predictions generated"
    assert not np.isnan(predictions).any(), "Predictions contain NaN"

    # Generate signals
    backtest_config = BacktestConfig(
        long_threshold=0.001,
        short_threshold=-0.001
    )
    signals = generate_signals(predictions, config=backtest_config)
    assert len(signals) == len(predictions), "Signal count mismatch"

    print("  End-to-end pipeline: PASSED")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Performer Python Implementation Tests")
    print("=" * 60)
    print()

    tests = [
        ("FAVOR+ Attention", test_favor_plus_attention),
        ("Performer Encoder", test_performer_encoder),
        ("Performer Forecaster", test_performer_forecaster),
        ("Data Loading", test_data_loading),
        ("Signal Generation", test_signal_generation),
        ("Backtest Config", test_backtest_config),
        ("End-to-End", test_end_to_end),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
