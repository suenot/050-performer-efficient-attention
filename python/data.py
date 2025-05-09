"""
Data Loading and Preprocessing for Performer

Provides:
- BybitClient: API client for fetching crypto data
- PerformerDataset: PyTorch dataset for training
- Feature engineering utilities
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Kline:
    """Single candlestick data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BybitClient:
    """
    Simple Bybit API client for fetching market data.

    Example:
        client = BybitClient()
        klines = client.get_klines("BTCUSDT", "1h", limit=1000)
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str = "60",  # 60 = 1 hour
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """
        Fetch historical klines from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of Kline objects
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                logger.error(f"API error: {data.get('retMsg')}")
                return []

            klines = []
            for item in data.get("result", {}).get("list", []):
                klines.append(Kline(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5])
                ))

            # Bybit returns newest first, reverse for chronological order
            return list(reversed(klines))

        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return []

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker information"""
        endpoint = f"{self.BASE_URL}/v5/market/tickers"

        params = {
            "category": "linear",
            "symbol": symbol
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") == 0:
                tickers = data.get("result", {}).get("list", [])
                return tickers[0] if tickers else None
            return None

        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return None


def load_bybit_data(
    symbol: str,
    interval: str = "60",
    days: int = 30,
    client: Optional[BybitClient] = None
) -> pd.DataFrame:
    """
    Load historical data from Bybit.

    Args:
        symbol: Trading pair
        interval: Candle interval
        days: Number of days of history
        client: Optional pre-existing client

    Returns:
        DataFrame with OHLCV data
    """
    if client is None:
        client = BybitClient()

    # Calculate time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    current_end = end_time

    # Fetch in batches (Bybit has 1000 limit)
    while current_end > start_time:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=1000,
            end_time=current_end
        )

        if not klines:
            break

        all_klines = klines + all_klines
        current_end = klines[0].timestamp - 1

        # Rate limiting
        time.sleep(0.1)

    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': k.timestamp,
        'open': k.open,
        'high': k.high,
        'low': k.low,
        'close': k.close,
        'volume': k.volume
    } for k in all_klines])

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

    return df


def compute_features(df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute technical features for Performer input.

    Args:
        df: DataFrame with OHLCV data
        feature_list: List of features to compute

    Returns:
        DataFrame with computed features
    """
    if feature_list is None:
        feature_list = ['log_return', 'volume_change', 'volatility', 'rsi', 'ma_ratio', 'high_low_range']

    features = pd.DataFrame(index=df.index)

    # Log returns
    if 'log_return' in feature_list:
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volume change ratio
    if 'volume_change' in feature_list:
        vol_ma = df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'] / vol_ma.replace(0, np.nan)

    # Realized volatility (20-period)
    if 'volatility' in feature_list:
        log_ret = np.log(df['close'] / df['close'].shift(1))
        features['volatility'] = log_ret.rolling(20).std()

    # RSI (14-period)
    if 'rsi' in feature_list:
        features['rsi'] = compute_rsi(df['close'], 14) / 100.0  # Normalize to [0, 1]

    # Moving average ratio
    if 'ma_ratio' in feature_list:
        ma_50 = df['close'].rolling(50).mean()
        features['ma_ratio'] = df['close'] / ma_50.replace(0, np.nan)

    # High-low range (normalized)
    if 'high_low_range' in feature_list:
        features['high_low_range'] = (df['high'] - df['low']) / df['close']

    # MACD
    if 'macd' in feature_list:
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema_12 - ema_26) / df['close']

    # Bollinger Band position
    if 'bb_position' in feature_list:
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        upper = ma_20 + 2 * std_20
        lower = ma_20 - 2 * std_20
        features['bb_position'] = (df['close'] - lower) / (upper - lower).replace(0, np.nan)

    return features


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class PerformerDataset(Dataset):
    """
    PyTorch Dataset for Performer training.

    Creates overlapping sequences from time series data.

    Example:
        dataset = PerformerDataset(X, y, seq_len=256)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 256,
        stride: int = 1
    ):
        """
        Args:
            X: Feature array [num_samples, num_features]
            y: Target array [num_samples, horizon] or [num_samples]
            seq_len: Sequence length for each sample
            stride: Step size between sequences
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.stride = stride

        # Calculate valid indices
        self.indices = list(range(seq_len, len(X), stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end_idx = self.indices[idx]
        start_idx = end_idx - self.seq_len

        x_seq = self.X[start_idx:end_idx]
        y_target = self.y[end_idx - 1] if self.y.dim() == 1 else self.y[end_idx - 1]

        return x_seq, y_target


def prepare_performer_data(
    symbols: List[str],
    lookback: int = 512,
    horizon: int = 24,
    features: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    interval: str = "60",
    days: int = 90
) -> Dict:
    """
    Prepare data for Performer training.

    Performer's linear complexity allows for longer lookback windows
    compared to standard Transformers.

    Args:
        symbols: List of trading pairs
        lookback: Sequence length (can be large with Performer!)
        horizon: Prediction horizon
        features: Features to compute
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        interval: Candle interval
        days: Days of history to fetch

    Returns:
        Dictionary with train/val/test data and metadata
    """
    if features is None:
        features = ['log_return', 'volume_change', 'volatility', 'rsi', 'ma_ratio', 'high_low_range']

    client = BybitClient()
    all_features = []
    all_targets = []

    for symbol in symbols:
        logger.info(f"Loading data for {symbol}...")
        df = load_bybit_data(symbol, interval=interval, days=days, client=client)

        if df.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        # Compute features
        feat_df = compute_features(df, features)

        # Target: future log returns
        target = np.log(df['close'].shift(-horizon) / df['close'])

        # Combine and drop NaN
        combined = pd.concat([feat_df, target.rename('target')], axis=1).dropna()

        all_features.append(combined[features].values)
        all_targets.append(combined['target'].values)

    if not all_features:
        raise ValueError("No data loaded for any symbol")

    # For single symbol, use directly; for multiple, could concatenate or create multi-asset input
    X = all_features[0]  # [num_samples, num_features]
    y = all_targets[0]   # [num_samples]

    # Create sequences
    sequences = []
    targets = []

    for i in range(lookback, len(X) - horizon):
        sequences.append(X[i-lookback:i])
        targets.append(y[i])

    X_seq = np.array(sequences)  # [num_sequences, lookback, num_features]
    y_seq = np.array(targets)    # [num_sequences]

    # Split data
    n_samples = len(X_seq)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    return {
        'X_train': X_seq[:train_end],
        'y_train': y_seq[:train_end],
        'X_val': X_seq[train_end:val_end],
        'y_val': y_seq[train_end:val_end],
        'X_test': X_seq[val_end:],
        'y_test': y_seq[val_end:],
        'symbols': symbols,
        'features': features,
        'lookback': lookback,
        'horizon': horizon,
        'num_features': len(features)
    }


def create_dataloaders(
    data: Dict,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from prepared data.

    Args:
        data: Output from prepare_performer_data
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PerformerDataset(
        data['X_train'], data['y_train'],
        seq_len=data['lookback']
    )
    val_dataset = PerformerDataset(
        data['X_val'], data['y_val'],
        seq_len=data['lookback']
    )
    test_dataset = PerformerDataset(
        data['X_test'], data['y_test'],
        seq_len=data['lookback']
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Test with synthetic data (to avoid API calls in test)
    print("\nCreating synthetic test data...")
    np.random.seed(42)

    # Simulate price data
    n_samples = 10000
    prices = 50000 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.001))

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
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()}")

    # Create dataset
    X = features.dropna().values
    y = np.random.randn(len(X))

    dataset = PerformerDataset(X, y, seq_len=256)
    print(f"Dataset size: {len(dataset)}")

    x_sample, y_sample = dataset[0]
    print(f"Sample input shape: {x_sample.shape}")
    print(f"Sample target shape: {y_sample.shape}")

    # Create dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch input shape: {batch_x.shape}")
    print(f"Batch target shape: {batch_y.shape}")

    print("\nAll data tests passed!")
