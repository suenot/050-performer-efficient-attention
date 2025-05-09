"""
Trading Strategy and Backtesting for Performer

Provides:
- Signal generation from model predictions
- Backtesting framework with risk management
- Performance metrics calculation
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    LONG = 1
    SHORT = -1
    HOLD = 0


@dataclass
class Signal:
    """Single trading signal"""
    timestamp: int
    signal_type: SignalType
    confidence: float
    predicted_return: float


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    use_stop_loss: bool = True
    stop_loss_level: float = 0.05  # 5% stop loss
    use_take_profit: bool = False
    take_profit_level: float = 0.10  # 10% take profit
    long_threshold: float = 0.001  # Minimum predicted return for long
    short_threshold: float = -0.001  # Maximum predicted return for short
    allow_short: bool = True


@dataclass
class BacktestResult:
    """Results from backtesting"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    capital_history: List[float] = field(default_factory=list)
    returns_history: List[float] = field(default_factory=list)
    positions_history: List[float] = field(default_factory=list)
    signals_history: List[Signal] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary string"""
        return f"""
Backtest Results:
================
Total Return: {self.total_return * 100:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown * 100:.2f}%
Win Rate: {self.win_rate * 100:.2f}%
Profit Factor: {self.profit_factor:.2f}
Number of Trades: {self.num_trades}
"""


def generate_signals(
    predictions: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    config: Optional[BacktestConfig] = None
) -> List[Signal]:
    """
    Generate trading signals from model predictions.

    Args:
        predictions: Array of predicted returns [num_samples]
        confidences: Optional confidence scores [num_samples]
        config: Backtest configuration

    Returns:
        List of Signal objects
    """
    if config is None:
        config = BacktestConfig()

    if confidences is None:
        confidences = np.ones_like(predictions)

    signals = []

    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        if pred > config.long_threshold:
            signal_type = SignalType.LONG
        elif pred < config.short_threshold and config.allow_short:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.HOLD

        signals.append(Signal(
            timestamp=i,
            signal_type=signal_type,
            confidence=float(conf),
            predicted_return=float(pred)
        ))

    return signals


def calculate_sharpe_ratio(returns: np.ndarray, annualization_factor: float = 252) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.sqrt(annualization_factor) * np.mean(returns) / np.std(returns)


def calculate_sortino_ratio(returns: np.ndarray, target: float = 0, annualization_factor: float = 252) -> float:
    """Calculate annualized Sortino ratio"""
    excess = returns - target
    downside = returns[returns < target]

    if len(downside) == 0:
        return float('inf') if np.mean(excess) > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return float('inf') if np.mean(excess) > 0 else 0.0

    return np.sqrt(annualization_factor) * np.mean(excess) / downside_std


def calculate_max_drawdown(capital_history: List[float]) -> float:
    """Calculate maximum drawdown"""
    if len(capital_history) == 0:
        return 0.0

    peak = capital_history[0]
    max_dd = 0.0

    for capital in capital_history:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        max_dd = max(max_dd, drawdown)

    return max_dd


def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor (gross profit / gross loss)"""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


class PerformerStrategy:
    """
    Trading strategy using Performer model predictions.

    Example:
        strategy = PerformerStrategy(model, config)
        signals = strategy.generate_signals(data)
        result = strategy.backtest(data, actual_returns)
    """

    def __init__(self, model, config: Optional[BacktestConfig] = None):
        """
        Args:
            model: Trained Performer model
            config: Backtest configuration
        """
        self.model = model
        self.config = config or BacktestConfig()
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'

    def predict(
        self,
        x: torch.Tensor,
        return_confidence: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions from model.

        Args:
            x: Input tensor [batch, seq_len, features]
            return_confidence: Whether to return confidence estimates

        Returns:
            predictions: Array of predicted returns
            confidences: Optional array of confidence scores
        """
        self.model.eval()

        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            predictions = output['predictions'].cpu().numpy()

            # Handle multi-horizon predictions (use first step)
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predictions = predictions[:, 0]

            confidences = None
            if return_confidence and 'confidence' in output and output['confidence'] is not None:
                confidences = output['confidence'].cpu().numpy()
                if len(confidences.shape) > 1:
                    confidences = confidences[:, 0]

        return predictions, confidences

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty estimation via MC Dropout.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean_predictions: Mean predictions
            std_predictions: Standard deviation (uncertainty)
        """
        self.model.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            x = x.to(self.device)
            for _ in range(n_samples):
                output = self.model(x)
                pred = output['predictions'].cpu().numpy()
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]
                predictions.append(pred)

        self.model.eval()

        predictions = np.stack(predictions)
        return predictions.mean(axis=0), predictions.std(axis=0)


def backtest_performer_strategy(
    model,
    test_loader: DataLoader,
    config: Optional[BacktestConfig] = None,
    verbose: bool = True
) -> BacktestResult:
    """
    Backtest Performer-based trading strategy.

    Args:
        model: Trained Performer model
        test_loader: DataLoader with test data
        config: Backtest configuration
        verbose: Print progress

    Returns:
        BacktestResult with all metrics
    """
    if config is None:
        config = BacktestConfig()

    strategy = PerformerStrategy(model, config)

    capital = config.initial_capital
    position = 0.0  # Current position (-1 to 1)
    entry_price = 0.0

    capital_history = [capital]
    returns_history = []
    positions_history = []
    signals_history = []

    num_trades = 0

    for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
        # Get predictions
        predictions, confidences = strategy.predict(batch_x)
        actual_returns = batch_y.numpy()

        # Process each sample in batch
        for i in range(len(predictions)):
            pred_return = predictions[i] if np.isscalar(predictions[i]) else predictions[i].item()
            actual_return = actual_returns[i] if np.isscalar(actual_returns[i]) else actual_returns[i].item()
            confidence = 1.0 if confidences is None else (
                confidences[i] if np.isscalar(confidences[i]) else confidences[i].item()
            )

            # Generate signal
            if pred_return > config.long_threshold:
                target_position = min(1.0, confidence) * config.max_position_size
            elif pred_return < config.short_threshold and config.allow_short:
                target_position = -min(1.0, confidence) * config.max_position_size
            else:
                target_position = 0.0

            # Record signal
            signal = Signal(
                timestamp=batch_idx * test_loader.batch_size + i,
                signal_type=SignalType.LONG if target_position > 0 else (
                    SignalType.SHORT if target_position < 0 else SignalType.HOLD
                ),
                confidence=confidence,
                predicted_return=pred_return
            )
            signals_history.append(signal)

            # Calculate trading costs
            position_change = abs(target_position - position)
            if position_change > 0.01:  # Only count as trade if significant change
                num_trades += 1
                trading_cost = position_change * (config.commission + config.slippage) * capital
            else:
                trading_cost = 0.0

            # Check stop loss
            if config.use_stop_loss and position != 0:
                current_pnl = position * actual_return
                if current_pnl < -config.stop_loss_level:
                    target_position = 0.0
                    trading_cost += abs(position) * (config.commission + config.slippage) * capital
                    num_trades += 1

            # Update position
            position = target_position

            # Calculate PnL
            pnl = position * actual_return * capital - trading_cost
            capital += pnl

            # Record history
            if capital_history:
                period_return = pnl / capital_history[-1]
            else:
                period_return = 0.0
            returns_history.append(period_return)
            capital_history.append(capital)
            positions_history.append(position)

        if verbose and (batch_idx + 1) % 100 == 0:
            logger.info(f"Processed {batch_idx + 1} batches, Capital: ${capital:,.2f}")

    # Calculate metrics
    returns_array = np.array(returns_history)

    result = BacktestResult(
        total_return=(capital - config.initial_capital) / config.initial_capital,
        sharpe_ratio=calculate_sharpe_ratio(returns_array),
        sortino_ratio=calculate_sortino_ratio(returns_array),
        max_drawdown=calculate_max_drawdown(capital_history),
        win_rate=(returns_array > 0).sum() / len(returns_array) if len(returns_array) > 0 else 0.0,
        profit_factor=calculate_profit_factor(returns_array),
        num_trades=num_trades,
        capital_history=capital_history,
        returns_history=returns_history,
        positions_history=positions_history,
        signals_history=signals_history
    )

    if verbose:
        print(result.summary())

    return result


def plot_backtest_results(result: BacktestResult, save_path: Optional[str] = None):
    """
    Plot backtest results.

    Args:
        result: BacktestResult object
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Capital curve
    axes[0].plot(result.capital_history, label='Portfolio Value', color='blue')
    axes[0].set_ylabel('Capital ($)')
    axes[0].set_title('Performer Strategy Backtest Results')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Returns distribution
    returns = np.array(result.returns_history)
    cumulative_returns = (1 + returns).cumprod()
    axes[1].plot(cumulative_returns, label='Cumulative Returns', color='green')
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Cumulative Return')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Positions
    axes[2].fill_between(
        range(len(result.positions_history)),
        result.positions_history,
        alpha=0.5,
        label='Position'
    )
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Position')
    axes[2].set_xlabel('Time Step')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved backtest plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Test strategy components
    print("Testing strategy components...")

    # Test signal generation
    predictions = np.array([0.002, -0.003, 0.0005, 0.001, -0.0008])
    signals = generate_signals(predictions)
    print(f"Generated {len(signals)} signals")
    for s in signals:
        print(f"  {s.signal_type.name}: pred={s.predicted_return:.4f}")

    # Test metrics
    returns = np.random.randn(252) * 0.01
    print(f"\nSharpe Ratio: {calculate_sharpe_ratio(returns):.2f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(returns):.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown([100 + sum(returns[:i]) for i in range(len(returns))]):.2%}")
    print(f"Profit Factor: {calculate_profit_factor(returns):.2f}")

    print("\nAll strategy tests passed!")
