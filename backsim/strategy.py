"""
Base Strategy implementation.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any

from .universe import DataSliceRequest
from .portfolio import Portfolio


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    def __init__(self):
        """Initialize strategy."""
        pass

    @abstractmethod
    def generate_orders(
        self,
        price_slice: Dict[str, Dict[str, List[float]]],
        portfolio: Portfolio,
        timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """
        Generate orders based on current market data and portfolio state.

        Args:
            price_slice: Price/volume data for relevant symbols
            portfolio: Current portfolio state
            timestamp: Current simulation timestamp

        Returns:
            List of order dictionaries
        """
        pass

    @abstractmethod
    def data_slicer(self, timestamp: datetime) -> DataSliceRequest:
        """
        Define data requirements for the strategy.

        Args:
            timestamp: Current simulation timestamp

        Returns:
            DataSliceRequest specifying required data
        """
        pass


class SimpleMovingAverageCrossover(Strategy):
    """
    Example strategy implementing a simple moving average crossover.
    """
    def __init__(
        self,
        symbol: str,
        short_window: int = 10,
        long_window: int = 20
    ):
        """
        Initialize strategy.

        Args:
            symbol: Trading symbol
            short_window: Short moving average window
            long_window: Long moving average window
        """
        super().__init__()
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window

    def data_slicer(self, timestamp: datetime) -> DataSliceRequest:
        """Define data requirements."""
        return DataSliceRequest(
            symbols=[self.symbol],
            fields=["close"],
            lookback=self.long_window,
            frequency="1d"
        )

    def generate_orders(
        self,
        price_slice: Dict[str, Dict[str, List[float]]],
        portfolio: Portfolio,
        timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on MA crossover."""
        if not price_slice or self.symbol not in price_slice:
            return []

        closes = price_slice[self.symbol]["close"]
        if len(closes) < self.long_window:
            return []

        # Calculate moving averages
        short_ma = sum(closes[-self.short_window:]) / self.short_window
        long_ma = sum(closes[-self.long_window:]) / self.long_window

        # Generate orders based on crossover
        current_position = portfolio.positions.get(self.symbol)
        orders = []

        if short_ma > long_ma and not current_position:
            # Buy signal
            orders.append({
                "symbol": self.symbol,
                "quantity": 100,  # Fixed position size
                "side": "BUY",
                "timestamp": timestamp
            })
        elif short_ma < long_ma and current_position:
            # Sell signal
            orders.append({
                "symbol": self.symbol,
                "quantity": current_position.quantity,
                "side": "SELL",
                "timestamp": timestamp
            })

        return orders
