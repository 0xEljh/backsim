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
        timestamp: datetime,
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
