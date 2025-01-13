"""
Broker implementation for order execution and fill simulation.
"""
from datetime import datetime
from typing import Protocol, Optional, List
import logging
import pandas as pd
import numpy as np
from .portfolio import Portfolio, Order, OrderStatus, OrderType, OrderSide
from .universe import AssetUniverse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SlippageModel(Protocol):
    """Protocol for slippage models."""
    def get_fill_price(
        self,
        order: Order,
        market_price: float,
        timestamp: datetime
    ) -> float:
        """Calculate fill price including slippage."""
        ...


class FixedSlippage:
    """Simple fixed percentage slippage model."""
    def __init__(self, slippage_pct: float = 0.01):
        self.slippage_pct = slippage_pct

    def get_fill_price(
        self,
        order: Order,
        market_price: float,
        timestamp: datetime
    ) -> float:
        """
        Calculate fill price with fixed percentage slippage.

        Args:
            order: Order to fill
            market_price: Current market price
            timestamp: Current simulation time

        Returns:
            Fill price including slippage
        """
        slippage_factor = 1 + (
            self.slippage_pct if order.side == OrderSide.BUY else -self.slippage_pct
        )
        return market_price * slippage_factor



class PriceMatrixView:
    """
    Provides a clean interface for price matrix operations.
    Wraps a price matrix DataFrame with (datetime, symbol) multi-index.
    """
    
    def __init__(self, price_matrix: pd.DataFrame):
        """
        Initialize with a price matrix DataFrame.
        
        Args:
            price_matrix: DataFrame with (datetime, symbol) multi-index
        """
        self._price_matrix = price_matrix.ffill()
        
    def get_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get the latest price for a symbol at the given timestamp."""
        try:
            return (
                self._price_matrix
                .loc[:timestamp, symbol]
                .iloc[-1:]
                .squeeze() # convert to scalar
            )
        except (KeyError, IndexError):
            return None
        
    def get_prices(self, timestamp: datetime) -> pd.Series:
        """Get all prices for all symbols at or before the given timestamp."""
        return (
            self._price_matrix
            .loc[:timestamp]
            .reset_index(level=0)  
            .groupby('symbol', sort=False)
            .last()  
            .squeeze(axis=1)  # Return as Series
        )


class Broker:
    """
    Simulates order execution and generates fills.
    """
    def __init__(
        self,
        asset_universe: AssetUniverse,
        price_matrix: pd.DataFrame,
        slippage_model: Optional[SlippageModel] = None,
        fill_delay: int = 0
    ):
        """
        Initialize broker.

        Args:
            asset_universe: AssetUniverse for price data
            price_matrix: DataFrame with price data
            slippage_model: Optional model for price slippage
            fill_delay: Number of time steps to delay fills
        """
        self.asset_universe = asset_universe
        self.price_view = PriceMatrixView(price_matrix)
        self.slippage_model = slippage_model
        self.fill_delay = fill_delay

    def process_fills(self, portfolio: Portfolio, timestamp: datetime):
        """
        Process fills for open orders.

        Args:
            portfolio: Portfolio containing orders to process
            timestamp: Current simulation timestamp
        """
        logger.debug(f"Processing fills at {timestamp}. Open orders: {len(portfolio.open_orders)}")
        
        for order in portfolio.open_orders[:]:  # Copy list to allow modification
            logger.debug(f"Processing order: {order.symbol} {order.side} {order.order_type} {order.quantity}")
            
            # Check for expired orders
            if order.is_expired(timestamp):
                logger.info(f"Order expired: {order.symbol} {order.side} {order.order_type}")
                order.status = OrderStatus.CANCELLED
                portfolio.open_orders.remove(order)
                continue

            # Get current market price
            market_price = self.price_view.get_price(order.symbol, timestamp)
            
            # if market_price is None:
            #     market_price = self.asset_universe.get_last_price(
            #         order.symbol,
            #         timestamp=timestamp
            #     )
                
            # For limit orders, check if price conditions are met
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and market_price > order.limit_price:
                    continue
                if order.side == OrderSide.SELL and market_price < order.limit_price:
                    continue
            
            # Calculate fill price with slippage
            # TODO: rewrite this terrible implementation
            # slippage_model should be much more flexible in its approach
            # if self.slippage_model is None:
            #     fill_price = market_price
            # else:
            #     fill_price = self.slippage_model.get_fill_price(
            #         order,
            #         market_price,
            #         timestamp
            #     )
            
            # Process the fill
            portfolio.fill_order(order, fill_price, self)
