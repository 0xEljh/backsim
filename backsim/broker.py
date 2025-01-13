"""
Broker implementation for order execution and fill simulation.
"""
from datetime import datetime
from typing import Protocol, Optional
import logging
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


class Broker:
    """
    Simulates order execution and generates fills.
    """
    def __init__(
        self,
        asset_universe: AssetUniverse,
        slippage_model: Optional[SlippageModel] = None,
        fill_delay: int = 0
    ):
        """
        Initialize broker.

        Args:
            asset_universe: AssetUniverse for price data
            slippage_model: Optional model for price slippage
            fill_delay: Number of time steps to delay fills
        """
        self.asset_universe = asset_universe
        self.slippage_model = slippage_model or FixedSlippage()
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
            market_price = self.asset_universe.get_last_price(
                order.symbol,
                timestamp=timestamp
            )
            logger.debug(f"Market price for {order.symbol}: {market_price}")

            # Check if limit order can be filled
            if order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and market_price > order.limit_price) or \
                   (order.side == OrderSide.SELL and market_price < order.limit_price):
                    logger.debug(f"Limit order price not met. Market: {market_price}, Limit: {order.limit_price}")
                    continue  # Skip if limit price not met

            # Calculate fill price with slippage
            fill_price = self.slippage_model.get_fill_price(
                order,
                market_price,
                timestamp
            )
            logger.debug(f"Fill price with slippage: {fill_price}")

            # For limit orders, ensure fill price respects limit price
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and fill_price > order.limit_price:
                    fill_price = order.limit_price
                elif order.side == OrderSide.SELL and fill_price < order.limit_price:
                    fill_price = order.limit_price
                logger.debug(f"Adjusted fill price for limit order: {fill_price}")

            # Fill the order in portfolio
            logger.info(f"Filling order: {order.symbol} {order.side} {order.order_type} at {fill_price}")
            portfolio.fill_order(order, fill_price)
