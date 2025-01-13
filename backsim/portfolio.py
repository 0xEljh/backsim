"""
Portfolio implementation for tracking positions and orders.
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    quantity: float
    side: OrderSide
    timestamp: datetime
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    expiry: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0

    def is_expired(self, current_time: datetime) -> bool:
        """Check if order has expired."""
        return self.expiry is not None and current_time > self.expiry


@dataclass
class Position:
    """Represents a position in a single asset."""

    symbol: str
    quantity: float
    cost_basis: float
    unrealized_pnl: float = 0.0

    # TODO: track unrealized P&L


class Portfolio:
    """
    Tracks positions, orders, and cash balance.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.filled_orders: List[Order] = []

    def add_orders(self, orders: List[Dict]):
        """
        Add new orders to the portfolio.

        Args:
            orders: List of order dictionaries
        """
        for order_dict in orders:
            order = Order(
                symbol=order_dict["symbol"],
                quantity=order_dict["quantity"],
                side=OrderSide(order_dict["side"]),
                timestamp=order_dict.get("timestamp", datetime.now()),
                order_type=OrderType(order_dict.get("order_type", "MARKET")),
                limit_price=order_dict.get("limit_price"),
                expiry=order_dict.get("expiry"),
            )
            self.open_orders.append(order)

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        return self.cash + sum(
            pos.quantity * pos.get("market_value", pos.cost_basis)
            for pos in self.positions.values()
        )

    def fill_order(self, order: Order, fill_price: float):
        """
        Process a fill for an order.

        Args:
            order: Order object to fill
            fill_price: Price at which the order was filled
        """
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity

        # Update position
        # TODO: update cash and value logic; support short selling
        position = self.positions.get(order.symbol)
        if order.side == OrderSide.BUY:
            if position is None:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol, quantity=order.quantity, cost_basis=fill_price
                )
            else:
                # Update existing position
                new_quantity = position.quantity + order.quantity
                position.cost_basis = (
                    position.quantity * position.cost_basis
                    + order.quantity * fill_price
                ) / new_quantity
                position.quantity = new_quantity
        else:  # SELL
            if position is None:
                raise ValueError(f"No position found for {order.symbol}")
            position.quantity -= order.quantity
            if position.quantity == 0:
                del self.positions[order.symbol]

        # Update cash
        trade_value = order.quantity * fill_price
        self.cash += trade_value if order.side == OrderSide.SELL else -trade_value

        # Move order to filled orders
        self.open_orders.remove(order)
        self.filled_orders.append(order)

    def get_snapshot(self, timestamp: datetime) -> Dict:
        """
        Get current portfolio state.

        Args:
            timestamp: Current simulation timestamp

        Returns:
            Dict containing portfolio state
        """
        return {
            "timestamp": timestamp,
            "cash": self.cash,
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "cost_basis": pos.cost_basis,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for pos in self.positions.values()
            ],
        }
