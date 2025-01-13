"""
Portfolio implementation for tracking positions and orders.
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


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

    @property
    def position_side(self):
        return PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT


@dataclass
class Position:
    """Represents a position in a single asset."""

    symbol: str
    quantity: float
    cost_basis: float
    # unrealized_pnl: float = 0.0

    # TODO: track unrealized P&L

    @property
    def side(self):
        return PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT


    def get_unrealized_pnl(self, current_price: float):
        return (current_price - self.cost_basis) * self.quantity


class QuantityMatrix:
    """Stores and manages quantity matrices for efficient position tracking.
    Because pandas is slow for single-row updates, we'll only add values when there
    is a change, and use ffill to populate the rest of the matrix when needed.
    This is effectively a "sparse" matrix.
    
    """
    
    def __init__(self, symbols: List[str], start_time: datetime, frequency: str = "1d"):
        self.symbols = symbols
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
        self.start_time = start_time
        self.frequency = frequency
        self._quantity_matrix = pd.DataFrame(
            0.0,  # Initialize with zeros instead of NaN since no position = 0
            index=pd.date_range(start=start_time, freq=frequency, closed="left"),
            columns=symbols
        )

        self.current_row = {} # internal storage for current row
        self.current_time = start_time
    
    def update_quantity(self, symbol: str, timestamp: datetime, quantity: float):
        """Update quantity for a symbol at given timestamp.
        If the timestamp is the same as the current time, update the current row.
        Else, update the quantity matrix.
        """
        if timestamp == self.current_time:
            self.current_row[symbol] = quantity
            return current_row
        
        # commit the current row
        if self.current_row:
            self._quantity_matrix.loc[self.current_time] = self.current_row
            self.current_row = {}
        
        # update current time and row
        self.current_time = timestamp
        self.current_row[symbol] = quantity
        
        return self.current_row
    
    @property
    def matrix(self):
        # commit current row and return matrix
        matrix = self._quantity_matrix.copy()
        if self.current_row:
            matrix.loc[self.current_time] = self.current_row
        # forward fill the matrix, ensuring also that it has all rows for each timestamp
        return (
            matrix
            .ffill()
            .reindex(pd.date_range(start=self.start_time, freq=self.frequency, closed="left"), method="ffill")
            .fillna(0.0)
        )
    
    def get_matrix(self, up_to_timestamp: datetime):
        # similar to matrix property, but may have to either slice or continue forward fill
        if up_to_timestamp < self.current_time:
            return (
                self.matrix
                .loc[:up_to_timestamp]
            )
        
        return (
            self.matrix
            # add new row with nan values
            .append(pd.DataFrame(np.nan, index=[up_to_timestamp], columns=self.symbols))
            .ffill()
            .reindex(pd.date_range(start=self.start_time, freq=self.frequency, closed="left"), method="ffill")
            .fillna(0.0)
        )
            


class Portfolio:
    """
    Tracks positions, orders, and cash balance.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.quantity_matrix = None # to be initialized later

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

    def update_position(self, symbol: str, quantity: float, price: float, timestamp: datetime, broker):
        """
        Update position after a fill.
        
        Args:
            symbol: Asset symbol
            quantity: Quantity filled (positive for buys, negative for sells)
            price: Fill price
            timestamp: Fill timestamp
            broker: Broker instance for matrix store updates
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=0, cost_basis=price)
            
        position = self.positions[symbol]
        
        # Update position quantity and cost basis
        old_quantity = position.quantity
        new_quantity = old_quantity + quantity
        
        # Update cost basis
        if new_quantity != 0:
            if old_quantity == 0:
                position.cost_basis = price
            else:
                # Weighted average for cost basis
                position.cost_basis = (
                    (old_quantity * position.cost_basis + quantity * price) / new_quantity
                )
        
        position.quantity = new_quantity
        

    def get_portfolio_value(self, timestamp: datetime, broker) -> float:
        """
        Calculate total portfolio value including cash.
        
        Args:
            timestamp: Current timestamp
            broker: Broker instance for matrix store access
            
        Returns:
            Total portfolio value
        """
        total_value, position_values = broker.matrix_store.calculate_portfolio_value(timestamp)
        return total_value + self.cash

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
        self.update_position(order.symbol, order.quantity, fill_price, order.timestamp)

        # Update cash
        trade_value = order.quantity * fill_price
        self.cash += trade_value if order.side == OrderSide.SELL else -trade_value

        # Move order to filled orders
        self.open_orders.remove(order)
        self.filled_orders.append(order)

    def get_snapshot(self, timestamp: datetime, broker) -> Dict:
        """
        Get current portfolio state.

        Args:
            timestamp: Current simulation timestamp
            broker: Broker instance for matrix store access

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
                }
                for pos in self.positions.values()
            ],
        }
