"""
Portfolio implementation for tracking positions and orders.
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from backsim.universe import QuantityMatrix

import logging
logger = logging.getLogger(__name__)

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
    leverage_ratio: float = 1.0
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    fees_incurred: float = 0.0

    def is_expired(self, current_time: datetime) -> bool:
        """Check if order has expired."""
        return self.expiry is not None and current_time > self.expiry

    # @property
    # def position_side(self):
    #     return PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT

    @property
    def margin(self):
        # Calculate margin for the order
        # use preferentially filled price
        if self.filled_price is not None:
            return abs(self.quantity * self.filled_price) / self.leverage_ratio
        if self.limit_price is None:
            # neither has been provided yet; checking margin of an unfilled market order
            logger.warning("Attempting to get margin of unfilled market order; Margin cannot be defined")
            return None
        return abs(self.quantity * self.limit_price) / self.leverage_ratio


@dataclass
class Position:
    """Represents a position in a single asset."""

    symbol: str
    quantity: float
    cost_basis: float
    initial_margin: float = 0.0
    # maintenance margin is a calculated property based on initial margin, for flexibility

    @property
    def side(self):
        return PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT

    @property
    def leverage(self):
        """Calculated effective leverage for the position.
        Uses the initial margin and cost basis
        """
        if self.is_zero:
            return 0.0 # avoid division by zero
        inital_position_value = abs(self.quantity * self.cost_basis)
        return inital_position_value / self.initial_margin
    
    @property
    def is_zero(self):
        # either define by quantity or define by value
        return self.quantity == 0

    def __repr__(self):
        return f"Position(symbol={self.symbol}, quantity={self.quantity}, cost_basis={self.cost_basis}, initial_margin={self.initial_margin}, leverage={self.leverage})"

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for the position."""
        return (current_price - self.cost_basis) * self.quantity

    # def check_margin_breach(self, current_price: float, maintenance_margin_ratio: float = 0.75) -> bool:
    #     """Check if position has breached maintenance margin requirement."""
    #     equity = self.initial_margin + self.get_unrealized_pnl(current_price)
    #     return equity < self.initial_margin * maintenance_margin_ratio

    def update_from_order(order: Order) -> float:
        """Update position based on a filled order. Returns PnL if position was closed (even partially)"""
        if order.status != OrderStatus.FILLED or order.filled_price is None:
            raise ValueError("Order must be filled before updating position")

        order_quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
        old_quantity = self.quantity
        new_quantity = self.quantity + order_quantity

        if self.is_zero:
            # opening new position
            self.cost_basis = order.filled_price
            self.initial_margin = order.margin
            self.quantity = new_quantity
            return 0.0
        
        if new_quantity == 0:
            # closing position
            realized_pnl = self.get_unrealized_pnl(order.filled_price)
            self.quantity = new_quantity
            return realized_pnl
        
        # Update cost basis and margin for 3 scenarios
        # 1. increasing position
        # 2. reducing position
        # 3. flipping position
        
        if (old_quantity > 0 and new_quantity < 0) or (old_quantity < 0 and new_quantity > 0): 
            # Flipping position
            realized_pnl = (order.filled_price - self.cost_basis) * old_quantity
            # Start the new position with remaining quantity
            self.quantity = new_quantity  
            self.cost_basis = order.filled_price
            self.initial_margin = abs(new_quantity * order.filled_price) / order.leverage_ratio

        if abs(new_quantity) > abs(old_quantity):
            # Increasing position
            self.cost_basis = (self.cost_basis * old_quantity + order.filled_price * new_quantity) / new_quantity
            self.quantity = new_quantity
            # add margin of new order
            self.initial_margin += order.margin
        
        if abs(new_quantity) < abs(old_quantity):
            # Reducing position
            closed_quantity = old_quantity - new_quantity
            realized_pnl = (order.filled_price - self.cost_basis) * closed_quantity
            self.quantity = new_quantity
            # proportinal reduction in margin requirement
            self.initial_margin = abs(new_quantity * self.cost_basis) / self.leverage
        
        return realized_pnl

class Portfolio:
    """
    Tracks positions, orders, and cash balance.
    """

    def __init__(self, initial_cash: float, quantity_matrix):
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.quantity_matrix = quantity_matrix

    def add_orders(self, orders: List[Dict]):
        """
        Add new orders to the portfolio.

        Args:
            orders: List of order dictionaries containing order details
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
                leverage_ratio=order_dict.get("leverage_ratio", 1.0),
            )
            self.open_orders.append(order)

    # def update_position(self, filled_order: Order):
    #     """
    #     Update position after a fill.
        
    #     Args:
    #         filled_order: Filled order to use for position update
        
    #     Raises:
    #         ValueError: If order is not filled
    #     """
    #     if filled_order.status != OrderStatus.FILLED or filled_order.filled_price is None:
    #         raise ValueError("Order must be filled before updating position")

    #     symbol = filled_order.symbol
    #     quantity = filled_order.quantity
    #     price = filled_order.filled_price
    #     leverage_ratio = filled_order.leverage_ratio

    #     if symbol not in self.positions:
    #         position_value = abs(quantity * price)
    #         initial_margin = position_value / leverage_ratio
    #         self.positions[symbol] = Position(
    #             symbol=symbol,
    #             quantity=0,
    #             cost_basis=price,
    #             initial_margin=initial_margin
    #         )
    #     else:
    #         self.positions[symbol].update_from_order(filled_order)
            
    #     position = self.positions[symbol]

    #     # update quantity matrix
    #     self.quantity_matrix.update_quantity(symbol, filled_order.timestamp, position.quantity)
        
    #     # Remove position if quantity is zero
    #     if position.is_zero:
    #         del self.positions[symbol]


    def fill_order(self, order: Order, fill_price: float, fill_quantity: float):
        """
        Process a fill for an order.

        Args:
            order: Order object to fill
            fill_price: Price at which the order was filled
            fill_quantity: Quantity of the order that was filled
        """
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = fill_quantity

        # Update position, if applicable
        realized_pnl = 0.0
        if order.symbol in self.positions:
            realized_pnl = self.positions[order.symbol].update_from_order(order)
            # update quantity matrix
            self.quantity_matrix.update_quantity(order.symbol, order.timestamp, self.positions[order.symbol].quantity)
            if self.positions[order.symbol].is_zero:
                # close the position
                self.positions.pop(order.symbol)
        else:
            # opening new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity if order.side == OrderSide.BUY else -order.quantity,
                cost_basis=fill_price,
                initial_margin=order.margin
            )
            # update quantity matrix
            self.quantity_matrix.update_quantity(order.symbol, order.timestamp, self.positions[order.symbol].quantity)
        
        # update cash: deduct fees, determine if this was a reduction in position
        self.cash += realized_pnl - order.fees_incurred

        # Move order to filled orders
        self.open_orders.remove(order)
        self.filled_orders.append(order)

        if self.cash < 0:
            raise ValueError("Insufficient cash to continue trading")

    

    # TODO: shift this responsibility to broker
    # def get_margin_status(self, current_prices: Dict[str, float]) -> Dict[str, Dict]:
    #     """
    #     Get margin status for all positions.

    #     Args:
    #         current_prices: Dictionary mapping symbols to current prices

    #     Returns:
    #         Dictionary containing margin information for each position
    #     """
    #     margin_status = {}
    #     for symbol, position in self.positions.items():
    #         if symbol in current_prices:
    #             current_price = current_prices[symbol]
    #             margin_status[symbol] = {
    #                 "initial_margin": position.initial_margin,
    #                 "maintenance_margin": position.initial_margin * self.maintenance_margin_ratio,
    #                 "unrealized_pnl": position.get_unrealized_pnl(current_price),
    #                 "margin_breach": position.check_margin_breach(current_price, self.maintenance_margin_ratio),
    #                 "leverage": position.leverage
    #             }
    #     return margin_status

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
            "positions": [str(pos) for pos in self.positions.values()],
        }

    def get_portfolio_value(self, timestamp: datetime) -> float:
        """
        Calculate total portfolio value including cash.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Total portfolio value
        """
        # TODO: Implement portfolio value calculation using quantity_matrix
        pass