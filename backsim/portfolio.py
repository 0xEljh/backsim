"""
Portfolio implementation for tracking positions and orders.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
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
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # TODO: handle this case


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

    @property
    def margin(self):
        # Calculate margin for the order
        # use preferentially filled price
        if self.filled_price is not None:
            return abs(self.quantity * self.filled_price) / self.leverage_ratio
        if self.limit_price is None:
            # neither has been provided yet; checking margin of an unfilled market order
            logger.warning(
                "Attempting to get margin of unfilled market order; Margin cannot be defined"
            )
            raise ValueError("Margin cannot be defined without a reference price.")
        return abs(self.quantity * self.limit_price) / self.leverage_ratio

    @property
    def is_fully_filled(self):
        # if self.status != OrderStatus.FILLED:
        #     return False
        return self.filled_quantity == self.quantity

    def __str__(self):
        return (
            f"Order(symbol={self.symbol}, quantity={self.quantity}, side={self.side}, timestamp={self.timestamp}, "
            f"order_type={self.order_type}, limit_price={self.limit_price}, expiry={self.expiry}, "
            f"status={self.status}, filled_price={self.filled_price}, filled_quantity={self.filled_quantity}, "
            f"fees_incurred={self.fees_incurred})"
        )


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
            return 0.0  # avoid division by zero
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

    def update_from_order(self, order: Order) -> float:
        """Update position based on a filled order. Returns PnL if any from the occurence of the order"""
        if order.filled_price is None or order.filled_quantity is None:
            raise ValueError("Order must be filled before updating position")

        order_quantity = order.quantity * (
            1 if order.side in (OrderSide.BUY, "BUY") else -1
        )
        old_quantity = self.quantity
        new_quantity = self.quantity + order_quantity

        realized_pnl = 0.0

        if self.is_zero:
            # opening new position
            self.cost_basis = order.filled_price
            self.initial_margin = order.margin
            self.quantity = new_quantity
            return realized_pnl

        if new_quantity == 0:
            # closing position
            realized_pnl = self.get_unrealized_pnl(order.filled_price)
            self.quantity = new_quantity
            return realized_pnl

        # Update cost basis and margin for 3 scenarios
        # 1. increasing position
        # 2. reducing position
        # 3. flipping position

        if (old_quantity > 0 and new_quantity < 0) or (
            old_quantity < 0 and new_quantity > 0
        ):
            # Flipping position
            realized_pnl = (order.filled_price - self.cost_basis) * old_quantity
            # Start the new position with remaining quantity
            self.quantity = new_quantity
            self.cost_basis = order.filled_price
            self.initial_margin = (
                abs(new_quantity * order.filled_price) / order.leverage_ratio
            )
            return realized_pnl

        if abs(new_quantity) > abs(old_quantity):
            # Increasing position
            self.cost_basis = (  # careful of negative qty
                self.cost_basis * abs(old_quantity)
                + order.filled_price * abs(order_quantity)
            ) / abs(new_quantity)

            self.quantity = new_quantity
            # add margin of new order
            self.initial_margin += order.margin
            return realized_pnl

        if abs(new_quantity) < abs(old_quantity):
            # Reducing position
            closed_quantity = old_quantity - new_quantity
            realized_pnl = (order.filled_price - self.cost_basis) * closed_quantity
            new_margin = (
                abs(new_quantity * self.cost_basis) / self.leverage
            )  # proportinal reduction in margin requirement
            self.quantity = new_quantity
            self.initial_margin = new_margin
            return realized_pnl


class Portfolio:
    """
    Tracks positions, orders, and cash balance.
    """

    def __init__(self, initial_cash: float, quantity_matrix):
        self._cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.quantity_matrix = quantity_matrix

    @property
    def total_margin(self):
        return sum(position.initial_margin for position in self.positions.values())

    @property
    def cash(self):
        """
        Available cash; i.e. cash not tied up as collateral for positions via margin.
        Not to be confused with buying power.
        """
        return self._cash - self.total_margin

    def add_orders(self, orders: List[Dict]):
        """
        Add new orders to the portfolio.

        Args:
            orders: List of order dictionaries containing order details
        """
        for order_dict in orders:
            # Handle fields with sensible defaults
            # we purposely avoid pop() here since we don't want to mutate order_dict
            timestamp = order_dict.get("timestamp", datetime.now())
            order_type = OrderType(order_dict.get("order_type", "MARKET"))
            leverage_ratio = order_dict.get("leverage_ratio", 1.0)

            # Filter out handled keys to avoid duplicates
            remaining_keyword_args = {
                k: v
                for k, v in order_dict.items()
                if k not in {"timestamp", "order_type", "leverage_ratio"}
            }

            order = Order(
                timestamp=timestamp,
                order_type=order_type,
                leverage_ratio=leverage_ratio,
                **remaining_keyword_args,
            )
            self.open_orders.append(order)

    def close_order(self, order: Order, non_existent_order_ok=False):
        """
        Closes the order by removing from open_orders (if present) and
        appending to closed_orders. This is intended for terminal states:
        FILLED, REJECTED, EXPIRED, CANCELED, etc.
        """
        # Defensive remove: only remove if still in open_orders
        if order not in self.open_orders and not non_existent_order_ok:
            raise ValueError(
                f"Tried to close an order ({order}) that does not exist in open_orders({self.open_orders})"
            )
        if order in self.open_orders:
            self.open_orders.remove(order)
        self.closed_orders.append(order)
        logger.debug(f"Order closed: {order}")

    def update_order(self, order: Order):
        """
        Updates existing orders (e.g. change in filled_price)
        If the order ends up in a terminal status or is fully-filled,
        move it to closed_orders.
        """
        if (
            order.status
            in [OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FILLED]
            or order.is_fully_filled  # consider if it should be this or OrderStatus.FILLED
        ):
            # Mark the status as FILLED if not already (e.g. a partial fill
            # that just became complete). Or keep REJECTED/EXPIRED as is.
            if order.is_fully_filled:
                order.status = OrderStatus.FILLED
            self.close_order(order)
        else:
            # Otherwise, it is still active (e.g. PARTIALLY_FILLED)
            # so keep it in open_orders. If we re-sort or
            # re-store open orders, modify this to accomodate
            logger.debug(f"Order updated but still open: {order}")

    def fill_order(self, order: Order):
        """
        Process a fill for an order.
        TODO: needs to be updated to handle partial fills.

        Args:
            order: Order object to fill
            fill_price: Price at which the order was filled
            fill_quantity: Quantity of the order that was filled
        """

        # Update position, if applicable
        realized_pnl = 0.0
        if order.symbol in self.positions:
            realized_pnl = self.positions[order.symbol].update_from_order(order)
            # update quantity matrix
            self.quantity_matrix.update_quantity(
                order.symbol, order.timestamp, self.positions[order.symbol].quantity
            )
            if self.positions[order.symbol].is_zero:
                # close the position
                self.positions.pop(order.symbol)
        else:
            # opening new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=(
                    order.quantity * (1 if order.side in (OrderSide.BUY, "BUY") else -1)
                ),
                cost_basis=order.filled_price,
                initial_margin=order.margin,
            )
            # update quantity matrix
            self.quantity_matrix.update_quantity(
                order.symbol, order.timestamp, self.positions[order.symbol].quantity
            )

        # update cash: deduct fees, determine if this was a reduction in position
        self._cash += realized_pnl - order.fees_incurred

        self.update_order(order)

        if self.cash < 0:  # TODO: decide if cash or _cash
            raise ValueError("Insufficient cash to continue trading")

    # TODO: shift this responsibility to broker
    # def get_margin_status(self, current_prices: Dict[str, float]) -> Dict[str, Dict]:
    #     """
    #     Get margin status for all positions.

    #     Args:
    #         current_prices: Dictionary mapping symbols to current prices

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
            "cash": self._cash,
            "positions": [str(pos) for pos in self.positions.values()],
        }

    def get_positions_value(
        self, timestamp: datetime, price_array: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate total position value.

        Args:
            timestamp: Current timestamp

        Returns:
            Total position value
        """
        # get last row of quantity matrix
        quantity_array = self.quantity_matrix.get_matrix(
            up_to_timestamp=timestamp
        ).iloc[-1]

        # depending on type of price array, calculate dot product
        if isinstance(price_array, np.ndarray):
            return np.dot(quantity_array, price_array)
        elif isinstance(price_array, pd.Series):
            return quantity_array.dot(price_array)
        else:
            raise ValueError("Invalid price array type")

    def get_portfolio_value(
        self, timestamp: datetime, price_array: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate total portfolio value including cash. i.e. total position value + cash

        Args:
            timestamp: Current timestamp

        Returns:
            Total portfolio value
        """
        # Calculate the total value of all positions
        positions_value = self.get_positions_value(timestamp, price_array)

        # Add the cash balance to the positions value
        portfolio_value = self._cash + positions_value

        # portfolio value should never be negative
        if portfolio_value < 0:
            logger.warning(
                f"Portfolio value is negative, something is wrong: {portfolio_value}"
            )

        return portfolio_value

    def cleanup_orders(self):
        """
        Remove any orders that have been filled, expired, cancelled, or rejected
        """
        orders_to_remove = [
            order for order in self.open_orders if order.status != OrderStatus.PENDING
        ]

        logger.debug(f"Orders to remove due to cleanup: {len(orders_to_remove)}")

        for order in orders_to_remove:
            self.open_orders.remove(order)
            self.closed_orders.append(order)
