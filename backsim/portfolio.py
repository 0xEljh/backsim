"""
Portfolio implementation for tracking positions and orders.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from backsim.universe import QuantityMatrix

import logging

logger = logging.getLogger(__name__)

EPSILON = 1e-12


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
class Fill:
    """Represents a single fill event for an order."""

    quantity: float
    price: float
    timestamp: datetime


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
    filled_price: Optional[float] = None  # average fill price
    filled_quantity: float = 0.0
    fees_incurred: float = 0.0
    fills: List[Fill] = field(default_factory=list)

    def add_fill(self, quantity: float, price: float, timestamp: datetime):
        """Add a new fill and update average price."""
        self.fills.append(Fill(quantity, price, timestamp))
        self.filled_quantity += quantity

        # Update average price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.filled_price = (
            total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0
        )

        # update status if now fully filled
        if self.is_fully_filled:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity

    def is_expired(self, current_time: datetime) -> bool:
        """Check if order has expired."""
        return self.expiry is not None and current_time > self.expiry

    @property
    def margin(self):
        # Calculate margin for the order
        # use preferentially filled price
        if self.filled_price is not None:
            return abs(self.filled_quantity * self.filled_price) / self.leverage_ratio
        if self.limit_price is None:
            # neither has been provided yet; checking margin of an unfilled market order
            logger.warning(
                "Attempting to get margin of unfilled market order; Margin cannot be defined"
            )
            raise ValueError("Margin cannot be defined without a reference price.")
        return abs(self.quantity * self.limit_price) / self.leverage_ratio

    @property
    def is_fully_filled(self):
        return self.filled_quantity == self.quantity

    def __str__(self):
        return (
            f"Order(symbol={self.symbol}, quantity={self.quantity}, side={self.side}, timestamp={self.timestamp}, "
            f"order_type={self.order_type}, limit_price={self.limit_price}, expiry={self.expiry}, "
            f"status={self.status}, filled_price={self.filled_price}, filled_quantity={self.filled_quantity}, "
            f"fees_incurred={self.fees_incurred})"
        )

    # allow dictionary like access of Order
    def __getitem__(self, key: str):
        """Get attribute using dictionary-like access."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Set attribute using dictionary-like access."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if attribute exists using 'in' operator."""
        return hasattr(self, key)


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    quantity: float
    cost_basis: float
    initial_margin: float = 0.0

    def update_from_fill(self, fill: Fill, order: Order) -> float:
        """Update position based on a filled order. Returns PnL if any from the occurence of the order"""
        if order.filled_price is None or order.filled_quantity is None:
            raise ValueError("Order must be filled before updating position")

        fill_quantity = fill.quantity * (
            1 if order.side in (OrderSide.BUY, "BUY") else -1
        )
        old_quantity = self.quantity
        new_quantity = self.quantity + fill_quantity

        realized_pnl = 0.0

        if self.is_zero:
            # opening new position
            self.cost_basis = order.filled_price
            self.initial_margin = order.margin
            self.quantity = new_quantity
            return realized_pnl

        if new_quantity == 0:
            # closing position
            realized_pnl = self.get_unrealized_pnl(fill.price)
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
            realized_pnl = (fill.price - self.cost_basis) * old_quantity
            # Start the new position with remaining quantity
            self.quantity = new_quantity
            self.cost_basis = fill.price
            self.initial_margin = abs(new_quantity * fill.price) / order.leverage_ratio
            return realized_pnl

        if abs(new_quantity) > abs(old_quantity):
            # Increasing position
            self.cost_basis = (  # careful of negative qty
                self.cost_basis * abs(old_quantity) + fill.price * abs(fill.quantity)
            ) / abs(new_quantity)

            self.quantity = new_quantity
            # add margin of new fill
            self.initial_margin += (
                fill.price * abs(fill.quantity) / order.leverage_ratio
            )
            return realized_pnl

        if abs(new_quantity) < abs(old_quantity):
            # Reducing position
            closed_quantity = old_quantity - new_quantity
            realized_pnl = (fill.price - self.cost_basis) * closed_quantity
            new_margin = (
                abs(new_quantity * self.cost_basis) / self.leverage
            )  # proportinal reduction in margin requirement
            self.quantity = new_quantity
            self.initial_margin = new_margin
            return realized_pnl

    def update_from_order(self, order: Order) -> float:
        if not order.fills:
            return 0.0

        # Use the latest fill
        latest_fill = order.fills[-1]

        # calculate fill cash flow; if short, we want to credit cash
        sign = 1 if order.side in (OrderSide.SELL, "SELL") else -1
        fill_cash_flow = sign * latest_fill.quantity * latest_fill.price

        return self.update_from_fill(latest_fill, order), fill_cash_flow

    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L for the position.
        For longs: (current_price - cost_basis) * quantity.
        For shorts: (cost_basis - current_price) * abs(quantity).
        """
        if self.quantity > 0:
            return (current_price - self.cost_basis) * self.quantity
        elif self.quantity < 0:
            return (self.cost_basis - current_price) * abs(self.quantity)
        else:
            return 0.0

    # def margin_usage(self, current_price: float) -> float:
    #     """
    #     Compute the current margin usage. In this simple model,
    #     margin scales with the price relative to the cost basis.
    #     """
    #     return self.initial_margin * (current_price / self.cost_basis)

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.cost_basis

    @property
    def is_zero(self) -> bool:
        """Check if position is effectively zero."""
        return self.notional_value < EPSILON

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

    def __repr__(self) -> str:
        return (
            f"Position(symbol={self.symbol}, quantity={self.quantity}, "
            f"cost_basis={self.cost_basis}, initial_margin={self.initial_margin}, side={self.side})"
        )


class Portfolio:
    """
    Tracks positions, orders, and cash balance.
    """

    def __init__(self, initial_cash: float, quantity_matrix: QuantityMatrix):
        self._cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.quantity_matrix: QuantityMatrix = quantity_matrix
        self.latest_prices: Optional[Union[np.ndarray, pd.Series]] = None

    def add_orders(self, orders: List[Union[Order, dict]]):
        """
        Add new orders to the portfolio.

        Args:
            orders: List of order dictionaries containing order details
        """
        for order_dict in orders:
            if type(order_dict) == Order:
                order = order_dict
            else:
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

            # no negative/zero quantity for orders, only positions.
            if order.quantity <= 0:
                # reject order
                order.status = OrderStatus.REJECTED
                logger.warning(f"Rejected invalid quantity order: {order}")
                self.closed_orders.append(order)
                continue

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

    def update_prices(self, prices: pd.Series):
        """
        Update the latest market prices.

        Args:
            prices: A pandas Series with symbols as index and current prices as values.
        """
        self.latest_prices = prices

    def fill_order(self, order: Order):
        if order.symbol in self.positions:
            realized_pnl, fill_cash_flow = self.positions[
                order.symbol
            ].update_from_order(order)

        else:
            realized_pnl = 0.0  # new position, no PnL
            # if short, we want to credit cash
            fill_cash_flow = (
                order.filled_quantity
                * order.filled_price
                * (-1 if order.side in (OrderSide.BUY, "BUY") else 1)
            )

            # open new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=(
                    order.filled_quantity
                    * (1 if order.side in (OrderSide.BUY, "BUY") else -1)
                ),
                cost_basis=order.filled_price,
                initial_margin=order.margin,
            )

        # add cash flow, subtract fees
        self._cash += fill_cash_flow - order.fees_incurred

        self.update_order(order)

        # TODO: log realized_pnl from the order.
        # self.log_realized_pnl(order, realized_pnl)

        # update quantity matrix
        if self.positions[order.symbol].is_zero:  # position is effectively closed
            self.positions.pop(order.symbol)
            self.quantity_matrix.update_quantity(order.symbol, order.timestamp, 0.0)
            return

        self.quantity_matrix.update_quantity(
            order.symbol, order.timestamp, self.positions[order.symbol].quantity
        )

    @property
    def positions_value(self) -> float:
        """
        Total mark-to-market value of all positions using latest price series
        """
        if not self.positions:
            return 0.0

        if self.latest_prices is None:
            # Fallback: use cost_basis if current prices are not available.
            return sum(pos.quantity * pos.cost_basis for pos in self.positions.values())

        # Get the latest quantities from the QuantityMatrix
        latest_quantities = self.quantity_matrix.matrix.iloc[-1]

        # Calculate the total value using vectorized operations
        return (latest_quantities * self.latest_prices).sum()

    @property
    def portfolio_value(self) -> float:
        """
        Net portfolio value = positions_value + cash.
        Short positions add negative contributions automatically.
        """

        return self.positions_value + self._cash

    @property
    def used_margin(self) -> float:
        """
        Sum of the per-position margin usage (maintenance margin).
        """
        if not self.positions:
            return 0.0
        if self.latest_prices is None:
            return sum(pos.initial_margin for pos in self.positions.values())

        # Create a DataFrame from positions.
        df = pd.DataFrame(
            {
                "cost_basis": {
                    sym: pos.cost_basis for sym, pos in self.positions.items()
                },
                "initial_margin": {
                    sym: pos.initial_margin for sym, pos in self.positions.items()
                },
            }
        )

        if self.latest_prices is None:
            logger.warning("get_total_margin_usage: latest_prices is None")
            return sum(pos.initial_margin for pos in self.positions.values())

        # Get current prices for these symbols.
        current_prices = self.latest_prices.reindex(df.index, fill_value=0.0)

        # Calculate margin usage for each position; guard against division by zero.
        df["margin_usage"] = df["initial_margin"] * (
            current_prices / df["cost_basis"]
        ).replace([np.inf, -np.inf], 1.0)

        return df["margin_usage"].sum()

    @property
    def available_margin(self) -> float:
        return self.portfolio_value - self.used_margin

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

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol. Returns None if position doesn't exist.

        Args:
            symbol: Symbol to get position for

        Returns:
            Position object if exists, None otherwise
        """
        return self.positions.get(symbol)

    def get_position_quantity(self, symbol: str) -> float:
        """
        Get position quantity for a symbol. Returns 0 if position doesn't exist.

        Args:
            symbol: Symbol to get position quantity for

        Returns:
            Position quantity if exists, 0 otherwise
        """
        position = self.get_position(symbol)
        return position.quantity if position else 0.0

    def get_position_side(self, symbol: str) -> Optional[PositionSide]:
        """
        Get position side for a symbol. Returns None if position doesn't exist.

        Args:
            symbol: Symbol to get position side for

        Returns:
            Position side if exists, None otherwise
        """
        position = self.get_position(symbol)
        return position.side if position else None
