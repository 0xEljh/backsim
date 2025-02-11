from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from backsim.universe import QuantityMatrix
from backsim.portfolio.models import (
    Order,
    Position,
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
)
from backsim.callback import Callback

import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Tracks positions, orders, and cash balance.
    """

    def __init__(
        self,
        initial_cash: float,
        quantity_matrix: QuantityMatrix,
        callbacks: Optional[List[Callback]] = None,
    ):
        self._cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.quantity_matrix: QuantityMatrix = quantity_matrix
        self.latest_prices: Optional[Union[np.ndarray, pd.Series]] = None
        self.callbacks = callbacks or []

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
                # TODO: datetime.now() is wrong... need to figure out how best to do this
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

        self._trigger_callbacks("on_prices_update", self)

        return self.portfolio_value  # for logging

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

        # update quantity matrix
        if self.positions[order.symbol].is_zero:  # position is effectively closed
            self.positions.pop(order.symbol)
            self.quantity_matrix.update_quantity(order.symbol, order.timestamp, 0.0)
            return

        self.quantity_matrix.update_quantity(
            order.symbol, order.timestamp, self.positions[order.symbol].quantity
        )

        # callbacks
        self._trigger_callbacks(
            "on_order_filled", portfolio=self, order=order, realized_pnl=realized_pnl
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

    def register_callback(self, callback: Callback):
        """Allow users to register additional callbacks."""
        self.callbacks.append(callback)

    def _trigger_callbacks(self, method_name: str, *args, **kwargs):
        """Helper method to trigger callbacks."""
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

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
