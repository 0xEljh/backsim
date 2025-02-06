"""
Broker implementation for order execution and fill simulation.
"""

import abc
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, Optional, List
import logging
import pandas as pd
import numpy as np
from math import copysign
from .portfolio import Portfolio, Order, OrderStatus, OrderType, OrderSide
from .universe import AssetUniverse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

EPSILON = 1e-12


class MarginModel(abc.ABC):
    """
    Determines how an order's margin requirements interact
    with the current portfolio and whether the order can be filled fully or partially.
    """

    @abc.abstractmethod
    def can_fill_order(
        self,
        portfolio: "Portfolio",
        order: "Order",
        fill_result: "FillResult",
    ) -> bool:
        """
        Return True if the portfolio can support filling the given quantity
        at the proposed fill price, False otherwise.
        """
        pass

    @abc.abstractmethod
    def compute_fillable_quantity(
        self, portfolio: "Portfolio", order: "Order", fill_result: "FillResult"
    ) -> float:
        """
        Given the order and proposed fill, return how many units can actually be filled
        based on margin/cash constraints. Could be the full fill_result.quantity
        or some partial amount.
        """
        pass

    def __call__(
        self, portfolio: "Portfolio", order: "Order", fill_result: "FillResult"
    ) -> float:
        return self.compute_fillable_quantity(portfolio, order, fill_result)


class SimpleCashMargin(MarginModel):
    """
    Example: The simplest margin model that just checks if we have enough margin
    to buy (or short) the requested quantity at the fill price.
    """

    def can_fill_order(
        self,
        portfolio: "Portfolio",
        order: "Order",
        fill_result: "FillResult",
    ) -> bool:
        required_margin = (
            abs(fill_result.quantity * fill_result.price) / order.leverage_ratio
        )
        return portfolio.available_margin >= required_margin

    def compute_fillable_quantity(
        self, portfolio: "Portfolio", order: "Order", fill_result: "FillResult"
    ) -> float:
        if self.can_fill_order(portfolio, order, fill_result):
            return fill_result.quantity

        # Calculate how much quantity we can afford at this price
        max_quantity = (portfolio.available_margin * order.leverage_ratio) / abs(
            fill_result.price
        )
        # Return the minimum of what we can afford and what's available to fill
        # sign of quantity needs to be handled appropriately
        # using math instead of numpy here for performance (TODO: check if it matters, and if correct)
        return copysign(
            min(max_quantity, abs(fill_result.quantity)), fill_result.quantity
        )


@dataclass
class FillResult:
    price: float
    quantity: float


class FillModel(abc.ABC):
    """
    Determines the fill price (and optionally partial fill quantity)
    for a given order, based on slippage or other logic.

    The model can freely query the provided AssetUniverse any way it likes.
    """

    @abc.abstractmethod
    def get_fill(
        self, order: "Order", timestamp: datetime, universe: "AssetUniverse"
    ) -> "FillResult":
        """
        Return the final fill price and how many units are filled at 'timestamp' for the given order.
        Implementation may gather relevant data from 'universe' as needed.
        Could incorporate volume, bar data, slippage, or more advanced logic.
        """
        pass

    def __call__(self, order: "Order", timestamp: datetime, universe: "AssetUniverse"):
        return self.get_fill(order, timestamp, universe)


class NaiveCloseFillModel(FillModel):
    """
    Example fill model that always fills at the last known close price prior to (or at) timestamp.
    """

    def get_fill(
        self, order: "Order", timestamp: datetime, universe: "AssetUniverse"
    ) -> "FillResult":
        df_slice = universe.slice_data(
            symbols=[order.symbol],
            end=timestamp,
            lookback=1,
            fields=["close", "volume"],
        )

        if df_slice.empty:
            return FillResult(price=float("nan"), quantity=0.0)

        price = price = df_slice["close"].iloc[-1].squeeze()

        # only proceed further if limit order:
        if order.order_type == OrderType.MARKET:
            return FillResult(price, order.quantity)

        # 1) Check if limit is triggered
        triggered = False
        if order.side == OrderSide.BUY and price <= order.limit_price:
            triggered = True
        elif order.side == OrderSide.SELL and price >= order.limit_price:
            triggered = True

        if not triggered:
            return FillResult(price=0.0, quantity=0.0)  # no fill

        return FillResult(price, order.quantity)


class VolumeAwareLimitFillModel(FillModel):
    """
    Example fill model:
    - Checks if the bar's high/low crosses the limit.
    - Uses volume to determine partial or full fill.
    - Potentially includes slippage logic.
    """

    def get_fill(
        self, order: Order, timestamp: datetime, universe: AssetUniverse
    ) -> FillResult:
        bar = universe.slice_data(
            symbols=[order.symbol],
            end=timestamp,
            lookback=1,
            fields=["open", "high", "low", "close", "volume"],
        )
        if bar.empty:
            return FillResult(price=float("nan"), quantity=0.0)

        # Extract row data
        row = bar.iloc[-1]
        bar_open = row["open"].squeeze()
        bar_high = row["high"].squeeze()
        bar_low = row["low"].squeeze()
        bar_volume = row["volume"].squeeze()

        limit_price = order.limit_price
        qty_left = order.quantity - order.filled_quantity

        # 1) Check if limit is triggered
        triggered = False
        if order.side == OrderSide.BUY and bar_low <= limit_price:
            triggered = True
        elif order.side == OrderSide.SELL and bar_high >= limit_price:
            triggered = True

        if not triggered:
            return FillResult(price=0.0, quantity=0.0)  # no fill

        # # 2) Decide fill price
        # # Suppose we fill at whichever is better: bar_open or the limit price
        # if order.side == OrderSide.BUY:
        #     fill_price = min(bar_open, limit_price, bar_low)
        # else:
        #     fill_price = max(bar_open, limit_price, bar_high)

        # fill at the limit for most simplistic approximation (else assumption of frequency is too high)

        # 3) Partial fill based on bar volume, if needed
        # For simplicity, assume if the bar volume < order qty, we partially fill
        # Or do more advanced volume fraction logic
        fillable_qty = min(qty_left, bar_volume)  # naive approach

        return FillResult(price=limit_price, quantity=fillable_qty)


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
                self._price_matrix.loc[:timestamp, symbol]
                .iloc[-1:]
                .squeeze()  # convert to scalar
            )
        except (KeyError, IndexError):
            return None

    def get_prices(self, timestamp: datetime) -> pd.Series:
        """Get all prices for all symbols at or before the given timestamp."""
        return (
            self._price_matrix.loc[:timestamp]
            .reset_index(level=0)
            .groupby("symbol", sort=False)
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
        # price_matrix: pd.DataFrame,
        fill_model: Optional[FillModel] = None,
        margin_model: Optional[MarginModel] = None,
        fill_delay: timedelta = timedelta(
            0
        ),  # set to non-zero value to prevent look-ahead
        allow_partial_margin_fills: bool = False,
        fallback_price_field: str = "close",  # or "open", etc
    ):
        """
        Initialize broker.

        Args:
            asset_universe: AssetUniverse for price data (OHLCV, etc.)
            price_matrix: DataFrame with price data
            fill_model: Optional fill model
            margin_model: Optional margin model
            maintenance_margin_ratio: Maintenance margin ratio
            fill_delay: Time delay before fills are processed (simulates latency, default: no delay)
        """
        self.asset_universe = asset_universe
        self.fill_model = fill_model
        self.margin_model = margin_model
        self.fill_delay = fill_delay
        self.allow_partial_margin_fills = allow_partial_margin_fills
        self.fallback_price_field = fallback_price_field

    def process_fills(self, portfolio: Portfolio, timestamp: datetime):
        """
        Process fills for open orders.

        Args:
            portfolio: Portfolio containing orders to process
            timestamp: Current simulation timestamp
        """
        logger.debug(
            f"Processing fills at {timestamp}. Open orders: {len(portfolio.open_orders)}"
        )

        # copy orders list to avoid modifying it concurrently
        ready_orders = [
            order
            for order in portfolio.open_orders[:]
            if (timestamp - order.timestamp) >= self.fill_delay
        ]

        for order in ready_orders:
            logger.debug(
                f"Processing order: {order.symbol} {order.side} {order.order_type} {order.quantity}"
            )

            # Check for expired orders
            if order.is_expired(timestamp):
                logger.info(
                    f"Order expired: {order.symbol} {order.side} {order.order_type}"
                )
                order.status = OrderStatus.EXPIRED
                portfolio.close_order(order)
                continue

            # If user has provided a FillModel, we delegate to it
            if self.fill_model:
                fill_result = self.fill_model(order, timestamp, self.asset_universe)
                fill_price = fill_result.price
                fill_quantity = fill_result.quantity
            else:
                # No custom fill model => fallback
                fill_price, fill_quantity = self._fallback_fill_logic(order, timestamp)

            # If the fill model said 0 quantity filled, skip
            if fill_quantity <= 0:
                continue

            # Check margin requirements
            if self.margin_model is not None:
                if not self.margin_model.can_fill_order(portfolio, order, fill_result):
                    if not self.allow_partial_margin_fills:
                        order.status = OrderStatus.REJECTED
                        portfolio.close_order(order)
                    continue

                # Update fill quantity based on margin constraints
                fillable_quantity = self.margin_model.compute_fillable_quantity(
                    portfolio, order, fill_result
                )
            else:
                fillable_quantity = fill_quantity  # no check

            # Determine final fill quantity
            final_quantity = min(
                fill_quantity,
                fillable_quantity,
                order.quantity - order.filled_quantity,  # Remaining quantity
            )

            # Process the fill
            try:
                if final_quantity > 0:
                    order.add_fill(
                        quantity=final_quantity, price=fill_price, timestamp=timestamp
                    )
                    portfolio.fill_order(order)

            except ValueError as e:
                # Handle insufficient cash
                order.status = OrderStatus.REJECTED

                portfolio.close_order(order)
                continue

    def _fallback_fill_logic(self, order: Order, timestamp: datetime):
        fill_quantity = order.quantity - order.filled_quantity

        fill_price = self._get_price(order.symbol, timestamp)

        # If the order is a limit, do the same naive limit check
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return (0.0, 0.0)  # skip
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return (0.0, 0.0)  # skip

        return (fill_price, fill_quantity)

    def _get_prices(self, symbols: list[str], timestamp: datetime) -> float:
        df_slice = self.asset_universe.slice_data(
            symbols=symbols, end=timestamp, lookback=1, fields=["close"]
        )
        return df_slice[self.fallback_price_field].iloc[-1].squeeze()

    def _get_price(self, symbol: str, timestamp: datetime) -> float:
        """
        Simple fallback if no fill_model is specified:
        use the last known close price from the AssetUniverse.
        """
        df_slice = self.asset_universe.slice_data(
            symbols=[symbol], end=timestamp, lookback=1, fields=["close"]
        )
        if df_slice.empty:
            raise ValueError(
                f"Could not retrieve last close price of {symbol} at {timestamp}"
            )
        return df_slice[self.fallback_price_field].iloc[-1].squeeze()
