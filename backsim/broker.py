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
from .portfolio import Portfolio, Order, OrderStatus, OrderType, OrderSide
from .universe import AssetUniverse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    ) -> bool:
        """
        Return True if the portfolio can support filling the given quantity
        of the order, False otherwise.
        """
        pass

    @abc.abstractmethod
    def compute_fillable_quantity(
        self, portfolio: "Portfolio", order: "Order"
    ) -> float:
        """
        Given the entire order, return how many units can actually be filled
        based on margin/cash constraints. Could be the full order.quantity
        or some partial amount.
        """
        pass

    def __call__(self, portfolio: "Portfolio", order: "Order") -> float:
        return self.compute_fillable_quantity(portfolio, order)


class SimpleCashMargin(MarginModel):
    """
    Example: The simplest margin model that just checks if we have enough cash
    to buy (or short) the requested quantity at some nominal price.
    """

    def can_fill_order(
        self,
        portfolio: "Portfolio",
        order: "Order",
    ) -> bool:
        return portfolio.cash > order.margin

    def compute_fillable_quantity(
        self, portfolio: "Portfolio", order: "Order"
    ) -> float:
        # For demonstration, fill the entire quantity if we have enough cash, else zero.
        if self.can_fill_order(portfolio, order, order.quantity):
            return order.quantity
        return 0.0


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
    ) -> float:
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
    ) -> FillResult:
        df_slice = universe.slice_data(
            symbols=[order.symbol],
            end=timestamp,
            lookback=1,
            fields=["close", "volume"],
        )

        if df_slice.empty:
            return FillResult(price=float("nan"), quantity=0.0)

        return FillResult(
            price=df_slice["close"].iloc[-1].squeeze(), quantity=order.quantity
        )


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
        fill_delay: timedelta = timedelta(0),
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
            if (timestamp - order.timestamp) < self.fill_delay
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

            # check for margin requirements
            if self.margin_model:
                fillable_quantity = self.margin_model(portfolio, order)
                if fillable_quantity <= 0.0:
                    # can't fill because of margin requirements
                    order.status = OrderStatus.REJECTED
                    portfolio.close_order(order)
                    logger.info(f"Order rejected due to insufficient margin: {order}")
                    continue
            else:
                # if no margin model, assume full quantity is fillable
                fillable_quantity = order.quantity

            # check fill
            if self.fill_model:
                fill_result = self.fill_model(order, timestamp, self.asset_universe)
                fill_price = fill_result.price
                fillable_quantity = fill_result.quantity
                final_quantity = min(
                    fill_result.quantity,
                    fillable_quantity,
                    order.quantity - order.filled_quantity,  # Remaining quantity
                )

            else:
                # If no fill model, do naive fill at close price
                fill_price = self._get_close_price(order.symbol, timestamp)
                final_quantity = fillable_quantity

            # Limit order price validation
            if order.order_type == OrderType.LIMIT:
                price_valid = (
                    order.side == OrderSide.BUY and fill_price <= order.limit_price
                ) or (order.side == OrderSide.SELL and fill_price >= order.limit_price)
                if not price_valid:
                    continue

            # Update order state
            if final_quantity > 0:
                order.filled_price = fill_price
                order.filled_quantity += final_quantity
                order.status = (
                    OrderStatus.PARTIALLY_FILLED
                    if order.filled_quantity < order.quantity
                    else OrderStatus.FILLED
                )
                portfolio.fill_order(order)

    def _get_close_price(self, symbol: str, timestamp: datetime) -> float:
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
        return df_slice["close"].iloc[-1]
