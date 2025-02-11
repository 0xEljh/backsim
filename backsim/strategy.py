"""
Base Strategy implementation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Union
import pandas as pd
from .portfolio import Portfolio, Order
from .universe import AssetUniverse


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    def __init__(self, name: str = "UnnamedStrategy"):
        # name for logging/results generation purposes
        self.name = name

    def on_simulation_start(self, universe, **kwargs):
        """
        Optional: Prepare or cache data before simulation starts, etc.
        Using this as opposed to a __enter__ and __exit__ context manager pattern
        for this to be more intuitive for users even though this is 'less pythonic'.
        Should revisit this assumption later.
        """
        pass

    def on_simulation_end(self):
        """Optional: cleanup, logging, etc."""
        pass

    @abstractmethod
    def generate_orders(
        self,
        data_slice,  # can be any data type, up to user
        portfolio: Portfolio,
        current_time: datetime,
    ) -> List[Union[Dict[str, Any], Order]]:
        """
        Generate orders based on the current data slice and portfolio state.

        Args:
            data_slice: Data slice the strategy acts on
            portfolio: Current portfolio state
            current_time: Current simulation timestamp

        Returns:
            List of order dictionaries
        """
        pass

    @abstractmethod
    def get_data_fetcher(
        self, universe: Any, **kwargs
    ) -> Callable[[datetime], pd.DataFrame]:
        """
        Return a callable that, given a timestamp, returns a data slice
        (DataFrame or otherwise) used by `generate_orders()`.
        """
        pass


# Example:
class SMACrossoverStrategy(Strategy):
    """
    A simple moving average crossover strategy:
    - We'll track a short and long SMA on the 'close' price
    - If short SMA > long SMA => BUY
    - If short SMA < long SMA => SELL
    """

    def __init__(
        self,
        symbols: List[str],
        short_window: int = 10,
        long_window: int = 30,
        name: str = "SMA Crossover",
    ):
        super().__init__(name=name)
        self.symbols = symbols
        self.short_window = short_window
        self.long_window = long_window
        self.cached_data: Optional[pd.DataFrame] = None

    def on_simulation_start(self, universe, **kwargs):
        """
        Preload all necessary data from the universe for our symbols.
        This way, we avoid repeated heavy loads each timestep.
        """
        # Personally would never do an in-place modification, so I would use a view
        # but we'll use a .copy() just in case
        self.cached_data = universe.df.loc[self.symbols].copy().sort_index()

    def get_data_fetcher(
        self, universe: AssetUniverse, **kwargs
    ) -> Callable[[datetime], pd.DataFrame]:
        """
        Returns a function that, given a timestamp, returns the last `long_window` bars
        for each symbol.
        """

        def fetch_data(current_time: datetime) -> pd.DataFrame:
            if self.cached_data is None:
                raise ValueError(
                    "Data not cached. Did you forget to call on_simulation_start?"
                )

            # Calculate the cutoff date.
            cutoff_date = current_time - pd.DateOffset(days=self.long_window)

            # print("Cutoff Date:", cutoff_date)
            # print("Current Time:", current_time)

            # Instead of slicing with pd.IndexSlice, use boolean indexing.
            dt_level = self.cached_data.index.get_level_values("datetime")
            mask = (dt_level >= cutoff_date) & (dt_level <= current_time)
            sliced_data = self.cached_data[mask]
            return sliced_data

        return fetch_data

    def generate_orders(
        self, data_slice: pd.DataFrame, portfolio, current_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        For each symbol, compute short/long SMAs from data_slice, compare them, and create buy/sell orders.
        Uses vectorized operations and pandas chaining.
        """
        if data_slice.empty:
            return []

        import numpy as np  # needed for np.where

        # Compute the short and long SMAs, take the last row per symbol, and build order parameters.
        orders_df = (
            data_slice
            # Compute the SMAs for each symbol using rolling windows.
            .assign(
                short_sma=lambda df: df.groupby(level="symbol")["close"].transform(
                    lambda x: x.rolling(self.short_window, min_periods=1).mean()
                ),
                long_sma=lambda df: df.groupby(level="symbol")["close"].transform(
                    lambda x: x.rolling(self.long_window, min_periods=1).mean()
                ),
            )
            # For each symbol, keep the last available row.
            .groupby(level="symbol")
            .last()
            .reset_index()
            # Compute the order quantity and determine the side of the order.
            .assign(
                order_size=portfolio.available_margin / 20,
                order_quantity=lambda df: (df.order_size / df.close).astype(int),
                side=lambda df: np.where(
                    df.short_sma > df.long_sma,
                    "BUY",
                    np.where(df.short_sma < df.long_sma, "SELL", None),
                ),
            )
            # Remove rows where SMA calculations are missing.
            .dropna(subset=["short_sma", "long_sma"])
        )

        # Filter out any rows that do not have a valid order side.
        orders_df = orders_df[orders_df.side.notna() & (orders_df.side != "")]

        # Add timestamp and order type columns and rename order_quantity to quantity.
        orders = (
            orders_df.assign(timestamp=current_time, order_type="MARKET")[
                ["timestamp", "symbol", "order_quantity", "side", "order_type"]
            ]
            .rename(columns={"order_quantity": "quantity"})
            .to_dict(orient="records")
        )

        return orders

    def on_simulation_end(self, **kwargs):
        """
        Cleanup or final logging if desired.
        """
        print(
            f"Strategy '{self.name}' finished. Cached data was shape: {self.cached_data.shape if self.cached_data is not None else '(None)'}"
        )
