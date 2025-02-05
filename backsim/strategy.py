"""
Base Strategy implementation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
import pandas as pd
from .portfolio import Portfolio
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

    def on_end(self):
        """Optional: cleanup, logging, etc."""
        pass

    @abstractmethod
    def generate_orders(
        self,
        price_slice: Dict[str, Dict[str, List[float]]],
        portfolio: Portfolio,
        timestamp: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Generate orders based on the current data slice and portfolio state.

        Args:
            price_slice: Price/volume data for relevant symbols
            portfolio: Current portfolio state
            timestamp: Current simulation timestamp

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

    # def on_simulation_start(self, universe, **kwargs):
    #     """
    #     Preload all necessary data from the universe for our symbols.
    #     This way, we avoid repeated heavy loads each timestep.
    #     """
    #     df = universe._df.loc[self.symbols].copy().sort_index()
    #     self.cached_data = df

    def on_simulation_start(self, universe, **kwargs):
        """
        Preload all necessary data from the universe for our symbols.
        This way, we avoid repeated heavy loads each timestep.
        """
        # Filter data for the specified symbols
        df = universe._df.loc[self.symbols].copy()

        # Explicitly rebuild the MultiIndex so that the datetime part is converted.
        df.index = pd.MultiIndex.from_tuples(
            [(symbol, pd.to_datetime(dt)) for symbol, dt in df.index],
            names=df.index.names,
        )

        # Debug: Print the index and type of the datetime values
        print("Cached Data Index (after conversion):", df.index)
        print(
            "Type of first datetime element:",
            type(df.index.get_level_values("datetime")[0]),
        )

        # Sort the index
        df = df.sort_index()

        # Cache the data
        self.cached_data = df

    # def get_data_fetcher(
    #     self, universe: AssetUniverse, **kwargs
    # ) -> Callable[[datetime], pd.DataFrame]:
    #     """
    #     Returns a function that, given a timestamp, returns the last `long_window` bars
    #     for each symbol. We'll compute SMAs in `generate_orders` below.
    #     """

    #     def fetch_data(current_time: datetime) -> pd.DataFrame:
    #         if self.cached_data is None:
    #             raise ValueError(
    #                 "Data not cached. Did you forget to call on_simulation_start?"
    #             )

    #         # Get the earliest date we need to include
    #         cutoff_date = current_time - pd.DateOffset(days=self.long_window)

    #         # Slice data to include only up to current_time and after cutoff_date
    #         sliced_data = self.cached_data.loc[
    #             (slice(None), slice(cutoff_date, current_time))
    #         ].sort_index()

    #         return sliced_data

    #     return fetch_data
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

            # Debug: Print the cutoff_date and current_time
            print("Cutoff Date:", cutoff_date)
            print("Current Time:", current_time)

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
                order_size=portfolio.cash / 10,
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
