"""
AssetUniverse implementation for storing OHLCV data in a multi-index DataFrame.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np


@dataclass
class DataSliceRequest:
    """Data slice request parameters."""

    symbols: List[str]
    fields: List[str]
    lookback: int
    frequency: str


OHLCV_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class DataSliceRequest:
    """
    Data slice request parameters.
    """

    symbols: List[str]
    fields: List[str]  # e.g. ["open", "high", "low", "close", "volume"]
    lookback: int  # how many bars to look back
    frequency: str


class AssetUniverse:
    """
    Stores all asset data in a single multi-index DataFrame:

        Index: (symbol, datetime)
        Columns: [open, high, low, close, volume]

    Provides multiple classmethods for easy instantiation from user data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a multi-index DataFrame. Typically, you won't call this directly;
        instead, use a classmethod (e.g. from_dict_of_dataframes, from_flat_df, etc.).

        Args:
            df: DataFrame with a MultiIndex (symbol, datetime) and columns = OHLCV_COLS
        """
        self._df = df
        self._validate_df()

    @classmethod
    def from_dict_of_dataframes(cls, data: Dict[str, pd.DataFrame]) -> AssetUniverse:
        """
        Create an AssetUniverse from a dictionary of {symbol -> DataFrame},
        each DataFrame having a DateTimeIndex and columns [open, high, low, close, volume].

        Args:
            data: dict of symbol -> DataFrame with OHLCV columns

        Returns:
            AssetUniverse instance
        """
        # Build a list of (symbol, sub_df) pairs, converting each to multi-index
        df_list = []
        for symbol, df_symbol in data.items():
            # Ensure columns match expected
            # (You can handle missing columns or rename as needed)
            df_symbol = df_symbol.copy()
            df_symbol = df_symbol[OHLCV_COLS]

            # Add symbol to the index
            # We'll build a multi-index: (symbol, datetime)
            df_symbol.index = pd.MultiIndex.from_product(
                [[symbol], df_symbol.index], names=["symbol", "datetime"]
            )
            df_list.append(df_symbol)

        # Concatenate all into a single DataFrame
        if df_list:
            df_all = pd.concat(df_list)
        else:
            # If empty, create an empty frame
            df_all = pd.DataFrame(columns=OHLCV_COLS)
            df_all.index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["symbol", "datetime"]
            )

        # Sort the index for consistency
        df_all.sort_index(inplace=True)
        return cls(df_all)

    @classmethod
    def from_flat_df(
        cls,
        df: pd.DataFrame,
        symbol_col: str = "symbol",
        datetime_col: str = "datetime",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> AssetUniverse:
        """
        Create an AssetUniverse from a single "flat" DataFrame with columns for symbol, datetime, and OHLCV.

        Args:
            df: A DataFrame with columns for symbol, datetime, open, high, low, close, volume
            symbol_col: Name of the column containing symbols
            datetime_col: Name of the column containing datetimes
            open_col, high_col, low_col, close_col, volume_col: Column names for OHLCV

        Returns:
            AssetUniverse instance
        """
        # Rename columns to our standard set
        renames = {
            symbol_col: "symbol",
            datetime_col: "datetime",
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
            volume_col: "volume",
        }
        df_renamed = df.rename(columns=renames)

        # Ensure we only keep the columns we need
        df_renamed = df_renamed[["symbol", "datetime"] + OHLCV_COLS]

        # Set a multi-index
        df_renamed.set_index(["symbol", "datetime"], inplace=True)
        df_renamed.sort_index(inplace=True)
        return cls(df_renamed)

    @classmethod
    def from_multiindex_df(cls, df: pd.DataFrame) -> AssetUniverse:
        """
        Create an AssetUniverse directly from a DataFrame that already has:
            - A MultiIndex with (symbol, datetime)
            - Columns = [open, high, low, close, volume]

        Args:
            df: Already well-formed multi-index DataFrame

        Returns:
            AssetUniverse instance
        """
        return cls(df)

    def _validate_df(self):
        """
        Internal consistency checks, e.g. ensuring columns = [open, high, low, close, volume]
        and that the index is a MultiIndex of (symbol, datetime).
        """
        # Check columns
        expected = set(OHLCV_COLS)
        actual = set(self._df.columns)
        if expected != actual:
            raise ValueError(f"DataFrame must have columns {expected}, got {actual}")
        # Check multi-index
        if not isinstance(self._df.index, pd.MultiIndex):
            raise ValueError("DataFrame index must be a MultiIndex (symbol, datetime).")

        idx_names = list(self._df.index.names)
        if idx_names != ["symbol", "datetime"]:
            raise ValueError("MultiIndex must be named ['symbol', 'datetime'].")

    # -------------------------------------------------------------------------
    # Properties and Accessors
    # -------------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        """Unique list of symbols present in the universe."""
        return list(self._df.index.levels[0])

    @property
    def datetimes(self) -> List[datetime]:
        """Unique list of datetimes present in the universe."""
        return list(self._df.index.levels[1])

    @property
    def df(self) -> pd.DataFrame:
        """
        Access the underlying multi-index DataFrame directly.
        """
        return self._df

    def get_price_matrix(
        self, price_agg_func: Optional[str | callable] = "close"
    ) -> pd.DataFrame:
        """
        Return a price matrix has the form of (symbol, datetime) -> price.

        Args:
            price_agg_func: Optional function to aggregate price ohlc data (default: "close" )

        Returns:
            pd.DataFrame with (symbol, datetime) -> price
        """

        if callable(price_agg_func):
            return self._df.groupby(level="symbol").agg(price_agg_func).reset_index()

        if price_agg_func not in OHLCV_COLS:
            raise ValueError(f"Invalid price_agg_func: {price_agg_func}")

        return (
            self._df
            # select the price column (open or close)
            .groupby(level="symbol")
            .agg({price_agg_func: "last"})
            .reset_index()
        )

    def get_data_slice(
        self, slice_request: DataSliceRequest, timestamp: datetime
    ) -> pd.DataFrame:
        """
        Return a data slice for the requested symbols, fields, and lookback.

        We'll return the slice as a DataFrame with the multi-index intact,
        but filtered to the symbols, fields, and the lookback bars up to `timestamp`.

        Args:
            slice_request: DataSliceRequest
            timestamp: Current simulation timestamp

        Returns:
            pd.DataFrame (multi-index) containing the requested slice
        """
        # Step 1: Filter the universe to the requested symbols
        idx = self._df.index.get_level_values("symbol").isin(slice_request.symbols)
        df_filtered_symbols = self._df[idx]

        # Step 2: Within each symbol, filter to rows up to `timestamp`
        # We can group by symbol and then slice
        frames = []
        for symbol in slice_request.symbols:
            # Try to slice for this symbol
            try:
                df_sym = df_filtered_symbols.loc[symbol]
            except KeyError:
                continue

            df_sym_up_to_ts = df_sym.loc[:timestamp]
            if slice_request.lookback > 0:
                df_sym_up_to_ts = df_sym_up_to_ts.iloc[-slice_request.lookback :]

            # Now only keep the requested fields
            df_sym_up_to_ts = df_sym_up_to_ts[slice_request.fields]

            # Reattach symbol to index so we preserve the multi-index shape
            df_sym_up_to_ts.index = pd.MultiIndex.from_arrays(
                [[symbol] * len(df_sym_up_to_ts), df_sym_up_to_ts.index],
                names=["symbol", "datetime"],
            )
            frames.append(df_sym_up_to_ts)

        if not frames:
            return pd.DataFrame(
                columns=OHLCV_COLS,
                index=pd.MultiIndex(
                    levels=[[], []], codes=[[], []], names=["symbol", "datetime"]
                ),
            )

        df_slice = pd.concat(frames)
        df_slice.sort_index(inplace=True)
        return df_slice


class QuantityMatrix:
    """Stores and manages quantity matrices for efficient position tracking.
    Because pandas is slow for single-row updates, we'll only add values when there
    is a change, and use ffill to populate the rest of the matrix when needed.
    This is effectively a "sparse" matrix.

    Key Features:
    - **Sparse Updates**: Only updates the current row until the timestamp changes, reducing
      unnecessary operations on the full matrix.
    - **Forward-Filling**: Automatically fills missing values using forward-fill (`ffill`) to
      ensure the matrix is complete and consistent.

    In normal use, it should be initialized using metadata from AssetUniverse.

    """

    def __init__(self, symbols: List[str], start_time: datetime, frequency: str = "1d"):
        self.symbols = symbols
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
        self.start_time = start_time
        self.frequency = frequency
        self._quantity_matrix = pd.DataFrame(
            0.0,  # Initialize with zeros instead of NaN since no position = 0
            index=pd.date_range(  # simply a date array of [start_time]
                start=start_time, freq=frequency, periods=1, inclusive="left"
            ),
            columns=symbols,
        )

        self.current_row = {}  # internal storage for current row
        self.current_time = start_time

    def update_quantity(self, symbol: str, timestamp: datetime, quantity: float):
        """Update quantity for a symbol at given timestamp.
        If the timestamp is the same as the current time, update the current row.
        Else, update the quantity matrix.
        """
        if timestamp == self.current_time:
            self.current_row[symbol] = quantity
            return self.current_row

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
            matrix.ffill()
            .reindex(
                pd.date_range(
                    start=self.start_time,
                    end=self.current_time,
                    freq=self.frequency,
                    inclusive="left",
                ),
                method="ffill",
            )
            .fillna(0.0)
        )

    def get_matrix(self, up_to_timestamp: datetime):
        # similar to matrix property, but may have to either slice or continue forward fill
        if up_to_timestamp < self.current_time:
            return self.matrix.loc[:up_to_timestamp]

        return (
            self.matrix
            # add new row with nan values
            # .append(pd.DataFrame(np.nan, index=[up_to_timestamp], columns=self.symbols))
            .ffill()
            .reindex(
                pd.date_range(
                    start=self.start_time,
                    end=up_to_timestamp,
                    freq=self.frequency,
                    inclusive="left",
                ),
                method="ffill",
            )
            .fillna(0.0)
        )
