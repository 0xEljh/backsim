"""
AssetUniverse implementation for storing OHLCV data in a multi-index DataFrame.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
import pandas as pd


OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class AssetUniverse:
    """
    Stores all asset data in a single multi-index DataFrame:

        Index: (symbol, datetime)
        Columns: [open, high, low, close, volume]

    Provides multiple classmethods for easy instantiation from user data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        require_ohlcv=True,
        price_matrix_builder: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        """
        Initialize with a multi-index DataFrame. Typically, you won't call this directly;
        instead, use a classmethod (e.g. from_dict_of_dataframes, from_flat_df, etc.).

        Args:
            df: DataFrame with a MultiIndex (symbol, datetime) and columns = OHLCV_COLS
            require_ohlcv: If True, raise an error if the DataFrame does not have the expected columns.
            price_matrix_builder: Optional function to build a price matrix from the DataFrame, instead of the default
        """
        self._df = df
        self._price_matrix_builder = price_matrix_builder
        if require_ohlcv:
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
            # Skip empty DataFrames
            if df_symbol.empty:
                continue

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
        required = set(["open", "high", "low", "close", "volume"])
        actual = set(self._df.columns)
        missing = required - actual
        if missing:  # extra columns are ok
            raise ValueError(f"Missing required columns: {missing}")
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
    def datetimes(self) -> pd.Series:
        """Unique datetimes present in the universe as a pandas Series."""
        unique_datetimes = self._df.index.get_level_values("datetime").unique()
        unique_datetimes = (
            pd.Series(unique_datetimes).sort_values().reset_index(drop=True)
        )
        return unique_datetimes

    @property
    def df(self) -> pd.DataFrame:
        """
        Access the underlying multi-index DataFrame directly.
        """
        return self._df

    @property
    def price_matrix(self) -> pd.DataFrame:
        """
        Returns the price matrix of the asset universe.

        If a custom price_matrix_extractor is provided, it will be used.
        Otherwise, a default implementation is used that unstacks the
        DataFrame and selects the 'close' column.

        Returns:
            A pandas DataFrame representing the price matrix, with
            'datetime' as index and 'symbol' as columns.
        """
        if self._price_matrix_builder:
            return self._price_matrix_builder(self._df)
        else:
            # Default price matrix extraction logic
            return self._df.unstack(level="symbol").loc[:, "close"]

    def slice_data(
        self,
        symbols=None,
        fields=None,
        start=None,
        end=None,
        lookback=None,
        resample_freq=None,
    ) -> pd.DataFrame:
        def _filter_symbols(df):
            if symbols is None:
                return df
            # return df.loc[pd.IndexSlice[symbols, :], :]
            return df.query("symbol in @symbols")

        def _filter_end(df):
            if end is None:
                return df
            return df.loc[pd.IndexSlice[:, :end], :]

        def _apply_lookback(df):
            if lookback is None:
                return df
            return df.groupby(level="symbol").tail(lookback)

        def _filter_start(df):
            if start is None:
                return df
            slice_end = end if end else None
            return df.loc[pd.IndexSlice[:, start:slice_end], :]

        def _select_fields(df):
            if fields is None:
                return df
            valid_fields = df.columns.intersection(fields, sort=False)
            return df[valid_fields]

        def _resample(df):
            if not resample_freq:
                return df
            return (
                df.groupby(level="symbol", group_keys=False)
                .resample(resample_freq, level="datetime")
                .last()
            )

        return (
            self._df.pipe(_filter_symbols)
            .pipe(_filter_end)
            .pipe(_apply_lookback)
            .pipe(_filter_start)
            .pipe(_select_fields)
            .pipe(_resample)
            .sort_index()
        )

    def append_data(self, new_data: pd.DataFrame) -> None:
        # Expects multi-index with (symbol, datetime).
        # We could validate or just trust the user.
        self._df = pd.concat([self._df, new_data], verify_integrity=False)
        self._df.sort_index(inplace=True)


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

    def __init__(
        self,
        symbols: List[str],
        start_time: datetime,
        frequency: Union[str, pd.Timedelta] = "1d",
    ):
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
