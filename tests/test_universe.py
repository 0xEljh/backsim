"""
Tests for the AssetUniverse class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backsim.universe import AssetUniverse, OHLCV_COLS


# Test fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Create a sample OHLCV DataFrame for a single symbol."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "close": [102.0, 103.0, 104.0, 105.0, 106.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def multi_symbol_data(sample_ohlcv_data):
    """Create sample data for multiple symbols."""
    return {
        "AAPL": sample_ohlcv_data,
        "GOOGL": sample_ohlcv_data * 2,  # Different prices for second symbol
    }


# Test Instantiation and Validation
def test_empty_universe_creation():
    """Test creating an empty universe."""
    empty_df = pd.DataFrame(columns=OHLCV_COLS)
    empty_df.index = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["symbol", "datetime"]
    )
    universe = AssetUniverse(empty_df)
    assert len(universe.symbols) == 0
    assert len(universe.datetimes) == 0


def test_missing_columns():
    """Test that ValueError is raised when required columns are missing."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            # Missing 'close' and 'volume'
        }
    )
    df.index = pd.MultiIndex.from_tuples(
        [("AAPL", datetime(2024, 1, 1))], names=["symbol", "datetime"]
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        AssetUniverse(df)


def test_incorrect_multiindex():
    """Test that ValueError is raised when index structure is incorrect."""
    # Test with single index
    df = pd.DataFrame({col: [100.0] for col in OHLCV_COLS}, index=["AAPL"])
    with pytest.raises(ValueError, match="must be a MultiIndex"):
        AssetUniverse(df)

    # Test with wrong index names
    df.index = pd.MultiIndex.from_tuples(
        [("AAPL", datetime(2024, 1, 1))], names=["wrong", "date"]
    )
    with pytest.raises(ValueError, match="must be named"):
        AssetUniverse(df)


# Test Classmethods
def test_from_dict_of_dataframes(multi_symbol_data):
    """Test creating universe from dictionary of DataFrames."""
    universe = AssetUniverse.from_dict_of_dataframes(multi_symbol_data)
    assert set(universe.symbols) == {"AAPL", "GOOGL"}
    assert len(universe.datetimes) == 5

    # Test with empty data for one symbol
    data_with_empty = {
        "AAPL": multi_symbol_data["AAPL"],
        "EMPTY": pd.DataFrame(columns=OHLCV_COLS),
    }
    universe = AssetUniverse.from_dict_of_dataframes(data_with_empty)
    assert set(universe.symbols) == {"AAPL"}  # Empty symbol should be excluded


def test_from_flat_df(sample_ohlcv_data):
    """Test creating universe from flat DataFrame with custom column names."""
    # Create flat df with custom column names
    flat_df = sample_ohlcv_data.reset_index()
    flat_df["symbol"] = "AAPL"
    flat_df.columns = ["date", "o", "h", "l", "c", "v", "sym"]  # Custom names

    universe = AssetUniverse.from_flat_df(
        flat_df,
        symbol_col="sym",
        datetime_col="date",
        open_col="o",
        high_col="h",
        low_col="l",
        close_col="c",
        volume_col="v",
    )

    assert len(universe.symbols) == 1
    assert len(universe.datetimes) == 5
    assert all(col in universe.df.columns for col in OHLCV_COLS)


def test_from_multiindex_df(multi_symbol_data):
    """Test creating universe from an already well-formed multi-index DataFrame."""
    # Create a proper multi-index df
    dfs = []
    for symbol, df in multi_symbol_data.items():
        df = df.copy()
        df.index = pd.MultiIndex.from_product(
            [[symbol], df.index], names=["symbol", "datetime"]
        )
        dfs.append(df)

    multi_df = pd.concat(dfs)
    universe = AssetUniverse.from_multiindex_df(multi_df)

    assert set(universe.symbols) == {"AAPL", "GOOGL"}
    assert len(universe.datetimes) == 5
    assert universe.df.equals(multi_df)  # Should be identical
