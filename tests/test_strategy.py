import pandas as pd
import numpy as np
import pytest
from backsim.universe import AssetUniverse
from backsim.strategy import SMACrossoverStrategy


def build_test_universe_data():
    """Helper function to create a small test universe with synthetic data."""
    # Create date range for 10 days to have enough data for moving averages
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    symbols = ["AAPL", "MSFT"]

    # Create multi-index for the DataFrame with correct order (symbol, datetime)
    index = pd.MultiIndex.from_product(
        [symbols, dates],  # Note: symbols first, then dates
        names=["symbol", "datetime"],  # Note: symbol first, then datetime
    )

    # Generate some synthetic price data
    np.random.seed(42)  # For reproducibility
    n_rows = len(dates) * len(symbols)
    data = {
        "open": np.random.uniform(100, 200, n_rows),
        "high": np.random.uniform(100, 200, n_rows),
        "low": np.random.uniform(100, 200, n_rows),
        "close": np.random.uniform(100, 200, n_rows),
        "volume": np.random.randint(1000, 10000, n_rows),
    }

    # Ensure high is highest and low is lowest for each row
    data["high"] = np.maximum(np.maximum(data["open"], data["close"]), data["high"])
    data["low"] = np.minimum(np.minimum(data["open"], data["close"]), data["low"])

    return pd.DataFrame(data, index=index)


def test_on_simulation_start():
    # Create a tiny in-memory universe (2 symbols, 10 days)
    df = build_test_universe_data()
    universe = AssetUniverse(df)

    strategy = SMACrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=5)
    strategy.on_simulation_start(universe)

    assert strategy.cached_data is not None
    assert not strategy.cached_data.empty
    assert "AAPL" in strategy.cached_data.index.get_level_values("symbol")
    assert len(strategy.cached_data.index.get_level_values("datetime").unique()) == 10


def test_get_data_fetcher():
    df = build_test_universe_data()
    universe = AssetUniverse(df)

    strategy = SMACrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=5)
    strategy.on_simulation_start(universe)

    fetcher = strategy.get_data_fetcher(universe)
    assert callable(fetcher)

    # Test with a timestamp in the middle of our data range
    test_timestamp = pd.Timestamp("2023-01-06")
    sliced_data = fetcher(test_timestamp)

    # Assert the returned data is a subset up to the test_timestamp
    assert not sliced_data.empty
    # Check that the max datetime in the slice <= test_timestamp
    max_dt = sliced_data.index.get_level_values("datetime").max()
    assert max_dt <= test_timestamp


def test_generate_orders():
    df = build_test_universe_data()
    universe = AssetUniverse(df)

    # For simplicity, let's say short_window=2, long_window=3
    strategy = SMACrossoverStrategy(symbols=["AAPL"], short_window=2, long_window=3)
    strategy.on_simulation_start(universe)
    fetcher = strategy.get_data_fetcher(universe)

    # Build a minimal mock Portfolio
    class MockPortfolio:
        def __init__(self, available_margin):
            self.available_margin = available_margin

    portfolio = MockPortfolio(available_margin=10000)

    # Pick a timestamp in the middle of our data range
    current_time = pd.Timestamp("2023-01-06")
    data_slice = fetcher(current_time)
    orders = strategy.generate_orders(data_slice, portfolio, current_time)

    # We can test if it returns any orders (depending on the data)
    # e.g., we expect either a BUY or SELL for "AAPL"
    assert isinstance(orders, list)
    if orders:  # Only check if we got orders
        for order in orders:
            assert order["symbol"] == "AAPL"
            assert order["timestamp"] == current_time
            assert order["order_type"] == "MARKET"
            assert order["quantity"] > 0  # from the calculation in generate_orders
            # side is either BUY or SELL
            assert order["side"] in ["BUY", "SELL"]


def test_generate_orders_empty_slice():
    df = build_test_universe_data()
    universe = AssetUniverse(df)

    strategy = SMACrossoverStrategy(symbols=["AAPL"])
    strategy.on_simulation_start(universe)

    data_slice = pd.DataFrame()  # simulate no data
    mock_portfolio = type("MockPortfolio", (), {"available_margin": 10000})()

    # Test with a timestamp outside our data range
    orders = strategy.generate_orders(
        data_slice, mock_portfolio, pd.Timestamp("2023-01-15")
    )
    assert orders == []  # Expect no orders if there's no data


def test_on_simulation_end(capfd):
    df = build_test_universe_data()
    universe = AssetUniverse(df)

    strategy = SMACrossoverStrategy(symbols=["AAPL"])
    strategy.on_simulation_start(universe)
    strategy.on_simulation_end()

    # Capture stdout to check for the printed message
    captured = capfd.readouterr()
    assert "finished. Cached data was shape:" in captured.out
