import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
from backsim.universe import AssetUniverse, QuantityMatrix
from backsim.portfolio import Portfolio
from backsim.broker import Broker
from backsim.strategy import SMACrossoverStrategy
from backsim.engine import SimulationEngine

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def generate_sample_data(
    symbols: list[str], start_date: str, end_date: str, freq: str
) -> Dict[str, pd.DataFrame]:
    """
    Generates a sample dictionary of DataFrames for testing.

    Args:
        symbols: List of symbols (e.g., ['AAPL', 'GOOG', 'MSFT']).
        start_date: Start date for the data (e.g., '2024-01-01').
        end_date: End date for the data (e.g., '2024-01-10').
        freq: Frequency of the data (e.g., '1H' for hourly).

    Returns:
        A dictionary of {symbol -> DataFrame} with OHLCV columns.
    """

    data = {}
    for symbol in symbols:
        dates = pd.date_range(start_date, end_date, freq=freq)
        num_rows = len(dates)
        df = pd.DataFrame(
            {
                "open": np.random.rand(num_rows),
                "high": np.random.rand(num_rows) + 0.1,
                "low": np.random.rand(num_rows) - 0.1,
                "close": np.random.rand(num_rows),
                "volume": np.random.randint(100, 1000, num_rows),
            },
            index=dates,
        )
        data[symbol] = df
    return data


if __name__ == "__main__":
    symbols = [
        "AAPL",
        "GOOG",
        "MSFT",
        "TSLA",
        "AMZN",
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "ADA",
    ]
    data = generate_sample_data(symbols, "2023-01-01", "2023-06-30", "1h")
    freq = pd.Timedelta(hours=1)

    universe = AssetUniverse.from_dict_of_dataframes(data)
    quantity_matrix = QuantityMatrix(
        symbols=symbols, start_time=universe.datetimes[0], frequency=freq
    )
    portfolio = Portfolio(initial_cash=100000, quantity_matrix=quantity_matrix)
    broker = Broker(universe)
    strategy = SMACrossoverStrategy(symbols=symbols, short_window=12, long_window=26)

    engine = SimulationEngine(
        universe=universe,
        portfolio=portfolio,
        broker=broker,
        strategies=[strategy],
        step_size=freq,
    )

    engine.run()

    print(f"Final Portfolio Value: {portfolio.portfolio_value:.2f}")
    print(f"Total Return: {(portfolio.portfolio_value / 100000 - 1):.2%}")
