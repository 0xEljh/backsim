import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
from backsim.universe import AssetUniverse, QuantityMatrix
from backsim.portfolio import Portfolio
from backsim.broker import Broker
from backsim.strategy import SMACrossoverStrategy
from backsim.engine import SimulationEngine
from backsim.callback import LoggingCallback
from backsim.callback.stats_collector import StatsCollectorCallback
import logging

logger = logging.getLogger(__name__)

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
    start_date = "2023-01-01"
    end_date = "2023-06-30"
    freq = pd.Timedelta(days=1)
    data = generate_sample_data(symbols, start_date, end_date, freq)

    universe = AssetUniverse.from_dict_of_dataframes(data)
    strategy = SMACrossoverStrategy(symbols=symbols, short_window=12, long_window=26)

    stats_callback = StatsCollectorCallback()

    backsim = SimulationEngine(
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2023, 6, 30),
        step_size=freq,
        initial_cash=100000,
        callbacks=[LoggingCallback(logger), stats_callback],
        epoch_size=50,
    )

    portfolio = backsim.run(strategies=[strategy], asset_universe=universe)

    print(portfolio.portfolio_value)

    # compute stats
    df_timeseries = stats_callback.get_timeseries_df()
    print("Sharpe ratio:", stats_callback.compute_sharpe_ratio())
    print("Max Drawdown:", stats_callback.compute_max_drawdown())

    stats_callback.plot_portfolio_value()
