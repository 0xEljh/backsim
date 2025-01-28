"""
Core event engine (Clock) implementation for backsim.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd

from .universe import AssetUniverse
from .portfolio import Portfolio
from .broker import Broker
from .strategy import Strategy


class SimulationEngine:
    """
    Event engine that drives the backtesting simulation by advancing time
    and coordinating data flow between components.
    """

    def __init__(
        self,
        asset_universe: AssetUniverse,
        portfolio: Portfolio,
        broker: Broker,
        strategies: List[Strategy],
        start_time: datetime,
        end_time: datetime,
        step_size: str = "1D",  # e.g. '1D' for daily. Using pandas frequency strings
    ):
        self.asset_universe = asset_universe
        self.portfolio = portfolio
        self.broker = broker
        self.strategies = strategies
        self.start_time = start_time
        self.end_time = end_time
        self.step_size = step_size

        # Will store cash at each timestep. Key = datetime, Value = float
        self.cash_series = {}

        # Store a list of (timestamp, order/fill) for logging
        self.orders_log: List[Dict[str, Any]] = []
        # self.fills_log: List[Dict[str, Any]] = []

        # The current_time will be set during .run()
        self.current_time: Optional[datetime] = None

    def run(self) -> Dict[str, Any]:
        """
        Run the backtesting simulation from start_time to end_time.

        Returns:
            Dict containing simulation results and statistics
        """
        # Create a date range based on start/end/step_size
        date_range = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=self.step_size,
            inclusive="both",  # or "left"/"right"
        )

        # Event loop
        for current_time in date_range:
            self.current_time = current_time

            # 1. Generate orders from all strategies
            self._generate_and_submit_orders(current_time)

            # 2. Process fills through the broker
            self.broker.process_fills(self.portfolio, current_time)

            # 3. Record the portfolio’s *current* cash. The quantity matrix takes care
            #    of positions changes, so we only track cash changes here.
            self.cash_series[current_time] = self.portfolio.cash

        # After stepping through the entire simulation, do a vectorized PnL calculation
        results = self._compile_results()

        return results

    def _generate_and_submit_orders(self, current_time: datetime):
        """Helper method to generate new orders from each strategy and add them to the portfolio."""
        # Collect all orders from all strategies
        orders = []
        for strategy in self.strategies:
            # TODO: update this when figuring out shape of strategy
            # 1. Figure out the data slice needed for this strategy
            #    The strategy might define a helper function data_slicer(current_time)
            #    that returns something like (symbol, slice_of_data).
            price_slice = self.asset_universe.get_data_slice(
                strategy.data_slicer(current_time), current_time
            )

            # 2. Ask the strategy for orders
            strategy_orders = strategy.generate_orders(
                price_slice, self.portfolio, current_time
            )
            orders.extend(strategy_orders)

        # TODO: consider strategies that might want to be aware of other orders being posted, or a meta strategy to manage orders
        # (Perhaps that's just a strategy stack)
        # 3. Add them to the portfolio
        self.portfolio.add_orders(orders)

        # 4. Logging
        for o in orders:
            # order may be a dict or Order, which can also be accessed like a dict
            self.orders_log.append(
                {
                    # copy, since order is mutable
                    "timestamp": o.get("timestamp", current_time),
                    "symbol": o["symbol"],
                    "quantity": o["quantity"],
                    "side": o["side"],
                    "order_type": o.get("order_type", "MARKET"),
                    "limit_price": o.get("limit_price"),
                    "expiry": o.get("expiry"),
                }
            )

    def _compile_results(self) -> Dict[str, Any]:
        """
        Do a final pass to compile simulation results:
          - vectorized portfolio value timeseries
          - final metrics
          - logs
        """

        # TODO: see if win rate can be added

        # 1) Convert cash_series dict -> a pandas Series
        cash_series = pd.Series(self.cash_series).sort_index()

        # 2) Compute positions value over time (vectorized) using the quantity matrix
        #    and the relevant slice of prices from AssetUniverse
        full_quantity_matrix = (
            self.portfolio.quantity_matrix.matrix
        )  # .matrix is ffilled
        date_index = full_quantity_matrix.index

        # Gather a matching panel of prices from AssetUniverse
        price_df = self.asset_universe.get_price_matrix(
            start=date_index.min(),
            end=date_index.max(),
            field="close",  # or "adjusted_close", etc
            freq=self.step_size,
        )

        # quantity matrix and price df should likely already be aligned (same freq); TODO consider if necessary
        price_df_aligned = price_df.reindex(date_index, method="ffill").fillna(
            method="ffill"
        )

        # Multiply quantity_matrix by price_df to get value of each symbol over time
        #   Then sum across columns to get total positions value
        positions_value_series = (full_quantity_matrix * price_df_aligned).sum(axis=1)

        # 3) Combine positions value with cash to get total portfolio value.
        #    We must align the indexes, so let’s reindex `cash_series` to date_index as well.
        cash_series_aligned = cash_series.reindex(date_index, method="ffill").fillna(
            method="ffill"
        )
        portfolio_value_series = positions_value_series + cash_series_aligned

        # 4) Compute final metrics
        metrics = self._calculate_metrics(portfolio_value_series)

        # 5) Return everything in a dictionary, so user can do further analysis
        return {
            "portfolio_value": portfolio_value_series,
            "cash": cash_series_aligned,
            "positions_value": positions_value_series,
            "orders_log": self.orders_log,
            "fills_log": self.fills_log,  # TODO: populate fills log
            "metrics": metrics,
        }

    def _calculate_metrics(self, portfolio_value_series: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics from the final portfolio value series.
        This can be as sophisticated as you want (CAGR, drawdowns, Sharpe, etc.)
        """
        if portfolio_value_series.empty:
            return {}

        start_val = portfolio_value_series.iloc[0]
        end_val = portfolio_value_series.iloc[-1]
        returns = portfolio_value_series.pct_change().dropna()

        total_return = (end_val / start_val) - 1.0
        annualized_return = self._annualized_return(returns)
        max_drawdown = self._max_drawdown(portfolio_value_series)

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            # TODO: Add more
        }

    @staticmethod
    def _annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Approximate annualized return from daily returns (or whichever period length is used).
        """
        avg_return = returns.mean()
        return (1 + avg_return) ** periods_per_year - 1

    @staticmethod
    def _max_drawdown(value_series: pd.Series) -> float:
        """
        Calculate maximum drawdown from a series of portfolio values.
        """
        running_max = value_series.cummax()
        drawdown = (value_series - running_max) / running_max
        return drawdown.min()
