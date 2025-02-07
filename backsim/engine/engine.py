"""
Core event engine (Clock) implementation for backsim.
"""

from datetime import datetime
from typing import List, Optional
import pandas as pd

from backsim.universe import AssetUniverse
from backsim.portfolio import Portfolio
from backsim.broker import Broker
from backsim.strategy import Strategy
from backsim.engine import SimulationEngineCallback
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Event engine that drives the backtesting simulation by advancing time
    and coordinating data flow between components.
    """

    def __init__(
        self,
        universe: AssetUniverse,
        portfolio: Portfolio,
        broker: Broker,
        strategies: List[Strategy],
        step_size: pd.Timedelta,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,  # end time inclusive
        callbacks: Optional[List[SimulationEngineCallback]] = None,
        **kwargs,  # for any additional datasets, etc
    ):
        self.universe = universe
        self.portfolio = portfolio
        self.broker = broker
        self.strategies = strategies

        self.step_size = step_size

        if start_time is None:
            start_time = self.universe.datetimes.min()
        if end_time is None:
            end_time = self.universe.datetimes.max()
        self.start_time = start_time
        self.end_time = end_time

        self.date_range = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=self.step_size,
            inclusive="both",  # or "left"/"right"
        )

        self.data_fetchers = {
            strategy.name: strategy.get_data_fetcher(universe, **kwargs)
            for strategy in self.strategies
        }
        self.price_matrix = universe.price_matrix

        self.callbacks = callbacks or []

        self.kwargs = kwargs

    def run(self):
        # --- Simulation Start ---
        # Allow the strategy to do any pre-simulation setup.
        for strategy in self.strategies:
            strategy.on_simulation_start(self.universe, **self.kwargs)

        for cb in self.callbacks:
            cb.on_simulation_start(self)

        # --- Simulation Loop ---
        initial_value = self.portfolio.portfolio_value
        from tqdm import tqdm

        pbar = tqdm(self.date_range, desc="Simulating", leave=True)
        for current_time in pbar:
            # --- Step Start Callback ---
            for cb in self.callbacks:
                cb.on_step_start(self, current_time)

            self._run_single_step(current_time)

            # --- Step End Callback ---
            for cb in self.callbacks:
                cb.on_step_end(self, current_time)

            # Update progress bar information
            current_value = self.portfolio.portfolio_value
            port_return = (current_value / initial_value) - 1
            pbar.set_postfix(
                {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "portfolio_value": f"{current_value:.2f}",
                    "return": f"{port_return:.2%}",
                }
            )

        # --- Simulation End ---
        for cb in self.callbacks:
            cb.on_simulation_end(self)

        for strategy in self.strategies:
            strategy.on_simulation_end()

        return self

    def _run_single_step(self, current_time):
        """
        Advances the simulation by a single step.

        Fetches data for each strategy using the previously configured data fetcher.
        Runs each strategy and generates orders.
        Pushes all orders to portfolio.
        Has broker process any fills.
        Updates portfolio with latest prices.
        """
        orders = []

        for strategy in self.strategies:
            try:
                data_slice = self.data_fetchers[strategy.name](current_time)
            except Exception as e:
                logger.error(f"Error fetching data for strategy {strategy.name}: {e}")
                continue

            try:
                orders.extend(
                    strategy.generate_orders(
                        data_slice=data_slice,
                        portfolio=self.portfolio,
                        current_time=current_time,
                    )
                )
            except Exception as e:
                # Run the strategy and get the orders
                logger.error(
                    f"Error generating orders for strategy {strategy.name}: {e}"
                )
                continue

        # push all orders to portfolio
        self.portfolio.add_orders(orders)

        # Have broker process any fills
        self.broker.process_fills(self.portfolio, current_time)

        # update portfolio with latest prices
        latest_prices = self.price_matrix.asof(current_time)
        self.portfolio.update_prices(latest_prices)
