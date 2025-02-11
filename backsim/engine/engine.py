"""
Core event engine (Clock) implementation for backsim.
"""

from datetime import datetime
from typing import List, Optional
import pandas as pd

from backsim.universe import AssetUniverse, QuantityMatrix
from backsim.portfolio import Portfolio
from backsim.broker import Broker
from backsim.strategy import Strategy
from backsim.callbacks import Callback
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
        initial_cash: float,
        step_size: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        callbacks: Optional[List[Callback]] = None,
        epoch_size: int = 100
    ):
        self.initial_cash = initial_cash
        self.step_size = step_size
        self.start_time = start_time
        self.end_time = end_time
        self.callbacks = callbacks or []
        self.epoch_size = epoch_size

        # Internal placeholders that will be set up in run()
        self.universe = None
        self.price_matrix = None
        self.portfolio = None
        self.broker = None
        self.current_strategy = None
        self.date_range = None
        self.kwargs = None  # additonal datasets, etc.

    def on_simulation_start(self, strategies, universe, **kwargs):
        # create all relevant objects
        self.universe = universe
        self.price_matrix = universe.price_matrix
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            quantity_matrix=QuantityMatrix(
                symbols=self.universe.symbols,
                start_time=self.start_time,
                frequency=self.step_size,
            ),
            callbacks=self.callbacks,  # simply pass down callbacks
        )
        self.broker = Broker(universe)
        self.strategies = strategies
        self.kwargs = kwargs

        if self.start_time is None:
            self.start_time = self.universe.datetimes.min()
        if self.end_time is None:
            self.end_time = self.universe.datetimes.max()

        self.date_range = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=self.step_size,
            inclusive="both",  # or "left"/"right"
        )

        # Allow the strategy to do any pre-simulation setup.
        for strategy in self.strategies:
            strategy.on_simulation_start(self.universe, **self.kwargs)

        # get data fetchers from strategies
        self.data_fetchers = {
            strategy.name: strategy.get_data_fetcher(universe, **kwargs)
            for strategy in self.strategies
        }

        for cb in self.callbacks:
            cb.on_simulation_start(engine=self)

    def on_simulation_end(self):

        for cb in self.callbacks:
            cb.on_simulation_end(self)

        for strategy in self.strategies:
            strategy.on_simulation_end()

    def run(self, strategies, asset_universe, **kwargs):
        # --- Simulation Start ---
        self.on_simulation_start(strategies, asset_universe, **kwargs)

        # --- Simulation Loop ---
        initial_value = self.portfolio.portfolio_value
        from tqdm import tqdm

        pbar = tqdm(self.date_range, desc="Simulating", leave=True)
        step_count = 0
        total_steps = len(self.date_range)
        for current_time in pbar:
            step_count += 1
            # --- Step Start Callback ---
            for cb in self.callbacks:
                cb.on_step_start(self, current_time)

            self._run_single_step(current_time)

            # --- Step End Callback ---
            for cb in self.callbacks:
                cb.on_step_end(self, current_time)

            if step_count % self.epoch_size == 0 or step_count == total_steps:
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
        self.on_simulation_end()

        return self.portfolio

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
