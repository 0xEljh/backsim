"""
Core event engine (Clock) implementation for backsim.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

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
        step_size: str = "1d",
    ):
        self.asset_universe = asset_universe
        self.portfolio = portfolio
        self.broker = broker
        self.strategies = strategies
        self.start_time = start_time
        self.end_time = end_time
        self.step_size = step_size
        self.current_time: Optional[datetime] = None

    def run(self) -> Dict[str, Any]:
        """
        Run the backtesting simulation from start_time to end_time.

        Returns:
            Dict containing simulation results and statistics
        """
        self.current_time = self.start_time
        results = {"portfolio_history": [], "metrics": {}}

        while self.current_time <= self.end_time:
            # 1. Generate orders from strategies
            orders = []
            for strategy in self.strategies:
                # Get data slices based on strategy requirements
                price_slice = self.asset_universe.get_data_slice(
                    strategy.data_slicer(self.current_time), self.current_time
                )

                # Generate and collect orders
                strategy_orders = strategy.generate_orders(
                    price_slice, self.portfolio, self.current_time
                )
                orders.extend(strategy_orders)

            # 2. Submit orders to portfolio
            self.portfolio.add_orders(orders)

            # 3. Process fills through broker
            self.broker.process_fills(self.portfolio, self.current_time)

            # 4. Record state
            results["portfolio_history"].append(
                self.portfolio.get_snapshot(self.current_time)
            )

            # 5. Advance time
            self.current_time = self._increment_time(self.current_time)

        # Calculate final metrics
        results["metrics"] = self._calculate_metrics(results["portfolio_history"])
        return results

    def _increment_time(self, current_time: datetime) -> datetime:
        """
        Advance the simulation time by step_size.
        """
        # TODO: Implement proper time advancement logic based on step_size
        # This is a simplified version
        if self.step_size == "1d":
            return current_time.replace(day=current_time.day + 1)
        raise NotImplementedError(f"Step size {self.step_size} not implemented")

    def _calculate_metrics(self, portfolio_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate performance metrics from portfolio history.

        Args:
            portfolio_history: List of portfolio snapshots

        Returns:
            Dict containing calculated metrics
        """
        if not portfolio_history:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        # Calculate daily returns
        equity_curve = [
            snapshot["cash"]
            + sum(
                pos["quantity"] * pos.get("market_value", pos["cost_basis"])
                for pos in snapshot["positions"]
            )
            for snapshot in portfolio_history
        ]

        returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
        ]

        if not returns:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Sharpe ratio (assuming daily data, risk-free rate = 0)
        returns_mean = sum(returns) / len(returns)
        returns_std = (
            sum((r - returns_mean) ** 2 for r in returns) / len(returns)
        ) ** 0.5
        sharpe_ratio = returns_mean / returns_std * (252**0.5) if returns_std > 0 else 0

        # Maximum drawdown
        peak = equity_curve[0]
        max_drawdown = 0.0
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Win rate
        winning_days = sum(1 for r in returns if r > 0)
        win_rate = winning_days / len(returns) if returns else 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }
