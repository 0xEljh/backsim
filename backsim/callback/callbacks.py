from .callback import Callback
import csv
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING
from backsim.portfolio.models import OrderSide

if TYPE_CHECKING:
    from backsim.engine.engine import SimulationEngine
    from backsim.portfolio import Portfolio, Order


class LoggingCallback(Callback):
    """Example callback implementation for logging key events"""

    def __init__(self, logger):
        self.logger = logger

    def on_simulation_start(self, engine: "SimulationEngine"):
        self.logger.info("🚀 Simulation started")

    def on_step_start(self, engine: "SimulationEngine", timestamp: datetime):
        self.logger.info(f"⏲️ Step started at {timestamp}")

    def on_order_filled(self, portfolio, order, realized_pnl):
        self.logger.info(
            # f"💰 Order filled at {timestamp}: {order} "
            f"Realized PnL: {realized_pnl:.2f} from {"buying" if order.side == OrderSide.BUY else "selling"} {order.symbol}"
        )

    def on_prices_update(self, portfolio):
        self.logger.info(
            # f"📈 Prices updated at {timestamp}\n"
            f"Value: {portfolio.portfolio_value:.2f} | "
            f"Margin: {portfolio.available_margin:.2f}"
        )

    def on_simulation_end(self, engine: "SimulationEngine"):
        self.logger.info("🏁 Simulation ended")


class BacktestArtifactSaver(Callback):
    def __init__(self, output_dir: str = "backtest_artifacts"):
        """
        Initialize the artifact saver.

        Args:
            output_dir: Directory to save the artifacts (default: "backtest_artifacts").
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize files
        self.orders_file = self.output_dir / "orders.csv"
        self.portfolio_file = self.output_dir / "portfolio.csv"

        # Write headers
        self._write_headers()

    def _write_headers(self):
        """Write headers to CSV files."""
        with open(self.orders_file, "w") as f:
            writer = csv.writer(f)
            # TODO: this way of doing it is not extensible, update it
            # can use a more pythonic method
            writer.writerow(
                [
                    "timestamp",
                    "symbol",
                    "quantity",
                    "side",
                    "order_type",
                    "status",
                    "filled_price",
                    "filled_quantity",
                    "fees_incurred",
                    "realized_pnl",
                ]
            )

        with open(self.portfolio_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "portfolio_value",
                    "cash",
                    "positions_value",
                    "available_margin",
                ]
            )

    def on_order_filled(self, portfolio, order, realized_pnl):
        """
        Save order details when an order is filled.
        """
        # Convert Order to dictionary using its to_dict method
        order_dict = order.to_dict()
        with open(self.orders_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    order_dict["timestamp"],
                    order_dict["symbol"],
                    order_dict["quantity"],
                    order_dict["side"],
                    order_dict["order_type"],
                    order_dict["status"],
                    order_dict["filled_price"],
                    order_dict["filled_quantity"],
                    order_dict["fees_incurred"],
                    realized_pnl,
                ]
            )

    def on_prices_update(self, portfolio):
        """
        Save portfolio details when prices are updated.
        """
        with open(self.portfolio_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    portfolio.portfolio_value,
                    portfolio._cash,
                    portfolio.positions_value,
                    portfolio.available_margin,
                ]
            )
