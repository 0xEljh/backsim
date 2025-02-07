from abc import ABC, abstractmethod
from pathlib import Path
import csv


class PortfolioCallback(ABC):
    @abstractmethod
    def on_order_filled(self, portfolio: "Portfolio", order: "Order", realized_pnl):
        """
        Called when an order is completely or partially filled.
        :param portfolio: The Portfolio instance.
        :param order: The Order that was filled.
        :param realized_pnl: Realized profit and loss from the fill.
        """
        pass

    @abstractmethod
    def on_prices_update(self, portfolio: "Portfolio"):
        """
        Called when an order is closed (terminal state).
        """
        pass


class LoggingCallback(PortfolioCallback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_order_filled(self, portfolio: "Portfolio", order: "Order", realized_pnl):
        self.logger.info(f"Order filled: {order} Realized PnL: {realized_pnl}")

    def on_prices_update(self, portfolio: "Portfolio"):
        self.logger.info(
            f"Portfolio update with new prices\nPortfolio value: {portfolio.portfolio_value}\nMargin available: {portfolio.available_margin}"
        )


# TODO: perhaps consider a more global artifact saver
# that way, we also have a more definite global place to save the artifacts to


class BacktestArtifactSaver:
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
