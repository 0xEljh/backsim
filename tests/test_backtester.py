"""
Integration test for the backtester.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List

from backsim.engine import SimulationEngine
from backsim.universe import AssetUniverse, DataSliceRequest
from backsim.portfolio import Portfolio
from backsim.broker import Broker, FixedSlippage
from backsim.strategy import Strategy


class MockDataSource:
    """Mock data source for testing."""

    def __init__(self):
        self.data = {
            "AAPL": {
                datetime(2024, 1, 1): {"close": 100.0},
                datetime(2024, 1, 2): {"close": 101.0},
                datetime(2024, 1, 3): {"close": 102.0},
                datetime(2024, 1, 4): {"close": 101.5},
                datetime(2024, 1, 5): {"close": 103.0},
            }
        }


class MockAssetUniverse(AssetUniverse):

    def setup(self, data_source: MockDataSource):
        super().__init__([data_source])

    @classmethod
    def setUpClass(cls):
        cls.instance = cls(MockDataSource())

    @classmethod
    def tearDownClass(cls):
        del cls.instance

    @classmethod
    def get_instance(cls):
        return cls.instance

    def get_last_price(
        self, symbol: str, field: str = "close", timestamp: datetime = None
    ) -> float:
        """Get price for symbol at timestamp."""
        return self.data_sources.data[symbol][timestamp][field]


class MockStrategy(Strategy):
    """
    Simple test strategy that places both market and limit orders.

    Renamed from 'TestStrategy' for the same reason.
    """

    def setup(self):
        super().__init__()
        # Initialize any necessary attributes

    @classmethod
    def setUpClass(cls):
        cls.instance = cls()

    @classmethod
    def tearDownClass(cls):
        del cls.instance

    @classmethod
    def get_instance(cls):
        return cls.instance

    def data_slicer(self, timestamp: datetime) -> DataSliceRequest:
        return DataSliceRequest(
            symbols=["AAPL"], fields=["close"], lookback=1, frequency="1d"
        )

    def generate_orders(
        self,
        price_slice: Dict[str, Dict[str, List[float]]],
        portfolio: Portfolio,
        timestamp: datetime,
    ) -> List[Dict]:
        """Generate test orders."""
        orders = []

        # Place a market buy order on day 1
        if timestamp.day == 1:
            orders.append(
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "side": "BUY",
                    "order_type": "MARKET",
                    "timestamp": timestamp,
                }
            )

        # Place a limit sell order on day 3
        elif timestamp.day == 3:
            orders.append(
                {
                    "symbol": "AAPL",
                    "quantity": 50,
                    "side": "SELL",
                    "order_type": "LIMIT",
                    "limit_price": 102.5,
                    "expiry": timestamp + timedelta(days=2),
                    "timestamp": timestamp,
                }
            )

        return orders


class TestBacktester(unittest.TestCase):
    """Test the complete backtesting system."""

    def setUp(self):
        """Set up test environment."""
        self.initial_cash = 100000.0
        self.data_source = MockDataSource()
        self.universe = MockAssetUniverse(self.data_source)
        self.portfolio = Portfolio(initial_cash=self.initial_cash)
        self.broker = Broker(self.universe, slippage_model=FixedSlippage(0.001))
        self.strategy = MockStrategy()

    def test_backtest_execution(self):
        """Test full backtest execution with market and limit orders."""
        engine = SimulationEngine(
            asset_universe=self.universe,
            portfolio=self.portfolio,
            broker=self.broker,
            strategies=[self.strategy],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            step_size="1d",
        )

        results = engine.run()

        # Verify we have portfolio history
        self.assertTrue(len(results["portfolio_history"]) > 0)

        # Verify metrics are calculated
        metrics = results["metrics"]
        self.assertIsNotNone(metrics["total_return"])
        self.assertIsNotNone(metrics["sharpe_ratio"])
        self.assertIsNotNone(metrics["max_drawdown"])
        self.assertIsNotNone(metrics["win_rate"])

        # Verify order execution
        filled_orders = self.portfolio.filled_orders
        self.assertTrue(len(filled_orders) > 0)
        # note that order_type is an Enum, so we need to compare the value
        self.assertTrue(
            any(order.order_type.value == "MARKET" for order in filled_orders)
        )
        self.assertTrue(
            any(order.order_type.value == "LIMIT" for order in filled_orders)
        )

    def test_portfolio_value(self):
        """Test portfolio value calculation."""
        engine = SimulationEngine(
            asset_universe=self.universe,
            portfolio=self.portfolio,
            broker=self.broker,
            strategies=[self.strategy],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            step_size="1d",
        )
        results = engine.run()

        final_portfolio_value = results["portfolio_history"][-1]["portfolio_value"]
        initial_portfolio_value = self.portfolio.initial_cash

        self.assertAlmostEqual(
            final_portfolio_value, initial_portfolio_value * 1.05, places=2
        )

    def test_order_execution(self):
        """Test order execution process."""
        engine = SimulationEngine(
            asset_universe=self.universe,
            portfolio=self.portfolio,
            broker=self.broker,
            strategies=[self.strategy],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            step_size="1d",
        )
        results = engine.run()

        # Check if market buy order was executed on day 1
        self.assertTrue(
            any(
                order["symbol"] == "AAPL"
                and order["side"] == "BUY"
                and order["order_type"] == "MARKET"
                for order in results["filled_orders"]
                if order["timestamp"].day == 1
            )
        )

        # Check if limit sell order was executed on day 3
        self.assertTrue(
            any(
                order["symbol"] == "AAPL"
                and order["side"] == "SELL"
                and order["order_type"] == "LIMIT"
                for order in results["filled_orders"]
                if order["timestamp"].day == 3
            )
        )

    def test_metrics_calculation(self):
        """Test metrics calculation."""
        engine = SimulationEngine(
            asset_universe=self.universe,
            portfolio=self.portfolio,
            broker=self.broker,
            strategies=[self.strategy],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            step_size="1d",
        )
        results = engine.run()

        # Test total_return
        initial_cash = self.portfolio.initial_cash
        final_value = results["portfolio_history"][-1]["portfolio_value"]
        expected_total_return = (final_value - initial_cash) / initial_cash
        self.assertAlmostEqual(
            results["metrics"]["total_return"], expected_total_return, places=4
        )

        # Test sharpe_ratio
        daily_returns = [result["returns"] for result in results["portfolio_history"]]
        std_dev = np.std(daily_returns) * np.sqrt(
            252
        )  # Assuming 252 trading days per year
        expected_sharpe_ratio = results["metrics"]["sharpe_ratio"]
        self.assertAlmostEqual(
            expected_sharpe_ratio, daily_returns[-1] / std_dev, places=4
        )


if __name__ == "__main__":
    unittest.main()
