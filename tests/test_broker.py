"""Unit tests for the Broker class."""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from backsim.broker import (
    Broker,
    FillModel,
    MarginModel,
    FillResult,
    SimpleCashMargin,
    NaiveCloseFillModel,
    VolumeAwareLimitFillModel,
)
from backsim.portfolio import Portfolio, Order, OrderStatus, OrderType, OrderSide
from backsim.universe import AssetUniverse, QuantityMatrix


# Test fixtures
@pytest.fixture
def sample_universe_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start="2025-01-15", periods=5, freq="D")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100, 101, 100, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 99, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        ),
        "GOOGL": pd.DataFrame(
            {
                "open": [200, 202, 204, 206, 208],
                "high": [210, 212, 214, 216, 218],
                "low": [190, 192, 194, 196, 198],
                "close": [204, 206, 208, 210, 212],
                "volume": [2000, 2200, 2400, 2600, 2800],
            },
            index=dates,
        ),
        "SMOL": pd.DataFrame(
            {
                "open": [200, 202, 204, 206, 208],
                "high": [210, 212, 214, 216, 218],
                "low": [190, 192, 194, 196, 198],
                "close": [204, 206, 208, 210, 212],
                "volume": [20, 22, 24, 26, 28],
            },
            index=dates,
        ),
    }
    return AssetUniverse.from_dict_of_dataframes(data)


@pytest.fixture
def start_time():
    """Start time for the test period."""
    return datetime(2025, 1, 15, 10, 0, 0)


@pytest.fixture
def end_time(start_time):
    """End time for the test period."""
    return start_time + timedelta(days=5)


@pytest.fixture
def quantity_matrix(sample_universe_data, start_time):
    """Create quantity matrix."""
    return QuantityMatrix(sample_universe_data.symbols, start_time=start_time)


@pytest.fixture
def portfolio(quantity_matrix):
    """Create a portfolio with initial cash."""
    initial_cash = 10000.0
    return Portfolio(initial_cash, quantity_matrix)


@pytest.fixture
def broker(sample_universe_data):
    """Create a broker with default models."""
    return Broker(
        asset_universe=sample_universe_data,
        fill_model=NaiveCloseFillModel(),
        margin_model=SimpleCashMargin(),
        fill_delay=timedelta(0),
    )


# Custom fill model for testing
class FixedPriceFillModel(FillModel):
    """Fill model that uses a fixed price for testing"""

    def __init__(self, fill_price: float):
        super().__init__()  # Call parent constructor if needed
        self.fill_price = fill_price

    def get_fill(self, order, timestamp, universe):
        # Implementation that returns the fixed price
        return FillResult(self.fill_price, order.quantity)


# Market Order Tests
def test_market_order_full_fill(broker, portfolio, start_time):
    """Test that market order gets fully filled when enough margin."""
    order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.MARKET,
        "side": OrderSide.BUY,
        "timestamp": start_time,
    }
    portfolio.add_orders([order_dict])

    assert portfolio.open_orders[0].status == OrderStatus.PENDING

    broker.process_fills(portfolio, start_time)

    assert portfolio.closed_orders[0].status == OrderStatus.FILLED
    assert portfolio.closed_orders[0].filled_quantity == 10
    assert len(portfolio.positions) == 1
    assert portfolio.positions["AAPL"].quantity == 10


def test_market_order_insufficient_margin(broker, portfolio, start_time):
    """Test that order is rejected when insufficient margin."""
    # Create order that requires more cash than available
    large_quantity = 1000  # Will require more cash than portfolio has
    order_dict = {
        "symbol": "AAPL",
        "quantity": large_quantity,
        "order_type": OrderType.MARKET,
        "side": OrderSide.BUY,
        "timestamp": start_time,
    }
    portfolio.add_orders([order_dict])
    broker.process_fills(portfolio, start_time)

    assert portfolio.closed_orders[0].status == OrderStatus.REJECTED
    assert len(portfolio.positions) == 0


# Limit Order Tests
def test_limit_order_execution(broker, portfolio, start_time):
    """Test limit order fills when price conditions are met."""
    # Set limit price above current market price for a sell order
    limit_price = 104.5  # above 3rd day close price, below 4th day close price

    # First establish a position
    market_order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.MARKET,
        "side": OrderSide.BUY,
        "timestamp": start_time,
    }
    portfolio.add_orders([market_order_dict])
    broker.process_fills(portfolio, start_time)

    assert len(portfolio.open_orders) == 0

    # Now try to sell with limit
    limit_order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.LIMIT,
        "side": OrderSide.SELL,
        "timestamp": start_time + timedelta(days=2),
        "limit_price": limit_price,
    }
    portfolio.add_orders([limit_order_dict])

    # Process at a time when price is below limit
    broker.process_fills(portfolio, start_time + timedelta(days=2))
    assert len(portfolio.open_orders) == 1
    assert len(portfolio.closed_orders) == 1

    # Process at a time when price is above limit
    broker.process_fills(portfolio, start_time + timedelta(days=3))

    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 2

    # Get the limit order (should be the second order)
    limit_order = portfolio.closed_orders[-1]
    assert limit_order.status == OrderStatus.FILLED
    assert limit_order.filled_quantity == 10
    assert "AAPL" not in portfolio.positions  # Position should be closed


def test_limit_order_expiry(broker, portfolio, start_time):
    """Test that limit orders expire correctly."""
    order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.LIMIT,
        "side": OrderSide.BUY,
        "timestamp": start_time,
        "limit_price": 90.0,  # Below market price
        "expiry": start_time + timedelta(days=1),
    }
    portfolio.add_orders([order_dict])

    # Process fills after expiry
    broker.process_fills(portfolio, start_time + timedelta(days=2))
    assert portfolio.closed_orders[0].status == OrderStatus.EXPIRED


def test_limit_order_directionality_long(broker, portfolio, start_time):
    """Test limit order directionality for going long."""
    # Buy positive quantity to establish long position
    # AAPL (close) prices: 102, 99, 104, 105, 106
    limit_price = 101.0  # Above 2nd day close price
    order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.LIMIT,
        "side": OrderSide.BUY,  # Buy to open long
        "timestamp": start_time,
        "limit_price": limit_price,
    }
    portfolio.add_orders([order_dict])

    # Process at a time when price is above limit (should not fill)
    broker.process_fills(portfolio, start_time)
    assert len(portfolio.open_orders) == 1
    assert len(portfolio.closed_orders) == 0

    # Process at a time when price is below limit
    broker.process_fills(portfolio, start_time + timedelta(days=1))
    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 1
    assert portfolio.positions["AAPL"].quantity == 10  # Long position established

    # Now sell positive quantity to reduce long
    sell_limit_price = 104.5  # Below 4th day close price, above 3rd
    sell_order_dict = {
        "symbol": "AAPL",
        "quantity": 5,  # Positive quantity to reduce long
        "order_type": OrderType.LIMIT,
        "side": OrderSide.SELL,  # Sell to reduce long
        "timestamp": start_time + timedelta(days=2),
        "limit_price": sell_limit_price,
    }
    portfolio.add_orders([sell_order_dict])

    broker.process_fills(portfolio, start_time + timedelta(days=2))
    assert len(portfolio.open_orders) == 1

    # Process fills when price rises above limit
    broker.process_fills(portfolio, start_time + timedelta(days=3))
    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 2
    assert portfolio.positions["AAPL"].quantity == 5  # Long position reduced


def test_limit_order_directionality_short(broker, portfolio, start_time):
    """Test limit order directionality for going short."""
    # Buy negative quantity to establish short position
    # AAPL (close) prices: 102, 99, 104, 105, 106
    limit_price = 103.0  # Below 3rd day close price; above prev days
    order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.LIMIT,
        "side": OrderSide.SELL,  # Sell to open short
        "timestamp": start_time,
        "limit_price": limit_price,
    }
    portfolio.add_orders([order_dict])

    broker.process_fills(portfolio, start_time)
    assert len(portfolio.open_orders) == 1

    # Process at a time when price is above limit (should fill)
    broker.process_fills(portfolio, start_time + timedelta(days=2))
    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 1
    assert portfolio.positions["AAPL"].quantity == -10  # Short position established

    # Now buy positive quantity to reduce short
    buy_limit_price = 105.0
    buy_order_dict = {
        "symbol": "AAPL",
        "quantity": 5,  # Positive quantity to reduce short
        "order_type": OrderType.LIMIT,
        "side": OrderSide.BUY,  # Buy to reduce short
        "timestamp": start_time + timedelta(days=3),
        "limit_price": buy_limit_price,
    }
    portfolio.add_orders([buy_order_dict])

    # Process fills when price drops to limit
    broker.process_fills(portfolio, start_time + timedelta(days=3))
    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 2
    assert portfolio.positions["AAPL"].quantity == -5  # Short position reduced


def test_negative_quantity_rejection(broker, portfolio, start_time):
    """Test that negative quantity orders are rejected."""
    order_dict = {
        "symbol": "AAPL",
        "quantity": -10,  # Negative quantity should be rejected
        "order_type": OrderType.LIMIT,
        "side": OrderSide.BUY,
        "timestamp": start_time,
        "limit_price": 100.0,
    }
    portfolio.add_orders([order_dict])

    broker.process_fills(portfolio, start_time)
    assert len(portfolio.closed_orders) == 1
    assert portfolio.closed_orders[0].status == OrderStatus.REJECTED


def test_zero_quantity_order(broker, portfolio, start_time):
    """Test that zero quantity orders are rejected."""
    order_dict = {
        "symbol": "AAPL",
        "quantity": 0,
        "order_type": OrderType.MARKET,
        "side": OrderSide.BUY,
        "timestamp": start_time,
    }
    portfolio.add_orders([order_dict])
    broker.process_fills(portfolio, start_time)

    assert portfolio.closed_orders[0].status == OrderStatus.REJECTED


# Custom Fill Model Tests
def test_custom_fill_model(sample_universe_data, portfolio, start_time):
    """Test that custom fill model's price is used."""
    expected_fill_price = 150.0
    custom_fill_model = FixedPriceFillModel(fill_price=expected_fill_price)

    broker = Broker(
        asset_universe=sample_universe_data,
        fill_model=custom_fill_model,
        margin_model=SimpleCashMargin(),
    )

    order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.MARKET,
        "side": OrderSide.BUY,
        "timestamp": start_time,
    }
    portfolio.add_orders([order_dict])
    broker.process_fills(portfolio, start_time)

    assert portfolio.closed_orders[0].filled_price == expected_fill_price


def test_volume_aware_limit_fill_model(sample_universe_data, portfolio, start_time):
    """Test VolumeAwareLimitFillModel's fill behavior."""
    broker = Broker(
        asset_universe=sample_universe_data,
        fill_model=VolumeAwareLimitFillModel(),
        margin_model=SimpleCashMargin(),
    )

    # Test 1: Buy limit order - should fill when low price <= limit_price
    buy_order_dict = {
        "symbol": "AAPL",
        "quantity": 5,  # Requires ~$490 margin at $98/share
        "order_type": OrderType.LIMIT,
        "side": OrderSide.BUY,
        "timestamp": start_time,
        "limit_price": 98.0,  # Between low (95) and high (105) of first bar
    }
    portfolio.add_orders([buy_order_dict])
    broker.process_fills(portfolio, start_time)

    # Verify fill
    assert len(portfolio.closed_orders) == 1
    filled_buy = portfolio.closed_orders[0]
    assert filled_buy.status == OrderStatus.FILLED
    assert filled_buy.filled_price == 98.0  # Should fill at limit
    assert filled_buy.filled_quantity == 5  # Should fill fully as quantity < volume

    # Test 2: Sell limit order - should fill when high price >= limit_price
    sell_order_dict = {
        "symbol": "SMOL",  # Using SMOL to test low volume behavior
        "quantity": 30,  # More than bar volume (20) to test partial fills
        "order_type": OrderType.LIMIT,
        "side": OrderSide.SELL,
        "timestamp": start_time,
        "limit_price": 205.0,  # Between low (190) and high (210) of first bar
    }
    portfolio.add_orders([sell_order_dict])
    broker.process_fills(portfolio, start_time)

    # Verify partial fill
    assert len(portfolio.closed_orders) == 1  # a partial fill order is not closed
    partial_filled_sell = portfolio.open_orders[1]
    assert partial_filled_sell.status == OrderStatus.PARTIALLY_FILLED
    assert partial_filled_sell.filled_price == 205.0  # Should fill at limit
    assert partial_filled_sell.filled_quantity == 20  # Limited by bar volume

    # Test 3: Buy limit order - should not fill when low price > limit_price
    no_fill_order_dict = {
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": OrderType.LIMIT,
        "side": OrderSide.BUY,
        "timestamp": start_time,
        "limit_price": 94.0,  # Below low price (95)
    }
    portfolio.add_orders([no_fill_order_dict])
    broker.process_fills(portfolio, start_time)

    # Verify no fill
    assert len(portfolio.open_orders) == 1
    assert portfolio.open_orders[0].status == OrderStatus.PENDING


def test_volume_aware_limit_fill_model_empty_bar(
    sample_universe_data, portfolio, start_time
):
    """Test VolumeAwareLimitFillModel's behavior with empty bar data."""
    broker = Broker(
        asset_universe=sample_universe_data,
        fill_model=VolumeAwareLimitFillModel(),
        margin_model=SimpleCashMargin(),
    )

    # Try to fill at a timestamp with no data
    future_time = start_time + timedelta(days=10)
    order_dict = {
        "symbol": "SMOL",
        "quantity": 100,
        "order_type": OrderType.LIMIT,
        "side": OrderSide.BUY,
        "timestamp": future_time,
        "limit_price": 100.0,
    }
    portfolio.add_orders([order_dict])
    broker.process_fills(portfolio, future_time)

    # Should not fill due to empty bar
    assert len(portfolio.open_orders) == 1
    assert portfolio.open_orders[0].status == OrderStatus.PENDING
