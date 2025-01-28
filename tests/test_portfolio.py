"""Unit tests for the Portfolio class."""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from backsim.portfolio import (
    Portfolio,
    Order,
    Position,
    OrderSide,
    OrderType,
    OrderStatus,
)
from backsim.universe import QuantityMatrix


@pytest.fixture
def symbols():
    return ["AAPL", "GOOGL"]


@pytest.fixture
def start_time():
    """Start time for the test period."""
    return datetime(2025, 1, 15, 10, 0, 0)


@pytest.fixture
def end_time(start_time):
    """End time for the test period."""
    return start_time + timedelta(days=5)


@pytest.fixture
def quantity_matrix(symbols, start_time):
    """Create quantity matrix with appropriate time range."""
    return QuantityMatrix(symbols, start_time=start_time)


@pytest.fixture
def portfolio(quantity_matrix):
    initial_cash = 10000.0
    return Portfolio(initial_cash, quantity_matrix)


def test_portfolio_initialization(portfolio):
    """Test that portfolio initializes with correct default values."""
    assert portfolio.cash == 10_000.0
    assert len(portfolio.positions) == 0
    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 0


@pytest.mark.parametrize(
    "order_details",
    [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "side": OrderSide.BUY,
            "order_type": "MARKET",
        },
        {
            "symbol": "GOOGL",
            "quantity": 50,
            "side": OrderSide.SELL,
            "limit_price": 150.0,
            "order_type": "LIMIT",
        },
    ],
)
def test_add_orders(portfolio, start_time, order_details):
    """Test adding different types of orders to portfolio."""
    order_details["timestamp"] = start_time
    portfolio.add_orders([order_details])

    assert len(portfolio.open_orders) == 1
    order = portfolio.open_orders[0]
    assert order.symbol == order_details["symbol"]
    assert order.quantity == order_details["quantity"]
    assert order.side == OrderSide(order_details["side"])
    assert order.order_type == OrderType(order_details["order_type"])


@pytest.mark.parametrize(
    "fill_price,expected_quantity,expected_cash",
    [
        (150.0, 10, 10000.0 - (150.0 * 10)),  # regular buy
        (200.0, 50, 10_000.0 - (200.0 * 50)),  # buy with all cash
    ],
)
def test_fill_buy_order(
    portfolio, start_time, fill_price, expected_quantity, expected_cash
):
    """Test filling buy orders with different prices and quantities; leverage = 1."""
    orders = [
        {
            "symbol": "AAPL",
            "quantity": expected_quantity,
            "side": "BUY",
            "timestamp": start_time,
            "order_type": "MARKET",
        }
    ]
    portfolio.add_orders(orders)
    order = portfolio.open_orders[0]

    # Add fill
    order.add_fill(quantity=expected_quantity, price=fill_price, timestamp=start_time)
    portfolio.fill_order(order)

    assert "AAPL" in portfolio.positions
    position = portfolio.positions["AAPL"]
    assert position.quantity == expected_quantity
    assert position.cost_basis == fill_price
    assert portfolio.cash == expected_cash
    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 1


def test_fill_sell_order_after_long(portfolio, start_time):
    """Test filling a sell order after establishing a long position."""
    # Create initial long position
    buy_price = 150.0
    buy_qty = 10
    portfolio.add_orders(
        [
            {
                "symbol": "AAPL",
                "quantity": buy_qty,
                "side": "BUY",
                "timestamp": start_time,
                "order_type": "MARKET",
            }
        ]
    )
    buy_order = portfolio.open_orders[0]
    buy_order.add_fill(quantity=buy_qty, price=buy_price, timestamp=start_time)
    portfolio.fill_order(buy_order)
    initial_cash = portfolio._cash  # cash, without factoring in margin

    # Sell half the position
    sell_price = 160.0
    sell_qty = 5
    portfolio.add_orders(
        [
            {
                "symbol": "AAPL",
                "quantity": sell_qty,
                "side": "SELL",
                "timestamp": start_time,
                "order_type": "MARKET",
            }
        ]
    )
    sell_order = portfolio.open_orders[0]
    sell_order.add_fill(quantity=sell_qty, price=sell_price, timestamp=start_time)
    portfolio.fill_order(sell_order)

    position = portfolio.positions["AAPL"]
    assert position.quantity == buy_qty - sell_qty

    realized_pnl = (sell_price - buy_price) * sell_qty
    expected_cash = initial_cash + realized_pnl

    expected_margin = (buy_qty * buy_price) / 2  # half sold -> half margin

    assert portfolio._cash == expected_cash
    assert portfolio.total_margin == expected_margin
    assert portfolio.cash == (expected_cash - expected_margin)


def test_portfolio_value_calculation(portfolio, start_time, symbols):
    """Test portfolio value calculation with multiple positions."""
    initial_cash = portfolio.cash
    # Create a position
    buy_price = 150.0
    buy_qty = 10
    portfolio.add_orders(
        [
            {
                "symbol": "AAPL",
                "quantity": buy_qty,
                "side": "BUY",
                "timestamp": start_time,
                "order_type": "MARKET",
            }
        ]
    )
    buy_order = portfolio.open_orders[0]
    buy_order.add_fill(quantity=buy_qty, price=buy_price, timestamp=start_time)
    portfolio.fill_order(buy_order)

    # Test with price increase
    current_prices = pd.Series([160.0, 0.0], index=symbols)
    position_value = buy_qty * 160.0
    expected_portfolio_value = initial_cash + position_value

    assert (
        portfolio.get_portfolio_value(start_time, current_prices)
        == expected_portfolio_value
    )


def test_negative_cash_handling(portfolio, start_time):
    """Test that portfolio raises error when cash becomes negative."""
    orders = [
        {
            "symbol": "AAPL",
            "quantity": 1000,
            "side": "BUY",
            "timestamp": start_time,
            "order_type": "MARKET",
        }
    ]
    portfolio.add_orders(orders)
    order = portfolio.open_orders[0]
    order.add_fill(quantity=1000, price=100.0, timestamp=start_time)

    with pytest.raises(ValueError, match="Insufficient cash to continue trading"):
        portfolio.fill_order(order)


@pytest.mark.parametrize(
    "initial_position,flip_order,expected_quantity",
    [
        # Long to Short
        (
            {"quantity": 10, "side": "BUY", "price": 100.0},
            {"quantity": 15, "side": "SELL", "price": 110.0},
            -5,
        ),
        # Short to Long
        (
            {"quantity": 10, "side": "SELL", "price": 100.0},
            {"quantity": 15, "side": "BUY", "price": 90.0},
            5,
        ),
    ],
)
def test_position_flipping(
    portfolio, start_time, initial_position, flip_order, expected_quantity
):
    """Test flipping positions from long to short and vice versa."""
    # Establish initial position
    portfolio.add_orders(
        [
            {
                "symbol": "AAPL",
                "quantity": initial_position["quantity"],
                "side": initial_position["side"],
                "timestamp": start_time,
                "order_type": "MARKET",
            }
        ]
    )
    order = portfolio.open_orders[0]
    order.add_fill(
        quantity=initial_position["quantity"],
        price=initial_position["price"],
        timestamp=start_time,
    )
    portfolio.fill_order(order)

    # Flip the position
    portfolio.add_orders(
        [
            {
                "symbol": "AAPL",
                "quantity": flip_order["quantity"],
                "side": flip_order["side"],
                "timestamp": start_time,
                "order_type": "MARKET",
            }
        ]
    )
    order = portfolio.open_orders[0]
    order.add_fill(
        quantity=flip_order["quantity"], price=flip_order["price"], timestamp=start_time
    )
    portfolio.fill_order(order)

    position = portfolio.positions["AAPL"]
    assert position.quantity == expected_quantity
    assert position.cost_basis == flip_order["price"]


def test_partial_fill_handling(portfolio, start_time):
    """Test handling of partial fills with proper fill tracking."""
    # Create an order for 100 shares
    order_details = {
        "symbol": "AAPL",
        "quantity": 10,
        "side": OrderSide.BUY,
        "timestamp": start_time,
        "order_type": OrderType.MARKET,
    }
    portfolio.add_orders([order_details])
    order = portfolio.open_orders[0]

    # First partial fill
    order.add_fill(quantity=4, price=150.0, timestamp=start_time)
    portfolio.fill_order(order)

    # Check position after first fill
    assert "AAPL" in portfolio.positions
    position = portfolio.positions["AAPL"]
    assert position.quantity == 4
    assert position.cost_basis == 150.0
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == 4
    assert order.filled_price == 150.0

    # Second partial fill at different price
    order.add_fill(quantity=6, price=155.0, timestamp=start_time)
    portfolio.fill_order(order)

    # Check position after second fill
    position = portfolio.positions["AAPL"]
    assert position.quantity == 10
    assert abs(position.cost_basis - 153.0) < 1e-10  # (40*150 + 60*155)/100 = 153
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 10
    assert abs(order.filled_price - 153.0) < 1e-10


def test_fill_tracking(portfolio, start_time):
    """Test that fills are properly tracked in the order."""
    order_details = {
        "symbol": "AAPL",
        "quantity": 50,
        "side": OrderSide.BUY,
        "timestamp": start_time,
        "order_type": OrderType.MARKET,
    }
    portfolio.add_orders([order_details])
    order = portfolio.open_orders[0]

    # Add multiple fills
    fills = [
        (20, 150.0, start_time),
        (20, 151.0, start_time + timedelta(minutes=1)),
        (10, 152.0, start_time + timedelta(minutes=2)),
    ]

    for qty, price, time in fills:
        order.add_fill(quantity=qty, price=price, timestamp=time)
        portfolio.fill_order(order)

    # Verify fills are tracked
    assert len(order.fills) == 3
    assert [f.quantity for f in order.fills] == [20, 20, 10]
    assert [f.price for f in order.fills] == [150.0, 151.0, 152.0]

    # Verify average price calculation
    expected_avg = (20 * 150.0 + 20 * 151.0 + 10 * 152.0) / 50
    assert abs(order.filled_price - expected_avg) < 1e-10


def test_position_update_from_fills(portfolio, start_time):
    """Test that position is updated correctly from fills."""
    # Create initial long position
    buy_order = {
        "symbol": "AAPL",
        "quantity": 10,
        "side": OrderSide.BUY,
        "timestamp": start_time,
        "order_type": OrderType.MARKET,
    }
    portfolio.add_orders([buy_order])
    order = portfolio.open_orders[0]

    # Fill in parts
    order.add_fill(quantity=6, price=150.0, timestamp=start_time)
    portfolio.fill_order(order)
    order.add_fill(quantity=4, price=155.0, timestamp=start_time)
    portfolio.fill_order(order)

    # Verify position
    position = portfolio.positions["AAPL"]
    assert position.quantity == 10
    expected_cost = (6 * 150.0 + 4 * 155.0) / 10
    assert abs(position.cost_basis - expected_cost) < 1e-10

    # now fully filled, so there should be no open orders
    assert len(portfolio.open_orders) == 0
    assert len(portfolio.closed_orders) == 1

    # Now sell partially
    sell_order = {
        "symbol": "AAPL",
        "quantity": 8,
        "side": OrderSide.SELL,
        "timestamp": start_time,
        "order_type": OrderType.MARKET,
    }
    portfolio.add_orders([sell_order])
    order = portfolio.open_orders[0]

    # Partial sells
    order.add_fill(quantity=5, price=160.0, timestamp=start_time)
    portfolio.fill_order(order)
    order.add_fill(quantity=3, price=162.0, timestamp=start_time)
    portfolio.fill_order(order)

    # Verify final position
    position = portfolio.positions["AAPL"]
    assert position.quantity == 2  # 10 - 8
    assert (
        position.cost_basis == expected_cost
    )  # Cost basis shouldn't change for remaining shares
