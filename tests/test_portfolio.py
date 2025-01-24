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
            "filled_price": fill_price,
            "filled_quantity": expected_quantity,
        }
    ]
    portfolio.add_orders(orders)
    order = portfolio.open_orders[0]

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
                "filled_price": buy_price,
                "filled_quantity": buy_qty,
                "status": OrderStatus.FILLED,
            }
        ]
    )
    portfolio.fill_order(portfolio.open_orders[0])
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
                "filled_price": sell_price,
                "filled_quantity": sell_qty,
                "status": OrderStatus.FILLED,
            }
        ]
    )
    portfolio.fill_order(portfolio.open_orders[0])

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
                "filled_price": buy_price,
                "filled_quantity": buy_qty,
            }
        ]
    )
    portfolio.fill_order(portfolio.open_orders[0])

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
            "filled_price": 100.0,  # High enough to make cash negative
            "filled_quantity": 1000,
        }
    ]
    portfolio.add_orders(orders)

    with pytest.raises(ValueError, match="Insufficient cash to continue trading"):
        portfolio.fill_order(portfolio.open_orders[0])


@pytest.mark.parametrize(
    "initial_position,flip_order,expected_quantity",
    [
        # Long to Short
        ((100, 15.0, "BUY"), (200, 16.0, "SELL"), -100),
        # Short to Long
        ((100, 15.0, "SELL"), (200, 14.0, "BUY"), 100),
    ],
)
def test_position_flipping(
    portfolio, start_time, initial_position, flip_order, expected_quantity
):
    """Test flipping positions from long to short and vice versa."""
    init_qty, init_price, init_side = initial_position
    flip_qty, flip_price, flip_side = flip_order

    # Create initial position
    portfolio.add_orders(
        [
            {
                "symbol": "AAPL",
                "quantity": init_qty,
                "side": init_side,
                "timestamp": start_time,
                "order_type": "MARKET",
                "filled_price": init_price,
                "filled_quantity": init_qty,
            }
        ]
    )
    portfolio.fill_order(portfolio.open_orders[0])

    position = portfolio.positions["AAPL"]

    assert position.quantity == init_qty * (1 if init_side == "BUY" else -1)
    assert position.cost_basis == init_price

    # Flip position
    portfolio.add_orders(
        [
            {
                "symbol": "AAPL",
                "quantity": flip_qty,
                "side": flip_side,
                "timestamp": start_time,
                "order_type": "MARKET",
                "filled_price": flip_price,
                "filled_quantity": flip_qty,
            }
        ]
    )
    portfolio.fill_order(portfolio.open_orders[0])

    position = portfolio.positions["AAPL"]
    assert position.quantity == expected_quantity
    assert position.cost_basis == flip_price
