from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List


import logging

logger = logging.getLogger(__name__)

EPSILON = 1e-12


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # TODO: handle this case


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Fill:
    """Represents a single fill event for an order."""

    quantity: float
    price: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert Fill to a dictionary."""
        return {
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Fill":
        """Create a Fill from a dictionary."""
        return cls(
            quantity=data["quantity"],
            price=data["price"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    quantity: float
    side: OrderSide
    timestamp: datetime
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    expiry: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    leverage_ratio: float = 1.0
    filled_price: Optional[float] = None  # average fill price
    filled_quantity: float = 0.0
    fees_incurred: float = 0.0
    fills: List[Fill] = field(default_factory=list)

    def add_fill(self, quantity: float, price: float, timestamp: datetime):
        """Add a new fill and update average price."""
        self.fills.append(Fill(quantity, price, timestamp))
        self.filled_quantity += quantity

        # Update average price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.filled_price = (
            total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0
        )

        # update status if now fully filled
        if self.is_fully_filled:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity

    def is_expired(self, current_time: datetime) -> bool:
        """Check if order has expired."""
        return self.expiry is not None and current_time > self.expiry

    @property
    def margin(self):
        # Calculate margin for the order
        # use preferentially filled price
        if self.filled_price is not None:
            return abs(self.filled_quantity * self.filled_price) / self.leverage_ratio
        if self.limit_price is None:
            # neither has been provided yet; checking margin of an unfilled market order
            logger.warning(
                "Attempting to get margin of unfilled market order; Margin cannot be defined"
            )
            raise ValueError("Margin cannot be defined without a reference price.")
        return abs(self.quantity * self.limit_price) / self.leverage_ratio

    @property
    def is_fully_filled(self):
        return self.filled_quantity == self.quantity

    def __str__(self):
        return (
            f"Order(symbol={self.symbol}, quantity={self.quantity}, side={self.side}, timestamp={self.timestamp}, "
            f"order_type={self.order_type}, limit_price={self.limit_price}, expiry={self.expiry}, "
            f"status={self.status}, filled_price={self.filled_price}, filled_quantity={self.filled_quantity}, "
            f"fees_incurred={self.fees_incurred})"
        )

    # allow dictionary like access of Order
    def __getitem__(self, key: str):
        """Get attribute using dictionary-like access."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Set attribute using dictionary-like access."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if attribute exists using 'in' operator."""
        return hasattr(self, key)

    @classmethod
    def from_dict(cls, data: Dict) -> "Order":
        """Create an Order from a dictionary."""
        return cls(
            symbol=data["symbol"],
            quantity=data["quantity"],
            side=OrderSide(data["side"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            order_type=OrderType(data["order_type"]),
            limit_price=data["limit_price"],
            expiry=datetime.fromisoformat(data["expiry"]) if data["expiry"] else None,
            status=OrderStatus(data["status"]),
            leverage_ratio=data["leverage_ratio"],
            filled_price=data["filled_price"],
            filled_quantity=data["filled_quantity"],
            fees_incurred=data["fees_incurred"],
            fills=[Fill.from_dict(fill) for fill in data["fills"]],
        )

    def to_dict(self) -> Dict:
        """Convert Order to a dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side.value,
            "timestamp": self.timestamp.isoformat(),
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "status": self.status.value,
            "leverage_ratio": self.leverage_ratio,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "fees_incurred": self.fees_incurred,
            "fills": [fill.to_dict() for fill in self.fills],
        }


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    quantity: float
    cost_basis: float
    initial_margin: float = 0.0

    def update_from_fill(self, fill: Fill, order: Order) -> float:
        """Update position based on a filled order. Returns PnL if any from the occurence of the order"""
        if order.filled_price is None or order.filled_quantity is None:
            raise ValueError("Order must be filled before updating position")

        fill_quantity = fill.quantity * (
            1 if order.side in (OrderSide.BUY, "BUY") else -1
        )
        old_quantity = self.quantity
        new_quantity = self.quantity + fill_quantity

        realized_pnl = 0.0

        if self.is_zero:
            # opening new position
            self.cost_basis = order.filled_price
            self.initial_margin = order.margin
            self.quantity = new_quantity
            return realized_pnl

        if new_quantity == 0:
            # closing position
            realized_pnl = self.get_unrealized_pnl(fill.price)
            self.quantity = new_quantity
            return realized_pnl

        # Update cost basis and margin for 3 scenarios
        # 1. increasing position
        # 2. reducing position
        # 3. flipping position

        if (old_quantity > 0 and new_quantity < 0) or (
            old_quantity < 0 and new_quantity > 0
        ):
            # Flipping position
            realized_pnl = (fill.price - self.cost_basis) * old_quantity
            # Start the new position with remaining quantity
            self.quantity = new_quantity
            self.cost_basis = fill.price
            self.initial_margin = abs(new_quantity * fill.price) / order.leverage_ratio
            return realized_pnl

        if abs(new_quantity) > abs(old_quantity):
            # Increasing position
            self.cost_basis = (  # careful of negative qty
                self.cost_basis * abs(old_quantity) + fill.price * abs(fill.quantity)
            ) / abs(new_quantity)

            self.quantity = new_quantity
            # add margin of new fill
            self.initial_margin += (
                fill.price * abs(fill.quantity) / order.leverage_ratio
            )
            return realized_pnl

        if abs(new_quantity) < abs(old_quantity):
            # Reducing position
            closed_quantity = old_quantity - new_quantity
            realized_pnl = (fill.price - self.cost_basis) * closed_quantity
            new_margin = (
                abs(new_quantity * self.cost_basis) / self.leverage
            )  # proportinal reduction in margin requirement
            self.quantity = new_quantity
            self.initial_margin = new_margin
            return realized_pnl

    def update_from_order(self, order: Order) -> float:
        if not order.fills:
            return 0.0

        # Use the latest fill
        latest_fill = order.fills[-1]

        # calculate fill cash flow; if short, we want to credit cash
        sign = 1 if order.side in (OrderSide.SELL, "SELL") else -1
        fill_cash_flow = sign * latest_fill.quantity * latest_fill.price

        return self.update_from_fill(latest_fill, order), fill_cash_flow

    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L for the position.
        For longs: (current_price - cost_basis) * quantity.
        For shorts: (cost_basis - current_price) * abs(quantity).
        """
        if self.quantity > 0:
            return (current_price - self.cost_basis) * self.quantity
        elif self.quantity < 0:
            return (self.cost_basis - current_price) * abs(self.quantity)
        else:
            return 0.0

    # def margin_usage(self, current_price: float) -> float:
    #     """
    #     Compute the current margin usage. In this simple model,
    #     margin scales with the price relative to the cost basis.
    #     """
    #     return self.initial_margin * (current_price / self.cost_basis)

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.cost_basis

    @property
    def is_zero(self) -> bool:
        """Check if position is effectively zero."""
        return self.notional_value < EPSILON

    @property
    def side(self):
        return PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT

    @property
    def leverage(self):
        """Calculated effective leverage for the position.
        Uses the initial margin and cost basis
        """
        if self.is_zero:
            return 0.0  # avoid division by zero
        inital_position_value = abs(self.quantity * self.cost_basis)
        return inital_position_value / self.initial_margin

    def __repr__(self) -> str:
        return (
            f"Position(symbol={self.symbol}, quantity={self.quantity}, "
            f"cost_basis={self.cost_basis}, initial_margin={self.initial_margin}, side={self.side})"
        )
