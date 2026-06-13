"""The single guarded order-write surface (issue #9).

Every order — buy AND sell — must pass through ``submit_order`` here, and
``submit_order`` routes through ``guard_sell_against_death_spiral`` BEFORE the
order ever reaches an executor. There is deliberately:

  * exactly ONE public write method (``submit_order``). No ``sell`` /
    ``liquidate`` / ``close_position`` convenience that could bypass the guard;
  * NO live executor in this module. The default executor is a paper recorder
    that touches no network and no real money;
  * a fail-closed construction gate: building a non-paper broker requires an
    explicit ``allow_live=True`` AND ``ALLOW_ALPACA_LIVE_TRADING=1`` in the env,
    and even then it does NOT acquire the live-writer lock here (HARD RULE 2:
    porting a process that can win the live lock is a separate, reviewed step).

The moray audit confirmed both live writers funnelled through the death-spiral
guard; this replicates that boundary so no un-guarded sell path is reachable.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from core import config
from core.alpaca_singleton import (
    guard_sell_against_death_spiral,
    record_buy_price,
)

logger = logging.getLogger("keel.broker")

# Broker-neutral live-enable gate (K3 #13). The check accepts any of
# ``config.LIVE_ENABLE_ENV_VARS`` (KEEL_* primary, legacy ALPACA name aliased).
# ``ALLOW_LIVE_ENV_VAR`` is kept as the canonical single name for callers/tests
# that set one variable; it is the legacy name so existing units keep working.
ALLOW_LIVE_ENV_VAR = "ALLOW_ALPACA_LIVE_TRADING"

_BUY = "buy"
_SELL = "sell"
_VALID_SIDES = (_BUY, _SELL)


class LiveBrokerForbiddenError(RuntimeError):
    """Raised when something tries to construct a live broker without the gate."""


class OrderRejectedError(RuntimeError):
    """Raised when an order is malformed (bad side / qty / price)."""


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str        # "buy" or "sell"
    qty: float
    price: float


class OrderExecutor(Protocol):
    """The narrow seam an executor must satisfy. Receives only guarded orders."""

    def execute(self, order: Order) -> None: ...


@dataclass
class PaperExecutor:
    """Default executor: records orders in-memory. No network, no real money."""

    filled: List[Order] = field(default_factory=list)

    def execute(self, order: Order) -> None:
        self.filled.append(order)
        logger.info("[paper-exec] %s %s qty=%.6f @ %.4f",
                    order.side.upper(), order.symbol, order.qty, order.price)


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Broker:
    """The one order write surface. Construct, then call ``submit_order``.

    ``paper`` defaults to ``core.config.PAPER``. A live broker is only buildable
    with ``allow_live=True`` AND ``ALLOW_ALPACA_LIVE_TRADING=1`` set; otherwise
    construction raises. No live executor ships with this module, so a live
    broker has nothing to execute against by default — the boundary exists, the
    cutover does not.
    """

    executor: OrderExecutor = field(default_factory=PaperExecutor)
    paper: bool = field(default=None)  # type: ignore[assignment]
    allow_live: bool = False

    def __post_init__(self) -> None:
        if self.paper is None:
            self.paper = config.PAPER
        if not self.paper:
            # Fail closed: a live broker needs BOTH the explicit kwarg and the
            # env gate (any accepted broker-neutral name). This module never wins
            # the live-writer lock itself.
            if not self.allow_live or not config.live_trading_enabled():
                raise LiveBrokerForbiddenError(
                    "Refusing to construct a LIVE broker: requires allow_live=True "
                    f"AND one of {config.LIVE_ENABLE_ENV_VARS}=1. Live cutover is a "
                    "separate, reviewed step (HARD RULE 2: exactly one live writer)."
                )
            logger.warning("[broker] LIVE broker constructed (gated) — no live "
                           "executor ships here; orders go to the injected executor.")

    def submit_order(self, symbol: str, side: str, qty: float, price: float) -> Order:
        """The ONLY way to write an order. Guards first, then executes.

        Sells are checked against the death-spiral guard (a refusal raises
        RuntimeError and propagates). Buys record their price so a later sell
        has a reference floor. There is no path to an executor that skips this.
        """
        side_norm = str(side).strip().lower()
        if side_norm not in _VALID_SIDES:
            raise OrderRejectedError(f"invalid side {side!r}; expected buy/sell")
        if not symbol:
            raise OrderRejectedError("missing symbol")
        if not (qty > 0):
            raise OrderRejectedError(f"qty must be > 0, got {qty!r}")
        if not (price > 0):
            raise OrderRejectedError(f"price must be > 0, got {price!r}")

        # GUARD FIRST — before any executor sees the order. Sell-only inside;
        # a refused death-spiral sell raises RuntimeError and stops here.
        guard_sell_against_death_spiral(symbol, side_norm, float(price))

        order = Order(symbol=symbol, side=side_norm, qty=float(qty), price=float(price))
        self.executor.execute(order)

        if side_norm == _BUY:
            # Record only after a successful execute so the guard floor reflects
            # positions we actually hold.
            record_buy_price(symbol, float(price))

        return order


__all__ = [
    "Broker",
    "Order",
    "OrderExecutor",
    "PaperExecutor",
    "LiveBrokerForbiddenError",
    "OrderRejectedError",
    "ALLOW_LIVE_ENV_VAR",
]
