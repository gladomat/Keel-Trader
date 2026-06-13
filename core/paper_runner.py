"""Paper-only decision loop — deliberately NOT a live writer.

This wires the Phase-5 strategy to the safety spine in a way that can run a paper
decision cycle while being structurally incapable of touching real money:

  * it asserts ``core.config.PAPER is True`` at construction and refuses to run
    otherwise;
  * it NEVER calls ``enforce_live_singleton`` / ``acquire_alpaca_account_lock`` —
    it cannot win the single-writer lock (HARD RULE 2: exactly one live writer,
    and this is not it);
  * it still imports and exercises ``record_buy_price`` +
    ``guard_sell_against_death_spiral`` so the paper path rehearses the exact
    guards the (future, separately-reviewed) live path will use.

Orders are routed through an injected ``submit`` callable; the default is a
no-op/log sink. There is intentionally no real Alpaca client here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from core import config
from core.alpaca_singleton import (
    guard_sell_against_death_spiral,
    record_buy_price,
)
from models.xgb.strategy import Position, StrategyConfig, SymbolSignal, decide

logger = logging.getLogger("keel.paper_runner")


class LiveWriteAttemptError(RuntimeError):
    """Raised if anything tries to make this paper runner act as a live writer."""


@dataclass
class PaperOrder:
    symbol: str
    side: str          # "buy" or "sell"
    price: float
    target_weight: float


def _log_sink(order: PaperOrder) -> None:
    logger.info("[paper] %s %s @ %.4f (target_weight=%.4f)",
                order.side.upper(), order.symbol, order.price, order.target_weight)


@dataclass
class PaperRunner:
    """A paper-mode strategy loop. Construct, then ``step(signals, prices)``."""

    strategy_cfg: StrategyConfig = field(default_factory=StrategyConfig)
    submit: Callable[[PaperOrder], None] = _log_sink
    _held: dict = field(default_factory=dict)  # symbol -> last target weight

    def __post_init__(self):
        # Fail closed: a paper runner must never come up in live mode.
        if not config.PAPER:
            raise LiveWriteAttemptError(
                "PaperRunner refuses to run with PAPER=False; it is paper-only and "
                "must never become a live writer (set ALP_PAPER=1 or unset it)."
            )

    # No acquire_live_lock / enforce_live_singleton method exists here BY DESIGN.

    def step(self, signals: List[SymbolSignal], prices: dict) -> List[PaperOrder]:
        """Run one decision cycle: strategy -> guarded paper orders.

        ``prices`` maps symbol -> current price. Buys record the price (so the
        death-spiral guard has a reference); sells are checked against the guard
        exactly as the live path would, and a refusal propagates (RuntimeError).
        """
        targets = {p.symbol: p.weight for p in decide(signals, self.strategy_cfg)}
        orders: List[PaperOrder] = []

        # Exit names we no longer want (guarded sells).
        for sym in list(self._held):
            if sym not in targets:
                price = prices.get(sym)
                if price is None:
                    continue
                # Same guard the live path uses; refusal raises and stops the loop.
                guard_sell_against_death_spiral(sym, "sell", float(price))
                order = PaperOrder(sym, "sell", float(price), 0.0)
                self.submit(order)
                orders.append(order)
                del self._held[sym]

        # Enter / adjust desired names (record buys for the guard).
        for sym, weight in targets.items():
            price = prices.get(sym)
            if price is None:
                continue
            if sym not in self._held:
                record_buy_price(sym, float(price))
                order = PaperOrder(sym, "buy", float(price), weight)
                self.submit(order)
                orders.append(order)
            self._held[sym] = weight

        return orders


def main():  # pragma: no cover - tiny demo harness, not a live entry point
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Run a paper decision cycle (demo)")
    ap.add_argument("--symbol", default="DEMO")
    ap.add_argument("--score", type=float, default=0.2)
    ap.add_argument("--vol", type=float, default=0.02)
    ap.add_argument("--price", type=float, default=100.0)
    args = ap.parse_args()

    runner = PaperRunner()
    runner.step(
        [SymbolSignal(args.symbol, args.score, args.vol)],
        {args.symbol: args.price},
    )


if __name__ == "__main__":
    main()
