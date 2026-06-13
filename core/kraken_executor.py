"""Kraken live execution adapter (K6, #16) — STAGED, FAIL-CLOSED, NOT ENABLED.

This is the real-money order seam. It is written for human review only; per the
issue it must NOT be enabled until K5 (live-data paper) is clean over a
meaningful window and this adapter has been HITL-reviewed.

What keeps it safe while it sits in the tree:

  * it is **not wired anywhere** — no live entry point imports it, and it is NOT
    added to ``LIVE_WRITER_UNITS`` in ``ops/deploy_live_trader.sh`` (that empty
    registry is what the deploy handshake checks). The default ``Broker`` still
    ships a ``PaperExecutor``;
  * **fail-closed construction**: building a ``KrakenExecutor`` requires
    ``allow_live=True`` AND the broker-neutral live-enable env gate
    (``config.live_trading_enabled()``). Otherwise it raises;
  * it implements ONLY the ``OrderExecutor.execute(order)`` seam, so every order
    it ever sees has already passed through ``Broker.submit_order`` ->
    ``guard_sell_against_death_spiral``. There is no un-guarded sell path here;
  * it does NOT acquire the single-writer lock itself — the reviewed live ENTRY
    POINT must call ``enforce_live_singleton`` (HARD RULE 2: exactly one live
    writer), then inject this executor into a gated ``Broker``;
  * ccxt + the private API keys are read **lazily at execute time** only, so
    merely importing or constructing this module touches no secret and no
    network. No keys belong on the box until this is reviewed.

Wiring checklist (do in the SAME reviewed commit that enables live — NOT here):
  1. confirm the champion cleared the crypto-retuned gate on unseen data;
  2. confirm a clean K5 paper run over a meaningful window;
  3. add the live unit to ``LIVE_WRITER_UNITS`` in ``ops/deploy_live_trader.sh``;
  4. the live entry point calls ``enforce_live_singleton(force_live=True)`` and
     injects ``KrakenExecutor(allow_live=True)`` into ``Broker(paper=False,
     allow_live=True, executor=...)``;
  5. set the live env gates only in the supervised unit, never ad hoc;
  6. ``ops/deploy_live_trader.sh --live`` reports OK before walking away.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from core import config
from core.broker import Order

logger = logging.getLogger("keel.kraken_executor")

# Private API key env vars (read lazily, only at execute time). Never committed.
KRAKEN_KEY_ENV = "KRAKEN_API_KEY"
KRAKEN_SECRET_ENV = "KRAKEN_API_SECRET"

# ccxt symbol/type for a market order on Kraken spot.
_ORDER_TYPE = "market"


class LiveExecutorForbiddenError(RuntimeError):
    """Raised when something tries to construct the live executor without the gate."""


@dataclass
class KrakenExecutor:
    """Fail-closed ccxt ``create_order`` adapter behind the ``OrderExecutor`` seam.

    Construct ONLY from a reviewed live entry point, with ``allow_live=True`` and
    the live-enable env gate set. It is then injected into a gated ``Broker`` so
    every order routes through the death-spiral guard before reaching ``execute``.
    """

    allow_live: bool = False
    # Reuse the single live-writer account/lock identity (the entry point holds
    # the lock; this executor never acquires it).
    account_name: str = "alpaca_live_writer"
    _client: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Fail closed: BOTH the explicit kwarg and the env gate are required.
        if not self.allow_live or not config.live_trading_enabled():
            raise LiveExecutorForbiddenError(
                "Refusing to construct the Kraken LIVE executor: requires "
                f"allow_live=True AND one of {config.LIVE_ENABLE_ENV_VARS}=1. "
                "K6 is HITL-deferred — do not enable until K5 is signed off."
            )
        logger.warning(
            "[kraken_executor] LIVE executor constructed (gated). It does NOT hold "
            "the writer lock; the entry point must enforce_live_singleton first."
        )

    def _ccxt_client(self):  # pragma: no cover - needs ccxt + private keys
        if self._client is not None:
            return self._client
        import ccxt

        key = os.environ.get(KRAKEN_KEY_ENV, "")
        secret = os.environ.get(KRAKEN_SECRET_ENV, "")
        if not key or not secret:
            raise LiveExecutorForbiddenError(
                f"missing {KRAKEN_KEY_ENV}/{KRAKEN_SECRET_ENV}; live keys are read "
                "only at execute time and only in the supervised unit."
            )
        self._client = ccxt.kraken({
            "apiKey": key, "secret": secret, "enableRateLimit": True,
        })
        return self._client

    def execute(self, order: Order) -> None:  # pragma: no cover - real money path
        """Place a real Kraken market order. Only reached AFTER the broker guard.

        ``order`` has already passed ``guard_sell_against_death_spiral`` inside
        ``Broker.submit_order`` (sell-only refusal raises before we get here), so
        this method does not re-implement the guard — it must remain the single
        guarded surface.
        """
        client = self._ccxt_client()
        logger.warning("[kraken_executor] LIVE %s %s qty=%.8f @ ~%.4f",
                       order.side.upper(), order.symbol, order.qty, order.price)
        client.create_order(
            symbol=order.symbol, type=_ORDER_TYPE, side=order.side, amount=order.qty,
        )


__all__ = ["KrakenExecutor", "LiveExecutorForbiddenError",
           "KRAKEN_KEY_ENV", "KRAKEN_SECRET_ENV"]
