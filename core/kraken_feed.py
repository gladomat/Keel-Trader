"""Real-time Kraken price feed — PUBLIC endpoints only (K5, #15).

The forward-test data source: pulls live hourly bars + last price from Kraken's
**public** market-data endpoints via ccxt. It is structurally incapable of
trading:

  * it only ever calls ``fetch_ohlcv`` / ``fetch_ticker`` — public, no-auth
    market-data methods. No ``create_order``, no private/keyed endpoint, no API
    key is read or accepted (the K1-K5 safety boundary: no live keys on the box
    until K6 is reviewed);
  * it is **offline tooling** (needs network + ccxt, not installed in the test
    env) and is never imported by ``make test``. Verify edits with
    ``python3 -m py_compile``.

The paper loop (``core/kraken_paper.py``) consumes ``snapshot()`` and simulates
every fill locally through the ONE C engine — zero orders reach Kraken.
"""
from __future__ import annotations

from dataclasses import dataclass

# The 5 locked USD majors + cadence (stdlib-only module — no numpy in the import
# chain, so this stays importable under make test).
from sim.universe import KRAKEN_USD_MAJORS, TIMEFRAME


@dataclass(frozen=True)
class Bar:
    """One hourly OHLCV bar (the unit the C fill engine resolves a fill against)."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_ms: int


class KrakenPublicFeed:
    """Streams live Kraken prices over public market-data endpoints only.

    ``snapshot()`` returns ``{symbol: {"bar": Bar, "price": float}}`` where
    ``bar`` is the most recent CLOSED hourly bar (used for the local C-engine
    fill simulation) and ``price`` is the live last-trade price (the decision
    price). No authentication, ever.
    """

    def __init__(self, symbols=None):
        self.symbols = list(symbols) if symbols else list(KRAKEN_USD_MAJORS)
        self._ex = None

    def _exchange(self):  # pragma: no cover - needs ccxt + network
        if self._ex is None:
            import ccxt

            # Public client: no apiKey/secret. enableRateLimit so we behave.
            self._ex = ccxt.kraken({"enableRateLimit": True})
        return self._ex

    def snapshot(self) -> dict:  # pragma: no cover - needs ccxt + network
        """One pull of {symbol: {bar, price}} from public endpoints."""
        ex = self._exchange()
        out: dict[str, dict] = {}
        for sym in self.symbols:
            ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=2)  # public
            if not ohlcv:
                continue
            # Prefer the last fully CLOSED bar (index -2) when available.
            row = ohlcv[-2] if len(ohlcv) >= 2 else ohlcv[-1]
            ts, o, h, l, c, v = row
            bar = Bar(float(o), float(h), float(l), float(c), float(v), int(ts))
            ticker = ex.fetch_ticker(sym)  # public
            price = ticker.get("last") or ticker.get("close") or c
            out[sym] = {"bar": bar, "price": float(price)}
        return out
