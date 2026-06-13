"""Issue #15 (K5) — the live-data paper loop is paper-only and fills through C.

Stdlib-only (no pytest/numpy/ccxt): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_kraken_paper.py``. The live Kraken feed is
network tooling and is NOT exercised here; this pins the local fill-simulation +
guard wiring with synthetic snapshots.

Pins:
  (a) KrakenPaperTrader is paper-only and exposes NO live-writer surface;
  (b) buys/sells route through the one guarded broker, fills resolve through the
      ONE C engine (within the bar), and a paper ledger + PnL is persisted;
  (c) a death-spiral sell is refused exactly as the live path would refuse it.
"""
import os
import sys
import tempfile

# Isolate state + force paper BEFORE importing the spine (mirrors test_safety_spine).
os.environ["STATE_DIR"] = tempfile.mkdtemp(prefix="keel_paper_state_")
os.environ.pop("ALP_PAPER", None)
os.environ.pop("KEEL_PAPER", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config  # noqa: E402
from core.alpaca_singleton import forget_all_buys  # noqa: E402
from core.broker import PaperExecutor  # noqa: E402
from core.kraken_feed import Bar  # noqa: E402
from core.kraken_paper import KrakenPaperTrader, LiveWriteAttemptError  # noqa: E402


def _bar(price: float, ts: int = 0) -> Bar:
    return Bar(price, price * 1.01, price * 0.99, price, 1000.0, ts)


def _snap(symbol: str, price: float, ts: int = 0) -> dict:
    return {symbol: {"bar": _bar(price, ts), "price": price}}


def raises_runtime(fn) -> bool:
    try:
        fn()
        return False
    except RuntimeError:
        return True


def test_paper_only_no_live_surface():
    assert config.PAPER is True
    for forbidden in ("enforce_live_singleton", "acquire_alpaca_account_lock",
                      "acquire_live_lock", "go_live", "create_order"):
        assert not hasattr(KrakenPaperTrader, forbidden), forbidden
    # The default broker is a paper executor (no network, no real money).
    forget_all_buys()
    trader = KrakenPaperTrader(ledger_path=os.path.join(os.environ["STATE_DIR"], "l.jsonl"))
    assert isinstance(trader.broker.executor, PaperExecutor)
    print("ok test_paper_only_no_live_surface")


def test_buy_then_sell_fills_through_c_and_persists():
    forget_all_buys()
    scores = {"BTC/USD": 0.5}
    ledger = os.path.join(os.environ["STATE_DIR"], "ledger.jsonl")
    trader = KrakenPaperTrader(
        score_fn=lambda sym, bars: (scores.get(sym, -1.0), 0.02),
        ledger_path=ledger,
    )

    # Tick 1: conviction positive -> a buy, filled through the C engine.
    events = trader.step(_snap("BTC/USD", 100.0, ts=1))
    buys = [e for e in events if e.side == "buy"]
    assert len(buys) == 1 and buys[0].symbol == "BTC/USD"
    # C-engine fill sits within the bar's [low, high] = [99, 101].
    assert 99.0 <= buys[0].fill_price <= 101.0
    assert "BTC/USD" in trader._held
    # routed through the guarded broker's paper executor
    assert any(o.side == "buy" for o in trader.broker.executor.filled)
    # ledger persisted
    assert os.path.exists(ledger)
    with open(ledger) as fh:
        assert sum(1 for _ in fh) >= 1

    # Tick 2: drop conviction -> a guarded sell above the floor -> realized PnL.
    scores["BTC/USD"] = -1.0
    events2 = trader.step(_snap("BTC/USD", 99.0, ts=2))
    sells = [e for e in events2 if e.side == "sell"]
    assert len(sells) == 1
    assert "BTC/USD" not in trader._held
    import math
    assert math.isfinite(sells[0].realized_pnl)
    assert math.isfinite(trader.realized_pnl)
    print("ok test_buy_then_sell_fills_through_c_and_persists")


def test_death_spiral_sell_refused():
    forget_all_buys()
    scores = {"ETH/USD": 0.5}
    trader = KrakenPaperTrader(
        score_fn=lambda sym, bars: (scores.get(sym, -1.0), 0.02),
        ledger_path=os.path.join(os.environ["STATE_DIR"], "ds.jsonl"),
    )
    trader.step(_snap("ETH/USD", 100.0, ts=1))      # buy @ 100 (floor 97 @ 300bps)
    scores["ETH/USD"] = -1.0
    # Price 90 is below the death-spiral floor -> the guard must refuse the sell.
    assert raises_runtime(lambda: trader.step(_snap("ETH/USD", 90.0, ts=2)))
    forget_all_buys()
    print("ok test_death_spiral_sell_refused")


if __name__ == "__main__":
    test_paper_only_no_live_surface()
    test_buy_then_sell_fills_through_c_and_persists()
    test_death_spiral_sell_refused()
    print("all kraken-paper tests passed")
