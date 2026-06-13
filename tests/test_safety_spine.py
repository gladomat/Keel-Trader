"""Golden tests for the keel safety spine (core/alpaca_singleton + alpaca_account_lock).

Zero-dependency (plain asserts, no pytest) so `make test` stays toolchain-light.
Pins the three fail-closed gates and the time-aware death-spiral guard. Mirrors the
moray guard contract — change these and the guard in the same commit.

Run:  STATE_DIR=$(mktemp -d) python3 tests/test_safety_spine.py
"""
import os
import sys
import tempfile

# Isolate all state into a throwaway dir BEFORE importing the spine, and ensure
# paper mode (no ALP_PAPER => PAPER=True default in core.config).
os.environ["STATE_DIR"] = tempfile.mkdtemp(prefix="keel_test_state_")
os.environ.pop("ALP_PAPER", None)
os.environ.pop("ALPACA_DEATH_SPIRAL_OVERRIDE", None)
os.environ.pop("ALPACA_SINGLETON_OVERRIDE", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alpaca_singleton import (  # noqa: E402
    enforce_live_singleton,
    guard_sell_against_death_spiral,
    record_buy_price,
    forget_all_buys,
    DEFAULT_DEATH_SPIRAL_TOLERANCE_BPS,
)
from core.alpaca_account_lock import acquire_alpaca_account_lock  # noqa: E402
from core.broker import (  # noqa: E402
    Broker,
    PaperExecutor,
    LiveBrokerForbiddenError,
    OrderRejectedError,
    ALLOW_LIVE_ENV_VAR,
)
from core.trading_server import TradingServer, OrderRequestError  # noqa: E402


def raises_runtime(fn):
    try:
        fn()
        return False
    except RuntimeError:
        return True


def raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True


def test_paper_bypass():
    # Paper mode: no singleton lock required, returns None.
    lock = enforce_live_singleton(service_name="keel_test", force_live=False)
    assert lock is None, "paper mode must not acquire a live lock"
    print("  ok  paper bypass: no live lock in paper mode")


def test_death_spiral_guard():
    forget_all_buys()
    # No record on file => nothing to compare => allowed (guards round-trips only).
    guard_sell_against_death_spiral("AAPL", "sell", 50.0)

    record_buy_price("AAPL", 100.0)
    # K3: single crypto tolerance = 300 bps => floor 97.00 (no equity regime).
    assert DEFAULT_DEATH_SPIRAL_TOLERANCE_BPS == 300.0
    guard_sell_against_death_spiral("AAPL", "sell", 97.50)  # above floor => ok
    assert raises_runtime(lambda: guard_sell_against_death_spiral("AAPL", "sell", 96.50)), \
        "sell 300bps+ below buy must be refused"

    # Buys/holds are never guarded (guard is sell-only by design).
    guard_sell_against_death_spiral("AAPL", "buy", 1.0)

    # Explicit per-call tolerance still overrides the base.
    assert raises_runtime(lambda: guard_sell_against_death_spiral("AAPL", "sell", 99.85, tolerance_bps=10.0))
    guard_sell_against_death_spiral("AAPL", "sell", 99.95, tolerance_bps=10.0)  # within 10bps => ok
    print("  ok  death-spiral guard: 300bps crypto floor refuses, sell-only, explicit tol")


def test_volatility_aware_tolerance():
    forget_all_buys()
    record_buy_price("SOL", 100.0)
    # Base floor (300 bps) would REFUSE a sell at 96.0. But a volatile name with
    # atr_pct=0.03 (3%) and the default 3x multiplier widens the floor to
    # 9% below buy (91.00), so 96.0 is now allowed — vol-aware, never tightening.
    guard_sell_against_death_spiral("SOL", "sell", 96.0, atr_pct=0.03)
    # Still refuses below the widened floor.
    assert raises_runtime(lambda: guard_sell_against_death_spiral(
        "SOL", "sell", 90.0, atr_pct=0.03))
    # A tiny ATR can never tighten below the base 300 bps floor (97.00).
    assert raises_runtime(lambda: guard_sell_against_death_spiral(
        "SOL", "sell", 96.5, atr_pct=0.0001))
    # Legacy equity-regime kwargs are accepted and ignored (no crash).
    guard_sell_against_death_spiral("SOL", "sell", 97.5,
                                    stale_after_seconds=0, stale_tolerance_bps=500.0)
    print("  ok  volatility-aware tolerance: ATR widens the floor, never tightens")


def test_override_bypasses_guard():
    forget_all_buys()
    record_buy_price("TSLA", 100.0)
    os.environ["ALPACA_DEATH_SPIRAL_OVERRIDE"] = "1"
    try:
        guard_sell_against_death_spiral("TSLA", "sell", 1.0)  # absurd, but override allows it
    finally:
        os.environ.pop("ALPACA_DEATH_SPIRAL_OVERRIDE", None)
    print("  ok  break-glass override bypasses the guard (loudly)")


def test_inprocess_lock():
    acct = "keel_test_writer"
    lock = acquire_alpaca_account_lock(service_name="svc_a", account_name=acct)
    try:
        # Same service in-process is idempotent (returns the same handle).
        same = acquire_alpaca_account_lock(service_name="svc_a", account_name=acct)
        assert same is lock, "same service must reuse the in-process lock"
        # A DIFFERENT service can't steal the lock while it's held.
        assert raises_runtime(lambda: acquire_alpaca_account_lock(service_name="svc_b", account_name=acct)), \
            "second live writer must be refused"
    finally:
        lock.release()
    print("  ok  single-writer lock: idempotent same-service, refuses second writer")


def test_broker_is_paper_by_default():
    # In paper mode (no ALP_PAPER) the broker constructs freely and is paper.
    broker = Broker()
    assert broker.paper is True, "broker must default to paper"
    # The one write surface exists; there is NO un-guarded sell helper.
    assert hasattr(broker, "submit_order")
    for unsafe in ("sell", "liquidate", "close_position", "force_sell"):
        assert not hasattr(broker, unsafe), \
            f"broker must not expose an un-guarded {unsafe!r} helper"
    print("  ok  broker: paper by default, single submit_order surface")


def test_broker_buy_records_and_sell_is_guarded():
    forget_all_buys()
    exec_ = PaperExecutor()
    broker = Broker(executor=exec_)

    # A buy executes and records its price for the guard.
    broker.submit_order("NVDA", "buy", qty=10, price=100.0)
    assert len(exec_.filled) == 1 and exec_.filled[0].side == "buy"

    # A sell well below the buy floor (300bps => 97.00) is REFUSED — and the
    # executor never sees it (guard runs before execute).
    assert raises_runtime(lambda: broker.submit_order("NVDA", "sell", qty=10, price=96.0))
    assert len(exec_.filled) == 1, "refused sell must not reach the executor"

    # A sell above the floor passes through the guard and executes.
    broker.submit_order("NVDA", "sell", qty=10, price=99.90)
    assert len(exec_.filled) == 2 and exec_.filled[1].side == "sell"
    print("  ok  broker: buy records price, sell routes through death-spiral guard")


def test_broker_rejects_malformed_orders():
    broker = Broker()
    assert raises(lambda: broker.submit_order("AAPL", "hold", 1, 1.0), OrderRejectedError)
    assert raises(lambda: broker.submit_order("AAPL", "buy", 0, 1.0), OrderRejectedError)
    assert raises(lambda: broker.submit_order("AAPL", "buy", 1, 0.0), OrderRejectedError)
    assert raises(lambda: broker.submit_order("", "buy", 1, 1.0), OrderRejectedError)
    print("  ok  broker: rejects malformed side/qty/price/symbol")


def test_live_broker_construction_is_forbidden():
    # Without the explicit gate, a non-paper broker cannot be built.
    assert raises(lambda: Broker(paper=False), LiveBrokerForbiddenError)
    # allow_live alone is not enough — the env gate must also be set.
    assert raises(lambda: Broker(paper=False, allow_live=True),
                  LiveBrokerForbiddenError)
    # Even with both, this module ships no live executor (boundary, not cutover).
    os.environ[ALLOW_LIVE_ENV_VAR] = "1"
    try:
        gated = Broker(paper=False, allow_live=True, executor=PaperExecutor())
        assert gated.paper is False
    finally:
        os.environ.pop(ALLOW_LIVE_ENV_VAR, None)

    # K3 broker-neutral alias: the KEEL_* name gates construction just the same.
    os.environ["KEEL_ALLOW_LIVE_TRADING"] = "1"
    try:
        gated2 = Broker(paper=False, allow_live=True, executor=PaperExecutor())
        assert gated2.paper is False
    finally:
        os.environ.pop("KEEL_ALLOW_LIVE_TRADING", None)
    print("  ok  live broker: fail-closed unless explicitly gated (no cutover here)")


def test_trading_server_routes_through_broker():
    forget_all_buys()
    exec_ = PaperExecutor()
    server = TradingServer(Broker(executor=exec_))

    ok = server.handle_order({"symbol": "TSLA", "side": "buy", "qty": 5, "price": 100.0})
    assert ok["status"] == "accepted" and len(exec_.filled) == 1

    # The server has no order path of its own — a death-spiral sell still raises.
    assert raises_runtime(lambda: server.handle_order(
        {"symbol": "TSLA", "side": "sell", "qty": 5, "price": 90.0}))
    assert len(exec_.filled) == 1, "guard refusal must stop before the executor"

    # Malformed payloads are rejected by the server's own validation.
    assert raises(lambda: server.handle_order({"symbol": "TSLA"}), OrderRequestError)
    # No un-guarded sell helper on the server either.
    for unsafe in ("sell", "liquidate", "close_position"):
        assert not hasattr(server, unsafe)
    print("  ok  trading_server: delegates to the one guarded broker write path")


if __name__ == "__main__":
    print("keel safety-spine golden tests:")
    test_paper_bypass()
    test_death_spiral_guard()
    test_volatility_aware_tolerance()
    test_override_bypasses_guard()
    test_inprocess_lock()
    test_broker_is_paper_by_default()
    test_broker_buy_records_and_sell_is_guarded()
    test_broker_rejects_malformed_orders()
    test_live_broker_construction_is_forbidden()
    test_trading_server_routes_through_broker()
    print("ALL SAFETY-SPINE INVARIANTS HOLD.")
