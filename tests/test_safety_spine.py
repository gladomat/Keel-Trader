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


def raises_runtime(fn):
    try:
        fn()
        return False
    except RuntimeError:
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
    # Default intraday tol = 50 bps => floor 99.50.
    assert DEFAULT_DEATH_SPIRAL_TOLERANCE_BPS == 50.0
    guard_sell_against_death_spiral("AAPL", "sell", 99.90)  # above floor => ok
    assert raises_runtime(lambda: guard_sell_against_death_spiral("AAPL", "sell", 99.00)), \
        "sell 50bps+ below buy must be refused"

    # Buys/holds are never guarded (guard is sell-only by design).
    guard_sell_against_death_spiral("AAPL", "buy", 1.0)

    # Explicit per-call tolerance overrides regime selection.
    assert raises_runtime(lambda: guard_sell_against_death_spiral("AAPL", "sell", 99.85, tolerance_bps=10.0))
    guard_sell_against_death_spiral("AAPL", "sell", 99.95, tolerance_bps=10.0)  # within 10bps => ok
    print("  ok  death-spiral guard: intraday floor refuses, sell-only, explicit tol")


def test_time_aware_regime():
    forget_all_buys()
    record_buy_price("MSFT", 100.0)
    # Force the overnight regime (stale_after_seconds=0 => any buy is "stale").
    # Overnight tol 500 bps => floor 95.00, so a 96.00 sell that the tight 50bps
    # intraday rule would REFUSE is allowed under the overnight regime.
    guard_sell_against_death_spiral(
        "MSFT", "sell", 96.0, stale_after_seconds=0, stale_tolerance_bps=500.0
    )
    # Still refuses below the wide floor.
    assert raises_runtime(lambda: guard_sell_against_death_spiral(
        "MSFT", "sell", 94.0, stale_after_seconds=0, stale_tolerance_bps=500.0
    ))
    print("  ok  time-aware regime: overnight tolerance widens the floor")


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


if __name__ == "__main__":
    print("keel safety-spine golden tests:")
    test_paper_bypass()
    test_death_spiral_guard()
    test_time_aware_regime()
    test_override_bypasses_guard()
    test_inprocess_lock()
    print("ALL SAFETY-SPINE INVARIANTS HOLD.")
