"""Issue #16 (K6) — the live Kraken executor is fail-closed and NOT enabled.

Stdlib-only (no pytest/ccxt). This pins the safety boundary of the STAGED live
adapter without ever going near a real order: construction is refused unless
explicitly gated, there is no un-guarded sell path, and the live registry stays
empty. Run via ``make test`` / ``PYTHONPATH=. python3 tests/test_kraken_executor.py``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kraken_executor import (  # noqa: E402
    KrakenExecutor,
    LiveExecutorForbiddenError,
)


def raises(fn, exc) -> bool:
    try:
        fn()
        return False
    except exc:
        return True


def test_construction_is_fail_closed():
    # No gate at all -> refused.
    assert raises(lambda: KrakenExecutor(), LiveExecutorForbiddenError)
    # allow_live alone is not enough — the env gate must also be set.
    assert raises(lambda: KrakenExecutor(allow_live=True), LiveExecutorForbiddenError)
    # env gate alone (without allow_live) is not enough either.
    os.environ["KEEL_ALLOW_LIVE_TRADING"] = "1"
    try:
        assert raises(lambda: KrakenExecutor(allow_live=False), LiveExecutorForbiddenError)
        # Both gates -> constructs (gated). It still holds no lock, ships no keys.
        ex = KrakenExecutor(allow_live=True)
        assert ex.allow_live is True
        # No un-guarded sell/liquidate surface — only the narrow execute() seam.
        for unsafe in ("sell", "liquidate", "close_position", "create_order",
                       "enforce_live_singleton", "acquire_live_lock"):
            assert not hasattr(ex, unsafe), unsafe
        assert hasattr(ex, "execute")
    finally:
        os.environ.pop("KEEL_ALLOW_LIVE_TRADING", None)
    print("ok test_construction_is_fail_closed")


def test_not_registered_in_live_units():
    """The deploy registry must stay EMPTY — K6 is not wired/enabled."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "ops", "deploy_live_trader.sh")) as fh:
        script = fh.read()
    # The registry block exists but carries no active (uncommented) unit entry.
    start = script.index("LIVE_WRITER_UNITS=(")
    end = script.index(")", start)
    block = script[start:end]
    for line in block.splitlines()[1:]:
        stripped = line.strip()
        assert not stripped or stripped.startswith("#"), \
            f"a live unit appears registered — K6 must stay disabled: {stripped!r}"
    print("ok test_not_registered_in_live_units")


if __name__ == "__main__":
    test_construction_is_fail_closed()
    test_not_registered_in_live_units()
    print("all kraken-executor tests passed")
