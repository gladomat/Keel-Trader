"""Issue #2 — the out-of-sample gate (the single trusted judge).

Stdlib-only (no pytest/numpy): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_gate.py``.

Pins the three acceptance guards:
  (a) a flat policy earns ~0 return and is REJECTED (target 0.27 unmet);
  (b) the gate's fill path reproduces tests/test_fill_model.c on a 1-bar fixture
      (parity guard — the gate judges through the ONE C fill engine);
  (c) fail-fast bails on a deliberately bad policy in < N windows.

The gate needs a built libkeelsim.so and a sample .bin; both come from
``make build-sim`` + ``make data`` (the Makefile target runs build-sim first).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from research import policies as P
from research.eval import (
    PROMOTION_TARGET_MEDIAN_MONTHLY,
    evaluate,
    run_window,
)
from sim.keel_sim import MarketData, roundtrip_cost
from sim.make_sample_data import make_sample

# Golden cells from tests/test_fill_model.c (bar O=100 H=101 L=99 C=100, fee=10bps).
# (buffer_bps, slip_bps) -> (fill, entry, cost)
_GOLDEN_FILL = [
    (0.0, 0.0, 100.0000, 100.0000, 19.98),
    (0.0, 5.0, 100.0500, 100.0500, 29.96),
    (0.0, 20.0, 100.2000, 100.2000, 59.84),
    (5.0, 20.0, 100.2000, 100.2000, 59.84),
]


def _sample_md(tmp: Path, symbols=2, timesteps=600) -> MarketData:
    path = tmp / "gate_sample.bin"
    make_sample(path, num_symbols=symbols, num_timesteps=timesteps, seed=7)
    return MarketData.load(path)


def test_flat_policy_zero_return_and_rejected():
    with tempfile.TemporaryDirectory() as d:
        md = _sample_md(Path(d))
        try:
            # Individual windows: flat => no trades => ~0 return.
            for off in (0, 100, 200):
                wr = run_window(P.always_flat, md, off, window_steps=120, slippage_bps=10.0)
                assert abs(wr.total_return) < 1e-4, (off, wr.total_return)
                assert wr.num_trades == 0.0, (off, wr.num_trades)

            # Whole-gate verdict: a flat policy must be rejected (median << 0.27).
            verdict = evaluate(
                P.always_flat, md, n_windows=8, window_steps=120, seed=1,
                fail_fast=False,  # disable fail-fast so we judge on the median itself
            )
            assert not verdict.promote
            assert verdict.worst_cell_median_monthly < PROMOTION_TARGET_MEDIAN_MONTHLY
        finally:
            md.free()
    print("ok test_flat_policy_zero_return_and_rejected")


def test_fill_parity_guard():
    """The gate's C fill path must equal the golden fixture, cell-for-cell."""
    for buf, slip, exp_fill, exp_entry, exp_cost in _GOLDEN_FILL:
        fill, entry, cost = roundtrip_cost(100.0, 101.0, 99.0, 100.0, buf, slip)
        assert abs(fill - exp_fill) < 1e-3, (buf, slip, fill, exp_fill)
        assert abs(entry - exp_entry) < 1e-3, (buf, slip, entry, exp_entry)
        assert abs(cost - exp_cost) < 0.05, (buf, slip, cost, exp_cost)
    print("ok test_fill_parity_guard")


def test_fail_fast_triggers_early():
    """A hopeless (flat) policy makes the median target unreachable -> bail < N."""
    with tempfile.TemporaryDirectory() as d:
        md = _sample_md(Path(d))
        try:
            n = 8
            verdict = evaluate(
                P.always_flat, md, n_windows=n, window_steps=120, seed=2,
                fail_fast=True,
            )
            assert not verdict.promote
            assert verdict.failed_fast
            # The first (and only-run) cell bailed before exhausting all N windows.
            first_cell = verdict.cells[0]
            assert first_cell.failed_fast
            assert first_cell.windows_run < n, first_cell.windows_run
        finally:
            md.free()
    print("ok test_fail_fast_triggers_early")


if __name__ == "__main__":
    test_flat_policy_zero_return_and_rejected()
    test_fill_parity_guard()
    test_fail_fast_triggers_early()
    print("all gate tests passed")
