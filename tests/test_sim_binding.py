"""Smoke + parity test for the ctypes sim binding (issue #1).

Zero-dependency (plain asserts, no pytest) so `make test` stays toolchain-light.

Proves:
  (a) the binding loads a .bin and runs a vectorized step (sane obs/reward/terminal);
  (b) the binding's fill path reproduces tests/test_fill_model.c golden values —
      i.e. the ctypes driver wraps the SAME C core, not a second fill model.

Run:  make test-sim   (or: PYTHONPATH=. python3 tests/test_sim_binding.py)
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.keel_sim import MarketData, TradingEnv, roundtrip_cost, seed  # noqa: E402
from sim.make_sample_data import make_sample  # noqa: E402


def _approx(a, b, tol):
    return abs(a - b) <= tol


def test_load_and_step():
    with tempfile.TemporaryDirectory() as d:
        binp = Path(d) / "sample.bin"
        make_sample(binp, num_symbols=2, num_timesteps=400, seed=1)
        md = MarketData.load(binp)
        assert md.num_symbols == 2
        assert md.num_timesteps == 400
        assert md.features_per_sym == 16

        seed(123)
        env = TradingEnv(md, max_steps=64, decision_lag=2, forced_offset=0,
                         action_allocation_bins=1, action_level_bins=1)
        expected_obs = 2 * 16 + 5 + 2
        assert env.obs_size == expected_obs, f"obs_size {env.obs_size} != {expected_obs}"
        assert env.num_actions == 1 + 2 * 2, env.num_actions  # flat + 2 long + 2 short

        obs = env.reset()
        assert len(obs) == env.obs_size
        assert all(isinstance(x, float) for x in obs)

        # Run a vectorized-style rollout: go long sym 0 (action 1), then hold.
        steps = 0
        saw_terminal = False
        for i in range(64):
            action = 1 if i == 0 else 1  # open then keep
            obs, reward, term = env.step(action)
            steps += 1
            assert len(obs) == env.obs_size
            assert isinstance(reward, float)
            assert reward == reward, "reward must be finite (not NaN)"
            if term:
                saw_terminal = True
                break
        assert steps > 0
        assert saw_terminal, "a 64-step max_steps episode must terminate within the loop"

        log = env.log
        assert log["n"] >= 1.0, "terminal must accumulate one episode into the log"
        for k in ("total_return", "sortino", "max_drawdown"):
            assert log[k] == log[k], f"{k} must be finite"
        env.free()
        md.free()
        print("  ok  load + vectorized step: obs/reward/terminal/log all sane")


def test_fill_parity_with_golden_fixture():
    """The binding's fill path must reproduce tests/test_fill_model.c exactly."""
    # (buffer, slip, expected fill, expected entry, expected round-trip cost)
    golden = [
        (0.0, 0.0, 100.0000, 100.0000, 19.98),
        (5.0, 0.0, 100.0000, 100.0000, 19.98),
        (20.0, 0.0, 100.0000, 100.0000, 19.98),
        (0.0, 5.0, 100.0500, 100.0500, 29.96),
        (0.0, 20.0, 100.2000, 100.2000, 59.84),
        (5.0, 20.0, 100.2000, 100.2000, 59.84),
    ]
    for buf, slip, exp_fill, exp_entry, exp_cost in golden:
        fill, entry, cost = roundtrip_cost(100.0, 101.0, 99.0, 100.0, buf, slip)
        assert _approx(fill, exp_fill, 1e-3), f"fill {fill} != {exp_fill} (buf={buf},slip={slip})"
        assert _approx(entry, exp_entry, 1e-3), f"entry {entry} != {exp_entry}"
        assert _approx(cost, exp_cost, 0.05), f"cost {cost} != {exp_cost}"
    print("  ok  fill parity: binding reproduces the golden C fixture (ONE fill engine)")


if __name__ == "__main__":
    print("keel sim-binding tests:")
    test_load_and_step()
    test_fill_parity_with_golden_fixture()
    print("ALL SIM-BINDING INVARIANTS HOLD.")
