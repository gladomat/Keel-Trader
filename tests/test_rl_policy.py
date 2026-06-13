"""Issue #7 — the RL policy is judged by the SAME Phase-3 gate (no second sim).

Stdlib-only (no pytest/torch): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_rl_policy.py``.

Torch/PPO training is offline; here we pin the load-bearing seam: an RLPolicy
(the pure ``LinearPolicy`` stand-in for the trained torch MLP) can be wrapped as a
gate policy and evaluated by ``research.eval`` against the ONE C sim, yielding a
real verdict. We also pin the GAE helper used by the PPO update.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from models.rl.policy import LinearPolicy, as_gate_policy
from research.eval import evaluate
from sim.keel_sim import MarketData, TradingEnv
from sim.make_sample_data import make_sample


def test_linear_policy_argmax():
    # action 1 has the largest bias -> always chosen for any obs.
    pol = LinearPolicy(weights=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                       bias=[0.0, 1.0, -1.0])
    assert pol.act([0.5, -0.3]) == 1
    assert pol.num_actions == 3
    print("ok test_linear_policy_argmax")


def test_rl_policy_runs_through_gate():
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "rl.bin"
        make_sample(data, num_symbols=2, num_timesteps=400, seed=7)
        md = MarketData.load(data)
        try:
            # Size the policy to the env's actual obs/action dims.
            probe = TradingEnv(md, max_steps=32, decision_lag=2, forced_offset=0)
            obs_size, num_actions = probe.obs_size, probe.num_actions
            probe.free()

            # A zero policy -> argmax ties to action 0 (flat); deterministic + valid.
            weights = [[0.0] * obs_size for _ in range(num_actions)]
            bias = [0.0] * num_actions
            policy = as_gate_policy(LinearPolicy(weights, bias))

            verdict = evaluate(policy, md, n_windows=4, window_steps=64, seed=0,
                               fail_fast=False)
            # The gate produced a real verdict from the ONE C sim.
            assert isinstance(verdict.promote, bool)
            assert verdict.reason
            # A flat policy must not promote.
            assert not verdict.promote
        finally:
            md.free()
    print("ok test_rl_policy_runs_through_gate")


def test_gate_policy_clamps_invalid_action():
    # A policy that points at an out-of-range action must be clamped to flat.
    class _Bad:
        def act(self, obs):
            return 9999

    class _Env:
        num_actions = 5

    gp = as_gate_policy(_Bad())
    assert gp([0.0], _Env()) == 0
    print("ok test_gate_policy_clamps_invalid_action")


def test_gae_shapes_and_bootstrap():
    from models.rl.policy import compute_gae

    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    dones = [0.0, 0.0, 1.0]
    adv, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
    assert len(adv) == 3 and len(returns) == 3
    # Last step is terminal: delta = r - v = 1 - 0.5 = 0.5, no bootstrap.
    assert abs(adv[2] - 0.5) < 1e-9
    # returns = adv + values
    for a, v, r in zip(adv, values, returns):
        assert abs(r - (a + v)) < 1e-9
    print("ok test_gae_shapes_and_bootstrap")


if __name__ == "__main__":
    test_linear_policy_argmax()
    test_rl_policy_runs_through_gate()
    test_gate_policy_clamps_invalid_action()
    test_gae_shapes_and_bootstrap()
    print("all rl-policy tests passed")
