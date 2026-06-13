"""Issue #6 — backtest produces a real gate verdict + the paper loop stays paper.

Stdlib-only (no pytest/numpy): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_backtest_paper.py``.

Pins:
  (a) the XGB strategy run through the Phase-3 gate yields a *real* verdict
      (a finite number with a reason — pass or honest fail are both fine);
  (b) PaperRunner is paper-only: it constructs under PAPER=True, records buys and
      submits guarded orders, refuses a death-spiral sell, and exposes NO way to
      win the live-writer lock.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from core import config
from core.alpaca_singleton import forget_all_buys
from core.paper_runner import LiveWriteAttemptError, PaperOrder, PaperRunner
from models.xgb.backtest import baseline_score_fn, make_strategy_policy, run_backtest
from models.xgb.strategy import StrategyConfig, SymbolSignal
from research.eval import evaluate
from sim.keel_sim import MarketData
from sim.make_sample_data import make_sample


def test_backtest_produces_real_verdict():
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "bt.bin"
        make_sample(data, num_symbols=2, num_timesteps=600, seed=7)
        verdict = run_backtest(
            data, n_windows=6, window_steps=120, seed=0, fail_fast=False,
            verdict_out=Path(d) / "verdict.json",
        )
        # A real number came out of the real judge.
        assert math.isfinite(verdict.worst_cell_median_monthly)
        assert verdict.reason
        assert verdict.n_windows == 6
        assert isinstance(verdict.promote, bool)
        assert (Path(d) / "verdict.json").exists()
    print("ok test_backtest_produces_real_verdict")


def test_strategy_policy_emits_valid_actions():
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "bt.bin"
        make_sample(data, num_symbols=2, num_timesteps=300, seed=7)
        md = MarketData.load(data)
        try:
            policy = make_strategy_policy(baseline_score_fn(), md.num_symbols,
                                          md.features_per_sym, StrategyConfig())
            # Run a couple windows; every action the policy emits must be valid.
            from research.eval import TradingEnv
            env = TradingEnv(md, max_steps=64, forced_offset=0, decision_lag=2)
            obs = env.reset()
            for _ in range(64):
                a = policy(obs, env)
                assert 0 <= a < env.num_actions, a
                obs, _r, term = env.step(a)
                if term:
                    break
            env.free()
        finally:
            md.free()
    print("ok test_strategy_policy_emits_valid_actions")


def test_paper_runner_is_paper_only():
    # Sanity: the project default is paper.
    assert config.PAPER is True

    # No live-writer surface exists on the paper runner, by design.
    for forbidden in ("enforce_live_singleton", "acquire_alpaca_account_lock",
                      "acquire_live_lock", "go_live"):
        assert not hasattr(PaperRunner, forbidden), forbidden

    forget_all_buys()
    submitted = []
    runner = PaperRunner(submit=submitted.append)

    # A buy records the price and submits a paper order.
    orders = runner.step([SymbolSignal("AAA", 0.5, 0.02)], {"AAA": 100.0})
    assert any(o.side == "buy" and o.symbol == "AAA" for o in orders)
    assert any(o.side == "buy" for o in submitted)

    # Dropping the name triggers a guarded sell; well above the floor -> allowed.
    orders2 = runner.step([], {"AAA": 99.99})
    assert any(o.side == "sell" and o.symbol == "AAA" for o in orders2)
    print("ok test_paper_runner_is_paper_only")


def test_paper_runner_refuses_death_spiral_sell():
    forget_all_buys()
    runner = PaperRunner(submit=lambda o: None)
    runner.step([SymbolSignal("BBB", 0.5, 0.02)], {"BBB": 100.0})  # buy @ 100
    # Sell far below the remembered buy -> the guard must refuse (RuntimeError).
    try:
        runner.step([], {"BBB": 90.0})
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected death-spiral guard to refuse the sell")
    forget_all_buys()
    print("ok test_paper_runner_refuses_death_spiral_sell")


if __name__ == "__main__":
    test_backtest_produces_real_verdict()
    test_strategy_policy_emits_valid_actions()
    test_paper_runner_is_paper_only()
    test_paper_runner_refuses_death_spiral_sell()
    print("all backtest/paper tests passed")
