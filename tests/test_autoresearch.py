"""Issue #8 — autoresearch leaderboard is append-only + reproducible + gate-honest.

Stdlib-only (no pytest/numpy): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_autoresearch.py``.

Pins:
  (a) the leaderboard CSV is append-only (a second run grows it, header once);
  (b) trials are reproducible — same seed => identical metric row;
  (c) ranking uses generalization_score (overfit-penalized);
  (d) a trial that fails the Phase-3 gate is NOT promoted.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from research.autoresearch import (
    LEADERBOARD_COLUMNS,
    TrialConfig,
    generalization_score,
    mutation_grid,
    preset_pool,
    run_autoresearch,
    run_trial,
)
from sim.keel_sim import MarketData
from sim.make_sample_data import make_sample


def _read_rows(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_generalization_score_penalizes_overfit():
    # no overfit gap: score == test
    assert abs(generalization_score(0.1, 0.1) - 0.1) < 1e-9
    # train >> test: penalized below test
    assert generalization_score(0.5, 0.1) < 0.1
    assert abs(generalization_score(0.5, 0.1, overfit_penalty=1.0) - (0.1 - 0.4)) < 1e-9
    # test >= train (no overfit): no penalty
    assert abs(generalization_score(0.05, 0.2) - 0.2) < 1e-9
    print("ok test_generalization_score_penalizes_overfit")


def test_mutation_grid_expands_presets():
    presets = [TrialConfig("p")]
    grid = {"gross_exposure": [0.5, 1.0], "max_positions": [2, 3]}
    expanded = mutation_grid(presets, grid)
    assert len(expanded) == 4  # 1 preset x 2 x 2
    names = {c.name for c in expanded}
    assert len(names) == 4  # unique names
    print("ok test_mutation_grid_expands_presets")


def test_leaderboard_append_only_and_reproducible():
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "ar.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        lb = Path(d) / "leaderboard.csv"
        configs = preset_pool()

        md = MarketData.load(data)
        try:
            r1 = run_autoresearch(md, configs, lb, seed=0, n_windows=6, window_steps=100)
            rows_after_1 = _read_rows(lb)
            r2 = run_autoresearch(md, configs, lb, seed=0, n_windows=6, window_steps=100)
            rows_after_2 = _read_rows(lb)
        finally:
            md.free()

        # Append-only: second run adds rows, never truncates.
        assert len(rows_after_1) == len(configs)
        assert len(rows_after_2) == 2 * len(configs)

        # Header written exactly once (DictReader would choke otherwise); columns match.
        assert set(rows_after_2[0].keys()) == set(LEADERBOARD_COLUMNS)

        # Reproducible: same seed => identical metric columns per config.
        by_name_1 = {r.config_name: r for r in r1}
        by_name_2 = {r.config_name: r for r in r2}
        for name, a in by_name_1.items():
            b = by_name_2[name]
            assert abs(a.generalization_score - b.generalization_score) < 1e-9, name
            assert abs(a.test_median_monthly - b.test_median_monthly) < 1e-9, name
            assert abs(a.train_median_monthly - b.train_median_monthly) < 1e-9, name

        # Ranking is by generalization_score, descending.
        scores = [r.generalization_score for r in r1]
        assert scores == sorted(scores, reverse=True)
    print("ok test_leaderboard_append_only_and_reproducible")


def test_gate_failing_trial_not_promoted():
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "ar.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        md = MarketData.load(data)
        try:
            # An impossibly high conviction threshold -> always flat -> gate rejects.
            cfg = TrialConfig("hopeless", conviction_threshold=1e9)
            res = run_trial(md, cfg, seed=0, n_windows=6, window_steps=100)
            assert res.gate_promoted is False
            assert res.promoted is False
        finally:
            md.free()
    print("ok test_gate_failing_trial_not_promoted")


if __name__ == "__main__":
    test_generalization_score_penalizes_overfit()
    test_mutation_grid_expands_presets()
    test_leaderboard_append_only_and_reproducible()
    test_gate_failing_trial_not_promoted()
    print("all autoresearch tests passed")
