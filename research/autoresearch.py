"""Autoresearch: generate champions, rank them honestly, promote only gate-clearers.

The search loop: a **preset pool** of strategy configs, expanded by a **mutation
grid**, each run through the **trial pipeline** (train vs. unseen-test split, both
scored by the ONE Phase-3 gate / ONE C sim). Results are appended to an
**append-only leaderboard CSV** carrying a reproducibility manifest (git hash,
seed, hardware) and ranked on the overfit-penalized ``generalization_score``.

Honesty rules ported from moray's ``autoresearch_rl``:
  * the leaderboard is append-only — we never rewrite history, so a later good run
    can't quietly erase an earlier bad one;
  * ranking uses ``generalization_score`` (test performance minus an overfit
    penalty for train≫test), not raw train PnL;
  * **a trial is promoted only if it clears the Phase-3 gate on the unseen split** —
    a high score that fails the gate is believed-but-not-promoted.

Trials are deterministic given a seed (deterministic policy + seeded gate offsets),
so the same seed reproduces the same metric row.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from models.xgb.backtest import baseline_score_fn, make_strategy_policy
from models.xgb.strategy import StrategyConfig
from research.eval import evaluate
from sim.keel_sim import MarketData

LEADERBOARD_COLUMNS = [
    "timestamp", "git_hash", "hardware", "seed", "config_name", "params",
    "train_median_monthly", "test_median_monthly", "generalization_score",
    "gate_promoted", "promoted",
]


@dataclass(frozen=True)
class TrialConfig:
    name: str
    conviction_threshold: float = 0.0
    gross_exposure: float = 1.0
    max_position_weight: float = 0.34
    max_positions: int = 5

    def to_strategy(self) -> StrategyConfig:
        return StrategyConfig(
            conviction_threshold=self.conviction_threshold,
            gross_exposure=self.gross_exposure,
            max_position_weight=self.max_position_weight,
            max_positions=self.max_positions,
        )

    def params(self) -> dict:
        return {
            "conviction_threshold": self.conviction_threshold,
            "gross_exposure": self.gross_exposure,
            "max_position_weight": self.max_position_weight,
            "max_positions": self.max_positions,
        }


@dataclass
class TrialResult:
    config_name: str
    params: dict
    seed: int
    train_median_monthly: float
    test_median_monthly: float
    generalization_score: float
    gate_promoted: bool
    promoted: bool


def preset_pool() -> List[TrialConfig]:
    """A small spread of sane starting configs (the champions we mutate from)."""
    return [
        TrialConfig("baseline", conviction_threshold=0.0, gross_exposure=1.0),
        TrialConfig("selective", conviction_threshold=0.002, gross_exposure=0.8),
        TrialConfig("concentrated", max_position_weight=0.5, max_positions=2),
    ]


def mutation_grid(presets: Sequence[TrialConfig],
                  grid: Dict[str, Sequence]) -> List[TrialConfig]:
    """Expand each preset across the cartesian product of ``grid`` overrides."""
    if not grid:
        return list(presets)
    keys = list(grid.keys())
    out: List[TrialConfig] = []
    for preset in presets:
        for combo in itertools.product(*(grid[k] for k in keys)):
            overrides = dict(zip(keys, combo))
            suffix = "_".join(f"{k}{v}" for k, v in overrides.items())
            out.append(TrialConfig(
                name=f"{preset.name}__{suffix}",
                conviction_threshold=overrides.get("conviction_threshold",
                                                   preset.conviction_threshold),
                gross_exposure=overrides.get("gross_exposure", preset.gross_exposure),
                max_position_weight=overrides.get("max_position_weight",
                                                  preset.max_position_weight),
                max_positions=overrides.get("max_positions", preset.max_positions),
            ))
    return out


def generalization_score(train_score: float, test_score: float,
                         overfit_penalty: float = 1.0) -> float:
    """Test performance minus a penalty for the train≫test overfit gap."""
    overfit_gap = max(0.0, train_score - test_score)
    return test_score - overfit_penalty * overfit_gap


def run_trial(md: MarketData, config: TrialConfig, *, seed: int = 0,
              train_frac: float = 0.6, n_windows: int = 8, window_steps: int = 120,
              overfit_penalty: float = 1.0) -> TrialResult:
    """Score one config on a train split and the unseen test split (the gate)."""
    policy = make_strategy_policy(baseline_score_fn(), md.num_symbols,
                                  md.features_per_sym, config.to_strategy())

    # Train split: in-sample windows. fail_fast off so we always get a median.
    train_v = evaluate(policy, md, n_windows=n_windows, window_steps=window_steps,
                       seed=seed, fail_fast=False,
                       offset_lo_frac=0.0, offset_hi_frac=train_frac)
    # Test split: unseen windows -> THIS is the Phase-3 gate decision.
    test_v = evaluate(policy, md, n_windows=n_windows, window_steps=window_steps,
                      seed=seed, offset_lo_frac=train_frac, offset_hi_frac=1.0)

    gen = generalization_score(train_v.worst_cell_median_monthly,
                               test_v.worst_cell_median_monthly, overfit_penalty)
    return TrialResult(
        config_name=config.name,
        params=config.params(),
        seed=seed,
        train_median_monthly=train_v.worst_cell_median_monthly,
        test_median_monthly=test_v.worst_cell_median_monthly,
        generalization_score=gen,
        gate_promoted=test_v.promote,
        promoted=test_v.promote,  # promote only if the unseen gate clears
    )


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def append_to_leaderboard(path: Path, results: Sequence[TrialResult], *,
                          seed: int, git_hash: Optional[str] = None,
                          hardware: Optional[str] = None) -> Path:
    """Append rows to the leaderboard CSV (never truncates; writes header once)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    git_hash = git_hash if git_hash is not None else _git_hash()
    hardware = hardware if hardware is not None else platform.platform()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": ts,
                "git_hash": git_hash,
                "hardware": hardware,
                "seed": r.seed,
                "config_name": r.config_name,
                "params": json.dumps(r.params, sort_keys=True),
                "train_median_monthly": f"{r.train_median_monthly:.6f}",
                "test_median_monthly": f"{r.test_median_monthly:.6f}",
                "generalization_score": f"{r.generalization_score:.6f}",
                "gate_promoted": int(r.gate_promoted),
                "promoted": int(r.promoted),
            })
    return path


def run_autoresearch(md: MarketData, configs: Sequence[TrialConfig],
                     leaderboard_path: Path, *, seed: int = 0,
                     n_windows: int = 8, window_steps: int = 120,
                     **trial_kwargs) -> List[TrialResult]:
    """Run every config through the trial pipeline, append, return ranked-desc."""
    results = [run_trial(md, c, seed=seed, n_windows=n_windows,
                         window_steps=window_steps, **trial_kwargs)
               for c in configs]
    append_to_leaderboard(leaderboard_path, results, seed=seed)
    results.sort(key=lambda r: r.generalization_score, reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser(description="Run the autoresearch search loop")
    ap.add_argument("--data", default="sim/data/sample.bin")
    ap.add_argument("--leaderboard", default="artifacts/leaderboard.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--window-steps", type=int, default=720)
    args = ap.parse_args()

    configs = mutation_grid(preset_pool(), {
        "conviction_threshold": [0.0, 0.001],
        "gross_exposure": [0.8, 1.0],
    })

    md = MarketData.load(args.data)
    try:
        results = run_autoresearch(
            md, configs, Path(args.leaderboard), seed=args.seed,
            n_windows=args.windows, window_steps=args.window_steps,
        )
    finally:
        md.free()

    print(f"ran {len(results)} trials -> {args.leaderboard}")
    for r in results[:10]:
        flag = "PROMOTE" if r.promoted else "reject "
        print(f"  [{flag}] {r.config_name:<40} gen={r.generalization_score:+.4f} "
              f"(train={r.train_median_monthly:+.4f} test={r.test_median_monthly:+.4f})")


if __name__ == "__main__":
    main()
