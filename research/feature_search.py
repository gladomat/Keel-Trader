"""Auto-research the Kraken crypto edge: sweep the SIGNAL, not just the config.

``research.autoresearch`` searches strategy hyperparameters on ONE fixed baseline
signal (``chronos_close_delta_h24``). That can't find an edge if the edge lives in
a different feature (and it goes degenerate when that feature is zero). This sweep
searches the signal itself:

    for each FEATURE_SPEC feature f, each sign s in {+1,-1}, each strategy config:
        score = s * feature[f]              # the candidate signal
        train = gate over the in-sample window split   (offset 0 .. train_frac)
        test  = gate over the held-out window split    (offset train_frac .. 1)
        rank by generalization_score (test minus overfit penalty);
        a candidate is PROMOTED only if it clears the Phase-3 gate on the unseen
        split (same honest rule autoresearch uses).

Everything runs through the ONE C sim / ONE gate. Offline (needs the built sim +
a real ``.bin``); not part of ``make test``. Results append to a leaderboard CSV.

Run:  python3 -m research.feature_search --data sim/data/kraken_deep.bin
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from forecast.features import FEATURE_NAMES, FEATURE_SPEC
from models.xgb.strategy import StrategyConfig
from models.xgb.backtest import make_strategy_policy
from research.autoresearch import generalization_score
from research.eval import evaluate
from sim.keel_sim import MarketData

_VOL_FEATURE = "atr_pct_24h"  # inverse-vol risk proxy for sizing


@dataclass
class SearchResult:
    feature: str
    sign: int
    config_name: str
    train_median_monthly: float
    test_median_monthly: float
    generalization_score: float
    promoted: bool


def _score_fn(feat_idx: int, sign: int):
    def fn(feats: List[float]) -> float:
        return sign * feats[feat_idx]
    return fn


def make_long_short_policy(score_fn, n_symbols: int, features_per_sym: int):
    """Directional single-position long/short policy over the C action space.

    Picks the symbol with the largest |score|; positive -> long (action 1+sym),
    negative -> short (action 1+S+sym). Flat on a zero signal. This lets the gate
    judge SHORT signals too — long-only can't win a net-down crypto window.
    """
    def _policy(obs: List[float], env) -> int:
        best_sym, best_score = -1, 0.0
        for s in range(n_symbols):
            feats = obs[s * features_per_sym:(s + 1) * features_per_sym]
            sc = score_fn(feats)
            if abs(sc) > abs(best_score):
                best_score, best_sym = sc, s
        if best_sym < 0 or best_score == 0.0:
            return 0
        action = (1 + best_sym) if best_score > 0 else (1 + n_symbols + best_sym)
        return action if action < env.num_actions else 0
    return _policy


def _configs() -> list[tuple[str, StrategyConfig]]:
    """A small spread of long-only book shapes (top-1 / top-2 / top-3, exposure)."""
    return [
        ("top1", StrategyConfig(max_positions=1, gross_exposure=1.0, max_position_weight=1.0)),
        ("top2", StrategyConfig(max_positions=2, gross_exposure=1.0, max_position_weight=0.6)),
        ("top3", StrategyConfig(max_positions=3, gross_exposure=1.0, max_position_weight=0.4)),
        ("top2_half", StrategyConfig(max_positions=2, gross_exposure=0.5, max_position_weight=0.4)),
    ]


def run(data_path: Path, *, train_frac: float = 0.6, n_windows: int = 10,
        window_steps: int = 240, seed: int = 0, mode: str = "long",
        fee_rate: float | None = None, slip_bps: float | None = None,
        max_hold: int | None = None,
        leaderboard: Path | None = None) -> List[SearchResult]:
    md = MarketData.load(str(data_path))
    results: List[SearchResult] = []
    long_short = mode == "longshort"
    # Cost/turnover overrides (maker-fee + low-turnover experiments). When slip is
    # given we collapse the slippage matrix to that single cell.
    env_kw: dict = {}
    if fee_rate is not None:
        env_kw["fee_rate"] = fee_rate
    if max_hold is not None:
        env_kw["max_hold_hours"] = max_hold
    slippages = (slip_bps,) if slip_bps is not None else None
    # In long-short mode the policy direction encodes the signal sign, so the
    # explicit sign sweep would just mirror long<->short — keep both anyway so
    # always-positive features (confidence/vol) still get a short-side test.
    try:
        for fi, fname in enumerate(FEATURE_NAMES):
            for sign in (1, -1):
                score_fn = _score_fn(fi, sign)
                for cname, cfg in _configs():
                    if long_short:
                        policy = make_long_short_policy(score_fn, md.num_symbols,
                                                        md.features_per_sym)
                    else:
                        policy = make_strategy_policy(score_fn, md.num_symbols,
                                                      md.features_per_sym, cfg)
                    _slip = {"slippages_bps": slippages} if slippages else {}
                    train_v = evaluate(policy, md, n_windows=n_windows,
                                       window_steps=window_steps, seed=seed,
                                       fail_fast=False, offset_lo_frac=0.0,
                                       offset_hi_frac=train_frac, **_slip, **env_kw)
                    test_v = evaluate(policy, md, n_windows=n_windows,
                                      window_steps=window_steps, seed=seed,
                                      offset_lo_frac=train_frac, offset_hi_frac=1.0,
                                      **_slip, **env_kw)
                    gen = generalization_score(train_v.worst_cell_median_monthly,
                                               test_v.worst_cell_median_monthly)
                    results.append(SearchResult(
                        fname, sign, cname,
                        train_v.worst_cell_median_monthly,
                        test_v.worst_cell_median_monthly, gen, test_v.promote,
                    ))
    finally:
        md.free()

    results.sort(key=lambda r: r.generalization_score, reverse=True)
    if leaderboard is not None:
        _append(leaderboard, results, data_path)
    return results


def _append(path: Path, results: Sequence[SearchResult], data_path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cols = ["timestamp", "data", "feature", "sign", "config", "train_median_monthly",
            "test_median_monthly", "generalization_score", "promoted"]
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        for r in results:
            w.writerow({
                "timestamp": ts, "data": str(data_path), "feature": r.feature,
                "sign": r.sign, "config": r.config_name,
                "train_median_monthly": f"{r.train_median_monthly:.6f}",
                "test_median_monthly": f"{r.test_median_monthly:.6f}",
                "generalization_score": f"{r.generalization_score:.6f}",
                "promoted": int(r.promoted),
            })


def main():
    ap = argparse.ArgumentParser(description="Sweep signal feature x sign x config through the gate")
    ap.add_argument("--data", default="sim/data/kraken_deep.bin")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--window-steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["long", "longshort"], default="long")
    ap.add_argument("--fee-rate", type=float, default=None, help="override taker fee (e.g. 0.0010 maker)")
    ap.add_argument("--slip-bps", type=float, default=None, help="single slippage cell in bps")
    ap.add_argument("--max-hold", type=int, default=None, help="max hold hours (lower turnover)")
    ap.add_argument("--leaderboard", default="artifacts/kraken_feature_search.csv")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    results = run(Path(args.data), train_frac=args.train_frac, n_windows=args.windows,
                  window_steps=args.window_steps, seed=args.seed, mode=args.mode,
                  fee_rate=args.fee_rate, slip_bps=args.slip_bps, max_hold=args.max_hold,
                  leaderboard=Path(args.leaderboard))
    n_promoted = sum(1 for r in results if r.promoted)
    print(f"ran {len(results)} candidates on {args.data}; {n_promoted} PROMOTED (clear the gate OOS)")
    print(f"top {args.top} by generalization_score:")
    for r in results[:args.top]:
        flag = "PROMOTE" if r.promoted else "reject "
        print(f"  [{flag}] {r.feature:<24} sign={r.sign:+d} {r.config_name:<10} "
              f"gen={r.generalization_score:+.4f} "
              f"(train={r.train_median_monthly:+.4f} test={r.test_median_monthly:+.4f})")


if __name__ == "__main__":
    main()
