"""Walk-forward the volatility edge through the ONE C gate (regime-robustness).

The faint edge from feature_search (short the highest-vol major, low turnover,
maker fees) clears a single 60/40 split but is seed-fragile. This walks it across
N CONSECUTIVE out-of-sample folds spanning the whole history, so we see whether it
makes money in each regime or only got lucky in one test window.

Stays inside the single-position C sim / gate (no second fill model — the C env
has no multi-asset book, so a true long/short BASKET isn't expressible here; the
cross-sectional policy below is the single-position rank expression of it):

  * ``xsvol``  — each step, go LONG the symbol whose vol is furthest BELOW the
    cross-sectional mean, or SHORT the one furthest ABOVE, whichever is more
    extreme (a single-position cross-sectional vol-rank trade).
  * ``feature`` — the generic ``sign * feature[idx]`` long/short signal.

Run:  python3 -m research.walkforward --data sim/data/kraken_deep.bin
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from forecast.features import FEATURE_SPEC
from research.eval import evaluate
from research.feature_search import make_long_short_policy, _score_fn
from sim.keel_sim import MarketData


def make_xsvol_policy(vol_idx: int, n_symbols: int, features_per_sym: int):
    """Single-position cross-sectional vol-rank trade (long low-vol / short high-vol)."""
    def _policy(obs: List[float], env) -> int:
        vols = [obs[s * features_per_sym + vol_idx] for s in range(n_symbols)]
        mean = sum(vols) / len(vols)
        hi = max(range(n_symbols), key=lambda s: vols[s])
        lo = min(range(n_symbols), key=lambda s: vols[s])
        dev_hi = vols[hi] - mean          # how far the most-volatile is above mean
        dev_lo = mean - vols[lo]          # how far the least-volatile is below mean
        if dev_hi <= 0 and dev_lo <= 0:
            return 0
        if dev_hi >= dev_lo:
            action = 1 + n_symbols + hi   # short the most volatile
        else:
            action = 1 + lo               # long the least volatile
        return action if action < env.num_actions else 0
    return _policy


def walk_forward(data_path: Path, *, policy_kind: str, feature: str, sign: int,
                 folds: int = 8, n_windows: int = 8, window_steps: int = 240,
                 fee_rate: float = 0.0010, slip_bps: float = 5.0,
                 max_hold: int = 72, seed: int = 0) -> List[float]:
    md = MarketData.load(str(data_path))
    vol_idx = FEATURE_SPEC.index("atr_pct_24h")
    try:
        if policy_kind == "xsvol":
            policy = make_xsvol_policy(vol_idx, md.num_symbols, md.features_per_sym)
        else:
            policy = make_long_short_policy(_score_fn(FEATURE_SPEC.index(feature), sign),
                                            md.num_symbols, md.features_per_sym)
        meds: List[float] = []
        for k in range(folds):
            lo = k / folds
            hi = (k + 1) / folds
            v = evaluate(policy, md, n_windows=n_windows, window_steps=window_steps,
                         seed=seed, fail_fast=False, slippages_bps=(slip_bps,),
                         fee_rate=fee_rate, max_hold_hours=max_hold,
                         offset_lo_frac=lo, offset_hi_frac=hi)
            meds.append(v.worst_cell_median_monthly)
        return meds
    finally:
        md.free()


def main():
    ap = argparse.ArgumentParser(description="Walk-forward the vol edge across consecutive OOS folds")
    ap.add_argument("--data", default="sim/data/kraken_deep.bin")
    ap.add_argument("--policy", choices=["xsvol", "feature"], default="xsvol")
    ap.add_argument("--feature", default="atr_pct_24h")
    ap.add_argument("--sign", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window-steps", type=int, default=240)
    ap.add_argument("--fee-rate", type=float, default=0.0010)
    ap.add_argument("--slip-bps", type=float, default=5.0)
    ap.add_argument("--max-hold", type=int, default=72)
    args = ap.parse_args()

    meds = walk_forward(Path(args.data), policy_kind=args.policy, feature=args.feature,
                        sign=args.sign, folds=args.folds, n_windows=args.windows,
                        window_steps=args.window_steps, fee_rate=args.fee_rate,
                        slip_bps=args.slip_bps, max_hold=args.max_hold)
    pos = sum(1 for m in meds if m > 0)
    label = "xsvol(long-low/short-high)" if args.policy == "xsvol" else f"{args.feature} sign={args.sign:+d}"
    print(f"walk-forward {label} on {args.data} | maker fee={args.fee_rate} slip={args.slip_bps}bps hold={args.max_hold}h")
    print(f"  folds positive: {pos}/{args.folds}")
    print("  per-fold median monthly: " + "  ".join(f"{m:+.3f}" for m in meds))
    srt = sorted(meds)
    print(f"  median-of-folds={srt[len(srt)//2]:+.4f}  min={min(meds):+.3f}  max={max(meds):+.3f}")


if __name__ == "__main__":
    main()
