"""Score the XGB champion on the real judge: strategy -> Phase-3 gate -> verdict.

This runs the Phase-5 pure strategy (``models/xgb/strategy.py``) through the ONE
out-of-sample gate (``research/eval.py``), which drives the ONE C fill engine. The
point is a *real* number — clearing the 0.27 bar or honestly failing are both
acceptable outcomes; what matters is that the verdict comes from the same judge a
live candidate would face, with no separate backtest fill model.

The C sim holds a single position at a time, so the gate-facing policy collapses
the strategy's ranked book to its top conviction pick (the largest target weight)
and emits that symbol's long action — or goes flat when nothing clears conviction.
Scores come from either a trained XGB artifact (lazy xgboost load) or, with no
model, a deterministic feature-as-signal baseline so a real verdict is always
producible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Optional

from forecast.features import FEATURE_SPEC
from models.xgb.strategy import StrategyConfig, SymbolSignal, decide
from research.eval import Verdict, evaluate
from sim.keel_sim import MarketData

# Feature indices used by the no-model baseline signal.
_SCORE_FEATURE = "chronos_close_delta_h24"   # bullish if forecast close is up
_VOL_FEATURE = "atr_pct_24h"                  # risk proxy for inverse-vol sizing

ScoreFn = Callable[[List[float]], float]      # per-symbol feature vector -> score


def _symbol_features(obs: List[float], s: int, n_symbols: int,
                     features_per_sym: int) -> List[float]:
    """Slice symbol ``s``'s feature vector out of the obs (layout: [S*F | acct | pos])."""
    start = s * features_per_sym
    return obs[start: start + features_per_sym]


def baseline_score_fn() -> ScoreFn:
    """Deterministic feature-as-signal: score = forecast 24h close delta."""
    idx = FEATURE_SPEC.index(_SCORE_FEATURE)

    def _fn(feats: List[float]) -> float:
        return feats[idx]

    return _fn


def xgb_artifact_score_fn(artifact_dir: Path) -> ScoreFn:  # pragma: no cover - needs xgboost
    """Load a trained XGB artifact and score via P(up) - 0.5 (centered conviction)."""
    import numpy as np
    import xgboost as xgb

    meta = json.loads((Path(artifact_dir) / "metadata.json").read_text())
    if meta.get("feature_spec_version") != FEATURE_SPEC.version:
        raise ValueError(
            f"artifact spec {meta.get('feature_spec_version')} != {FEATURE_SPEC.version}"
        )
    booster = xgb.Booster()
    booster.load_model(str(Path(artifact_dir) / "model.json"))

    def _fn(feats: List[float]) -> float:
        dm = xgb.DMatrix(np.asarray([feats], dtype=np.float32),
                         feature_names=list(meta["feature_names"]))
        return float(booster.predict(dm)[0]) - 0.5

    return _fn


def make_strategy_policy(score_fn: ScoreFn, n_symbols: int, features_per_sym: int,
                         cfg: Optional[StrategyConfig] = None):
    """Adapt the pure strategy into a gate policy ``(obs, env) -> action``.

    The strategy ranks/sizes a book; the single-position C env can only hold one
    name, so we take the top-weight pick and emit its long action (``1 + sym``),
    falling back to flat (0) when the strategy deploys nothing.
    """
    cfg = cfg or StrategyConfig()
    vol_idx = FEATURE_SPEC.index(_VOL_FEATURE)

    def _policy(obs: List[float], env) -> int:
        signals = []
        for s in range(n_symbols):
            feats = _symbol_features(obs, s, n_symbols, features_per_sym)
            signals.append(SymbolSignal(
                symbol=str(s), score=score_fn(feats), volatility=abs(feats[vol_idx]),
            ))
        positions = decide(signals, cfg)
        if not positions:
            return 0  # flat
        top = max(positions, key=lambda p: p.weight)
        sym = int(top.symbol)
        action = 1 + sym
        return action if action < env.num_actions else 0

    return _policy


def run_backtest(data_path: Path, *, artifact_dir: Optional[Path] = None,
                 n_windows: int = 20, window_steps: int = 720, seed: int = 0,
                 strategy_cfg: Optional[StrategyConfig] = None,
                 verdict_out: Optional[Path] = None, **gate_kwargs) -> Verdict:
    """Run the strategy through the gate on ``data_path`` and return the verdict."""
    md = MarketData.load(data_path)
    try:
        score_fn = (xgb_artifact_score_fn(artifact_dir) if artifact_dir
                    else baseline_score_fn())
        policy = make_strategy_policy(score_fn, md.num_symbols, md.features_per_sym,
                                      strategy_cfg)
        verdict = evaluate(policy, md, n_windows=n_windows, window_steps=window_steps,
                           seed=seed, **gate_kwargs)
    finally:
        md.free()

    if verdict_out:
        verdict_out.parent.mkdir(parents=True, exist_ok=True)
        verdict_out.write_text(json.dumps({
            "promote": verdict.promote,
            "worst_cell_median_monthly": verdict.worst_cell_median_monthly,
            "promotion_target": verdict.promotion_target,
            "failed_fast": verdict.failed_fast,
            "reason": verdict.reason,
            "n_windows": verdict.n_windows,
            "window_steps": verdict.window_steps,
            "source": "xgb_artifact" if artifact_dir else "feature_baseline",
        }, indent=2))
    return verdict


def main():
    ap = argparse.ArgumentParser(description="Backtest the XGB strategy through the gate")
    ap.add_argument("--data", default="sim/data/sample.bin")
    ap.add_argument("--artifact-dir", default=None, help="trained XGB artifact (optional)")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--window-steps", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verdict-out", default=None, help="write verdict JSON here")
    args = ap.parse_args()

    verdict = run_backtest(
        Path(args.data),
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        n_windows=args.windows, window_steps=args.window_steps, seed=args.seed,
        verdict_out=Path(args.verdict_out) if args.verdict_out else None,
    )
    print(verdict.summary())
    raise SystemExit(0 if verdict.promote else 1)


if __name__ == "__main__":
    main()
