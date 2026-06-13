"""Pure signal -> pick -> size strategy for the XGB-daily champion.

This is the shape of moray's ``xgbnew/live_trader.py`` decision logic with every
broker call, network read, and global state stripped out: it is a *pure function*
of model scores + risk inputs, so it is exhaustively unit-testable and reusable by
both the backtester (Phase 6) and the live path (Phase 8) without divergence.

Pipeline:
  1. **conviction filter** — drop symbols whose model score is below threshold
     (long-only here: a non-positive edge is never sized in);
  2. **pick** — keep the top ``max_positions`` survivors by score;
  3. **size** — inverse-volatility weights (lower vol -> bigger slice), scaled to
     ``gross_exposure``, with a per-name cap whose overflow is redistributed to the
     uncapped names.

No I/O, no randomness, no broker. Stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SymbolSignal:
    """One symbol's model output + risk input at a decision point."""
    symbol: str
    score: float        # model conviction / expected edge; higher = more bullish
    volatility: float   # risk proxy for inverse-vol sizing (e.g. atr_pct_24h)


@dataclass(frozen=True)
class Position:
    """A target long allocation as a fraction of portfolio equity."""
    symbol: str
    weight: float


@dataclass(frozen=True)
class StrategyConfig:
    conviction_threshold: float = 0.0   # minimum score to be eligible (long-only)
    max_positions: int = 5              # cap on number of concurrent names
    gross_exposure: float = 1.0         # total fraction of equity deployed (<=1 = no leverage)
    max_position_weight: float = 0.34   # per-name cap as a fraction of equity
    min_volatility: float = 1e-4        # floor so inverse-vol can't divide by ~0


def filter_by_conviction(signals: List[SymbolSignal],
                         cfg: StrategyConfig) -> List[SymbolSignal]:
    """Keep only long-eligible signals: score above threshold AND strictly positive."""
    return [s for s in signals
            if s.score >= cfg.conviction_threshold and s.score > 0.0]


def select_picks(signals: List[SymbolSignal],
                 cfg: StrategyConfig) -> List[SymbolSignal]:
    """Top ``max_positions`` by score (ties broken by symbol for determinism)."""
    ranked = sorted(signals, key=lambda s: (-s.score, s.symbol))
    return ranked[: max(0, cfg.max_positions)]


def inverse_vol_weights(picks: List[SymbolSignal],
                        cfg: StrategyConfig) -> List[Position]:
    """Inverse-vol weights scaled to gross_exposure, per-name cap redistributed.

    Lower volatility earns a larger slice. The per-name cap overflow is pushed onto
    the remaining uncapped names iteratively until either nothing overflows or every
    name is capped (in which case total deployed may fall below gross_exposure — we
    never breach the cap to hit the exposure target).
    """
    if not picks:
        return []

    raw = {s.symbol: 1.0 / max(s.volatility, cfg.min_volatility) for s in picks}
    total_raw = sum(raw.values())
    if total_raw <= 0.0:
        return []

    cap = cfg.max_position_weight
    budget = cfg.gross_exposure
    capped: dict[str, float] = {}
    remaining = dict(raw)

    # Iteratively allocate the remaining budget across uncapped names by raw weight,
    # capping any that exceed the per-name limit and recycling their overflow.
    while remaining:
        rem_total = sum(remaining.values())
        rem_budget = budget - sum(capped.values())
        if rem_budget <= 0.0 or rem_total <= 0.0:
            break
        newly_capped = {}
        for sym, w in remaining.items():
            alloc = rem_budget * (w / rem_total)
            if alloc > cap + 1e-12:
                newly_capped[sym] = cap
        if not newly_capped:
            # nothing overflows: finalize the proportional split
            for sym, w in remaining.items():
                capped[sym] = rem_budget * (w / rem_total)
            break
        capped.update(newly_capped)
        for sym in newly_capped:
            del remaining[sym]

    positions = [Position(s.symbol, capped.get(s.symbol, 0.0)) for s in picks]
    return [p for p in positions if p.weight > 0.0]


def decide(signals: List[SymbolSignal], cfg: StrategyConfig | None = None) -> List[Position]:
    """Full pipeline: conviction filter -> pick -> inverse-vol size."""
    cfg = cfg or StrategyConfig()
    eligible = filter_by_conviction(signals, cfg)
    picks = select_picks(eligible, cfg)
    return inverse_vol_weights(picks, cfg)
