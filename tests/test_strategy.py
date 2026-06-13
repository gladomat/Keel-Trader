"""Issue #4 — the pure XGB strategy (signal -> pick -> size).

Stdlib-only (no pytest/numpy): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_strategy.py``.

Covers conviction filtering and inverse-vol sizing edge cases. The strategy is a
pure function, so no sim/.bin is needed.
"""
from __future__ import annotations

from models.xgb.strategy import (
    Position,
    StrategyConfig,
    SymbolSignal,
    decide,
    filter_by_conviction,
    inverse_vol_weights,
    select_picks,
)


def _wmap(positions):
    return {p.symbol: p.weight for p in positions}


def test_empty_and_no_conviction():
    cfg = StrategyConfig(conviction_threshold=0.5)
    assert decide([], cfg) == []
    # all below threshold -> nothing sized
    sigs = [SymbolSignal("A", 0.1, 0.02), SymbolSignal("B", -0.3, 0.02)]
    assert decide(sigs, cfg) == []
    print("ok test_empty_and_no_conviction")


def test_conviction_filter_requires_positive():
    cfg = StrategyConfig(conviction_threshold=-1.0)  # threshold below zero
    sigs = [SymbolSignal("A", 0.0, 0.02), SymbolSignal("B", 0.2, 0.02)]
    kept = filter_by_conviction(sigs, cfg)
    # score 0.0 is not strictly positive -> dropped even though it clears threshold
    assert [s.symbol for s in kept] == ["B"]
    print("ok test_conviction_filter_requires_positive")


def test_select_picks_top_k_by_score():
    cfg = StrategyConfig(max_positions=2)
    sigs = [
        SymbolSignal("A", 0.1, 0.02),
        SymbolSignal("B", 0.9, 0.02),
        SymbolSignal("C", 0.5, 0.02),
    ]
    picks = select_picks(sigs, cfg)
    assert [s.symbol for s in picks] == ["B", "C"]
    print("ok test_select_picks_top_k_by_score")


def test_inverse_vol_lower_vol_gets_more():
    cfg = StrategyConfig(gross_exposure=1.0, max_position_weight=1.0)
    picks = [SymbolSignal("LOWVOL", 0.5, 0.01), SymbolSignal("HIGHVOL", 0.5, 0.04)]
    w = _wmap(inverse_vol_weights(picks, cfg))
    assert w["LOWVOL"] > w["HIGHVOL"]
    # 1/0.01 : 1/0.04 = 4 : 1 -> 0.8 / 0.2 of gross
    assert abs(w["LOWVOL"] - 0.8) < 1e-6
    assert abs(w["HIGHVOL"] - 0.2) < 1e-6
    assert abs(sum(w.values()) - 1.0) < 1e-6
    print("ok test_inverse_vol_lower_vol_gets_more")


def test_gross_exposure_scales_total():
    cfg = StrategyConfig(gross_exposure=0.5, max_position_weight=1.0)
    picks = [SymbolSignal("A", 0.5, 0.02), SymbolSignal("B", 0.5, 0.02)]
    w = _wmap(inverse_vol_weights(picks, cfg))
    assert abs(sum(w.values()) - 0.5) < 1e-6
    assert abs(w["A"] - 0.25) < 1e-6 and abs(w["B"] - 0.25) < 1e-6
    print("ok test_gross_exposure_scales_total")


def test_per_name_cap_redistributes_overflow():
    # Equal vols would split 1/3 each; cap at 0.34 keeps it just legal.
    cfg = StrategyConfig(gross_exposure=1.0, max_position_weight=0.34, max_positions=3)
    picks = [
        SymbolSignal("A", 0.5, 0.02),
        SymbolSignal("B", 0.5, 0.02),
        SymbolSignal("C", 0.5, 0.02),
    ]
    w = _wmap(inverse_vol_weights(picks, cfg))
    for sym in ("A", "B", "C"):
        assert w[sym] <= 0.34 + 1e-9, (sym, w[sym])
    assert abs(sum(w.values()) - 1.0) < 1e-6

    # Now make one name so low-vol it would blow past the cap; overflow must move.
    cfg2 = StrategyConfig(gross_exposure=1.0, max_position_weight=0.5, max_positions=3)
    picks2 = [
        SymbolSignal("BIG", 0.5, 0.001),   # tiny vol -> wants almost everything
        SymbolSignal("M", 0.5, 0.02),
        SymbolSignal("N", 0.5, 0.02),
    ]
    w2 = _wmap(inverse_vol_weights(picks2, cfg2))
    assert abs(w2["BIG"] - 0.5) < 1e-6, w2          # capped
    # remaining 0.5 split between M and N (equal vol) -> 0.25 each
    assert abs(w2["M"] - 0.25) < 1e-6 and abs(w2["N"] - 0.25) < 1e-6, w2
    assert abs(sum(w2.values()) - 1.0) < 1e-6
    print("ok test_per_name_cap_redistributes_overflow")


def test_zero_volatility_does_not_blow_up():
    cfg = StrategyConfig(gross_exposure=1.0, max_position_weight=1.0, min_volatility=1e-4)
    picks = [SymbolSignal("Z", 0.5, 0.0), SymbolSignal("Y", 0.5, 0.0)]
    w = _wmap(inverse_vol_weights(picks, cfg))
    # both floored to the same vol -> equal split, finite
    assert abs(w["Z"] - 0.5) < 1e-6 and abs(w["Y"] - 0.5) < 1e-6
    print("ok test_zero_volatility_does_not_blow_up")


def test_single_pick_capped():
    cfg = StrategyConfig(gross_exposure=1.0, max_position_weight=0.34, max_positions=5)
    picks = [SymbolSignal("ONLY", 0.9, 0.02)]
    w = _wmap(inverse_vol_weights(picks, cfg))
    assert abs(w["ONLY"] - 0.34) < 1e-9  # one name can't exceed its cap
    print("ok test_single_pick_capped")


if __name__ == "__main__":
    test_empty_and_no_conviction()
    test_conviction_filter_requires_positive()
    test_select_picks_top_k_by_score()
    test_inverse_vol_lower_vol_gets_more()
    test_gross_exposure_scales_total()
    test_per_name_cap_redistributes_overflow()
    test_zero_volatility_does_not_blow_up()
    test_single_pick_capped()
    print("all strategy tests passed")
