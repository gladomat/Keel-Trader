"""Issue #3 — the ONE feature spec + its contract validator + .bin round-trip.

Stdlib-only (no pytest/numpy): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_features.py``.

What it pins:
  1. FEATURE_SPEC v1 is well-formed (16 ordered, unique fields; kinds split 8/8).
  2. validate_feature_contract rejects an out-of-spec model and accepts a
     conforming one (incl. strict positional order).
  3. The packed .bin round-trips: features written via sim/binpack.write_market_bin
     read back identically through BOTH the pure-Python reader AND the C loader
     (keel_md_feature) — proving the Python packer and the C sim agree byte-for-byte.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from forecast.features import (
    FEATURE_NAMES,
    FEATURE_SPEC,
    FEATURES_PER_SYM,
    FeatureContractError,
    validate_feature_contract,
)
from forecast.technical import (
    TECHNICAL_FEATURES,
    assert_technical_in_spec,
    technical_features,
)
from sim.binpack import read_features, read_header, write_market_bin


def _approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol + tol * abs(b)


def test_spec_shape():
    assert FEATURE_SPEC.version == "v1"
    assert FEATURES_PER_SYM == 16
    assert len(FEATURE_NAMES) == 16
    # names are unique and index() round-trips
    assert len(set(FEATURE_NAMES)) == 16
    for i, name in enumerate(FEATURE_NAMES):
        assert FEATURE_SPEC.index(name) == i
    # 8 forecast + 8 technical, in that block order (indices 0-7 / 8-15)
    forecast = FEATURE_SPEC.names_of_kind("forecast")
    technical = FEATURE_SPEC.names_of_kind("technical")
    assert len(forecast) == 8 and len(technical) == 8
    assert FEATURE_NAMES[:8] == forecast
    assert FEATURE_NAMES[8:] == technical
    print("ok test_spec_shape")


def test_validator_accepts_conforming():
    # exact spec order = a conforming model
    assert validate_feature_contract(list(FEATURE_NAMES)) == list(FEATURE_NAMES)
    # a prefix in spec order passes strict_order
    prefix = list(FEATURE_NAMES[:6])
    assert validate_feature_contract(prefix, strict_order=True) == prefix
    # a dict carrying its own feature list is accepted
    assert validate_feature_contract({"features": prefix}) == prefix
    print("ok test_validator_accepts_conforming")


def test_validator_rejects_out_of_spec():
    bad = list(FEATURE_NAMES) + ["some_feature_not_in_spec"]
    try:
        validate_feature_contract(bad)
    except FeatureContractError as e:
        assert "some_feature_not_in_spec" in str(e)
    else:
        raise AssertionError("expected FeatureContractError for out-of-spec feature")

    # right names, wrong order -> strict_order rejects
    shuffled = list(FEATURE_NAMES)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    try:
        validate_feature_contract(shuffled, strict_order=True)
    except FeatureContractError:
        pass
    else:
        raise AssertionError("expected FeatureContractError for wrong feature order")

    # a model exposing no feature names is a hard error, not a silent pass
    try:
        validate_feature_contract(object())
    except FeatureContractError:
        pass
    else:
        raise AssertionError("expected FeatureContractError when names undeterminable")
    print("ok test_validator_rejects_out_of_spec")


def _make_fixture(path: Path, T: int, S: int, F: int):
    """Deterministic features/prices so we can assert exact read-back."""
    features = [[[float(t * 1000 + s * 100 + f) * 0.001 for f in range(F)]
                 for s in range(S)] for t in range(T)]
    prices = [[[float(t * 10 + s + p) for p in range(5)]
               for s in range(S)] for t in range(T)]
    write_market_bin(path, [f"SYM{s}" for s in range(S)], features, prices,
                     num_timesteps=T, features_per_sym=F, version=1)
    return features, prices


def test_bin_roundtrip_pure_python():
    T, S, F = 5, 2, FEATURES_PER_SYM
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rt.bin"
        features, _ = _make_fixture(path, T, S, F)
        hdr = read_header(path)
        assert hdr["num_symbols"] == S
        assert hdr["num_timesteps"] == T
        assert hdr["features_per_sym"] == F
        assert hdr["price_features"] == 5
        back = read_features(path)
        for t in range(T):
            for s in range(S):
                for f in range(F):
                    assert _approx(back[t][s][f], features[t][s][f]), (t, s, f)
    print("ok test_bin_roundtrip_pure_python")


def test_bin_roundtrip_c_loader():
    """The C sim must read the same feature bytes the Python packer wrote."""
    try:
        from sim.keel_sim import MarketData
    except Exception as e:  # pragma: no cover - lib not built
        print(f"SKIP test_bin_roundtrip_c_loader: {e}")
        return
    T, S, F = 5, 2, FEATURES_PER_SYM
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rt.bin"
        features, prices = _make_fixture(path, T, S, F)
        md = MarketData.load(path)
        try:
            assert md.num_symbols == S
            assert md.num_timesteps == T
            assert md.features_per_sym == F
            for t in range(T):
                for s in range(S):
                    for f in range(F):
                        assert _approx(md.feature(t, s, f), features[t][s][f]), (t, s, f)
                    for p in range(5):
                        assert _approx(md.price(t, s, p), prices[t][s][p]), (t, s, p)
        finally:
            md.free()
    print("ok test_bin_roundtrip_c_loader")


# ---------------------------------------------------------------------------
# K2 (#12): the pure technical-feature computation (FEATURE_SPEC indices 8-15)
# ---------------------------------------------------------------------------
def test_technical_features_land_in_spec():
    names = assert_technical_in_spec()
    # the 8 technical names, in spec order, == FEATURE_NAMES[8:]
    assert list(TECHNICAL_FEATURES) == names == FEATURE_NAMES[8:]
    assert len(names) == 8
    # forecast + technical partition the whole spec
    assert FEATURE_SPEC.names_of_kind("forecast") + names == FEATURE_NAMES
    print("ok test_technical_features_land_in_spec")


def test_technical_features_flat_series():
    """A perfectly flat OHLC series -> every technical feature collapses to ~0."""
    n = 100
    o = h = l = c = [100.0] * n
    rows = technical_features(o, h, l, c)
    assert len(rows) == n and all(len(r) == 8 for r in rows)
    last = rows[-1]
    # return_1h, return_24h, ma_delta_24h, ma_delta_72h, atr_pct, trend, drawdown -> 0
    for idx in (0, 1, 3, 4, 5, 6, 7):
        assert abs(last[idx]) < 1e-9, (idx, last[idx])
    # volatility (index 2): undefined at the first two bars -> 0.01 fallback, then 0
    assert abs(rows[0][2] - 0.01) < 1e-12
    assert abs(rows[1][2] - 0.01) < 1e-12
    assert abs(last[2]) < 1e-9
    print("ok test_technical_features_flat_series")


def test_technical_features_uptrend_and_clipping():
    """A constant +1%/bar uptrend -> known returns; clips and ordering hold."""
    n = 120
    c = [100.0 * (1.01 ** i) for i in range(n)]
    o = h = l = c  # degenerate bars; we only assert close-derived features
    rows = technical_features(o, h, l, c)

    # return_1h ~ +0.01 every bar after the first (well inside the +/-0.5 clip)
    for i in range(1, n):
        assert abs(rows[i][0] - 0.01) < 1e-6, (i, rows[i][0])
    # return_24h ~ 1.01**24 - 1 ~ 0.2697 once 24 bars are available
    assert abs(rows[40][1] - (1.01 ** 24 - 1)) < 1e-6
    # trend_72h would be 1.01**72 - 1 ~ 1.047 -> clipped to +1.0
    assert abs(rows[-1][6] - 1.0) < 1e-9
    # monotonic up: close == 72h rolling max -> drawdown 0; ma_delta positive
    assert abs(rows[-1][7]) < 1e-9
    assert rows[-1][3] > 0.0 and rows[-1][4] > 0.0
    # everything stays within its declared clip band
    for r in rows:
        assert -0.5 <= r[0] <= 0.5 and -1.0 <= r[1] <= 1.0
        assert -0.5 <= r[3] <= 0.5 and -0.5 <= r[4] <= 0.5
        assert 0.0 <= r[5] <= 0.5 and -1.0 <= r[6] <= 1.0 and -1.0 <= r[7] <= 0.0
    print("ok test_technical_features_uptrend_and_clipping")


def test_full_feature_set_validates():
    """A produced full row (forecast 0 + technical) carries spec-valid names."""
    # The .bin column identity is FEATURE_NAMES; a produced block of that width
    # must pass the contract validator in strict spec order.
    assert validate_feature_contract(list(FEATURE_NAMES), strict_order=True) == list(FEATURE_NAMES)
    # technical block occupies exactly indices 8..15
    for off, name in enumerate(TECHNICAL_FEATURES):
        assert FEATURE_SPEC.index(name) == 8 + off
    print("ok test_full_feature_set_validates")


if __name__ == "__main__":
    test_spec_shape()
    test_validator_accepts_conforming()
    test_validator_rejects_out_of_spec()
    test_bin_roundtrip_pure_python()
    test_bin_roundtrip_c_loader()
    test_technical_features_land_in_spec()
    test_technical_features_flat_series()
    test_technical_features_uptrend_and_clipping()
    test_full_feature_set_validates()
    print("all feature tests passed")
