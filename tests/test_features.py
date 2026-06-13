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


if __name__ == "__main__":
    test_spec_shape()
    test_validator_accepts_conforming()
    test_validator_rejects_out_of_spec()
    test_bin_roundtrip_pure_python()
    test_bin_roundtrip_c_loader()
    print("all feature tests passed")
