"""Issue #5 — the offline forecast cache's correctness guards.

Stdlib-only (no pytest/pandas): plain asserts, run via ``make test`` /
``PYTHONPATH=. python3 tests/test_forecast_cache.py``.

Pins the two load-bearing guards (the parquet write itself is offline-only):
  (a) the cache-timestamp ≤ labeled-bar leakage invariant accepts clean rows and
      rejects a row that peeked at (or past) its labeled bar;
  (b) the forecast features the cache feeds are exactly FEATURE_SPEC's 'forecast'
      kind (they land in the ONE spec);
  (c) MAE-vs-PnL objective divergence is detected and measured.
"""
from __future__ import annotations

from forecast.build_cache import (
    CacheLeakageError,
    Checkpoint,
    ForecastRow,
    assert_features_in_spec,
    assert_no_leakage,
    forecast_feature_names,
    measure_objective_divergence,
)
from forecast.features import FEATURE_SPEC

HOUR = 3600


def _row(ts: int, context_end: int, horizon: int) -> ForecastRow:
    return ForecastRow(
        symbol="SYM0", horizon=horizon, timestamp=ts, context_end=context_end,
        target_timestamp=ts + horizon * HOUR,
        predicted_close_p10=99.0, predicted_close_p50=100.0, predicted_close_p90=101.0,
        predicted_high_p50=101.5, predicted_low_p50=98.5,
    )


def test_leakage_accepts_clean_rows():
    rows = [_row(ts=t * HOUR, context_end=t * HOUR, horizon=1) for t in range(1, 6)]
    assert_no_leakage(rows)  # context_end == timestamp, target in future -> fine
    # context strictly before the bar is also fine
    rows2 = [_row(ts=t * HOUR, context_end=(t - 1) * HOUR, horizon=24) for t in range(1, 6)]
    assert_no_leakage(rows2)
    print("ok test_leakage_accepts_clean_rows")


def test_leakage_rejects_future_context():
    # context_end one hour AFTER the labeled bar -> the model peeked: must raise.
    bad = [_row(ts=10 * HOUR, context_end=11 * HOUR, horizon=1)]
    try:
        assert_no_leakage(bad)
    except CacheLeakageError as e:
        assert "context_end" in str(e)
    else:
        raise AssertionError("expected CacheLeakageError for future context")
    print("ok test_leakage_rejects_future_context")


def test_leakage_rejects_non_future_target():
    bad = ForecastRow(
        symbol="SYM0", horizon=1, timestamp=10 * HOUR, context_end=10 * HOUR,
        target_timestamp=10 * HOUR,  # not strictly in the future
        predicted_close_p10=99.0, predicted_close_p50=100.0, predicted_close_p90=101.0,
        predicted_high_p50=101.5, predicted_low_p50=98.5,
    )
    try:
        assert_no_leakage([bad])
    except CacheLeakageError as e:
        assert "target_timestamp" in str(e)
    else:
        raise AssertionError("expected CacheLeakageError for non-future target")
    print("ok test_leakage_rejects_non_future_target")


def test_forecast_features_land_in_spec():
    names = assert_features_in_spec()
    assert names == FEATURE_SPEC.names_of_kind("forecast")
    # all 8 forecast features (indices 0-7) and every one is in the spec
    assert len(names) == 8
    for n in names:
        assert n in FEATURE_SPEC.names
    assert forecast_feature_names() == names
    print("ok test_forecast_features_land_in_spec")


def test_objective_divergence_detected():
    # best MAE is 'a' (0.10) but best Sharpe is 'b' (1.5) -> divergence logged.
    cps = [
        Checkpoint("a", mae=0.10, sharpe=0.8),
        Checkpoint("b", mae=0.15, sharpe=1.5),
        Checkpoint("c", mae=0.20, sharpe=1.1),
    ]
    d = measure_objective_divergence(cps)
    assert d.best_mae_checkpoint == "a"
    assert d.best_sharpe_checkpoint == "b"
    assert d.diverged is True
    assert abs(d.mae_gap - 0.05) < 1e-9
    assert abs(d.sharpe_gap - 0.7) < 1e-9

    # when they agree, no divergence
    agree = [Checkpoint("x", mae=0.1, sharpe=2.0), Checkpoint("y", mae=0.3, sharpe=1.0)]
    d2 = measure_objective_divergence(agree)
    assert d2.diverged is False
    assert d2.best_mae_checkpoint == d2.best_sharpe_checkpoint == "x"
    print("ok test_objective_divergence_detected")


if __name__ == "__main__":
    test_leakage_accepts_clean_rows()
    test_leakage_rejects_future_context()
    test_leakage_rejects_non_future_target()
    test_forecast_features_land_in_spec()
    test_objective_divergence_detected()
    print("all forecast-cache tests passed")
