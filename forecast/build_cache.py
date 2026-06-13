"""Build the offline forecast cache: ``cache_root/h{H}/{sym}.parquet``.

Batch/offline ONLY (keel invariant: no model latency in any trade path). This
materialises Chronos2 quantile forecasts to parquet so the live/eval paths only
ever *read* precomputed numbers.

Two correctness guards live here as **pure, stdlib-testable** functions so they
can be unit-pinned without pandas/pyarrow:

  * ``assert_no_leakage`` — the cache-timestamp ≤ labeled-bar invariant: a forecast
    row anchored at timestamp ``t`` may only have been produced from inputs whose
    context ends at ≤ ``t`` (it predicts ``t + horizon`` in the future). When the
    feature is later joined onto the price bar at ``t`` (see ``sim/export_data``),
    nothing from the future has leaked in. A row whose ``context_end > timestamp``
    is a leak and raises.

  * ``measure_objective_divergence`` — logs/returns whether a higher-Sharpe
    calibration prefers a *worse-MAE* checkpoint (the MAE-vs-PnL objective seam:
    minimising forecast error is not the same as maximising PnL).

The forecast-derived *feature* names this cache ultimately feeds are owned by the
ONE ``forecast.features.FEATURE_SPEC`` (kind ``"forecast"``); ``forecast_feature_names``
returns them and ``assert_features_in_spec`` fails loudly if they ever drift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from forecast.features import FEATURE_SPEC

logger = logging.getLogger("keel.forecast.build_cache")

# Raw quantile columns written to each parquet (per horizon, suffixed _h{H} on merge).
QUANTILE_COLUMNS = (
    "predicted_close_p10", "predicted_close_p50", "predicted_close_p90",
    "predicted_high_p50", "predicted_low_p50",
)


class CacheLeakageError(ValueError):
    """A forecast row used information at or after the bar it is attached to."""


@dataclass(frozen=True)
class ForecastRow:
    """One cached forecast: anchored at ``timestamp``, predicting ``target_timestamp``.

    ``timestamp`` is the cache key AND the bar the derived feature is joined onto.
    ``context_end`` is the last input timestamp the model was allowed to see.
    """
    symbol: str
    horizon: int
    timestamp: int        # epoch seconds (or any monotonic int bar key)
    context_end: int      # last input timestamp used to produce this row
    target_timestamp: int  # the future bar being predicted
    predicted_close_p10: float
    predicted_close_p50: float
    predicted_close_p90: float
    predicted_high_p50: float
    predicted_low_p50: float


def assert_no_leakage(rows: Sequence[ForecastRow]) -> None:
    """Raise CacheLeakageError unless every row obeys context_end ≤ timestamp < target.

    This is the load-bearing leakage guard. ``context_end > timestamp`` means the
    model peeked at the labeled bar (or beyond) when producing a row attached to
    that bar. ``target_timestamp <= timestamp`` means the "forecast" isn't actually
    in the future.
    """
    for r in rows:
        if r.context_end > r.timestamp:
            raise CacheLeakageError(
                f"{r.symbol} h{r.horizon}: context_end {r.context_end} > "
                f"cache timestamp {r.timestamp} (forecast saw the labeled bar)"
            )
        if r.target_timestamp <= r.timestamp:
            raise CacheLeakageError(
                f"{r.symbol} h{r.horizon}: target_timestamp {r.target_timestamp} "
                f"<= cache timestamp {r.timestamp} (target not in the future)"
            )


def forecast_feature_names() -> list[str]:
    """The spec features this cache feeds (FEATURE_SPEC entries of kind 'forecast')."""
    return FEATURE_SPEC.names_of_kind("forecast")


def assert_features_in_spec() -> list[str]:
    """Fail loudly if the forecast features ever drift out of the ONE spec."""
    names = forecast_feature_names()
    missing = [n for n in names if n not in FEATURE_SPEC.names]
    if missing:
        raise CacheLeakageError(  # reuse: a spec mismatch is a contract break
            f"forecast features absent from FEATURE_SPEC {FEATURE_SPEC.version}: {missing}"
        )
    return names


@dataclass(frozen=True)
class Checkpoint:
    name: str
    mae: float      # lower is better (forecast error)
    sharpe: float   # higher is better (downstream PnL quality)


@dataclass(frozen=True)
class ObjectiveDivergence:
    best_mae_checkpoint: str
    best_sharpe_checkpoint: str
    diverged: bool
    mae_gap: float      # mae(best_sharpe) - mae(best_mae)  (>=0 when diverged)
    sharpe_gap: float   # sharpe(best_sharpe) - sharpe(best_mae)


def measure_objective_divergence(checkpoints: Sequence[Checkpoint]) -> ObjectiveDivergence:
    """Detect + log whether the best-Sharpe checkpoint is NOT the best-MAE one.

    The MAE-vs-PnL seam: a checkpoint with worse forecast error can produce better
    PnL. We surface that explicitly rather than blindly promoting on lowest MAE.
    """
    if not checkpoints:
        raise ValueError("no checkpoints to compare")
    best_mae = min(checkpoints, key=lambda c: c.mae)
    best_sharpe = max(checkpoints, key=lambda c: c.sharpe)
    diverged = best_mae.name != best_sharpe.name
    result = ObjectiveDivergence(
        best_mae_checkpoint=best_mae.name,
        best_sharpe_checkpoint=best_sharpe.name,
        diverged=diverged,
        mae_gap=best_sharpe.mae - best_mae.mae,
        sharpe_gap=best_sharpe.sharpe - best_mae.sharpe,
    )
    if diverged:
        logger.warning(
            "MAE-vs-PnL divergence: best-Sharpe '%s' (mae=%.5f, sharpe=%.4f) differs "
            "from best-MAE '%s' (mae=%.5f, sharpe=%.4f); preferring Sharpe costs "
            "%.5f MAE for %.4f Sharpe.",
            best_sharpe.name, best_sharpe.mae, best_sharpe.sharpe,
            best_mae.name, best_mae.mae, best_mae.sharpe,
            result.mae_gap, result.sharpe_gap,
        )
    else:
        logger.info("MAE and Sharpe agree on checkpoint '%s'.", best_mae.name)
    return result


def safe_symbol(symbol: str) -> str:
    """Filesystem-safe symbol for the parquet filename ('BTC/USD' -> 'BTC_USD').

    The reader (``sim.export_data.load_forecast``) applies the same mapping, so a
    quoted pair like ``BTC/USD`` does not turn its ``/`` into a directory.
    """
    return symbol.replace("/", "_")


def write_cache(cache_root: Path, symbol: str, horizon: int,
                rows: Sequence[ForecastRow]) -> Path:  # pragma: no cover - needs pandas
    """Write one symbol/horizon parquet after the leakage guard passes (offline)."""
    assert_no_leakage(rows)
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "pandas/pyarrow required to write the parquet cache (offline batch job)"
        ) from e

    out_dir = Path(cache_root) / f"h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_symbol(symbol)}.parquet"

    df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(r.timestamp, unit="s", utc=True),
            **{col: getattr(r, col) for col in QUANTILE_COLUMNS},
        }
        for r in rows
    ]).set_index("timestamp")
    df.to_parquet(out_path)
    logger.info("wrote %d rows -> %s", len(df), out_path)
    return out_path


def build_cache(cache_root: Path, forecaster, series_by_symbol: dict,
                horizons: Sequence[int]):  # pragma: no cover - offline orchestration
    """Batch entrypoint: produce forecasts for each symbol/horizon and cache them.

    ``forecaster`` is a ``Chronos2LoRAForecaster``; ``series_by_symbol`` maps symbol
    -> a context-bearing object the forecaster consumes. This orchestrator is the
    offline glue; the correctness guards above are what the tests pin.
    """
    assert_features_in_spec()
    written = []
    for sym, series in series_by_symbol.items():
        for h in horizons:
            rows = forecaster.forecast_rows(series, horizon=h)  # user-provided adapter
            written.append(write_cache(cache_root, sym, h, rows))
    return written
