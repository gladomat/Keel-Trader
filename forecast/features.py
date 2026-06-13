"""The ONE versioned feature spec + contract validator.

keel's invariant #3 (docs/REBUILD_HANDOFF.md §5.2): the old repo had *three* disjoint
feature sets — live XGB's 14 daily technicals, the RL C env's 16 hourly Chronos2
features, and binanceneural's 23-feature set — that silently diverged so a checkpoint
trained against one could not be served by another. Here there is exactly one
ordered, versioned ``FEATURE_SPEC`` and a ``validate_feature_contract`` that refuses
any model needing features the live path can't supply (the pattern moray's
``xgbnew/features.py`` already had).

All consumers import from here:
  - ``sim/export_data.py`` packs the ``.bin`` in this exact order (so the C sim and
    the model share one definition);
  - ``models/xgb`` and ``models/rl`` train on these names;
  - ``forecast/build_cache.py`` (Phase 6) registers its forecast columns here.

The spec is **append-only across versions**: never reorder or repurpose an index
within a version, or a packed ``.bin`` silently means something else to the C loader.
"""
from __future__ import annotations

from dataclasses import dataclass

PRICE_FEATURES = 5  # OHLCV, fixed by the .bin format (not part of FEATURE_SPEC)


@dataclass(frozen=True)
class FeatureField:
    name: str
    dtype: str          # numpy-style dtype string; the .bin is float32 throughout
    kind: str           # "forecast" (Chronos2-derived) or "technical" (price-derived)
    description: str = ""


@dataclass(frozen=True)
class FeatureSpec:
    version: str
    fields: tuple[FeatureField, ...]

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def n_features(self) -> int:
        return len(self.fields)

    def index(self, name: str) -> int:
        for i, f in enumerate(self.fields):
            if f.name == name:
                return i
        raise KeyError(f"feature {name!r} not in FEATURE_SPEC {self.version}")

    def names_of_kind(self, kind: str) -> list[str]:
        return [f.name for f in self.fields if f.kind == kind]


# ---------------------------------------------------------------------------
# FEATURE_SPEC v1 — ordering MUST match sim/export_data.py and the C obs layout
# (16 features per symbol; indices 0-7 forecast, 8-15 technical).
# ---------------------------------------------------------------------------
FEATURE_SPEC = FeatureSpec(
    version="v1",
    fields=(
        FeatureField("chronos_close_delta_h1", "float32", "forecast", "Chronos2 close p50 delta, 1h horizon"),
        FeatureField("chronos_high_delta_h1", "float32", "forecast", "Chronos2 high p50 delta, 1h"),
        FeatureField("chronos_low_delta_h1", "float32", "forecast", "Chronos2 low p50 delta, 1h"),
        FeatureField("chronos_close_delta_h24", "float32", "forecast", "Chronos2 close p50 delta, 24h"),
        FeatureField("chronos_high_delta_h24", "float32", "forecast", "Chronos2 high p50 delta, 24h"),
        FeatureField("chronos_low_delta_h24", "float32", "forecast", "Chronos2 low p50 delta, 24h"),
        FeatureField("forecast_confidence_h1", "float32", "forecast", "inverse p90-p10 spread, 1h"),
        FeatureField("forecast_confidence_h24", "float32", "forecast", "inverse p90-p10 spread, 24h"),
        FeatureField("return_1h", "float32", "technical", "1-bar pct return, clipped +/-0.5"),
        FeatureField("return_24h", "float32", "technical", "24-bar pct return, clipped +/-1.0"),
        FeatureField("volatility_24h", "float32", "technical", "rolling std of 1h returns, 24h"),
        FeatureField("ma_delta_24h", "float32", "technical", "(close-ma24)/ma24, clipped +/-0.5"),
        FeatureField("ma_delta_72h", "float32", "technical", "(close-ma72)/ma72, clipped +/-0.5"),
        FeatureField("atr_pct_24h", "float32", "technical", "ATR(24)/close, clipped 0..0.5"),
        FeatureField("trend_72h", "float32", "technical", "72-bar pct return, clipped +/-1.0"),
        FeatureField("drawdown_72h", "float32", "technical", "(close-rollmax72)/rollmax72, clipped -1..0"),
    ),
)

# Convenience re-exports (single source of truth for every consumer).
FEATURE_NAMES: list[str] = FEATURE_SPEC.names
FEATURES_PER_SYM: int = FEATURE_SPEC.n_features


class FeatureContractError(ValueError):
    """A model requires features the active FEATURE_SPEC can't supply."""


def model_required_features(model) -> list[str]:
    """Best-effort extraction of the feature names a model was trained on.

    Accepts: a plain list/tuple of names; an object exposing ``feature_names``,
    ``feature_names_in_`` (sklearn), or ``feature_names`` via xgboost Booster
    (``get_booster().feature_names``); or a dict with a ``"features"`` key.
    """
    if model is None:
        raise FeatureContractError("model is None; cannot validate feature contract")
    if isinstance(model, (list, tuple)):
        return list(model)
    if isinstance(model, dict) and "features" in model:
        return list(model["features"])
    for attr in ("feature_names", "feature_names_in_"):
        names = getattr(model, attr, None)
        if names is not None:
            return list(names)
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        names = getattr(booster(), "feature_names", None)
        if names:
            return list(names)
    raise FeatureContractError(
        "could not determine the model's required features; expose `feature_names` "
        "or pass a list of names"
    )


def validate_feature_contract(model, spec: FeatureSpec = FEATURE_SPEC,
                              *, strict_order: bool = False) -> list[str]:
    """Raise FeatureContractError unless every feature the model needs is in ``spec``.

    Returns the validated required-feature list on success. With
    ``strict_order=True`` the model's features must also appear as a prefix of the
    spec in the same order (the live path packs features positionally into the
    ``.bin``, so order is load-bearing for anything that reads the obs vector).
    """
    required = model_required_features(model)
    spec_names = spec.names
    spec_set = set(spec_names)

    missing = [f for f in required if f not in spec_set]
    if missing:
        raise FeatureContractError(
            f"model needs features absent from FEATURE_SPEC {spec.version}: {missing}. "
            f"The live path cannot supply these — retrain against the spec or extend it."
        )

    if strict_order:
        expected = spec_names[: len(required)]
        if required != expected:
            raise FeatureContractError(
                f"model feature order does not match FEATURE_SPEC {spec.version} prefix.\n"
                f"  model:    {required}\n  expected: {expected}"
            )
    return required
