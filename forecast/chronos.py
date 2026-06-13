"""Chronos2 LoRA forecaster wrapper — batch/offline ONLY.

Keel invariant: there is **zero model latency in any trade path**. The forecaster
is never called live; it runs in a batch job (``forecast/build_cache.py``) that
materialises quantile forecasts to parquet, and the live/eval paths only ever read
that cache. This module is the explicit seam for that offline fine-tune.

It LoRA-fine-tunes ``amazon/chronos-2`` (r=16, α=32, targeting the attention
``q/k/v/o`` projections) and can merge the adapter back for fast inference. Heavy
deps (torch/transformers/peft) are imported lazily so the stdlib test suite can
import this module for introspection without them installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LoRAConfig:
    base_model: str = "amazon/chronos-2"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Attention projection modules the adapter targets (q/k/v/o).
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class TrainConfig:
    context_length: int = 512
    horizons: tuple[int, ...] = (1, 24)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    learning_rate: float = 1e-4
    num_steps: int = 2000
    batch_size: int = 32
    seed: int = 0


class Chronos2LoRAForecaster:
    """Thin offline wrapper around a LoRA-adapted Chronos2 model.

    All heavy imports happen inside methods so importing this class is cheap and
    dependency-free; the methods raise a clear error if the offline deps are
    missing rather than failing at module import.
    """

    # FEATURE_SPEC forecast quantiles (p10/p50/p90) + the horizon step indices.
    QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)
    HOUR_SECONDS = 3600

    def __init__(self, lora: LoRAConfig | None = None, train: TrainConfig | None = None):
        self.lora = lora or LoRAConfig()
        self.train_cfg = train or TrainConfig()
        self._pipe = None      # the loaded chronos BaseChronosPipeline
        self._device = None

    @staticmethod
    def _require(mod: str):
        try:
            return __import__(mod)
        except ImportError as e:  # pragma: no cover - offline-only deps
            raise RuntimeError(
                f"{mod} is required for offline forecasting but is not installed. "
                "Chronos2 inference runs in the batch job, not the trade path."
            ) from e

    def _resolve_device(self):  # pragma: no cover - needs torch
        torch = self._require("torch")
        if torch.backends.mps.is_available():
            return "mps"   # Apple Silicon (handoff: bf16, no CUDA)
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_base(self):  # pragma: no cover - needs torch/chronos + model download
        """Load Chronos-2 via the official pipeline and move it onto the device.

        Zero-shot is enough to materialise the forecast cache; the LoRA fine-tune
        (``fit``) is an optional accuracy improvement, not a prerequisite.
        """
        self._require("torch")
        chronos = self._require("chronos")
        self._device = self._resolve_device()
        pipe = chronos.BaseChronosPipeline.from_pretrained(self.lora.base_model)
        try:
            pipe.model = pipe.model.to(self._device)
        except Exception:
            self._device = "cpu"
        self._pipe = pipe
        return pipe

    def fit(self, dataset) -> "Chronos2LoRAForecaster":  # pragma: no cover - offline
        """Optional LoRA fine-tune. Zero-shot inference is wired and sufficient
        for the forecast cache, so this remains an opt-in accuracy step."""
        if self._pipe is None:
            self.load_base()
        raise NotImplementedError(
            "LoRA fine-tune is optional; zero-shot predict_quantiles/forecast_rows "
            "are implemented and build the cache. Wire a trainer here to improve MAE."
        )

    def predict_quantiles(self, context, horizon: int,
                          quantiles: Sequence[float] | None = None):  # pragma: no cover
        """Quantile forecast of a single (close) series ``horizon`` steps ahead.

        ``context`` is a 1-D sequence of past values. Returns the final-step
        quantile vector (one value per requested quantile level).
        """
        torch = self._require("torch")
        if self._pipe is None:
            self.load_base()
        qlevels = list(quantiles or self.QUANTILE_LEVELS)
        x = torch.tensor([list(context)], dtype=torch.float32).reshape(1, 1, len(context))
        q, _mean = self._pipe.predict_quantiles(
            x, prediction_length=horizon, quantile_levels=qlevels)
        # q is a list[Tensor]; element shape (n_variates, pred_len, n_quantiles).
        final = q[0][0, horizon - 1]
        return [float(v) for v in final]

    def forecast_rows(self, series: dict, horizon: int, *,
                      context_length: int = 512, chunk: int = 16,
                      quantiles: Sequence[float] | None = None):  # pragma: no cover
        """Produce leakage-safe :class:`ForecastRow`s for one symbol/horizon.

        ``series`` = ``{"symbol", "timestamp" (epoch SECONDS), "close", "high",
        "low"}`` (equal-length lists, time-ascending). For each anchor bar ``i``
        with at least ``context_length`` history, the context ends AT bar ``i``
        (``context_end == timestamp``) and predicts ``i + horizon`` — so
        ``build_cache.assert_no_leakage`` holds. close/high/low are forecast
        jointly (Chronos-2 is multivariate) and batched for throughput.
        """
        torch = self._require("torch")
        if self._pipe is None:
            self.load_base()
        from forecast.build_cache import ForecastRow

        ts = series["timestamp"]
        close, high, low = series["close"], series["high"], series["low"]
        sym = series.get("symbol", "SYM")
        n = len(close)
        qlevels = list(quantiles or self.QUANTILE_LEVELS)
        step = horizon - 1
        anchors = list(range(context_length - 1, n))
        rows = []
        for s in range(0, len(anchors), chunk):
            batch = anchors[s:s + chunk]
            x = torch.tensor(
                [[close[i - context_length + 1:i + 1],
                  high[i - context_length + 1:i + 1],
                  low[i - context_length + 1:i + 1]] for i in batch],
                dtype=torch.float32,
            )  # (B, 3, context_length)
            q, _mean = self._pipe.predict_quantiles(
                x, prediction_length=horizon, quantile_levels=qlevels)
            for k, i in enumerate(batch):
                e = q[k]  # (3 variates, pred_len, n_quantiles); 0=close 1=high 2=low
                cp, hp, lp = e[0, step], e[1, step], e[2, step]
                t = int(ts[i])
                rows.append(ForecastRow(
                    symbol=sym, horizon=horizon, timestamp=t, context_end=t,
                    target_timestamp=t + horizon * self.HOUR_SECONDS,
                    predicted_close_p10=float(cp[0]), predicted_close_p50=float(cp[1]),
                    predicted_close_p90=float(cp[2]),
                    predicted_high_p50=float(hp[1]), predicted_low_p50=float(lp[1]),
                ))
        return rows
