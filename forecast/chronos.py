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

    def __init__(self, lora: LoRAConfig | None = None, train: TrainConfig | None = None):
        self.lora = lora or LoRAConfig()
        self.train_cfg = train or TrainConfig()
        self._model = None
        self._merged = False

    @staticmethod
    def _require(mod: str):
        try:
            return __import__(mod)
        except ImportError as e:  # pragma: no cover - offline-only deps
            raise RuntimeError(
                f"{mod} is required for offline forecasting but is not installed. "
                "Chronos2 fine-tune/inference runs in the batch job, not the trade path."
            ) from e

    def load_base(self):  # pragma: no cover - needs torch/transformers
        """Load the base Chronos2 model and wrap it with a PEFT LoRA adapter."""
        self._require("torch")
        transformers = self._require("transformers")
        peft = self._require("peft")

        model = transformers.AutoModel.from_pretrained(self.lora.base_model)
        lora_cfg = peft.LoraConfig(
            r=self.lora.r,
            lora_alpha=self.lora.alpha,
            lora_dropout=self.lora.dropout,
            target_modules=list(self.lora.target_modules),
            bias="none",
        )
        self._model = peft.get_peft_model(model, lora_cfg)
        self._merged = False
        return self._model

    def fit(self, dataset) -> "Chronos2LoRAForecaster":  # pragma: no cover - offline
        """Fine-tune the LoRA adapter on a time-series dataset (offline)."""
        if self._model is None:
            self.load_base()
        raise NotImplementedError(
            "wire to your Chronos2 trainer loop; this seam fixes the LoRA config "
            "(r=16/α=32, q/k/v/o) and the offline contract"
        )

    def merge_and_save(self, out_dir: str | Path):  # pragma: no cover - offline
        """Merge the LoRA adapter into the base weights and save for fast inference."""
        if self._model is None:
            raise RuntimeError("nothing to merge: call load_base()/fit() first")
        merged = self._model.merge_and_unload()
        self._merged = True
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out))
        return out

    def predict_quantiles(self, context, horizon: int,
                          quantiles: Sequence[float] | None = None):  # pragma: no cover
        """Return quantile forecasts ``horizon`` steps ahead (offline batch use)."""
        if self._model is None:
            raise RuntimeError("model not loaded; call load_base()/fit() first")
        raise NotImplementedError("wire to Chronos2 sampling/quantile decode")
