"""LoRA-fine-tune Chronos-2 on Kraken bars + measure the effect (offline, MPS).

Trains a LoRA adapter (official Chronos-2 ``fit(finetune_mode='lora')``) on the 5
USD majors' close/high/low series, with an honest held-out evaluation: forecast
the last ``--eval-n`` bars of each symbol BEFORE and AFTER fine-tuning and compare
close-price MAE. The eval tail is excluded from training, so the comparison is
out-of-sample.

Heavy + offline (torch/chronos/MPS); never imported by ``make test``.

Run:  make finetune-kraken
   or: python3 -m forecast.finetune_kraken --train-steps 500
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from forecast.chronos import Chronos2LoRAForecaster
from sim.kraken_data import (
    KRAKEN_USD_MAJORS, _since_ms, fetch_binance_backfill, fetch_kraken, merge_series,
)


def _series_by_symbol(merged: dict) -> dict:
    out = {}
    for sym, bars in merged.items():
        if not bars:
            continue
        ts = sorted(bars)
        out[sym] = {
            "close": np.array([bars[t][3] for t in ts], dtype=np.float32),
            "high": np.array([bars[t][1] for t in ts], dtype=np.float32),
            "low": np.array([bars[t][2] for t in ts], dtype=np.float32),
        }
    return out


def _close_mae(forecaster, series: dict, eval_n: int, context_length: int) -> float:
    """Mean abs error of the close p50 forecast over the held-out tail (per symbol),
    forecasting ``eval_n`` steps from the train/eval boundary."""
    import torch

    errs = []
    for sym, s in series.items():
        c, h, l = s["close"], s["high"], s["low"]
        cut = len(c) - eval_n
        ctx_lo = max(0, cut - context_length)
        x = torch.tensor(
            np.stack([c[ctx_lo:cut], h[ctx_lo:cut], l[ctx_lo:cut]]),
            dtype=torch.float32,
        ).reshape(1, 3, cut - ctx_lo)
        q, _ = forecaster._pipe.predict_quantiles(
            x, prediction_length=eval_n, quantile_levels=[0.1, 0.5, 0.9])
        pred_close_p50 = np.array([float(q[0][0, k, 1]) for k in range(eval_n)])
        actual = c[cut:cut + eval_n]
        errs.append(float(np.mean(np.abs(pred_close_p50 - actual))))
    return float(np.mean(errs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LoRA fine-tune Chronos-2 on Kraken bars")
    ap.add_argument("--symbols", default=",".join(KRAKEN_USD_MAJORS))
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--backfill", choices=["none", "binance"], default="binance")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=24, help="held-out tail bars for MAE")
    ap.add_argument("--train-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--prediction-length", type=int, default=24)
    ap.add_argument("--out", default="forecast/cache/kraken_lora", help="adapter output dir")
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    since_ms = _since_ms(None, args.days)
    sources = []
    if args.backfill == "binance":
        sources.append(fetch_binance_backfill(symbols, since_ms))
    sources.append(fetch_kraken(symbols, since_ms))
    series = _series_by_symbol(merge_series(*sources))
    for sym, s in series.items():
        print(f"  {sym}: {len(s['close'])} bars")

    # Train inputs EXCLUDE the eval tail (honest out-of-sample comparison).
    eval_n = args.eval_n
    train_inputs = [
        np.stack([s["close"][:-eval_n], s["high"][:-eval_n], s["low"][:-eval_n]])
        for s in series.values()
    ]

    f = Chronos2LoRAForecaster()
    f.load_base()
    print(f"  chronos loaded on device={f._device}")

    mae_zero = _close_mae(f, series, eval_n, args.context_length)
    print(f"[eval] zero-shot close MAE over last {eval_n} bars: {mae_zero:.4f}")

    print(f"[train] LoRA fit: steps={args.train_steps} batch={args.batch_size} "
          f"lr={args.lr} ctx={args.context_length}")
    f.fit(train_inputs, prediction_length=args.prediction_length,
          num_steps=args.train_steps, batch_size=args.batch_size,
          learning_rate=args.lr, context_length=args.context_length,
          output_dir=args.out)

    mae_tuned = _close_mae(f, series, eval_n, args.context_length)
    print(f"[eval] fine-tuned close MAE over last {eval_n} bars: {mae_tuned:.4f}")
    delta = mae_zero - mae_tuned
    pct = 100.0 * delta / mae_zero if mae_zero else 0.0
    verdict = "IMPROVED" if delta > 0 else "no improvement"
    print(f"[result] {verdict}: MAE {mae_zero:.4f} -> {mae_tuned:.4f} "
          f"({pct:+.1f}%); adapter saved under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
