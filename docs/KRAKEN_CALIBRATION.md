# Kraken (crypto) gate calibration — rationale

K4 (#14). The out-of-sample gate (`research/eval.py`) and the sim fee were ported
with **equity-daily-rotation** numbers. This file records why each crypto knob was
re-tuned, so a later reviewer sees reasoning, not just changed constants.

> These values are **provisional**. They were chosen by reasoning about Kraken's
> microstructure, not yet confirmed against a real-data gate run (that needs
> `ccxt`/`xgboost` installed — see the manual commands below). Confirm/adjust them
> once the real `sim/data/kraken_market.bin` exists and re-pin the tests.

## What changed

| Knob | Equity (was) | Crypto (now) | Why |
|------|--------------|--------------|-----|
| `DEFAULT_FEE_RATE` (K3) | `0.0010` (10 bps) | `0.0026` (26 bps) | Kraken spot **taker** fee per leg. Round trip ≈ 52 bps of friction before slippage — the gate must not price rosy fills. |
| `DEFAULT_SLIPPAGES_BPS` | `(0,5,10,20)` | `(0,10,20,30)` | Market orders on the majors (esp. SOL/XRP/LTC) see wider adverse slippage than liquid equities. The **worst cell** drives promotion, so the stress is pushed to 30 bps. |
| `PROMOTION_TARGET_MEDIAN_MONTHLY` | `0.27` | `0.10` | 27%/mo at the **worst** slippage cell, after ~52 bps round-trip crypto friction, is not sustainable. 10%/mo worst-cell median is aggressive-but-defensible and still rejects mediocre policies. Lowering it is paired with *higher* fee + *wider* slippage, so the bar as a whole is **not** loosened. |
| `FAIL_FAST_MAX_DD` | `0.20` | `0.30` | Crypto pullbacks inside a ~1-month window routinely exceed 20%; a 0.20 limit false-fails decent long-only books. 0.30 tolerates normal swings while still bailing on a genuine blow-up. |
| `DEFAULT_WINDOW_STEPS` | `720` | `720` | Unchanged. 24/7 crypto → ~730 h/month, so a 720-step hourly window is still ≈ one month (`BARS_PER_MONTH = 730`). |

The death-spiral guard recalibration (single 300 bps volatility-aware tolerance,
equity overnight regime removed) is documented in K3 (`core/alpaca_singleton.py`).

## How to run the judges on the real Kraken data (offline / manual)

These need network + offline deps (`ccxt`, optionally `xgboost`) and the built C
sim. They are **not** part of `make test` (which stays synthetic + stdlib-only).

```bash
# 0. one-time: build the sim, install offline deps in a venv
make build-sim
pip install ccxt xgboost pandas numpy          # + torch/transformers/peft for forecasts

# 1. build the Chronos-2 forecast cache on Kraken bars (offline, MPS; ~20 min)
#    fetches 120d hourly OHLCV (Kraken public + Binance deep-history backfill),
#    runs Chronos-2 inference, writes h1/h24 parquet under forecast/cache/kraken.
make build-cache-kraken

# 2. fetch real OHLCV -> .bin and JOIN the forecasts (full FEATURE_SPEC; K1+K2)
python3 -m sim.kraken_data --days 120 --backfill binance \
        --forecast-cache forecast/cache/kraken --output sim/data/kraken_market.bin
#   (without --forecast-cache, forecast features 0-7 stay honest zeros)

# 3. run the gate + backtest + autoresearch on the real .bin (K4)
make gate-kraken
make backtest-kraken
make autoresearch-kraken
```

> Verified end-to-end (2026-06-13): cache built (2369 rows/sym/horizon over 120d),
> forecasts 0-7 joined and non-zero, gate runs on real forecasts. The reference /
> baseline policies are honestly REJECTED (fail-fast on >30% drawdown over the
> 120-day window) — promotion needs a real trained champion, not a baseline.

## LoRA fine-tune (optional) — keep zero-shot for now

`Chronos2LoRAForecaster.fit` wires Chronos-2's official `fit(finetune_mode='lora')`
(LoRA on the q/k/v/o attention linears). `make finetune-kraken` trains on the 5
majors' close/high/low and reports out-of-sample close-MAE on the held-out tail
BEFORE vs AFTER tuning.

Empirical finding (2026-06-13, 120d / 5 series, last-24-bar holdout MAE):

| Config | close MAE | vs zero-shot |
|--------|-----------|--------------|
| zero-shot | 32.82 | — |
| LoRA, lr 1e-4, 500 steps | 72.41 | **−120% (catastrophic)** |
| LoRA, lr 1e-5, 300 steps | 32.91 | −0.2% (neutral) |

Takeaway: zero-shot Chronos-2 is already strong; LoRA on this small in-domain set
is neutral-to-harmful (aggressive LR causes catastrophic forgetting; gentle LR
barely adapts in a few hundred steps). **The cache stays zero-shot.** Levers that
would actually help before re-trying: much more/longer history, more symbols, a
scale-normalized loss/metric (the MAE above is BTC-scale-dominated), the package
default `lr≈1e-6`, and validation-based early stopping. The fine-tune path is in
place for when that data exists.

Record each promoted champion's evidence in `ops/prod.md` (append, never
overwrite). A champion may only go to K6 (live) after it clears this gate on
**unseen** data and a clean K5 paper run.
