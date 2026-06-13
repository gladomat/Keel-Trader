# Chronos2 Deep Dive — the forecast/feature supply chain for the RL track

**Date:** 2026-06-13 · **Branch:** `rl-deep-dive-mapping` · Read-only exploration.
Phase 3 of `DEEPDIVE_PLAN.md`. Sibling of the existing deep-dive set.

**Scope:** `chronos2_trainer.py`, `chronos2_full_finetune.py`, `chronos2_stock_augmentation.py`,
`chronos2_linear_calibration.py`, `chronos2_objective.py`, `chronos2_lora_improvement_sweep.py`,
`retrain_chronos2_hourly_loras.py`, `build_hourly_forecast_caches.py`. CLAUDE.md calls Chronos2
"important" — it produces the forecast features the RL C env consumes.

---

## 0. TL;DR + the one fact that re-frames the whole campaign

- **Chronos2 = Amazon `amazon/chronos-2` (Chronos-2, vendored under `chronos-forecasting/`), LoRA-tuned.**
  Not Chronos-Bolt, not a T5 you train from scratch. `DEFAULT_MODEL_ID = "amazon/chronos-2"`
  (`chronos2_trainer.py:24`), loaded via `Chronos2Pipeline.from_pretrained`
  (`:284`) and adapted with PEFT LoRA `r=16, alpha=32, targets=(q,k,v,o), dropout=0.05`
  (`:85-88, 292-296`), then `merge_and_unload()` after training.
- **Forecasts are BATCH-precomputed into a parquet cache**, never called online in the trade hotpath.
  `build_hourly_forecast_caches.py` writes `cache_root/h{horizon}/{symbol}.parquet` with
  `predicted_close_p10/p50/p90_h{h}` (+ high/low variants); `export_data.py` reads that cache to pack
  the 16-feature `.bin` (`export_data.py:67-79, 118-122`).
- **Re-framing the campaign:** these Chronos2 features feed the **RL C env** (hourly crypto/mixed) —
  **not the live XGB daily trader**, which *zeros* its Chronos columns (`live_trader.py:512-516`, see
  `LIVE_TRADER_DEEPDIVE.md`). So the whole Chronos2 → forecast-cache → `.bin` → RL pipeline is part of
  the **research track**, not the live money path. Chronos2 quality bottlenecks the RL experiments,
  not production PnL today.
- **Objective optimizes forecast accuracy (`pct_return_mae`), not trading PnL** — trading economics
  are bolted on downstream by a separate linear calibration grid-search.

---

## 1. Base model + LoRA (Q1)

- Base: `amazon/chronos-2` via `chronos.chronos2.{Chronos2Model,Chronos2Pipeline,Chronos2Trainer}`
  (`chronos2_trainer.py:314-316`) — the actual Chronos-2 stack vendored in `chronos-forecasting/`.
- LoRA (`LoraConfig`, `:292`): `r=16`, `lora_alpha=32` (2×r), `dropout=0.05`, `target_modules=(q,k,v,o)`
  — attention projections only; everything else frozen. `chronos2_full_finetune.py` is the heavier
  variant (also `merge_and_unload`, alpha=2×r).
- **`context_length=1024`, `prediction_length=1`** (`:77-78`) — single-step forecaster over a 1024-bar
  context (the sub-agent's "512" was wrong; the default is 1024). Multi-horizon (h1/h24) features come
  from running/aggregating the cache at different horizons, not a multi-step head.

## 2. Training data (Q2)

Daily stocks (`trainingdata/`) and hourly crypto (`binance_spot_hourly/`), loaded by
`chronos2_stock_augmentation.py` (`load_ohlc_csv`). OHLC as multivariate channels. Train/val/test split
holds out the last ~60+60 bars for early-stop + OOS. Optional **return-space variants** (level series
duplicated in %-return space for scale-invariance) and **sliding daily offsets** (hourly→daily at
7 phase alignments → ~7× data). `[uncertain]` exact default symbol universe; depends on CLI.

## 3. Pre-augmentation — leakage check passes (Q3)

`chronos2_stock_augmentation.py` applies ~17 augmentations **train-only** (hard gate
`if self.mode != DatasetMode.TRAIN: return` `:261`) → **no inference leakage**. Default-on:
amplitude jitter (`exp(N(0,0.30))`), relative noise (`±0.2%`), time dropout (NaN 2% of context),
return variants. Default-off (opt-in): detrend, channel dropout, time-warp, outlier/gap/trend/earnings/
structural-break injection, vol-regime shift, mean-reversion, washout, parabolic. All operate on the
**context (past) only**, so even when enabled they don't peek at the prediction target.

## 4. Calibration — forecasts → trade signal (Q4)

`chronos2_linear_calibration.py` is a post-hoc layer turning raw quantiles into a buy/sell signal.
From `q10/q50/q90` of the close (and high/low/open/step-2) it derives
`predicted_return, uncertainty=(q90−q10)/prev, skewness, midpoint_return, step2_return, open_return`
(`:913-929`). The trade `signal` is a **weighted linear combo** of these (+IC term
`predicted_return/uncertainty`) vs `buy_threshold/sell_threshold` (`CalibrationParams.apply:147-171`).
A multi-phase **grid search** (coarse thresholds → fine → signal-weight → auxiliary weights →
ultra-fine, `:939-1083`) maximizes **Sharpe / Sortino / Calmar** with an optional uncertainty filter.
So: Chronos2 learns forecasts; calibration learns the trade rule on top.

## 5. Objective — accuracy, not PnL (Q5)

`chronos2_objective.py:81-84`:
```
objective = pct_return_mae + smoothness_weight·pct_return_mae_smoothness − direction_bonus·direction_accuracy
```
Defaults `smoothness_weight=0, direction_bonus=0` ⇒ pure **`pct_return_mae`** (mean
|pred_return − actual_return|). The LoRA sweep (`chronos2_lora_improvement_sweep.py`) promotes a config
only if it beats baseline `pct_return_mae` by a margin (~5%). **This is the key seam:** the model is
selected for *forecast* accuracy, while trading is optimized *separately* in calibration — a model that
minimizes MAE need not maximize Sharpe. Deliberate, but a real objective-misalignment to keep in mind.

## 6. Forecast cache + parity (Q6,Q7)

`build_hourly_forecast_caches.py` reads per-symbol hyperparams (`hyperparams/chronos2/hourly/{sym}.json`),
runs **sliding-window** inference (context strictly *before* each prediction target — no
generation-time lookahead), and writes `cache_root/h{horizon}/{sym}.parquet`. Both
`retrain_chronos2_hourly_loras.py` and `export_data.py` read the **same** cache path → **path parity**
between train and downstream. Inference is **batch/offline** (nightly-style job); the trade-time path is
pure parquet lookups + linear calibration, **zero model latency in the hotpath**.

**The one real lookahead risk** `[uncertain]`: parity holds only if the cache row at timestamp `T` was
generated using data available *at or before* `T`. The generation loop is past-only, so the risk is not
in the code but in **scheduling/timestamp alignment** — if a cache is rebuilt over a window with
later-arriving data and then an eval replays that window, the forecasts encode mild future information.
Not provable from source; worth a check that cache build timestamps precede the bars they label.

---

## 7. Open questions / follow-ups
- Confirm cache-build timing vs labeled-bar timing (§6) — the only lookahead vector here.
- Quantify the objective seam (§5): does a higher-Sharpe calibration ever prefer a *worse*-MAE LoRA?
  If so, selecting LoRAs on MAE is leaving trading performance on the table.
- Map which `.bin` datasets (`crypto34_hourly`, `mixed40_daily`, `stocks20_daily`) are built from which
  forecast cache — i.e. which RL experiments actually depend on Chronos2 vs use raw price features only.
- Sibling Chronos dirs (`chronospnltrader`, `cutechronos`, `binancechronossolexperiment*`,
  `stockagentopus_chronos2`) are unexplored — likely experiment forks; `[uncertain]` if any is live-adjacent.

---
*Read-only synthesis; line cites as of branch `rl-deep-dive-mapping`, 2026-06-13. No code modified.*
