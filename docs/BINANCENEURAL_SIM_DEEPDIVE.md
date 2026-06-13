# binanceneural Soft-Fill Sim Deep Dive — the differentiable training-gradient sim

**Date:** 2026-06-13 · **Branch:** `rl-deep-dive-mapping` · Read-only exploration.
Phase 4 (final) of `DEEPDIVE_PLAN.md`. Completes the sim picture: **C binary = ground truth /
validation; binanceneural soft JAX = gradients / training.**

**Scope:** `binanceneural/jax_losses.py`, `marketsimulator.py`, `config.py`, `jax_trainer.py`,
`data.py`, `forecasts.py`. CLAUDE.md: *"soft sigmoid fills have lookahead bias — never trust training
sortino alone"*, *"binanceneural soft sim is for training gradients only"*,
*"fill_temperature=0.01 (reduced from 5e-4 to limit gradient leakage)"*.

---

## 0. TL;DR

- **The whole point of binanceneural is differentiability.** Hard binary fills (`low ≤ buy_price`) have
  zero gradient w.r.t. the limit price — you can't backprop a policy through them. binanceneural
  replaces the step with a **sigmoid** so JAX autodiff can flow reward → fill → limit price → policy.
- **The soft fill IS the lookahead.** Buy-fill probability at bar `t` is
  `sigmoid((buy_price − low_t)/(close_t·temperature))` (`jax_losses.py:59-69`). It uses **bar t's own
  realized low**. So `∂(fill_prob)/∂(buy_price) > 0` differentiably rewards setting `buy_price` just
  above the realized low — the policy *learns to fill at the extreme it shouldn't have known*. That's
  the gradient leak CLAUDE.md warns about.
- **`fill_temperature` controls leak sharpness, not its existence.** Smaller temp ⇒ sharper sigmoid ⇒
  stronger, more precisely-targeted gradient toward the realized low/high. Raising it 5e-4 → **0.01**
  (`config.py:171`) softens (≈20× flatter) the gradient to *limit* leakage — but never removes it.
- **Binary validation is mandatory and built in.** `validation_use_binary_fills=True`
  (`config.py:235`); training uses `simulate_hourly_trades` (soft), validation swaps to
  `simulate_hourly_trades_binary` (hard) (`jax_trainer.py:183-189`). The binary path is the same
  family as the C env's binary fill — that's why a model must clear the C binary sim at `lag≥2` before
  deploy (no soft Sortino is trusted).
- **Research/training only.** Consumes the same Chronos2 forecasts as the C env; not a live writer.
  Binance is paused per production memory.

---

## 1. The soft fill model (Q1)

`jax_losses.py:59-82` — two sigmoids, scale-normalized by `close`:
```python
def approx_buy_fill_probability(buy_price, low_price, close_price, temperature=5e-4):
    scale = clip(abs(close_price), 1e-4)
    score = (buy_price - low_price) / (scale * temperature)
    return sigmoid(score)                 # buy fills as buy_price rises above the bar low
def approx_sell_fill_probability(sell_price, high_price, close_price, temperature=5e-4):
    score = (high_price - sell_price) / (scale * temperature)
    return sigmoid(score)                 # sell fills as sell_price drops below the bar high
```
Used in `_simulate_core` (`:250-256`): when `probabilistic=True`, fill = these sigmoids (market-order
entries forced to prob 1); executed qty = `intensity · capacity · fill` so a *fractional* fill is
differentiable. (The eager `marketsimulator.py` is the non-JAX, hard-fill reference path:
`buy_fill = low <= buy_price·(1−buffer)`, `:210, 456`.)

## 2. The lookahead mechanism, precisely (Q2)

At the **execution bar `t`**, fill probability depends on `low_t / high_t / close_t` — values that are
only known *after* bar `t` completes. Because the fill is a smooth function of `buy_price`, the
training gradient `∂reward/∂buy_price` is nonzero and points toward `buy_price ≈ low_t` (max fill at
min cost). Over training the policy's price head **learns to hug the realized intrabar extreme** — a
fill it could not have achieved live, where it must commit the limit before the bar reveals its low.
That inflates training PnL/Sortino vs reality.

- **Temperature's role:** `score = gap / (close · temp)`. At `temp=5e-4` the sigmoid is near-step, so
  the gradient is a sharp spike exactly at the realized low → aggressive, precise leak. At `temp=0.01`
  the transition is ~20× wider → the policy can't tune its limit as tightly to the extreme → leak is
  damped. The repo deliberately *raised* temp (5e-4→0.01) trading a little gradient signal for a lot
  less lookahead.
- **`decision_lag` only fixes the input side.** Lagging the *observation* (so the policy can't read
  the execution bar's features when deciding) does **not** remove the fill-side leak — the soft fill
  at bar `t` still differentiably uses `low_t/high_t`. This is exactly why input-lag alone is
  insufficient and **binary-fill validation is required**, not optional.

## 3. Binary-fill mode + the train/val switch (Q3)

`_simulate_core` `probabilistic=False` branch (`jax_losses.py:258-263`):
`buy_fill = ((low ≤ buy_threshold) & (intensity>0))`, `sell_fill = (high ≥ sell_threshold)` — a hard
step, **no temperature, no gradient through the fill**. Dispatch:
- `jax_trainer.py:183-189`: `if train: simulate_hourly_trades(..., temperature=fill_temperature)` else
  `simulate_hourly_trades_binary(...)`.
- `config.py:235`: `validation_use_binary_fills=True` (default). Same flag honored in
  `hf_trainer_bridge.py`.

So every model is trained on soft fills but **scored on hard fills** — and then must additionally clear
the *external* pufferlib C binary sim at `lag≥2` (the true ground truth) before promotion.

## 4. Loss / gradient flow (Q4)

`compute_hourly_objective` (`jax_losses.py:454+`): differentiable **Sortino + `return_weight`·annual
return** (+ optional smoothness penalty):
```
downside_std = sqrt(mean(clip(-r,0)^2)+eps);  sortino = mean(r)/downside_std · sqrt(periods)
score = sortino + return_weight · (mean(r)·periods)
```
Gradient path: policy → `(buy_price, sell_price, intensity)` → soft `fill_prob` (sigmoid) → executed
qty → portfolio returns `r` → Sortino. All smooth, so JAX backprops end-to-end — *through the leaky
fill*. The JAX policy lives in `jax_policy.py`; trainer in `jax_trainer.py`.

## 5. Relation to the C env + Chronos2 (Q5)

- **Same forecast source, separate everything else.** binanceneural reads the **parquet forecast
  cache directly** (`forecasts.py:61` `read_parquet`, `data.py:271` `forecast_cache_root`) — it never
  touches the C env's `.bin`/`MktdData`. Both sit downstream of the same Chronos2 parquet cache but
  assemble features independently.
- **Feature vectors do NOT match — not a byte-match, not loadable [RESOLVED].** Compared
  `build_default_feature_columns`/`build_feature_frame` (`data.py:493-587`) against the C `.bin`
  layout (`export_data.py:18-33`):
  - **Count/order differ:** C = 16 fixed, chronos-first; binanceneural = **variable** (base-first +
    per-horizon chronos block + horizon-pair spreads → **23** with horizons {1,24}).
  - **Shared (~9):** `return_1h, return_24h, volatility_24h, chronos_{close,high,low}_delta_{h1,h24}`.
  - **C-only (7):** `forecast_confidence_h1/h24, ma_delta_24h/72h, atr_pct_24h, trend_72h, drawdown_72h`.
  - **binanceneural-only (13+):** `return_4h/48h/168h, range_pct, volume_z, hour_sin/cos, dow_sin/cos,
    chronos_asymmetry{_h}, chronos_range_width{_h}, forecast_delta_spread_h1_h24`.
  - **Different obs topology:** binanceneural is **per-symbol** (`marketsimulator.py:72,97` — each
    symbol simulated independently, combined under one shared cash balance), a per-asset torch/JAX
    policy. The C env packs **all symbols into one flattened obs** (`obs = S·F + 5 + S`). So a
    binanceneural checkpoint is **not loadable** by the C-eval/`eval_100d` path (different feature
    set, count, order, *and* obs shape).
- **The "two-sim contract" is methodological, not a checkpoint handoff.** binanceneural (JAX, soft,
  per-symbol, 23-feat) is the *gradient engine*; pufferlib C (binary, multi-symbol, 16-feat) is the
  *truth oracle*. They are **parallel research stacks** sharing only the upstream Chronos2 parquet
  cache — the discipline "validate neural ideas under binary fills at `lag≥2`" is a *practice*, not a
  literal load of the same policy object across sims. A C-env policy is born and validated entirely
  within the pufferlib stack; binanceneural is a separate differentiable-training playground.
- **Not production.** No Alpaca writer here; Binance paused per `moray-production-reality`. Pure
  research/training.

## 6. Realism params — defaults are lookahead-permissive (Q6)

`config.py` **defaults are NOT the production realism settings** — they're train-friendly and must be
overridden:

| param | config default | production (CLAUDE.md) |
|---|---|---|
| `maker_fee` | 0.0 | 10 bps |
| `fill_buffer_pct` | 0.0005 (5 bps) | 5 bps ✓ |
| `decision_lag_bars` | **0** (same-bar!) | **≥2** |
| `max_leverage` | 1.0 | (margin 6.25%) |
| `margin_annual_rate` | 0.0 | configured |
| `max_hold_hours` | 24 (PolicyConfig) | 6h |
| `fill_temperature` | 0.01 | 0.01 ✓ |
| `validation_use_binary_fills` | True | True ✓ |

⇒ The default config (`fee=0, decision_lag=0`, soft fills) is the **maximally optimistic** training
setting; the production realism (fee 10bps, lag≥2, max_hold 6h, binary validation) is imposed by
overrides. A naive run on defaults would report wildly inflated Sortino — precisely the trap
CLAUDE.md guards against.

---

## 7. Open questions / follow-ups
- ~~Does binanceneural's feature vector match the C env's `.bin`?~~ **[RESOLVED — §5]** No: 23 vs 16
  features, different order, different members, different obs topology (per-symbol vs flattened
  multi-symbol). Not loadable across stacks; they share only the upstream Chronos2 parquet cache.
- Quantify the soft→binary Sortino gap on a sample checkpoint (run train-mode then binary-mode on the
  same policy) — the size of that gap *is* the lookahead premium.
- Verify the `kernels/` dir (custom CUDA?) — referenced but not opened; may hold a faster fill kernel.

---
*Read-only synthesis; line cites as of branch `rl-deep-dive-mapping`, 2026-06-13. No code modified.*
