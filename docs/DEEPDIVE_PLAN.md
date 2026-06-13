# Moray Deep-Dive Campaign — Plan

**Date:** 2026-06-13 · **Branch:** `rl-deep-dive-mapping` · Read-only exploration.
**Rule:** every finding lands in a Markdown doc at the repo root, house-style matching the existing
deep-dive set. Cite exact `file:line`. Mark low-confidence claims `[uncertain]`. No code changes,
no running anything that imports `alpaca_wrapper` in LIVE mode.

## Already done (don't redo)
- `REPO_MAP.md` — repo overview, subsystem map, ~12 RL tracks, experiment taxonomy.
- `REBUILD_RECOMMENDATIONS.md` — keep/discard + how to run autoresearch.
- `RL_AUTORESEARCH_DEEPDIVE.md` — `autoresearch_rl.py` presets/mutation/generalization_score.
- `RL_TRAIN_PPO_DEEPDIVE.md` — `train.py` PPO loop internals.
- `RL_C_ENV_DEEPDIVE.md` — `pufferlib_market` C sim (fills/reward/lag) + train/eval parity risk.

## What's still dark (this campaign)
Everything documented so far is the **RL research track**, which is *not in production*. The gaps
below cover the live money system, the eval/parity numbers, and the signal supply chain.

---

## Phase 1 — Live XGB trader + Alpaca safety layer  → `LIVE_TRADER_DEEPDIVE.md`
**Why first:** the only real-money system; governed by the CLAUDE.md HARD RULES; never opened.
**Targets:** `xgbnew/live_trader.py` (2425 L), `alpaca_wrapper.py` (2309 L),
`src/alpaca_singleton.py` (494 L), `scripts/deploy_live_trader.sh`, `alpacaprod.md`,
`tests/test_alpaca_singleton.py`.

**Questions to answer (cite file:line):**
1. **Decision loop:** how does `live_trader.py` go signal → order? What model artifact does it load,
   at what cadence, what's the symbol universe, and where does sizing come from?
2. **The single-writer lock:** trace `enforce_live_singleton` — fcntl lock path, the exit-42 path,
   `ALP_PAPER` bypass, `LIVE_WRITER_UNITS` registry, and how `deploy_live_trader.sh` proves the lock
   holder PID matches the new supervisor.
3. **Death-spiral guard:** `guard_sell_against_death_spiral` — the time-aware tolerance (≤8h 50bps /
   older 500bps), buy-price tracking JSON (3-day TTL), the override env var, and exactly which order
   paths route through it.
4. **Order placement:** `alpaca_order_stock` — limit vs market, retries, partial fills, cancel logic,
   and how it reconciles Alpaca's fills back into local state.
5. **XGB model:** where is the live champion trained (`xgbnew/`), what features, and does it share the
   16-feature Chronos2 vector with the C env or a different feature set? (parity question)
6. **Failure modes:** what crashes the loop on purpose (guard RuntimeErrors), what's caught, and how
   the supervisor restarts.

## Phase 2 — Eval-sim parity audit  → `EVAL_SIM_PARITY_DEEPDIVE.md`
**Why:** closes the `[uncertain]` flag from `RL_C_ENV_DEEPDIVE.md` §7 with hard numbers.
**Targets:** `pufferlib_market/intrabar_replay.py` (1059 L), `hourly_replay.py` (1845 L),
`evaluate_multiperiod.py` (497 L), `scripts/eval_100d.py`.

**Questions:**
1. Line-by-line map of the Python fill model vs C `resolve_limit_fill_price`/`close_position`/
   `open_long` — enumerate every divergence (entry-slippage locus, SL/TP, intrabar OHLC walk,
   decision_lag mechanism).
2. **Numerical parity test [optional, read-only sim runs]:** feed both the C env (via
   `eval_generic.py`/binding) and `simulate_daily_policy_intrabar` the *same* policy + data at
   slippage 0/5/10/20 bps; tabulate total_return / Sortino / num_trades deltas. Quantify the gap.
3. How `eval_100d.py` chooses backend and what the promotion gate actually enforces (the 27%/month
   median rule from CLAUDE.md HARD RULE 1).

## Phase 3 — Chronos2 forecast / feature supply chain  → `CHRONOS2_DEEPDIVE.md`
**Why:** CLAUDE.md calls it "important"; it produces the 16 features XGB *and* the C env consume.
**Targets:** `chronos2_trainer.py`, `chronos2_full_finetune.py`, `chronos2_lora_improvement_sweep.py`,
`chronos2_stock_augmentation.py`, `chronos2_linear_calibration.py`, `chronos2_objective.py`,
`retrain_chronos2_hourly_loras.py`, plus `export_data.py`'s forecast-cache consumption.

**Questions:**
1. Base model + LoRA setup: which Chronos2 checkpoint, LoRA rank/targets, what's frozen.
2. Pre-augmentation pipeline (`chronos2_stock_augmentation.py`) — what transforms, and are they
   applied at train only or inference too (leakage risk).
3. Calibration (`chronos2_linear_calibration.py`) — how raw forecasts become the p90/p10/confidence
   features in `export_data.py`.
4. Forecast cache provenance: how the parquet cache (`--forecast-cache-root`) is produced and whether
   train/live use the same forecasts (parity).
5. The objective/eval: what `chronos2_objective.py` optimises and how it connects to trading PnL.

## Phase 4 — binanceneural soft-fill training sim  → `BINANCENEURAL_SIM_DEEPDIVE.md`
**Why:** the soft-fill counterpart CLAUDE.md warns has lookahead bias; completes the sim picture.
**Targets:** `binanceneural/`, `binanceneural_archsweep/`,
`scripts/run_deep_binanceneural_sweep.py`, `scripts/run_binanceneural_robustness_sweep.py`,
`tests/test_binanceneural_execution.py`.

**Questions:**
1. The soft/sigmoid fill model + `fill_temperature` — where the differentiable fill leaks lookahead,
   and how CLAUDE.md's `fill_temperature=0.01` clamps the gradient leak.
2. What this sim is *for* (gradient-based training) vs the C binary sim (validation) — the handoff
   contract between them.
3. The robustness/arch sweeps — what they vary and how results feed model selection.

---

## Method & sequencing
- **Order:** Phase 1 → 2 → 3 → 4 (production risk first, then parity, then signal, then gradient sim).
  Phases are independent; each ships its own md so partial progress is durable.
- **Tooling:** targeted `rg`/`grep` + Read; use **Explore** subagents to fan out over large surfaces
  (Phase 1 alpaca_wrapper, Phase 3 chronos2's ~15 files) without burning main context.
- **Each phase:** synthesize in own words, cite `file:line`, end with an "Open questions" section.
- **After each phase:** add a one-line pointer to the file-memory `MEMORY.md` index + a MuninnDB
  engram (REST `:8475/api/engrams`, `concept`+`content`+`tags`; the `concept` field is required for
  later recall — an empty concept stores but won't activate).

## Cross-cutting questions the campaign should resolve
- **Feature parity:** does the live XGB champion consume the *same* 16-feature Chronos2 vector as the
  RL C env, or a divergent feature set? (Phase 1 Q5 + Phase 3.)
- **Forecast parity:** are train-time and live forecasts produced by the same Chronos2 path? (Phase 3.)
- **Sim parity:** quantified C-vs-Python eval gap (Phase 2) — the cleanest place for training Sortino
  to overstate deployable Sortino.
