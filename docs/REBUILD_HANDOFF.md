# Rebuild Handoff — Build the Best Version (lessons + invariants + copy-list)

**Date:** 2026-06-13 · Distills the five deep-dive docs (`LIVE_TRADER`, `RL_C_ENV`,
`EVAL_SIM_PARITY`, `CHRONOS2`, `BINANCENEURAL`) into what a fresh agent needs to **build a clean
trading bot from scratch**, copying code where it earns its place. Companion to — and partial
**correction of** — `REBUILD_RECOMMENDATIONS.md` (keep/discard) and `REPO_MAP.md` (the map).

> Read order for the next builder: this doc → [`REBUILD_RECOMMENDATIONS.md`](REBUILD_RECOMMENDATIONS.md)
> (keep/discard) → the specific deep-dive when you touch that subsystem. The deep-dives are the "why"
> and hold the **bulk of the repo's documented knowledge**; this doc is the "what to do."

---

## Documentation map — where the repo knowledge actually lives

This handoff is a distillation; **the linked docs below carry the detailed findings, file:line cites,
and the reasoning.** Read the relevant one before touching that subsystem.

| Doc | Subsystem | What's in it |
|---|---|---|
| [`REPO_MAP.md`](REPO_MAP.md) | whole repo | Overview, subsystem map, ~12 RL tracks, experiment taxonomy, signal-vs-noise, open questions. **Start here for orientation.** |
| [`REBUILD_RECOMMENDATIONS.md`](REBUILD_RECOMMENDATIONS.md) | rebuild | Keep/discard table, target dir shape, how to run the autoresearch loop. |
| [`LIVE_TRADER_DEEPDIVE.md`](LIVE_TRADER_DEEPDIVE.md) | **live money** | The XGB daily daemon + the 3 Alpaca safety gates, singleton lock mechanics, death-spiral guard, deploy verification, feature set. (§3 copy-list draws from here.) |
| [`RL_C_ENV_DEEPDIVE.md`](RL_C_ENV_DEEPDIVE.md) | training sim | The C `trading_env.c`: action space, binary fills, reward shaping, obs layout, decision_lag, leverage. The ground-truth sim. |
| [`EVAL_SIM_PARITY_DEEPDIVE.md`](EVAL_SIM_PARITY_DEEPDIVE.md) | eval vs train | The train/eval fill divergence, **the empirical fill-parity run (§5)** proving the buffer gap + cost parity. Underpins §1 here. |
| [`CHRONOS2_DEEPDIVE.md`](CHRONOS2_DEEPDIVE.md) | forecaster | `amazon/chronos-2` + LoRA, augmentation (leakage-checked), calibration, the MAE-vs-PnL objective seam, parquet cache + lookahead vector. Underpins §5.1 here. |
| [`BINANCENEURAL_SIM_DEEPDIVE.md`](BINANCENEURAL_SIM_DEEPDIVE.md) | soft sim | The JAX differentiable sim, the sigmoid-fill lookahead mechanism, soft/binary switch, why a checkpoint isn't loadable by the C path. Underpins §2 + §5.2 here. |
| [`RL_AUTORESEARCH_DEEPDIVE.md`](RL_AUTORESEARCH_DEEPDIVE.md) | the loop | `autoresearch_rl.py`: preset pools, mutation grids, trial pipeline, the overfit-penalized `generalization_score` ranking formula. |
| [`RL_TRAIN_PPO_DEEPDIVE.md`](RL_TRAIN_PPO_DEEPDIVE.md) | PPO internals | `train.py` loss, GAE, advantage-norm modes (group-relative/GSPO), CUDA paths, stability guards. |
| [`DEEPDIVE_PLAN.md`](DEEPDIVE_PLAN.md) | meta | The campaign plan that produced the deep-dives — phase scoping + method, useful if you extend the exploration. |
| [`alpacaprod.md`](alpacaprod.md) / [`binanceprod.md`](binanceprod.md) | ops | What's actually running in production + deploy commands + marketsim scores. The source of truth for "what's live." |

---

## 0. The meta-lesson the old handoff buries: the *simple* model is what's live

Only **XGBoost-daily** trades real money (`xgbnew/live_trader.py`): 14 plain technical features
(returns/RSI/vol/ATR/dollar-vol/52w-range/day-of-week), **no Chronos2, no RL, no neural sim**. Every
sophisticated track — the PPO RL flagship, the Chronos2 forecasts, binanceneural's JAX sim, the ~20
LLM agents — is **research that has not beaten the boring incumbent into production**. The
daily-RL trader was even stopped (2026-04-30); Binance is paused.

**Build implication:** start from the boring champion + the gate. Make every sophistication *earn*
promotion against the gate on unseen data. Do not build the RL/forecast cathedral first and bolt
trading on later — that's exactly the residue this repo accumulated.

---

## 1. The one architectural rule that, broken, silently inflates every result

**ONE fill engine, used by BOTH training and validation, pinned by a golden parity test.**

The current repo has *four* fill implementations — C training env (`trading_env.c`), a Python
"intrabar" eval reimplementation (`intrabar_replay.py`), a JAX soft sim (`binanceneural`), and dead
C/C++ sims. They **do not agree**, and the disagreement favors the eval. I proved this by running the
real fill code from both sides over a grid ([`EVAL_SIM_PARITY_DEEPDIVE.md`](EVAL_SIM_PARITY_DEEPDIVE.md) §5):

- **Cost is at parity** (round-trip ≈ `2·(fee+slip)`; fold slippage into fee = price-shift, both legs).
- **But the promotable eval fills `fill_buffer` bps *better* than the C ground truth** — for the
  identical fill condition the C env fills at `open`, the Python intrabar fills at `open·(1−buffer)`.
  At the production 5 bps buffer that's a free 5 bps/leg the eval grants and live trading won't.

**Build implication:** write the fill model **once**. Expose it two ways — a hard/binary mode (truth)
and a differentiable/soft wrapper (gradients) over the *same* arithmetic — and ship a unit test that
asserts soft→hard agree as `temperature→0`, plus a fixture that pins fill price/cost at slip
{0,5,10,20}×buffer {0,5,20}. The repo's `REBUILD_RECOMMENDATIONS.md:22` claims the judge is a C++
binary sim; **it isn't** — the promotable `eval_100d --execution-granularity hourly_intrabar` path is
the Python reimplementation. Fix that in the rebuild: the judge must run the *same* core the policy
trained against.

---

## 2. Lookahead is structural in soft fills — design around it, don't trust around it

binanceneural's differentiable fill is `sigmoid((buy_price − low_t)/(close_t·temperature))`
(`jax_losses.py:59`). The fill probability at bar `t` uses **bar t's own realized low/high**, so the
training gradient teaches the policy to set its limit at the realized extreme — a fill it cannot get
live. **`decision_lag` on the observation does NOT fix this** (the obs lag and the fill leak are
different channels); lowering `fill_temperature` (5e-4→0.01) only damps the leak.

**Build implication:** a soft sim is a *gradient engine only*. Hard rules: (a) every model promotes on
**binary fills at `decision_lag≥2`**; (b) never report soft-fill Sortino as a result; (c) keep the
soft↔hard Sortino gap visible — it *is* the lookahead premium, and a healthy training setup keeps it
small.

---

## 3. Copy these verbatim — the crown jewels (hard to re-derive)

| Copy | Path | Why it's worth copying as-is |
|---|---|---|
| **Singleton + death-spiral guard** | `src/alpaca_singleton.py`, `src/alpaca_account_lock.py` | Three fail-closed gates: explicit-enable (`ALLOW_ALPACA_LIVE_TRADING=1`), fcntl single-writer lock (writes holder `{pid,host}` JSON), per-sell death-spiral guard with **time-aware** tolerance (50 bps intraday / 500 bps overnight after 8h). The overnight/intraday split encodes a real crash-loop incident — don't re-derive it. |
| **Lock-verifying deploy** | `scripts/deploy_live_trader.sh` | Stops every other registered writer, waits for the lock to clear, then **refuses to report OK unless the lock-holder PID == the new supervisor PID** (or descendant). `LIVE_WRITER_UNITS` registry is the source of truth. |
| **Guard tests** | `tests/test_alpaca_singleton.py` | Keep green; they're the executable spec of the safety layer. |
| **Binary-fill sim core** | `pufferlib_market/src/trading_env.c` + `include/trading_env.h` + `src/binding.c` | The ground-truth fill/reward/accounting. Make this the *only* sim; have the eval call it (not a Python twin). |
| **Data format** | `pufferlib_market/export_data.py` | The `.bin` packer (MKTD header + features + OHLCV). Simple, fast, mmap-friendly. |
| **The gate** | `scripts/eval_100d.py` | Fail-fast (`--fail-fast-max-dd 0.20`), slippage matrix (0/5/10/20, worst cell), median-monthly≥0.27 target, `decision_lag≥2` enforcement. **Fix:** point it at the C core. |
| **The loop** | `pufferlib_market/autoresearch_rl.py` + `*_leaderboard.csv` | Append-only leaderboard + manifest (git hash/seed/hardware) + overfit-penalized `generalization_score` ranking. Genuinely good reproducibility. |
| **Forecaster seam** | `chronos2_trainer.py` → `build_hourly_forecast_caches.py` → forecast-cache parquet | LoRA fine-tune → batch parquet cache → serve. Zero model latency in the trade hotpath. |
| **Live decision loop + feature contract** | `xgbnew/live_trader.py`, `xgbnew/features.py` | The incumbent. Note `features.py` already has a **feature-contract validator** that rejects models needing unsupported live features — copy that pattern (see §5). |
| **Broker boundary** | `src/trading_server/server.py` | HTTP single-writer surface; both live writers route every sell through the guard (verified). |

---

## 4. The production parameters are the contract — bake them in as defaults

From CLAUDE.md, confirmed across the deep-dives: `fee=10bps`, `margin=6.25%` (= the inverse of
`max_leverage`), `fill_buffer=5bps`, `max_hold=6h`, `decision_lag≥2`, `validation_use_binary_fills=True`,
`fill_temperature=0.01` (soft sim only), slippage test cells `{0,5,10,20}`, promotion target
**median monthly ≥ 27% on unseen windows, worst slippage cell**, fail-fast maxDD 0.20.

**Trap:** the soft-sim *config defaults* are lookahead-permissive (`decision_lag=0`, `fee=0`) — the
realism is imposed by overrides (`binanceneural/config.py`). In the rebuild make the **safe values the
defaults** and require an explicit flag to loosen them, so a naive run can't report fantasy Sortino.

---

## 5. Two seams to close (the current system leaves them open)

5.1 **Forecaster objective ≠ trading objective.** Chronos2 is selected on `pct_return_mae`
(`chronos2_objective.py:81`); the trade rule is a *separate* downstream linear calibration maximizing
Sharpe/Sortino (`chronos2_linear_calibration.py`). A model that minimizes MAE need not maximize PnL.
**Rebuild choice:** either train the forecaster against a PnL-aware/calibration-in-the-loop objective,
or keep the seam but make it explicit and measured (does a higher-Sharpe calibration ever prefer a
worse-MAE checkpoint? If so you're leaving money on the table by selecting on MAE).

5.2 **Three disjoint feature sets.** Live XGB uses 14 daily technical features; the RL C env uses 16
hourly Chronos2 features; binanceneural uses a 23-feature set of its own (different order, members,
*and* per-symbol vs flattened-multi-symbol obs topology — a binanceneural checkpoint is **not loadable**
by the C-eval path). **Rebuild choice:** one **versioned feature spec** with a contract validator
(xgbnew already has the pattern) feeding all consumers, so train/live/research cannot silently diverge.

---

## 6. Target shape (refines `REBUILD_RECOMMENDATIONS.md`)

~6 dirs, not 199: `core/` (safety spine + broker), `sim/` (the *one* fill engine + the gate that calls
it), `forecast/` (one LoRA forecaster + parquet cache), `models/` (xgb incumbent + one PPO),
`research/` (autoresearch loop + config-driven sweeps + append-only leaderboards), `ops/` (deploy +
prod docs). Collapse the hundreds of root one-offs (`sweep_*`, `train_v*`, `*_vs_chronos2`,
`quick_*`) into **one parametrized sweep entrypoint + a config file** — that sprawl is the single
biggest tax.

**Discard outright:** the dead sims (`cppsimulator`, `c_market_sim`, `market_sim_c`,
`marketsimulator`), superseded RL tracks (`pufferlibtraining{,2,3}`, `gpu_trading_env`, …), the LLM
agent zoo (keep one probe), and `binanceneural` *as a maintained codebase* — keep its soft-fill idea
as the differentiable wrapper over the one fill engine (§1), not a parallel stack.

---

## 7. Open empirical questions a rebuild should settle on day one

- **Policy-level C-vs-Python eval magnitude** — needs a trained checkpoint + built `.so`. *Moot if you
  follow §1* (one sim ⇒ nothing to diff). The fill-level gap is already quantified (§1).
- **Chronos2 cache-build timestamp vs labeled-bar alignment** — the one real lookahead vector in the
  forecast pipeline ([`CHRONOS2_DEEPDIVE.md`](CHRONOS2_DEEPDIVE.md) §6): forecasts must be generated from data available
  *at or before* the bar they label. Assert this in the cache builder.
- **soft↔binary Sortino gap** on a sample policy — instrument it as a standing metric (§2).

---

## 8. The one-paragraph version

Build the boring XGBoost-daily champion + the three-gate safety spine first, and copy that spine
verbatim — it's the crown jewel. Stand up **one** fill engine (binary for truth, a soft wrapper over
the *same* arithmetic for gradients) with a golden parity test, because the current repo's biggest
silent bug is that its eval sim fills `fill_buffer` bps better than its training sim. Promote nothing
that hasn't cleared the gate (median monthly ≥27% on unseen windows, worst of slippage {0,5,10,20},
`decision_lag≥2`, binary fills, fail-fast). Treat the forecaster as a cached, accuracy-optimized
component but close (or at least measure) the gap between its MAE objective and trading PnL, and feed
all models from one versioned feature spec. Everything else in the 199 directories is search residue —
git history, not code.

---
*Synthesis of the 2026-06-13 deep-dive campaign. Cites are to branch `rl-deep-dive-mapping`.*
