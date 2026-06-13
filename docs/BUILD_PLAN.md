# keel_trader — Build Plan (the rest of the implementation)

**Created:** 2026-06-13. The roadmap from "two crown jewels bootstrapped" to "a champion trading
on the gate." Companion to [`REBUILD_HANDOFF.md`](REBUILD_HANDOFF.md) (the brief / invariants) and the
deep-dive docs (the "why"). Each phase is **one or a few commits**, ends with a **green test/gate**,
and ports from the moray source where the handoff §3 copy-list says to.

## Invariants every phase must hold (the definition of done)
- `make test` stays green (golden fill fixture + safety spine + whatever the phase adds).
- **One fill engine.** Anything that simulates fills calls `sim/` — no second implementation, ever.
  New sim surfaces must extend `tests/test_fill_model.c` (or a Python parity test that diffs against
  it), per `EVAL_SIM_PARITY_DEEPDIVE.md` §5.
- **Safe values are defaults** (`decision_lag≥2`, binary fills, fee 10bps, fill_buffer 5bps,
  max_hold 6h); loosening needs an explicit flag.
- **No process that can win the live-writer lock** lands without an explicit, reviewed live-cutover
  step (Phase 8). Everything before that is paper / offline.
- Each phase commits its own tests; nothing is "done" until its proving gate runs.

## Status legend  ✅ done · 🔨 next · ⬜ planned

| Phase | What | Status |
|---|---|---|
| 0 | Skeleton + docs knowledge base | ✅ |
| 1 | The one fill engine + golden test (`sim/`, `tests/test_fill_model.c`) | ✅ |
| 2 | Safety spine (`core/`, paper-default) + golden tests | ✅ |
| 3 | **The gate** — build the binding, Python eval over the one sim | 🔨 |
| 4 | Data + one versioned feature spec + contract validator | ⬜ |
| 5 | The incumbent — xgb-daily model + paper decision loop | ⬜ |
| 6 | Forecaster seam — Chronos2 LoRA → parquet cache | ⬜ |
| 7 | RL track + autoresearch loop + leaderboards | ⬜ |
| 8 | Broker boundary + ops + **live cutover** (gated, reviewed) | ⬜ |

---

## Phase 3 — The gate (the judge) 🔨  *next*
**Goal:** "nothing trades until it clears the gate." A single trusted out-of-sample judge that runs a
policy through the **one** C sim at the production realism params and returns the promotion verdict.

**Deliverables**
- `sim/setup.py` + `make build-sim` → compile `sim/src/binding.c` into `keel_sim.*.so` (needs
  `pufferlib`/numpy headers — see Risks). Smoke: `binding.shared(data_path=...)` + a vec step.
- `research/eval.py` — port the load-bearing logic of moray's `scripts/eval_100d.py`:
  N random unseen windows, **slippage matrix {0,5,10,20} report-the-worst**, `decision_lag≥2`,
  binary fills, **fail-fast** (`--fail-fast-max-dd 0.20`, median-monthly-impossible early bail),
  promotion target **median monthly ≥ 0.27**. It must call the C sim (Phase 1), *not* a Python
  reimplementation (that divergence is the bug we're avoiding — `EVAL_SIM_PARITY_DEEPDIVE.md`).
- `research/policies.py` — trivial reference policies (always-flat, always-long-0, random-seeded) so
  the gate is testable without a trained model.

**Ports from:** `scripts/eval_100d.py` (gate logic), `pufferlib_market/binding.c` (already in `sim/`),
`pufferlib_market/setup.py`, `pufferlib_market/environment.py` (the `TradingEnvConfig` bridge).

**Proving gate:** `tests/test_gate.py` — (a) a flat policy yields ~0 return and is rejected;
(b) the gate's fill path reproduces `test_fill_model.c` values on a 1-bar fixture (parity guard);
(c) fail-fast triggers on a deliberately bad policy in < N windows.

**Risks / deps:** building the binding needs `pufferlib`'s `env_binding.h` + numpy dev headers. If
pufferlib is heavy to vendor, fall back to a thin standalone `c_step` driver (like moray's
`tests/smoke_c_env.c`) wrapped via `ctypes` — still the *same* C core, no second fill model.
Use `uv venv` + `uv pip install` (numpy, torch only when a real policy is needed).

**Definition of done:** `make build-sim && make test` green; a reference policy gets a verdict; the
parity guard ties the gate to the golden fill fixture.

---

## Phase 4 — Data + one versioned feature spec ⬜
**Goal:** kill the old repo's three-disjoint-featuresets problem. One spec, one validator, all
consumers downstream of it.

**Deliverables**
- `forecast/features.py` — a single `FEATURE_SPEC` (versioned: `v1`, ordered list + dtype) and a
  `validate_feature_contract(model, spec)` that refuses a model needing features the live path can't
  supply (port the pattern from `xgbnew/features.py`).
- `sim/export_data.py` already ported — wire it to the spec so the `.bin` and the model share one
  feature definition. Document the `.bin` format in `sim/README.md`.
- A small committed **sample dataset** (or a `make data` that regenerates one) so tests/backtests run
  without the full moray data tree. Large `.bin`/parquet stay git-ignored.

**Ports from:** `pufferlib_market/export_data.py`, `xgbnew/features.py` (the 14 daily cols + contract
validator), `RL_C_ENV_DEEPDIVE.md` §4 (obs layout) + `CHRONOS2_DEEPDIVE.md` (the forecast features).

**Proving gate:** `tests/test_features.py` — the spec round-trips through `export_data` → `.bin` →
`market_data_load` with identical values; the contract validator rejects an out-of-spec model.

---

## Phase 5 — The incumbent: xgb-daily + paper decision loop ⬜
**Goal:** start from the boring champion (it's what was actually live), score it on the Phase-3 gate.

**Deliverables**
- `models/xgb/train.py` — train the daily XGB on the Phase-4 features → a versioned artifact.
- `models/xgb/strategy.py` — the signal→pick→size logic (conviction filter, inverse-vol sizing,
  allocation), pure and unit-tested, **no broker calls**.
- `models/xgb/backtest.py` — run the strategy through the Phase-3 gate; record the verdict.
- `core/paper_runner.py` — a **paper-only** decision loop (imports the guard + `record_buy_price`,
  uses `core/config.PAPER=True`); deliberately NOT a live writer.

**Ports from:** `xgbnew/live_trader.py` (decision loop shape, sizing, the guard-call discipline — but
strip to paper), `xgbnew/features.py`, `LIVE_TRADER_DEEPDIVE.md` §2–4.

**Proving gate:** `tests/test_strategy.py` (sizing/conviction edge cases) + a backtest that clears or
honestly fails the gate (failing is fine — the point is the number is *real*).

---

## Phase 6 — Forecaster seam (optional features) ⬜
**Goal:** the Chronos2 LoRA → parquet-cache pattern, feeding research features. Keep the seam explicit.

**Deliverables**
- `forecast/chronos.py` — LoRA fine-tune wrapper (`amazon/chronos-2`, r=16/α=32, q/k/v/o) →
  `forecast/build_cache.py` writing `cache_root/h{H}/{sym}.parquet`. **Batch/offline only** — zero
  model latency in any trade path.
- Close (or at least measure) the **MAE-vs-PnL objective seam** flagged in `CHRONOS2_DEEPDIVE.md` §5:
  log whether a higher-Sharpe calibration ever prefers a worse-MAE checkpoint.
- Assert the **cache-timestamp ≤ labeled-bar** invariant in `build_cache.py` (the one lookahead
  vector, `CHRONOS2_DEEPDIVE.md` §6).

**Ports from:** `chronos2_trainer.py`, `build_hourly_forecast_caches.py`, `chronos2_objective.py`,
`chronos2_linear_calibration.py`.

**Proving gate:** `tests/test_forecast_cache.py` — cache build is leakage-safe (timestamp assertion)
and features land in the Phase-4 spec.

---

## Phase 7 — RL track + autoresearch loop ⬜
**Goal:** one PPO track + the search loop that generates champions, ranked honestly.

**Deliverables**
- `models/rl/train.py` — one PPO (port `pufferlib_market/train.py`); trains against the **one** C sim.
- `research/autoresearch.py` — preset pool + mutation grid + trial pipeline + **append-only
  leaderboard CSV** with manifest (git hash, seed, hardware), ranked on the overfit-penalized
  `generalization_score`. Every winner must clear the Phase-3 gate before it's believed.
- If a differentiable/soft sim is wanted for gradients, it is a **wrapper over the same fill
  arithmetic** as `sim/` with a `temperature→0 ⇒ binary` parity test — never a parallel stack
  (`BINANCENEURAL_SIM_DEEPDIVE.md` is the cautionary tale).

**Ports from:** `pufferlib_market/{train,autoresearch_rl}.py`, `RL_TRAIN_PPO_DEEPDIVE.md`,
`RL_AUTORESEARCH_DEEPDIVE.md`.

**Proving gate:** `tests/test_autoresearch.py` — leaderboard is append-only + reproducible (same
seed ⇒ same row); a trial that fails the gate is not promoted.

---

## Phase 8 — Broker boundary + ops + LIVE CUTOVER ⬜ (gated, reviewed)
**Goal:** the single HTTP write surface and a lock-verifying deploy — and only then, optionally, live.

**Deliverables**
- `core/broker.py` / `core/trading_server.py` — the broker boundary (port `src/trading_server/`).
  **Every order routes through the death-spiral guard** (the moray audit confirmed both live writers
  did; replicate that — no un-guarded sell helper reachable by a live unit).
- `ops/deploy_live_trader.sh` — port moray's: `LIVE_WRITER_UNITS` registry, stop-others, **verify the
  lock-holder PID == new supervisor PID**, append a history log.
- `ops/prod.md` — the live-state source of truth (what's running, deploy commands, marketsim scores).

**Live-cutover checklist (do not skip):**
1. The champion has cleared the Phase-3 gate on unseen data (median monthly ≥ 0.27, worst slip cell).
2. Paper run is clean for a meaningful window.
3. The live entry point imports the singleton and is added to `LIVE_WRITER_UNITS` **in the same
   commit** (HARD RULE 2).
4. If sharing an Alpaca account with any other system, both use the **same lock path + account name**.
5. `ALLOW_ALPACA_LIVE_TRADING=1` + `ALP_PAPER=0` set only in the supervised unit, never ad hoc.
6. `deploy_live_trader.sh` reports OK (lock-holder PID matches) before walking away.

**Proving gate:** `tests/test_safety_spine.py` extended to cover the broker's order paths;
`deploy_live_trader.sh` dry-run verifies the lock handshake.

---

## Suggested order & why
3 → 4 → 5 first: the gate + features + the boring champion get you a **measurable, honest baseline
trading on paper** with the smallest surface. 6 and 7 (forecaster, RL) are the research levers that
must *beat* that baseline on the gate to matter. 8 (live) is last and gated. Resist building 6/7
before 3/4/5 exist — that inversion is exactly how moray accreted 199 directories of unvalidated
research.

## One-line definition of "the system works"
A champion model, trained against the one fill engine, clears the gate on unseen windows at
`decision_lag≥2` / worst-slippage, runs clean on paper through the guarded broker, and is promoted to
a single live writer by a deploy that proves it holds the lock — with every fill in train, eval, and
backtest produced by the same code `tests/test_fill_model.c` pins.
