# RL C Market-Sim Deep Dive — `pufferlib_market` (the training "ground truth")

**Date:** 2026-06-13 · **Branch:** `rl-deep-dive-mapping` · Read-only exploration.
Sibling of `REPO_MAP.md`, `REBUILD_RECOMMENDATIONS.md`, `RL_AUTORESEARCH_DEEPDIVE.md`,
`RL_TRAIN_PPO_DEEPDIVE.md`. This one cracks open the **C env** the flagship PPO trains against.

Files: `pufferlib_market/include/trading_env.h` (205 L), `src/trading_env.c` (852 L),
`src/binding.c` (299 L), `tests/smoke_c_env.c`, `export_data.py`, `environment.py`.

CLAUDE.md calls this *"pufferlib C sim with binary fills is ground truth."* The Python PPO never
sees raw PnL — **all fills, accounting, and reward shaping happen in C** (`c_step`). The Python side
only normalises observations (Welford `RunningObsNorm`) and samples a `Categorical` action.

---

## 0. TL;DR (the load-bearing facts)

- **One position at a time, whole-portfolio.** `AgentState.position_sym` is a *single* int: `-1`
  flat, `0..S-1` long symbol s, `S..2S-1` short symbol s (`trading_env.h:67`). There is **no
  multi-asset book** — the agent holds exactly one symbol or cash. A new target auto-closes the old.
- **Discrete action factored as (side, symbol, allocation_bin, level_bin).**
  `num_actions = 1 + 2·S·alloc_bins·level_bins` (`binding.c:200`, `trading_env.h:150`).
- **Fills are binary limit fills** off the bar's OHLC (`resolve_limit_fill_price`,
  `trading_env.c:383`). `fill_buffer_bps` = pessimism gate, `fill_slippage_bps` = adverse haircut,
  `fee_rate` on both legs. The "soft sigmoid fills" CLAUDE.md warns about are **not in this C file**
  — they live in the *training-gradient* sim (`binanceneural`), not here. This C env is hard binary.
- **`decision_lag` is implemented in the *observation*, not the execution.** Agent sees features
  from `t-lag`, trades at bar `t` (`fill_observations`, `trading_env.c:174-176`). Default 2;
  `<2` emits a `UserWarning` (`binding.c:154`). This is the anti-lookahead guarantee.
- **Reward = clipped per-step % equity change, then 7 optional shaping penalties** (`c_step`
  `:684-760`). Sortino/drawdown/win-rate are accumulated into `Log` only at episode end (`:809-831`).
- **⚠ Train/eval are DIFFERENT code.** Training = this C `c_step`. The promotable eval
  (`eval_100d.py --execution-granularity hourly_intrabar`) is a **pure-Python reimplementation**
  (`intrabar_replay.py`) with divergences (see §7). The C env *is* reachable from eval only via the
  non-promotable `--execution-granularity daily` path. This is a real parity risk.

---

## 1. Action semantics (Q1)

**Layout** (`trading_env.h:105-115`, decoded `c_step:587-668`):

```
action 0                       → go flat (close any open position)
action 1 .. side_block         → LONG   actions
action side_block+1 .. 2·side_block → SHORT actions
   where side_block = S · alloc_bins · level_bins
```

Each side action index (after subtracting 1, and the short block offset) factors as
(`c_step:617-622`):

```
target_sym = idx / (alloc_bins·level_bins)
rem        = idx % (alloc_bins·level_bins)
alloc_idx  = rem / level_bins
level_idx  = rem % level_bins
```

- **allocation_pct** = `(alloc_idx+1)/alloc_bins`, clamped `[0.01,1.0]` (`action_allocation_pct:415`).
  With `alloc_bins=5` → 20/40/60/80/100 % of `cash·max_leverage`.
- **level_offset_bps** = `(2·level_idx/(level_bins-1) − 1)·action_max_offset_bps`, i.e. a symmetric
  limit offset in `[−max,+max]` bps around the bar **open** (`action_level_offset_bps:426`).
  `level_bins=1` ⇒ offset 0 (market-ish at open).

**Production default is the degenerate 1×1 grid** (`environment.py:30-32`, `binding.c:87-95`:
`action_allocation_bins=1, action_level_bins=1, action_max_offset_bps=0`). So the live action space
is just `1 + 2S` (flat / long-each / short-each at 100 % alloc, no limit offset). The
alloc/level bins are an *autoresearch* knob, not the default flagship.

**Re-selecting the held position = hold** (`c_step:628`: `if position_sym == target_pos_id →
hold_hours++`). Switching symbols closes-then-opens in one step (`:638-663`).

---

## 2. Fill model — the "binary fills" ground truth (Q2)

`resolve_limit_fill_price` (`trading_env.c:383-413`) is the core. Per order at bar `t`, symbol `s`:

1. **Reference = bar OPEN** (`open_long:440`, `open_short:474`, `close_position:344` all use
   `P_OPEN`). Rationale in comments: "order placed at start of bar."
2. **Target** = `open · (1 + level_offset_bps/1e4)` (`open_long:444`).
3. **Pessimistic acceptance gate** (`:394-401`) — market must *overshoot* the target by
   `fill_buffer_bps` before the fill is accepted:
   - buy: accept only if `low ≤ target·(1 − buffer)`
   - sell: accept only if `high ≥ target·(1 + buffer)`
   - `buffer=0` ⇒ classic "did the bar touch the target" semantics.
4. **Adverse slippage** (`:403-411`): buys fill at `target·(1+slip)`, sells at `target·(1−slip)`;
   if the slipped price exits `[low,high]` the fill is **rejected** (agent stays flat that step).
5. **Fee on both legs**: entry `denom = fill·(1+fee_rate)` (`open_long:455`); exit `effective_fee =
   fee_rate + slippage_bps/1e4` applied as a haircut on the close price (`close_position:356-364`).
   *Note the slippage is double-counted-ish by design:* on the limit fill it shifts the price, and on
   the **close** it's folded into `effective_fee`. Entry uses `fill_slippage` via the price shift;
   exit uses it via `effective_fee`.

**`fill_probability`** (`c_step:649-653`): independent Bernoulli gate per *open* — if
`fast_rand_float() > fill_probability`, the order is dropped and the agent goes flat that step.
Default 1.0 (always fill). Models liquidity, not adverse selection.

**`max_hold_hours`** (`c_step:563-573`): before action decode, if the open position's `hold_hours ≥
max_hold_hours` and the symbol is tradable, force-close at bar open. CLAUDE.md's 6h cap maps here.
If untradable, defers to next tradable bar.

**Tradable mask** (`is_tradable:193`): per `[t][s]` uint8; **only present for file version ≥2**.
`export_data.py` currently writes `VERSION = 1` (`export_data.py:50`) with **no mask**, so in the
shipped data `is_tradable` always returns 1 — the closed-market handling (`c_step:601-635`) is dead
code for v1 `.bin` files. `[uncertain]` whether any v2 file with a mask is actually produced
elsewhere; the exporter in-repo doesn't.

**No soft/sigmoid fills here.** Grep of `trading_env.c` shows no `fill_temperature`, no sigmoid,
no softmax over fills. The lookahead-biased soft fills CLAUDE.md warns about belong to the
`binanceneural` training-gradient sim, *not* this C env. So training-on-C and the warning are
consistent: C is the hard binary ground truth.

---

## 3. Reward construction (Q3)

Per step (`c_step:684-774`), in order:

1. `ret = (equity_after − equity_before)/equity_before` (`:686-688`). Equity marked at bar
   `t_new = offset+step` close (`compute_equity:226`, uses `P_CLOSE`).
2. `reward = ret · reward_scale` (default 10), then **clipped to ±reward_clip** (default 5)
   (`:710-715`).
3. **cash_penalty** (default 0.01): subtracted every step the agent is flat *and* any market open
   (`:718-720`). Opportunity-cost pressure to stay invested.
4. **drawdown_penalty** (default 0): `−penalty·current_drawdown` when below peak (`:723-725`).
5. **downside_penalty** (default 0): `−penalty·ret²` when `ret<0` (`:729-731`) — raises Sortino by
   punishing the downside leg.
6. **smooth_downside_penalty** (+temperature, default 0.02): softplus proxy for `max(0,−ret)`,
   squared, differentiable around 0 (`:735-749`). The numerically-stable softplus branches at ±20.
7. **trade_penalty** (default 0): `−penalty·trade_events` where each open/close counts (`:752-754`).
   Anti-churn; does **not** touch cash accounting.
8. **smoothness_penalty** (default 0): `−penalty·(ret−prev_ret)²` (`:757-760`). Steadier PnL curve.

Borrow fee on shorts/leveraged longs is deducted from **cash and equity_after** *before* `ret` is
computed (`borrow_fee:247`, applied `c_step:678-682`), so it shows up as realised drag in the reward.

All defaults at `binding.c:98-123`. The flagship's actual penalty mix is set by autoresearch presets
(see `RL_AUTORESEARCH_DEEPDIVE.md`), not these defaults.

**Episode-end metrics** (`c_step:809-831`) accumulate into the pufferlib `Log` struct
(total_return, sortino, max_drawdown, num_trades, win_rate, avg_hold_hours). Sortino is annualised:
`mean_ret/downside_dev·√periods_per_year` (`:811-818`, `ppy` default 8760 hourly). These are
*reporting* aggregates — they do **not** feed the per-step reward.

---

## 4. Observation layout (Q4)

`obs_size = S·F + 5 + S` (`binding.c:199`, `trading_env.h:168`). `F = features_per_sym` (16 in v1;
header-driven so v3+ could be 20, `market_data_load:77`). Built by `fill_observations:160`:

- `obs[s·F + 0..F-1]` — the symbol's **lagged** feature vector at `t_obs = t − decision_lag`
  (memcpy'd straight from the `.bin`, `:184`).
- `obs[S·F + 0]` = `cash/INITIAL_CASH`
- `obs[S·F + 1]` = `position_value/INITIAL_CASH` (signed; negative for short)
- `obs[S·F + 2]` = `unrealised_pnl/INITIAL_CASH`
- `obs[S·F + 3]` = `hold_hours/max_steps`
- `obs[S·F + 4]` = `step/max_steps` (episode progress)
- `obs[S·F + 5 .. 4+S]` = one-hot position (**+1** long, **−1** short, 0 flat) (`:214-219`)

Portfolio valuation in the obs uses **lagged** prices (`t_obs`, `:194`) — what a trader would see
before the current bar closes. Equity for *reward* uses **current** close (`t_new`). Consistent
anti-lookahead.

**The 16 features** (`export_data.py:18-33`, packed from Chronos2 + price):
`0-2` chronos close/high/low Δ h1, `3-5` chronos close/high/low Δ h24,
`6-7` forecast_confidence h1/h24 (from p90/p10 spread), `8-9` return 1h/24h,
`10` volatility_24h, `11-12` ma_delta 24h/72h, `13` atr_pct_24h, `14` trend_72h, `15` drawdown_72h.
These are *deltas/ratios* (scale-free), then Welford-normalised on the Python side.

**`.bin` format** (`export_data.py:248-273`, `market_data_load:42-147`): 64-byte header
(`<4sIIIII40s`: magic `MKTD`, version, num_symbols, num_timesteps, features_per_sym, price_features,
pad) → `S·16B` symbol names → `float32[T][S][F]` features → `float32[T][S][5]` OHLCV prices →
(v2+) `uint8[T][S]` tradable mask. The whole file is `malloc`'d and read once per worker
(`my_shared`, `binding.c:34`), shared read-only across all vec envs.

---

## 5. Leverage / shorts / margin (Q5)

- **Long sizing**: `buy_budget = cash · max_leverage · allocation_pct` (`open_long:452`).
  `max_leverage` default 1.0 ⇒ no leverage; the 6.25 % production *margin* is **not** a separate
  field — leverage is the inverse knob. `[uncertain]` whether 6.25 % margin maps to
  `max_leverage=16` somewhere in autoresearch presets, or is only enforced in production daemons.
- **Shorts**: `open_short:469` sets `position_sym = S+sym`, credits `qty·fill·(1−fee)` to cash;
  equity then subtracts `qty·price` (`compute_equity:236-242`). Closing buys back at
  `qty·price·(1+effective_fee)` (`close_position:358-360`).
- **Borrow cost** (`borrow_fee:247-273`): shorts charged on full notional `qty·price·apr/ppy` per
  step; leveraged longs charged on the borrowed portion `pos_val·(1−1/lev)`. Default
  `short_borrow_apr=0` ⇒ free shorting unless configured.
- **No per-symbol margin/liquidation engine** — single position, equity can't be split. Bankruptcy
  ends the episode at `equity < 1 % of INITIAL_CASH` (`c_step:780`).

---

## 6. Decision-lag implementation (Q6) — the anti-lookahead guarantee

`require_production_decision_lag` floors `decision_lag≥2` on the Python side; the C side enforces it
in **observation indexing only**:

- `fill_observations:174-176`: `t_obs = t − lag`, clamped ≥0. The agent's features come from
  `lag` bars *before* the bar it acts on.
- Execution (`open_long`/`open_short`/`close_position`) always uses bar `t = data_offset + step`
  (current bar). So **info is `lag` bars stale relative to the execution bar** → the policy cannot
  peek at the bar it trades into.
- `binding.c:148-158`: default 2; `<1` clamped to 1; `<2` raises a loud `UserWarning`
  ("lookahead bias; production must use >=2"). So a silent lookahead default can't re-enter prod.

**Subtlety / possible concern** `[uncertain]`: the lag is applied to *features* but the one-hot
position and portfolio scalars in the obs use the *agent's true current* state, and execution prices
are bar-`t` OPEN. That's correct (state is known; you only lag the market signal). But note the lag
is a fixed shift, not a queue of pending orders — there's no modelling of an order placed at `t−lag`
filling at `t`. The agent simply acts on stale features at the current bar. Adequate for
anti-lookahead; not a literal latency/queue model.

---

## 7. Train vs eval parity (the important cross-cutting finding)

`scripts/eval_100d.py` has two execution backends (`--execution-granularity`):

| backend | code | uses C `c_step`? | promotable? |
|---|---|---|---|
| `hourly_intrabar` (**default**) | `pufferlib_market/intrabar_replay.py::simulate_daily_policy_intrabar` | **No — pure Python/numpy** | **Yes** (default gate) |
| `daily` (legacy) | `fp4/bench/eval_generic.py` → `pufferlib_market.binding.vec_step` | **Yes — the real C env** | No, needs `--allow-daily-promotion` |

So **the promotable eval does not run the training sim.** `intrabar_replay.py` reimplements the fill
model in Python (`_resolve_limit_fill_price`, `_open_long/_short`, `_close_position` in
`hourly_replay.py`). Confirmed divergences from C (sub-agent, cite `intrabar_replay.py`):

1. **~~Entry slippage~~ [CORRECTED in `EVAL_SIM_PARITY_DEEPDIVE.md` §1].** The original claim here
   ("Python under-models entry slippage") is **wrong**. `eval_100d` folds slippage into `fee_rate`
   (`eval_100d.py:991`), which is charged on **both** legs, so round-trip cost ≈ `2·(fee+slip)` —
   the **same magnitude** as C. The real residual is that C bakes slip into the *recorded entry
   price* while Python keeps `entry_price` slip-free (charges it via fee), which only shifts SL/TP/
   death-spiral keying by a few bps. See Phase-2 doc for the full corrected analysis, plus a genuine
   **buffer-fill sign divergence** (Python fills `~fill_buffer` bps more favorably than C).
2. **Intrabar path / stop-loss / take-profit.** The Python sim walks intrabar hourly OHLC with
   gap-handling, SL/TP (`intrabar_replay.py:497-522`); the C env has **none of that** — it's
   bar-granular with only `max_hold_hours`. Eval exercises logic training never saw.
3. **decision_lag locus.** C lags the *observation features*; the Python eval applies delay between
   policy query and execution in `make_policy_fn` (`evaluate_multiperiod.py`). Same intent, different
   mechanism — worth a parity test.

**Recommendation for a rebuild:** make the promotable eval call the *same* C core as training (the
`daily`/`eval_generic.py` path already does), or hold the Python intrabar sim to a golden-parity test
against `c_step` at slippage 0/5/10/20 bps. CLAUDE.md already half-says this ("validate with
binary-fill marketsim at lag≥2 before deploying"); the gap is that the *default* eval backend isn't
the binary-fill C sim.

**Sibling sims** are dead/legacy for the live pipeline: `cppsimulator/`, `c_market_sim/`,
`market_sim_c/` unused by train/eval; `pufferlib_cpp_market_sim/` only in two bracket-parity tests,
not in `eval_100d.py` or `train.py`. So `pufferlib_market` (this C env) + `intrabar_replay`
(Python eval) are the only two that matter.

---

## 8. Open questions / flags for next pass

- `[uncertain]` Does any pipeline emit a **v2 `.bin` with a tradable mask**? The in-repo exporter
  writes v1 only, making all the market-closed handling dead for shipped data. If 24/7 crypto, fine;
  if equities with overnight gaps, the mask matters and may be silently absent.
- `[uncertain]` Where does the production **6.25 % margin** map into `max_leverage`? Not in the C
  defaults (1.0). Likely an autoresearch preset or only a live-daemon constraint.
- Verify the **entry-slippage parity gap** (§7.1) numerically — it's the cleanest place for the
  C-train Sortino to overstate the Python-eval Sortino.
- The `fill_slippage` **double application** (entry price-shift + exit `effective_fee`) in C
  (`:356-364` vs `:403-411`) is intentional per comments but worth a unit test asserting round-trip
  cost = `2·fee + 2·slip` so it can't silently drift.

---
*House style note: all line cites are to the files as of branch `rl-deep-dive-mapping`, 2026-06-13.
This doc is read-only synthesis; no code was modified.*
