# Live Trader Deep Dive — the only real-money system (XGB daily) + Alpaca safety layer

**Date:** 2026-06-13 · **Branch:** `rl-deep-dive-mapping` · Read-only exploration.
Phase 1 of `DEEPDIVE_PLAN.md`. Sibling of `REPO_MAP.md` & the RL deep-dive set.

**Scope:** `xgbnew/live_trader.py` (2425 L), `alpaca_wrapper.py` (2309 L),
`src/alpaca_singleton.py` (494 L), `src/alpaca_account_lock.py` (188 L),
`scripts/deploy_live_trader.sh`. This is the system actually placing real-money orders — everything
in the RL deep-dives is research that is *not* live.

---

## 0. TL;DR

- **Three independent safety gates**, all fail-closed: (1) `ALLOW_ALPACA_LIVE_TRADING=1` explicit
  enable (paper-first by default), (2) **one fcntl singleton lock** per account, (3) **per-sell
  death-spiral guard**. The first is a layer CLAUDE.md's HARD RULES don't even enumerate.
- **The live XGB daemon does NOT route writes through `alpaca_wrapper.alpaca_order_stock`.** It builds
  its *own* `TradingClient` (`live_trader.py:102`) and calls the three safety primitives
  (`enforce_live_singleton` / `guard_sell_against_death_spiral` / `record_buy_price`) **directly**.
  It is compliant because it calls `enforce_live_singleton(force_live=True)` itself
  (`live_trader.py:2378`) and is registered in `LIVE_WRITER_UNITS`. So "inherit the gate by importing
  `alpaca_wrapper`" (CLAUDE.md HARD RULE 2) is satisfied *via direct call*, not via the wrapper.
- **Feature-parity answer (a key cross-cutting question):** the live XGB champion uses a **14-column
  DAILY feature set** (returns / RSI / vol / ATR / dollar-vol / 52w-range / day-of-week), **NOT** the
  16 hourly Chronos2 forecast features the RL C env consumes. **Chronos2 columns are zeroed at live
  inference** (`live_trader.py:512-516`). The live system and the RL research track operate on
  *different feature universes and timeframes* (daily stocks vs hourly forecasts).
- **Strategy:** daily long-only rotation. Score the universe pre-open, buy top picks at 09:30, exit at
  15:50 (or carry overnight and rotate at next open under `--hold-through`).

---

## 1. Startup & the three safety gates (Q2,Q3 + a third gate)

`main()` (`live_trader.py:2385`) → `_enforce_live_startup_guards()` (`:2372`) before any trading:

1. **Explicit-enable gate** (`alpaca_account_lock.py:70` `require_explicit_live_trading_enable`):
   refuses unless `ALLOW_ALPACA_LIVE_TRADING=1`. "Repo is currently in paper-first safety mode."
   Called `live_trader.py:2377`. *(Not in CLAUDE.md's 4 HARD RULES but is a real fourth fail-closed
   gate.)*
2. **Singleton lock** (`enforce_live_singleton`, `alpaca_singleton.py:174`): `force_live=True`,
   `service_name="xgb_live_trader"`, `account_name="alpaca_live_writer"` (`live_trader.py:2378`).
3. **Death-spiral guard**: per-order, see §4.

**The fcntl lock** (`alpaca_account_lock.py:125`):
- Path `strategy_state/account_locks/alpaca_live_writer.lock` (CLAUDE.md says `<state>/account_locks/…`;
  resolved via `unified_orchestrator.state_paths.resolve_state_dir`).
- `flock(LOCK_EX | LOCK_NB)` — non-blocking. On contention raises `RuntimeError`, which
  `enforce_live_singleton` converts to **`SystemExit(42)`** (`alpaca_singleton.py:223-230`) so a
  server can't catch-and-continue past it.
- On success writes a JSON payload `{service_name, pid, hostname, started_at, cmdline}`
  (`alpaca_account_lock.py:55-63, 172-177`) + `fsync`. **This `pid` is what `deploy_live_trader.sh`
  reads to verify the holder.**
- **Idempotent in-process** (`_HELD_LOCKS`, `:110`): a process can acquire the same lock twice (import
  time + later call) without racing its own fd — required because `alpaca_wrapper` acquires at import
  AND other entry points re-acquire. A *different* `service_name` in the same process raises (`:145`).
- `atexit.register(lock.release)` (`:187`) frees it on clean exit; OS frees the flock on crash.
- **Paper bypass:** `env_real.PAPER` / `ALP_PAPER=1` ⇒ `enforce_live_singleton` is a no-op
  (`alpaca_singleton.py:202-208`) — unlimited paper instances.
- **Break-glass:** `ALPACA_SINGLETON_OVERRIDE=1` skips the lock, loudly logged (`:210-217`).

---

## 2. The decision loop (Q1)

`main` loop (`live_trader.py:2414-2419`): `while True: run_session(); break unless --loop; sleep
until next open`. One **session** = `run_session()` (`:2043`):

1. **Trading-day gate** `_is_today_trading_day()` queries Alpaca `/v2/clock` (`:2076`); skip
   weekends/holidays gracefully (no crash).
2. `_wait_for_market_open()` blocks to 09:30 ET (`:2345`); between sessions
   `_sleep_until_next_session()` polls every `--crypto-poll-seconds` (default 300s) (`:2304`).
3. **Score → pick → buy** at open (§3).
4. **Exit** at 15:50 ET (`:2144-2152`): query live positions, sell all held XGB picks. Under
   `--hold-through`, positions carry overnight and are sold only when rotated out next open
   (`:2056`, rotation diff logic `:1880-1881`).

Cadence is **daily**, poll-based; no intra-session reconciliation loop — positions are read live from
Alpaca at the open/close gates, no local position-state file (`_get_position_details:288`).

---

## 3. Model, features, sizing (Q2,Q3,Q4,Q5)

**Model** (`_load_models:2158`): default single pickle `analysis/xgbnew_daily/live_model.pkl`
(`DEFAULT_MODEL_PATH:79`), via `xgbnew.model_registry.load_any_model` → `XGBStockModel.load`
fallback (`:2210-2218`). Optional comma-separated **ensemble** (`--model-paths`), blended by
**probability mean** (`np.mean(score_matrix, axis=0)`, `:595`) with optional uncertainty penalty
`blended − penalty·score_std` (`:597`). **Loaded once at startup, never reloaded intra-session.**
No champion auto-selection — the path *is* the champion. A **feature-contract validator** rejects
models needing unsupported live features or offline FM latents and enforces identical feature sets
across ensemble seeds (`:2205-2287`).

**Features (the parity answer)** — 14 DAILY columns (`xgbnew/features.py:84` `DAILY_FEATURE_COLS`,
computed `build_features_for_symbol`, `live_trader.py:503`):
`ret_1d/2d/5d/10d/20d, rsi_14, vol_5d/20d, atr_14, cs_spread_bps, dolvol_20d_log,
price_vs_52w_high, price_vs_52w_range, day_of_week, last_close_log`. Optional add-ons: cross-sectional
**ranks** (`DAILY_RANK_FEATURE_COLS`), **dispersion** (`cs_iqr_ret5, cs_skew_ret5`), and **FM
latents** (33 cols, rejected for live by the contract validator). **Chronos2 forecast columns
(`chronos_oc_return`, `chronos_cc_return`, `chronos_pred_range`, `chronos_available`) are written as
ZEROS** at live inference (`:512-516`) — *the live daemon does not call Chronos2 at all.* ⇒ **No
feature overlap with the RL C env's 16 hourly Chronos2 features** (`export_data.py:18-33`); different
universes entirely.

**Universe:** `--symbols-file` default `symbol_lists/stocks_wide_1000_v1.txt` (~1000 names), filtered
by min dollar-volume (~$5M), max spread (~30 bps), vol band (`:520-562, 1090-1097`).

**Conviction filter:** `--min-score` threshold with a `--min-picks` floor (force ≥N picks)
(`:1255-1286`); optional `--no-picks-fallback` (e.g. SPY at reduced alloc) when nothing qualifies
(`:1416-1488`).

**Sizing** (`:2112-2127`, `_buy_notional_by_symbol:1491`):
`total_notional = portfolio_value · allocation · conviction_scale · spy_vol_scale`, then split
per-pick by weight (equal / score-norm / softmax) × **inverse-vol scale** (`_inv_vol_pick_scale`),
finally **clipped to buying power** (`:950`). Per-pick qty = `floor((notional/price)·1e4)/1e4`
(`:925`). `spy_vol_scale` targets a portfolio annualised vol vs SPY realized (`:749-827`);
`conviction_scale` ramps exposure between `--conviction-alloc-low/high`.

---

## 4. Order submission & death-spiral guard (Q4,Q7)

**Submission** (`_submit_limit_order:142`): builds a `LimitOrderRequest` (`TimeInForce.DAY`,
price rounded to 2dp, qty to 4dp) and calls **the daemon's own `client.submit_order(req)`** (`:154`)
— *not* `alpaca_wrapper`. It only borrows `alpaca_wrapper.latest_data` for read-only quotes
(`:160-162`). Limit prices sit at touch ± `aggressiveness_bps` (default 15 bps): buys at ask+15,
sells at bid−15 (`_stock_limit_price_near_market:178`). Marketable but bounded — never a true market
order.

**Buy path** (`_execute_buys:1534`): submit → poll fill (`_poll_filled_avg_price:216`, 30s) →
**`record_buy_price(sym, fill_price or last_close)`** (`:1600`, HARD RULE 3 memory). Submission
failures are caught per-symbol and logged (don't crash).

**Sell path** (`_execute_sells:1616`, rotation seller `:1741`): **every** sell first calls
**`guard_sell_against_death_spiral(sym, "sell", current_price)`** (`:1642`, `:1813`). The guard
(`alpaca_singleton.py:356`) refuses if `price < buy_price·(1 − tol/1e4)` using **time-aware
tolerance**: intraday 50 bps if buy ≤8h old, overnight 500 bps if older
(`DEFAULT_*` `:58-70`). A refusal raises **`RuntimeError` that is NOT caught** → crashes the daemon →
systemd marks the unit failed (the desired loud-stop behaviour). If no recorded buy exists for the
symbol, the guard allows the sell (it only prevents *round-trip* spirals, `:411-415`). Break-glass:
`ALPACA_DEATH_SPIRAL_OVERRIDE=1` (`:396`).

**Buy memory** is a per-account JSON at `<state>/alpaca_singleton/alpaca_live_writer_buys.json`,
fcntl-locked, 3-day TTL, atomic `tmp+fsync+os.replace` writes, corrupt files quarantined
(`alpaca_singleton.py:91, 247-346`).

---

## 5. `alpaca_wrapper` write surface & a guard-coverage caveat

`alpaca_wrapper` is the *other* live writer (the `trading-server` unit, §6). It acquires the singleton
lock **at import time** (`alpaca_wrapper.py:67-70`, service `alpaca_wrapper_{pid}`, account
`alpaca_live_writer`) and picks prod vs paper keys via `env_real.PAPER` (`:172-174`). Its canonical
order fn `alpaca_order_stock` (`:1394`) calls the death-spiral guard *before* submit (`:1473`) and
`record_buy_price` after buys (`:1512`) — matching CLAUDE.md HARD RULE 3.

**Caveat — not every helper calls the guard, but no LIVE unit reaches the un-guarded ones [RESOLVED].**
A sub-agent sweep found sell/close-capable helpers in `alpaca_wrapper` that submit without the guard:
`close_position_violently` (`:1146`), `close_position_near_market` (`:2228`),
`open_take_profit_position` sells (`:1574,1597`), and the crypto `crypto_alpaca_looper_api` HTTP path
(`:1489`). The guard is **sell-only by design**, so the *buy* helpers flagged "unguarded" are not gaps.
**Traced `trading-server`'s call graph (`src/trading_server/server.py`):** its *only* `alpaca_wrapper`
write paths are `open_order_at_price_or_all` (`server.py:1631`) — which **does** call
`guard_sell_against_death_spiral` (`alpaca_wrapper.py:741`) — and `cancel_order` (`server.py:1653`, a
cancel, N/A to the guard). It does **not** call any of the un-guarded sell/close helpers. So **both
registered live writers (XGB daemon + trading-server) route every sell through the death-spiral
guard.** The un-guarded helpers are reached only by non-registered scripts
(`trade_v4_e2e.py`, `alpaca_cli.py`, `deleverage_account_day_end.py`, `predict_stock_e2e.py`) — none in
`LIVE_WRITER_UNITS`, so they'd hit `SystemExit(42)` if run while a live writer holds the lock
(operator break-glass tools, effectively). Single `TradingClient` per process (`:213`) ⇒ no second
client bypasses the lock. `trading-server` adds its own live gate too: `cancel` requires
`live_ack == LIVE` **and** `ALLOW_ALPACA_LIVE_TRADING=1` (`server.py:1832-1837`).

---

## 6. Deployment & lock-holder verification (`deploy_live_trader.sh`)

`LIVE_WRITER_UNITS = (xgb-daily-trader-live, trading-server, daily-rl-trader)` (`:60-64`) — the
**source of truth** for who can win the lock (CLAUDE.md HARD RULE 2). Per production memory:
`xgb-daily-trader-live` is the live champion; `daily-rl-trader` was stopped 2026-04-30;
`trading-server` status `[uncertain]`.

Deploy sequence: validate target ∈ registry (`:298`) → record pre-state lock holder pid (`:369`) →
**`supervisorctl stop` every other registered unit** (`:373-386`, with a stop-phase-exempt carve-out
for the RL client-of-broker case `:310-318`) → wait ≤15s for the lock to clear if its holder died
(`:388-392`) → start target → **confirm the lock file's holder pid == the new supervisor pid (or a
descendant)** (`_lock_matches_unit:191`, exit 5 on mismatch) → append `deployments/live_trader_history.log`
(`:73`). Exit codes: 3 stop-failed, 5 lock-holder mismatch.

---

## 7. Failure modes (Q7)

- **Deliberate crash (uncaught):** death-spiral `RuntimeError` (`:1642,1813`); singleton
  `SystemExit(42)`; explicit-enable `RuntimeError`. All fail-closed → systemd-failed → manual re-enable.
- **Caught (continue):** per-symbol buy/sell submission errors (`:1611,1670`), `record_buy_price`
  failure (logged, buy already placed, `:1604`), quote-lookup failure (`:163`), non-trading-day (skip).
- **Feature-contract mismatch:** rejected at load (`:2245-2261`) before any trade.

---

## 8. Open questions / follow-ups

- ~~Does `trading-server` reach the un-guarded sell/close helpers?~~ **[RESOLVED]** No — its only write
  paths are `open_order_at_price_or_all` (guarded) and `cancel_order`. Both live writers guard every
  sell (§5).
- `[uncertain]` Is `trading-server` currently running, or is XGB-daily the sole live writer? Memory
  says only XGB-daily is live; confirm against `alpacaprod.md` + `live_trader_history.log`.
- The crypto `crypto_alpaca_looper_api` (`localhost:5050`) is a **separate process** outside the
  singleton/guard — but Binance/crypto is paused per memory, so likely dormant. Confirm.
- **Train/live mismatch worth stating loudly:** the live champion is **XGBoost on daily bars with a
  14-feature set**; all the marketsim/RL realism work (binary fills, decision_lag, Chronos2 features)
  validates a *different* system that isn't live. The 27%/month eval gate (HARD RULE 1) is RL-track;
  what guards the *daily XGB* champion's promotion is a separate question (Phase 2/3 territory).

---
*Read-only synthesis; line cites as of branch `rl-deep-dive-mapping`, 2026-06-13. No code modified.*
