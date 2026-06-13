# Eval-Sim Parity Deep Dive — C training env vs Python intrabar eval

**Date:** 2026-06-13 · **Branch:** `rl-deep-dive-mapping` · Read-only exploration.
Phase 2 of `DEEPDIVE_PLAN.md`. Follows up the parity risk flagged in `RL_C_ENV_DEEPDIVE.md` §7.

**Scope:** `pufferlib_market/intrabar_replay.py` (1059 L), `hourly_replay.py` (1845 L),
`scripts/eval_100d.py` (the `_evaluate_intrabar_hourly` path). Question: how does the
*promotable* eval differ from the C env the RL flagship trains on — and does it overstate deployable
PnL?

---

## 0. TL;DR + a correction

- **Training** = C `c_step` (`trading_env.c`). **Promotable eval** =
  `eval_100d.py --execution-granularity hourly_intrabar` → `simulate_daily_policy_intrabar`
  (`intrabar_replay.py:673`), a **pure-Python reimplementation**. The C env is reachable from eval
  only via the non-default `--execution-granularity daily` path (`fp4/bench/eval_generic.py`), which
  is **not promotable** without `--allow-daily-promotion`.
- **Correction to `RL_C_ENV_DEEPDIVE.md` §7.1:** the earlier claim that "Python under-models entry
  slippage" is **wrong**. `eval_100d` folds slippage into the fee: `fee_rate = fee_rate + bps/1e4`
  (`eval_100d.py:991`), and that fee is charged on **both** legs (`_open_long:349`, `_close_position:253`).
  So round-trip cost ≈ `2·(fee+slip)`, the **same magnitude** as the C env's `2·fee + 2·slip`
  (price-shift on entry `trading_env.c:406` + `effective_fee` on exit `:356`). Total transaction
  cost is at **parity**; the real divergences are structural (below), not a missing slippage leg.
- **Net read:** the eval is *more realistic* than training (hourly intrabar walk, SL/TP, max-hold at
  real times), and on the two points where it's *less* adverse than C — the limit-buffer fill price
  (§2) — it would mildly **overstate** PnL vs the C ground truth. No numerical run done (see §5);
  this is a code-level audit.

---

## 1. The two cost models are equivalent in magnitude (not in mechanism)

| | C env (`trading_env.c`) | Python intrabar (`eval_100d`→`intrabar_replay`) |
|---|---|---|
| fee | `fee_rate` on both legs | `fee_rate` on both legs |
| slippage input | `fill_slippage_bps` | folded into `fee_rate` at the call site (`eval_100d.py:991`) |
| entry cost | `target·(1+slip)·(1+fee)` (price shift + fee) | `target·(1+fee+slip)` (fee only, slip *in* fee) |
| exit proceeds | `open·(1 − fee − slip)` (`effective_fee`) | `open·(1 − fee − slip)` (slip in fee) |
| **round-trip** | `≈ 2·fee + 2·slip` | `≈ 2·(fee+slip)` |
| **recorded `entry_price`** | `target·(1+slip)` (slip-shifted) | bare `target` (no slip) |

**The one residual difference:** C bakes slippage into the *recorded entry price*; Python keeps
`entry_price` slip-free and charges slip via fee. So any logic that **keys off `entry_price`** —
stop-loss, take-profit, the death-spiral floor — sees a slightly different reference in the two sims.
Small (a few bps), but it means SL/TP trigger points are not bit-identical. `[low-impact]`

---

## 2. Limit-buffer fill price — a genuine sign divergence (eval is more optimistic)

This is the cleanest place the eval is *less adverse* than the C ground truth.

- **C** (`resolve_limit_fill_price:383`): buy gate `low ≤ target·(1−buffer)`; if it passes, fill at
  **`target`** (≈ bar open). The buffer only makes the fill **harder to get**; the price you pay is
  the (higher) target. Pessimistic on *fill probability*, neutral on *price*.
- **Python intrabar** (`intrabar_replay.py:908-916`): buy `target = open·(1+offset)·(1−buffer)`
  (buffer **lowers** the target), then fill if `low ≤ target`, filling **at the lowered target**.
  So a plain marketable buy (offset 0, buffer 5 bps) fills at **`open·(1−5bps)`** — a 5 bps *better*
  entry than C's `open`. Symmetric for sells (buffer raises the sell target → better sell price).

⇒ For the same `fill_buffer_bps`, **intrabar entries/exits are ~`fill_buffer` bps more favorable than
the C env.** With the production `fill_buffer_bps≈5`, that's ~5 bps/leg of optimism the eval grants
that training's ground truth does not. It partially offsets the cost parity in §1, in the *wrong*
direction for trustworthiness (eval looks better than truth). `[medium-confidence]` — directional
analysis from the code; would be pinned exactly by §5's numerical run.

Both sims agree on the **reference price = bar OPEN** for entries and exits
(`trading_env.c:344,440` vs `intrabar_replay.py:896,908`) — that part is at parity.

---

## 3. Structural divergences (eval is *more* realistic — by design)

3.1 **Execution granularity.** C advances one `.bin` bar per step (for daily data, one *day* per
step). `simulate_daily_policy_intrabar` queries the policy **once per calendar day** at a configured
trade hour, but **replays every hourly OHLC bar in between** (`intrabar_replay.py:694-699`). So fills
and exits occur at hourly granularity in eval vs daily-bar granularity in training. This is the
deliberate realism upgrade and the main reason the two PnLs differ.

3.2 **Stop-loss / take-profit / max-hold.** The C training env has **no SL/TP** — only
`max_hold_hours` force-close. Intrabar fully simulates SL/TP every hour, with bar-gap handling
(`_stop_loss_triggered:292`, `_take_profit_triggered:326`, max-hold `:515-522`). `eval_100d` passes
`stop_loss_pct`/`take_profit_pct` from the **loaded checkpoint config** (`:995-997`); if the RL policy
was trained without them (the common case) they're `None` and this path is inert — but any checkpoint
carrying SL/TP is evaluated under logic it was never trained against. `[uncertain]` whether the
flagship checkpoints set these.

3.3 **`decision_lag` locus.** C lags the *observation features* internally (`fill_observations:174`).
Intrabar applies the lag in the **policy wrapper** `make_policy_fn(..., decision_lag=…)`
(`eval_100d.py:981`, `evaluate_multiperiod.py`) — it delays between policy query and execution rather
than staling the obs. Same anti-lookahead intent, different mechanism; worth a unit test that both
produce identical action-timing for `lag=2`. `[uncertain]`

3.4 **Auto-reset / episode bookkeeping.** C accumulates Sortino/maxDD into a `Log` struct at episode
end and auto-resets (`c_step:809-838`). Intrabar returns a `DailyPolicyIntrabarResult` per window;
`eval_100d` aggregates across `n_windows` random starts and reports per-slippage cells
(`:1004-1052`). Different metric plumbing — compare totals, not internal counters.

---

## 4. The promotion gate (HARD RULE 1 lives here)

`eval_100d` loops slippages (default the 0/5/10/20 bps matrix from CLAUDE.md), each as a fee bump
(`:972, 991`), over `n_windows` random windows (`:968-969`). **Fail-fast** bails a cell when
`max_drawdown > fail_fast_max_dd` (default 0.20) or the median-monthly target becomes impossible
(`_median_target_impossible`, `:1023-1036`) — implementing CLAUDE.md HARD RULE 1 (median monthly
≥ 0.27, worst slip cell). The validation/gate layer (`eval_100d.py:188-298`) enforces
`decision_lag ≥ min_decision_lag`, `max_slippage_bps ≥ min`, `fee_rate ≥ min`, and a minimum
`hourly_fill_buffer_bps` for the intrabar backend. So the gate is well-guarded — but it guards the
**Python intrabar sim**, not the C training sim.

---

## 5. Numerical fill-parity test — RUN (real code, both sides)

A full `eval_100d` C-vs-Python run needs a trained checkpoint (none in tree) + the compiled `.so`. But
the §1/§2 claims live in the **fill functions**, which can be exercised directly. I built a harness that
drives the **real C functions** (`resolve_limit_fill_price`, `open_long`, `close_position` — via
`#include`ing `trading_env.c` to reach the statics) and the **real Python functions**
(`hourly_replay._resolve_limit_fill_price`, `_open_long_limit`, `_close_position`) over an identical
grid: fixed bar `O=100,H=101,L=99,C=100`, `fee=10bps`, buffer ∈ {0,5,20} bps, slip ∈ {0,5,20} bps, a
marketable buy. Results (buy fill price; round-trip cost = buy-then-sell-at-open):

| buffer | slip | C fill | py **daily** | py **intrabar** | C cost | py cost | C entry | py entry |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 100.0000 | 100.0000 | 100.0000 | 19.98 | 19.98 | 100.0000 | 100.0000 |
| 0 | 5 | 100.0500 | 100.0000 | 100.0000 | 29.96 | 29.96 | 100.0500 | 100.0000 |
| 0 | 20 | 100.2000 | 100.0000 | 100.0000 | 59.84 | 59.82 | 100.2000 | 100.0000 |
| 5 | 0 | 100.0000 | 100.0000 | **99.9500** | 19.98 | 19.98 | 100.0000 | 100.0000 |
| 5 | 20 | 100.2000 | 100.0000 | **99.9500** | 59.84 | 59.82 | 100.2000 | 100.0000 |
| 20 | 0 | 100.0000 | 100.0000 | **99.8000** | 19.98 | 19.98 | 100.0000 | 100.0000 |
| 20 | 20 | 100.2000 | 100.0000 | **99.8000** | 59.84 | 59.82 | 100.2000 | 100.0000 |

**Verdicts (empirical):**
- **§1 cost parity — CONFIRMED.** `C cost ≈ py cost` to <0.05 in every cell. Slip-as-price-shift (C)
  and slip-folded-into-fee (`eval_100d.py:991`) yield the **same** round-trip cost `≈2·(fee+slip)`.
  The "Python under-models entry slippage" claim is empirically refuted.
- **§2 buffer-fill divergence — CONFIRMED & QUANTIFIED.** C and the Python **daily** path both fill at
  `100.00` (parity). The Python **intrabar** path (the *promotable* eval) fills at exactly
  `open·(1−buffer/1e4)` = 99.95 @5bps, 99.80 @20bps — i.e. **`fill_buffer` bps better for the buyer,
  for the identical fill condition**. At the production buffer (~5 bps) the promotable eval grants
  every entry/exit a 5 bps edge the C ground truth does not.
- **Entry-price recording — CONFIRMED.** At slip>0, C `entry_price` is slip-shifted (100.05, 100.20)
  while Python keeps it bare (100.00) — so SL/TP/death-spiral thresholds key off prices that differ by
  the slip amount.

Harness: `/tmp/fill_parity.c` (+ `/tmp/fill_parity_c.csv`) and `/tmp/fill_parity.py` (run under a throwaway
`uv` venv with numpy/pandas). A clean rebuild should either run the C core in the promotable path or
pin the Python intrabar sim to a golden C parity fixture so the buffer edge can't silently inflate
eval PnL. (A full policy-level `eval_100d` diff still needs a checkpoint + built `.so` — the fill-level
result above already isolates the mechanism, which is what mattered.)

---

## 6. Open questions / follow-ups
- ~~Quantify §2's buffer-optimism gap numerically~~ **[DONE — §5]** = exactly `fill_buffer` bps per
  leg (5 bps at prod buffer); cost parity also empirically confirmed. Remaining: the *policy-level*
  PnL/Sortino gap over real windows (needs checkpoint + built `.so`).
- Confirm whether flagship checkpoints carry `stop_loss_pct`/`take_profit_pct` (§3.2); if so, eval
  exercises untrained logic.
- Unit-test `decision_lag` action-timing parity between the two loci (§3.3).
- Correct `RL_C_ENV_DEEPDIVE.md` §7.1's "entry slippage under-modeled" line — superseded by §1 here
  (cost is at parity; the divergence is entry-price *recording* + buffer fill price).

---
*Line cites as of branch `rl-deep-dive-mapping`, 2026-06-13. No repo code modified. §5 ran a throwaway
fill-parity harness against the real C + Python fill functions (artifacts under `/tmp`).*
