# Autoresearch Lab — Agentic Strategy-Discovery Loop

> Design spec. Adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
> (an AI agent rewriting code under a fixed budget, scored by one honest metric,
> looping autonomously overnight) to keel_trader. 2026-06-14.

## 1. Motivation

keel_trader's plumbing is done and honest, but every search so far has found **no
out-of-sample edge**. The recorded conclusion: *"making money is now a research
problem (label/strategy/features), not plumbing."*

The existing `research/autoresearch.py` is a **non-agentic grid + mutation sweep**.
It mutates only position-*sizing* knobs (`conviction_threshold`, `gross_exposure`,
`max_position_weight`, `max_positions`) on top of a **frozen** scoring policy. That
search space structurally excludes the signal, features, label, and model — i.e. the
only region where an edge could come from. Resizing a zero-edge signal yields a
zero-edge signal with different variance; this is why the sweeps found nothing, and
it is work an off-the-shelf optimizer (Optuna/CMA-ES) would do better anyway.

karpathy's insight is the missing lever: let an **AI agent rewrite the code that grid
search cannot reach** — the signal/feature/strategy logic itself — under a fixed
budget, scored by one honest metric, looping unattended. This spec builds that loop
as a new package, `research/lab/`, leaving the existing grid sweep untouched.

## 2. The honesty contract (frozen vs mutable)

The single most important property: **the agent must never be able to edit its own
scorer**, or it will learn to cheat the metric instead of finding an edge.

**Frozen — the agent may never edit these:**
- `sim/` — the ONE C fill engine (binary fills, fees, slippage).
- `research/eval.py` — the out-of-sample gate: N unseen windows, slippage matrix
  `{0,10,20,30}bps` worst-cell, fail-fast on drawdown/unreachable-median, 26bps
  Kraken-taker fee, promote iff worst-cell median-monthly ≥ 0.10.
- `generalization_score` (test minus overfit penalty) and the append-only leaderboard
  writer.
- The data `.bin`, including the upstream Chronos forecast feature precompute.

**Mutable — exactly one file, `research/lab/strategy.py`:**
- Exposes one fixed seam: `build_policy(md: MarketData, *, seed: int = 0) -> Policy`,
  where `Policy = Callable[[obs, env], int]`.
- Inside `build_policy` the agent has total freedom: derive/transform features from
  the obs vector, accumulate rolling state across bars, implement any signal, sizing,
  entry/exit and holding logic — as long as it returns a callable the frozen
  `evaluate()` can step.

**Why this is honest *and* maximal.** The C sim calls the policy **bar-by-bar**,
passing only the *current* bar's observation (`policy(obs, env)`), action space
`{0=flat, 1+sym=long sym}` (single position). The policy never sees the future, so
**lookahead is impossible by construction** regardless of what the agent writes. The
agent gets karpathy-style "rewrite the whole file" freedom; the C sim and the
unseen-split gate remain the unchangeable judge. The only thing out of reach is the
upstream feature *precompute* (frozen data); online derivation from obs is fair game.

## 3. Components

### 3.1 `research/lab/strategy.py` — the mutable file
- Ships at trial-0 as a copy of today's baseline
  (`make_strategy_policy(baseline_score_fn(), …)`) so the loop starts from a known,
  reproducible number.
- Must keep the `build_policy(md, *, seed=0) -> Policy` signature. Everything else is
  the agent's to rewrite.

### 3.2 `research/lab/harness.py` — the trial runner (frozen)
- `run_trial(md, build_policy, *, seed, train_frac, n_windows, window_steps,
  overfit_penalty) -> TrialResult`.
- Structurally identical to the existing `autoresearch.run_trial`, with one swap: the
  policy comes from the mutable module instead of a `StrategyConfig`.
- Calls the existing `evaluate()` twice:
  - in-sample: `offset_lo_frac=0.0, offset_hi_frac=train_frac, fail_fast=False`
  - unseen (the gate): `offset_lo_frac=train_frac, offset_hi_frac=1.0`
- Computes the existing `generalization_score(train_worst, test_worst, penalty)`.
- Promotes iff the unseen gate's `verdict.promote` is true.
- Returns a typed `TrialResult` (no stringly-typed metric dicts).

### 3.3 `research/lab/leaderboard.py` — append-only record (frozen)
- Reuses the existing leaderboard pattern (append-only CSV, header-once, never
  truncates) with a reproducibility manifest: `timestamp, git_hash, hardware, seed`.
- Extends the manifest with: **content hash of `strategy.py`**, and the frozen-module
  hashes (`sim/`, `research/eval.py`) so tampering is detectable.
- Archives the exact strategy file version to `artifacts/lab/strategies/<hash>.py`, so
  every row is byte-for-byte reproducible and any champion is recoverable.
- Columns: existing set + `strategy_hash`, `frozen_hash` (combined hash of the frozen
  modules), `trial_status` (ok/error/timeout), `gate_seed`.

### 3.4 `research/lab/drive.py` — the autonomous driver
`python -m research.lab.drive --trials 100`. Each iteration:
1. **Assemble context** — `program.md` + current `strategy.py` + last N leaderboard
   rows + previous verdict/metrics.
2. **Mutate** — invoke a **configurable mutator command** (default `claude -p`
   headless; `--mutator "<cmd>"`) that rewrites **only** `research/lab/strategy.py`.
   Pluggable so the loop is CLI-agnostic and testable with a deterministic stub.
3. **Run the trial in a subprocess** under a wall-clock **budget**
   (`--budget-seconds`, default 300 — karpathy's fixed-budget analog). Timeouts,
   exceptions, and malformed strategies are recorded as failed rows; the loop never
   crashes.
4. **Record & archive** — append leaderboard row + snapshot strategy by hash.
5. **Feed forward** — greedy keep-best: next context highlights the best-so-far
   champion (hill-climb); the leaderboard stays append-only so no history is lost.

Guardrails:
- The driver only ever writes `research/lab/strategy.py`; mutator output touching any
  other path is rejected and the trial is failed.
- Frozen-module hashes recorded per trial.
- **Per-trial gate-seed rotation** (recorded in the manifest). Because the same gate
  runs 100×, a fixed seed would let the agent overfit to one memorized draw of
  "unseen" windows. Rotating the seed means "clears the gate" = "clears on
  freshly-drawn unseen windows."
- `--dry-run` / stub-mutator mode runs the whole loop without an LLM.

### 3.5 `research/lab/program.md` — the human-steered brief
- Standing objective: clear the OOS gate net of ~52bps round-trip friction.
- The hard interface contract the agent must honor (`build_policy` signature; only
  edit this file; no lookahead — the obs is current-bar only).
- Accumulated scar tissue, e.g. *"resizing a zero-edge signal stays zero-edge; tuning
  sizing knobs alone has never cleared the gate — change the signal."*

## 4. Layout

```
research/lab/
  __init__.py
  strategy.py     # THE mutable file (starts = baseline); fixed build_policy() seam
  harness.py      # frozen: run_trial -> reuses evaluate() + generalization_score
  leaderboard.py  # frozen: append-only CSV + manifest + strategy archive
  drive.py        # the autonomous driver (subprocess + mutator + budget)
  program.md      # human-authored: objective, interface contract, scar tissue
artifacts/lab/
  leaderboard.csv
  strategies/<hash>.py
tests/test_lab.py
```

Makefile targets:
- `make lab-trial` — run one trial against the current `strategy.py`.
- `make lab-drive TRIALS=…` — run the autonomous driver.
- `test_lab` added to the `test` aggregate.

## 5. Testing

`tests/test_lab.py` (no LLM required — uses the stub mutator):
- **Frozen seam**: `build_policy` baseline reproduces the known baseline verdict.
- **Stub-mutator loop**: drive N trials with a deterministic stub that swaps in known
  strategy variants; assert append-only growth, correct ranking, keep-best.
- **Archive/repro**: a leaderboard row's `strategy_hash` matches the archived file;
  re-running that archived file reproduces the row's metric (seeded).
- **No-write-outside-strategy**: a stub mutator that tries to edit another path is
  rejected and the trial recorded as failed.
- **Budget**: a stub strategy that sleeps past the budget is killed and recorded as
  timeout.

## 6. Scope / non-goals (YAGNI)

- Do **not** modify or remove the existing `research/autoresearch.py` grid sweep.
- No principled optimizer (TPE/CMA-ES) — the agent *is* the search.
- No parallel/distributed trials in v1 (sequential loop; the budget bounds each).
- No live-trading or executor changes; this is offline research only.
- The default `claude -p` mutator is one option behind the `--mutator` seam; we do not
  hard-couple to any specific agent CLI.
