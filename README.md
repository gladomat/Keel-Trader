# keel_trader

A trading system rebuilt clean from the lessons of the `moray` monorepo. The name: a *keel* is the
structural backbone that keeps a boat upright and stable — which is what this system is about (the
safety spine is the crown jewel; the goal is low-drawdown, smooth-Sortino PnL).

> **Read `docs/REBUILD_HANDOFF.md` first** (the brief — what to build, what to copy, bugs not to
> reproduce), then **`docs/BUILD_PLAN.md`** (the sequenced roadmap for the rest of the
> implementation). The remaining `docs/` are the carried-over knowledge base from the moray deep-dive
> (the "why", with `file:line` cites).

## The non-negotiable invariants (don't break these)

1. **One fill engine, for training *and* evaluation, pinned by a golden test.** The single source of
   truth lives in `sim/` (binary-fill C core). Any future soft/differentiable wrapper or Python eval
   must reproduce `tests/test_fill_model.c` exactly. The old repo's biggest silent bug was an eval
   sim that filled ~`fill_buffer` bps better than its training sim — see
   `docs/EVAL_SIM_PARITY_DEEPDIVE.md` §5. **`make test` is the guard.**
2. **Safe values are the defaults.** `decision_lag ≥ 2`, binary fills, `fee = 10 bps`,
   `fill_buffer = 5 bps`, `max_hold = 6 h`. Loosening any of them requires an explicit flag, so a
   naive run can never report fantasy Sortino.
3. **One versioned feature spec** feeds all consumers (the old repo had three disjoint ones).
4. **Nothing trades until it clears the gate** — median monthly ≥ 27% on *unseen* windows, worst of
   slippage {0,5,10,20}, `decision_lag ≥ 2`, binary fills, fail-fast on max-drawdown.
5. **Exactly one live Alpaca writer**, enforced by the singleton lock + death-spiral guard (to be
   ported into `core/`). Default paper; live is opt-in and gated. (`docs/LIVE_TRADER_DEEPDIVE.md`.)

## Layout (target shape)

```
core/       safety spine (singleton lock + death-spiral guard) + config + paper loop   [done]
sim/        the ONE fill engine (binary-fill C core) + .bin format + Kraken adapter    [done]
forecast/   Chronos-2 forecaster (zero-shot + LoRA) + technical features + cache        [done]
models/     the incumbent (xgb daily) + one RL track                                    [done]
research/   gate + autoresearch + signal sweep + walk-forward validator                 [done]
ops/        deploy (lock-verifying) + prod docs                                         [done]
docs/       carried-over knowledge base + Kraken calibration + research findings        [done]
tests/      golden fixtures (fill + safety + features + gate + paper + ...) pinned       [done]
```

**Kraken (crypto) track + research:** the full chain (data → Chronos-2 forecasts →
features → gate → autoresearch → walk-forward → paper) is built and validated.
Honest outcome: **no robust tradeable edge** found in the 5 USD majors — see
[`docs/CRYPTO_RESEARCH_FINDINGS.md`](docs/CRYPTO_RESEARCH_FINDINGS.md) (and
[`docs/KRAKEN_CALIBRATION.md`](docs/KRAKEN_CALIBRATION.md) for the forecast/gate
calibration). Offline run targets: `make data-kraken build-cache-kraken
train-kraken gate-kraken feature-search walkforward paper-kraken`.

## Build / test

```bash
make test         # all golden fixtures: fill model (C) + safety spine (py)
make test-fill    # just the fill-model parity guard
make test-safety  # just the singleton lock + death-spiral guard
make test-asan    # fill model under AddressSanitizer/UBSan
```

## Status

Bootstrapped 2026-06-13. **Done:** repo skeleton, knowledge base, the single fill engine + golden
test, the safety spine (`core/` — three-gate singleton + time-aware death-spiral guard, **paper-default,
no live entry point wired**) + its golden tests. **Next:** the gate (`research/eval`, calling the one
sim), then the incumbent xgb-daily model. See `docs/BUILD_PLAN.md` for the full sequenced roadmap (phases 3–8) and
`docs/REBUILD_HANDOFF.md` §3 for the copy-list.

> **Live-writer note (HARD RULE 2):** `core/` ports the guard *logic* only. No process here can win
> the live-writer lock yet. Wiring a live entry point is a deliberate, reviewed step — and if keel
> ever trades the same Alpaca account as another system, both must share the same lock path/account
> so the singleton actually protects across them.
