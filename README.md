# keel_trader

A trading system rebuilt clean from the lessons of the `moray` monorepo. The name: a *keel* is the
structural backbone that keeps a boat upright and stable — which is what this system is about (the
safety spine is the crown jewel; the goal is low-drawdown, smooth-Sortino PnL).

> **Read `docs/REBUILD_HANDOFF.md` first.** It is the brief — what to build, what to copy, and the
> bugs not to reproduce. The rest of `docs/` is the carried-over knowledge base from the moray
> deep-dive (the "why", with `file:line` cites).

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
core/       safety spine (singleton lock + death-spiral guard) + broker boundary   [pending]
sim/        the ONE fill engine (binary-fill C core) + .bin data format            [done: ported]
forecast/   one LoRA forecaster + parquet cache                                    [pending]
models/     the incumbent (xgb daily) + one RL track                              [pending]
research/   autoresearch loop + config-driven sweeps + append-only leaderboards    [pending]
ops/        deploy (lock-verifying) + prod docs                                    [pending]
docs/       carried-over knowledge base (the moray deep-dive)                      [done]
tests/      golden fixtures (fill model pinned)                                    [done: fill model]
```

## Build / test

```bash
make test        # golden fill-model fixture (the parity guard)
make test-asan   # same, under AddressSanitizer/UBSan
```

## Status

Bootstrapped 2026-06-13. Done: repo skeleton, knowledge base, the single fill engine + its golden
test. Next: port the safety spine (`core/`, paper-default), then the gate (`research/eval`), then the
incumbent model. See `docs/REBUILD_HANDOFF.md` §3 for the copy-list.
