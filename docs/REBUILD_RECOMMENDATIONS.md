# Moray — Rebuild Recommendations (Keep / Discard)

> Opinionated guide for rebuilding this system from scratch. Companion to `REPO_MAP.md`.
> Generated 2026-06-13.

## The core insight

This repo is **~5% load-bearing system, ~95% search residue**. The 5% is a tight,
well-engineered production spine with two hard safety invariants and a single honest judge
(`scripts/eval_100d.py`). The 95% is the frozen exhaust of hundreds of experiments — dirs that
were forked, run once, beaten, and never deleted. A rebuild isn't "rewrite the system," it's
**delete the residue and keep the spine + the loop that generates it**.

The discipline worth preserving: *one ground-truth judge, append-only leaderboards, nothing
trades until it clears the judge.*

## Keep (the irreducible system)

| Keep | Why |
|---|---|
| **Safety spine**: `src/alpaca_singleton.py` (live-writer lock + death-spiral guard), `alpaca_wrapper.py`, `scripts/deploy_live_trader.sh` | The actual crown jewel — stops you blowing up real money. Hard-won, hard to re-derive. Non-negotiable. |
| **The judge**: `scripts/eval_100d.py` + **one** ground-truth sim (`pufferlib_cpp_market_sim/`, binary fills, `decision_lag≥2`) | A single trusted out-of-sample gate keeps the whole research process honest. Everything else is negotiable; this isn't. |
| **The loop**: `pufferlib_market/autoresearch_rl.py` + the `*_leaderboard.csv` convention | This *is* the autoresearch loop. Append-only CSV leaderboards + manifest JSON (git hash, seed, hardware) is a genuinely good reproducibility pattern. |
| **One forecaster path**: `chronos2_trainer.py` → `build_hourly_forecast_caches.py` → `src/forecast_cache_lookup.py` | Clean train→cache→serve seam. Keep the LoRA-fine-tune + parquet-cache pattern. |
| **Two model tracks that earn their place**: `xgbnew/` (simple, live, ~30 LOC of real logic) and **one** PPO (`pufferlib_market/`) | XGB because it's the incumbent and trivially cheap. One RL track because it's the most promising research lever. |
| **Broker boundary**: `src/trading_server/{server,client}.py` | The HTTP single-writer surface is the right abstraction. |

## Discard (the residue)

- **10 of the 12 RL tracks.** `pufferlibtraining{,2,3}`, `training/`, `gpu_trading_env`,
  `sharpnessadjustedproximalpolicy2`, `differentiable_market` as a separate thing — superseded or
  experimental. Keep `pufferlib_market`; archive `sharpnessadjustedproximalpolicy` (SAP) as *one*
  alternative optimizer behind a flag, not a parallel codebase.
- **~18 of ~20 `stockagent*` LLM variants + the GRPO stack** (`qwen_rl_trading`, `trltraining`).
  Expensive per-decision, unproven against the gate, huge surface. Keep one as a probe; discard
  the rest.
- **5 of 6 market sims.** `marketsimulator*`, `cppsimulator`, `c_market_sim`, `market_sim_c`,
  `frontiermarketsim`, the soft/differentiable sims — collapse to the one ground-truth C++ sim
  plus (optionally) one fast soft sim *clearly labeled training-only, never trusted for
  promotion*.
- **`neuraldailyv2..v4`, `binanceexp1`, `gstockagent`, `newnanoalpaca*`, `bags*`** — superseded
  versions and paused tracks. In a rebuild these are git history, not directories.
- **Hundreds of root one-offs**: `test_*_vs_chronos2.py`, `quick_*`, `compare_*`, `profile_*`,
  dozens of `sweep_*.py`/`train_v*.py`. Collapse into **one parametrized sweep entrypoint + a
  config file**. Single biggest source of sprawl.
- **Most `*progress*.md`/`*best.md`.** Keep `*prod.md` (current state) and one rolling research
  log.

**Target shape:** ~6 dirs instead of 199 — `core/` (spine+broker), `sim/` (the judge),
`forecast/`, `models/` (xgb + one rl), `research/` (autoresearch loop + config-driven sweeps +
leaderboards), `ops/` (deploy + prod docs).

## Running your own autoresearch loop

Pipeline is **export → sweep → gate**:

```bash
# 1. Pack forecast-augmented market data into the binary the C sim reads
make export        # or: python pufferlib_market/export_data.py --symbols ... --output .../market_data.bin

# 2. Run the loop — N trials, fixed wall-clock budget per trial, append to a fresh leaderboard
python pufferlib_market/autoresearch_rl.py \
  --train-data <train.bin> --val-data <val.bin> --holdout-data <holdout.bin> \
  --time-budget 300 --max-trials 50 \
  --holdout-n-windows 20 --holdout-max-leverage 1.0 \
  --leaderboard pufferlib_market/MY_run_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/MY_run
# (--stocks12 / --stocks / --a40-mode / --h100-mode select preset universes & sweep grids)

# 3. Gate the winners on the honest judge before believing anything
python scripts/eval_100d.py --checkpoint <best.pt> \
  --n-windows 30 --window-days 100 --monthly-target 0.27 \
  --decision-lag 2 --fail-fast-max-dd 0.20
```

Three things to get right or your stats will lie to you:
- **Trust the holdout, not the training Sortino.** The loop ranks on held-out robustness and
  down-ranks negative-holdout trials; `eval_100d.py` on *unseen* windows is the real number.
  Soft-fill training Sortino has lookahead bias.
- **Sweep slippage cells (0/5/10/20 bps) and report the worst**, at `decision_lag≥2`, binary
  fills. A strategy that only works at 0 bps / lag 0 is a data leak, not an edge.
- **Fail-fast** (`--fail-fast-max-dd`) so duds die in seconds — what makes a 50-trial loop cheap
  enough to run continuously.

> Note: a detailed breakdown of the actual preset grids / sweep knobs / holdout ranking formula
> is in `RL_AUTORESEARCH_DEEPDIVE.md`.
