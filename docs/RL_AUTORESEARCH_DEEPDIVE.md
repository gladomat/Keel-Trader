# RL Autoresearch — Deep Dive

> How `pufferlib_market/autoresearch_rl.py` (4,190 lines) actually works: the swept
> hyperparameters, the preset grids, the mutation logic, the trial pipeline, and the ranking
> formula. Companion to `REPO_MAP.md` and `REBUILD_RECOMMENDATIONS.md`. Generated 2026-06-13.
> Line numbers refer to `pufferlib_market/autoresearch_rl.py` unless noted.

## TL;DR

`autoresearch_rl.py` is a **hand-rolled hyperparameter search** (not Optuna/nevergrad). It:
1. Holds ~637 hand-curated preset configs in named pools, plus a random-mutation generator.
2. For each trial, shells out to `python -m pufferlib_market.train` with ~40 flags, under a
   wall-clock budget.
3. Then shells out to a chain of **out-of-sample evaluators** (quick val → holdout → market
   validation → replay).
4. Merges all metrics, computes an overfitting-penalised `generalization_score`, and appends a
   row to a leaderboard CSV.
5. Random mutations are seeded from the **best config seen so far** (greedy hill-climb), per
   asset "track".

The whole design philosophy is: *training return is not trusted; the leaderboard ranks on
penalised out-of-sample generalization.*

## 1. The search space — `TrialConfig` (lines 96–158)

A dataclass of ~50 fields. The ones that actually get swept, grouped:

**PPO core:** `lr` (3e-4), `anneal_lr` (True), `clip_eps` (0.2)/`clip_eps_end`/`anneal_clip`,
`ent_coef` (0.05)/`ent_coef_end`/`anneal_ent`, `gamma` (0.99), `gae_lambda` (0.95),
`ppo_epochs` (4), `vf_coef` (0.5), `clip_vloss`, `max_grad_norm` (0.5), `weight_decay` (0.0),
`lr_schedule` ("none"/"cosine") + `lr_warmup_frac`/`lr_min_ratio`.

**Advantage shaping (the GRPO/GSPO-inspired knob):** `advantage_norm` ∈
{`global`, `per_env`, `group_relative`}, plus `group_relative_size`, `group_relative_mix`,
`group_relative_clip`. This is the most interesting research lever — it imports group-relative
advantage normalisation (GRPO/GSPO ideas from LLM-RL) into a trading PPO.

**Reward shaping (where trading-domain knowledge lives):** `reward_scale` (10), `reward_clip` (5),
`cash_penalty` (0.01), `trade_penalty` (churn deterrent), `drawdown_penalty`, `downside_penalty`,
`smooth_downside_penalty` + `smooth_downside_temperature` (Sortino-like soft downside),
`smoothness_penalty`. **`fill_slippage_bps`** and `fee_rate` (10 bps default) are *trained-with*
so the policy learns a robust edge, not a frictionless one.

**Model / runtime:** `hidden_size` (256–2048), `arch` ("mlp"), `optimizer` ("adamw"/"muon",
+ `muon_norm_update` = NorMuon), `num_envs` (128), `rollout_len` (256), `minibatch_size` (2048),
`obs_norm`, `max_leverage` (1.0), `short_borrow_apr`, `seed`.

**Hardware/perf:** `use_bf16` (True), `cuda_graph_ppo` (True), `no_cuda_graph`, `no_tf32`,
`time_budget_override`. These are auto-forced by `--h100-mode`/`--a40-mode` (e.g. H100 →
`num_envs=256, minibatch=4096`; lines ~3910–3917).

**Stability guards:** `grad_norm_warn_threshold` (50), `grad_norm_skip_threshold` (1000),
`unstable_update_patience` (3), `lr_backoff_factor` (0.5), `min_lr` (1e-6).

## 2. The preset pools (~637 named configs)

`_resolve_experiment_pool` (lines 2599–2617) selects by CLI flag:

| Flag | Pool | Notes |
|---|---|---|
| (default) | `EXPERIMENTS + CRYPTO34_HOURLY_EXPERIMENTS` | "crypto mode" |
| `--stocks` | `STOCK_EXPERIMENTS` | Alpaca daily equities |
| `--stocks12` | `STOCK_EXPERIMENTS + (non-h100 H100_STOCK_EXPERIMENTS)` | the stocks12 benchmark |
| `--h100-mode` | `H100_STOCK_EXPERIMENTS` | large-model configs needing H100 |

- **`EXPERIMENTS`** (lines 162–636): the crypto/general grid. Progresses from vanilla baselines
  (`baseline_anneal_lr`, `obs_norm`, `cosine_lr`) → regularization sweeps (`wd_*`, `slip_*bps`,
  `trade_pen_*`, `downside_pen`) → the **"robust_reg" champion family** (`robust_reg_tp005_ent`
  etc: `wd=0.05, slip=8bps, obs_norm, trade_penalty=0.005` ± entropy anneal) → multi-seed
  validation of champions → larger models (`h1536/h2048_robust_ent`) → the **GSPO/GRPO block**
  (`gspo_like_*`, `per_env_adv_*` — group-relative advantage variants). The naming *is* the
  research history: each entry is a hypothesis that was run and logged.
- **`CRYPTO34_HOURLY_EXPERIMENTS`** (lines 637–646): generated programmatically as the cross
  product `_C34H_VARIANTS × _C34H_SEEDS` on a `_C34H_BASE` — i.e. a seed-robustness sweep over a
  34-symbol hourly crypto base config.
- **`STOCK_EXPERIMENTS`** (lines 660–1786): the largest pool. Stock-specific deltas vs crypto are
  documented inline (lines 647–659): ~10 bps Alpaca fee, daily bars (`periods_per_year=252`),
  long-bias via heavy `short_borrow_apr` (no long-only flag in `train.py`), `anneal_lr` mandatory.
- **`STOCK_TP03_SEED_EXPERIMENTS`** (1787–1814) and **`H100_STOCK_EXPERIMENTS`** (1815–2140):
  targeted seed sweeps and big-model variants.

`_select_from_pool` (2578) supports `--start-from N` (offset into the pool) and
`--descriptions a,b,c` (run only named subsets) — how you resume or cherry-pick a sweep.

## 3. Random mutation — `mutate_config` (lines 2153–2219)

After the presets are exhausted, the loop generates `random_mut_*` trials by mutating the
**best config found so far** (greedy hill-climb). It picks **2–3 params at random** (line 2210)
from a grid and reseeds. Three constrained modes encode hard-won failure knowledge ("scar
tissue") — the most valuable non-obvious content in the file:

- **Default mode** (2184–2209): wide grid over hidden_size, lr, ent_coef, weight_decay,
  slippage, gamma, advantage_norm, group_relative_mix, reward_scale, cash/trade/drawdown/
  downside/smoothness penalties, obs_norm, anneal flags.
- **`stocks_mode`** (`--stocks`): clamps `lr ∈ {1e-4, 3e-4}`, `slip ∈ {0, 5}` bps,
  `hidden_size ∈ {512, 1024}`, `anneal_lr` locked True. Inline comments record *why*:
  > `h=256` is catastrophic (−121 to −146) on stocks11_2012; `slip>5bps` (8/10/12/15) all
  > collapse to hold-cash; `anneal_lr=False` collapses. Best known formula: `lr=3e-4, wd=0.01,
  > tp=0.05, slip=5bps, anneal_lr` (score −4.05).
- **`per_env_focused`** (`--per-env-focused`): tight space around a proven stocks12 config
  (`rmu8597`, seed 1168). Locks `anneal_lr=True`, `advantage_norm ∈ {per_env, group_relative}`
  (global "collapses to NVDA-only"), `hidden_size ∈ {256, 512}`; varies only ent_coef, slippage,
  drawdown/smoothness penalties, gamma.
- **`seed_only`** (`--seed-only`): mutates *only* the seed, around a pre-seeded winning formula —
  for pure seed-robustness validation. In stocks+seed_only mode the loop pre-seeds `best_config`
  with the confirmed winner so trial 1 already uses it (lines ~3863–3878), avoiding the default
  `wd=0/tp=0/slip=0` config that collapses to hold-cash.

`h2048` auto-bumps `num_envs=256, minibatch=4096` for throughput (2213–2216).

## 4. The trial pipeline — `run_trial` (lines 2661–3457)

Each trial is a **subprocess chain**, all via `_run_capture` (a captured subprocess with
timeout):

1. **Train** (line 2726): `python -u -m pufferlib_market.train` with ~40 flags mapped 1:1 from
   `TrialConfig` (data path, total-timesteps, all PPO + reward-shaping knobs, checkpoint dir).
   This is where the C/CUDA env + PPO loop actually runs. Wall-clock bounded by `--time-budget`
   (default 300s) or per-config `time_budget_override`.
2. **Quick val eval** (line ~2865, `qcmd`, 60s timeout): fast in-sim validation return —
   `best_val_return` tracked separately on the same scale.
3. **Main eval** (line ~3112, `eval_cmd`): fuller in-sim evaluation (`eval_num_episodes`, default
   100).
4. **Holdout eval** (line 3157): `python -m pufferlib_market.evaluate_holdout` on a *separate*
   holdout `.bin` — produces `holdout_robust_score`, `holdout_negative_return_rate`, robust
   worst-window returns. `--holdout-n-windows` (default 20), `--holdout-max-leverage` (1.0),
   `--holdout-fill-buffer-bps` (5).
5. **Market validation** (line 3212, optional): `python -m unified_orchestrator.market_validation`
   `--asset-class … --days … --decision-cadence hourly` — runs the policy through a more
   production-like market replay.
6. **Replay eval** (the `replay_eval_*` params, 2689–2697): hourly-intrabar replay, optionally
   running the actual hourly policy with robust start-states and a 5 bps fill buffer — the
   closest in-loop proxy to the production `eval_100d.py` gate.

All resulting JSONs are summarised (`summarize_holdout_payload`,
`summarize_market_validation_payload`, `summarize_replay_eval_payload`, lines 2377–2513) and
merged into one metrics dict per trial.

## 5. The ranking formula (the part that keeps it honest)

**`compute_generalization_metrics` (lines 2301–2351)** is the core anti-overfit logic:

```
generalization_score = mean(replay_combo_score, holdout_robust_score)
                       − 0.5  · train_val_gap_pct          # penalise train≫val overfit
                       − 0.25 · val_replay_gap_pct          # penalise val≫replay leakage
                       − 25.0 · holdout_negative_return_rate # heavily punish negative holdout windows
```

The `25×` penalty on the fraction of negative holdout windows is the dominant term — a config
that's great on average but loses in many windows is crushed. `overfit_gap_score =
replay_combo_score − generalization_score` is also recorded as a diagnostic.

**`select_rank_score` (lines 2514–2565)** then picks the leaderboard sort key via an "auto"
priority ladder (first non-null wins):
```
generalization_score → smooth_score → replay_combo_score → market_goodness_score →
holdout_robust_score → replay_hourly_policy_robust_worst_annualized_return_pct →
… → replay_hourly_return_pct → val_return
```
So `generalization_score` dominates when available; everything degrades gracefully toward raw
val_return only if richer evals are missing. `_leaderboard_sort_value` (2568) sorts the CSV by
`rank_score`, falling back to `val_return`.

## 6. The driver loop — `main` (lines 3457–4190)

- Infers the **track** from the data path (`_infer_track`, ~3822): `stocks_daily` /
  `binance_crypto` / `mixed` / `hourly_crypto`. A `BestKnownTracker` persists the best combined
  score *per track* across runs (warm-starts mutation hill-climbing).
- Iterates the resolved preset pool first; entries already in the leaderboard are skipped
  (`desc in existing_trials`) so runs are **resumable/append-only**. Non-`random` presets run as
  named configs; `random_*` slots call `mutate_config(best_config, …)`.
- Applies CLI global overrides (`--periods-per-year`, `--max-steps-override`,
  `--fee-rate-override`, `--seed`, `--eval-num-episodes-override`) and hardware-mode forcing
  (`--h100-mode`, `--a40-mode`).
- Stops at `--max-trials` (default 50). Writes results to `--leaderboard`
  (default `pufferlib_market/autoresearch_leaderboard.csv`); checkpoints under
  `--checkpoint-root`.

## 7. Key CLI flags (from `main`, lines 3457–3535)

| Flag | Default | Meaning |
|---|---|---|
| `--train-data` / `--val-data` | required | training + in-sim validation `.bin` |
| `--holdout-data` | none | separate holdout for `evaluate_holdout` |
| `--time-budget` | 300 | wall-clock seconds per trial |
| `--max-trials` | 50 | stop after N |
| `--leaderboard` | `…/autoresearch_leaderboard.csv` | append-only results |
| `--checkpoint-root` | `…/checkpoints/autoresearch` | per-trial checkpoints |
| `--stocks` / `--stocks12` / `--h100-mode` / `--a40-mode` | off | pool + hardware presets |
| `--seed-only` / `--per-env-focused` | off | constrained mutation modes |
| `--start-from N` / `--descriptions a,b` | 0 / all | resume / cherry-pick presets |
| `--holdout-n-windows` | 20 | robustness windows |
| `--holdout-max-leverage` | 1.0 | leverage cap in holdout |
| `--periods-per-year` | 8760 | 8760 hourly / 252 daily — annualisation |

## 8. How this differs from a "normal" HPO loop (worth keeping vs. rebuilding)

**Genuinely good, keep:**
- Out-of-sample-first ranking with an explicit overfit penalty (the `generalization_score`
  formula) — most HPO loops rank on val loss and overfit the validator. This one doesn't.
- Append-only, resumable leaderboards keyed by description; per-track best-known warm-start.
- Friction (`fee`, `slippage`) baked into *training*, not just eval.
- The constrained mutation modes that encode "what collapses" — that scar-tissue knowledge is
  expensive to rediscover.

**Debt / would rebuild:**
- ~637 hand-curated preset dicts inline in a 4,190-line file. This should be a config file
  (YAML/JSON) + a small sampler, not 2,000 lines of literal dicts. The presets are really a
  *log* of past hypotheses masquerading as code.
- Greedy 2–3-param hill-climb from best-so-far is weak vs. a real optimiser (TPE/CMA-ES/BO).
  It works because the presets do the heavy lifting, but a principled sampler over the same
  space + the same `generalization_score` objective would likely search better.
- The subprocess-per-eval chain (train → quick → eval → holdout → market_validation → replay)
  is robust but slow and stringly-typed (metrics passed as JSON dicts with ~15 fallback keys in
  `select_rank_score`). A typed result object would remove a class of silent-null bugs.

**If you rebuild the loop:** keep the objective (`generalization_score` + slippage-cell worst-case
+ 25× negative-window penalty) and the append-only leaderboard; replace the inline preset dump
with a config-driven sampler; gate every leaderboard winner through `scripts/eval_100d.py` before
believing it.
