# Moray Monorepo — Map & Field Guide

> Read-only exploration deliverable. Nothing in the repo was modified; commands were inspection-only.
> Generated 2026-06-13.

## 1. Repo overview

A **production algorithmic-trading bot fused with a sprawling ML research lab** (298 MB,
~3,600 Python files, 199 top-level dirs, ~290 root scripts, ~150 markdown logs). It trades
US equities via **Alpaca** and crypto via **Binance**, and the bulk of the tree is
*experiments* searching for a strategy that clears a hard **27%/month-on-unseen-data** bar.
Stack: PyTorch + Lightning (+ some JAX), custom **C/CUDA market simulators** (pufferlib-style),
XGBoost, and time-series forecasters (**Chronos2** LoRAs, Toto). Three families of signal
generators compete: **RL policies** (PPO/GRPO), **XGBoost** directional classifiers, and
**LLM agents** (GPT/Claude/Qwen/Deepseek). The production spine is small and heavily guarded
(single-live-writer fcntl lock, death-spiral sell guard); everything else is research orbiting
it. State of play: the **XGBoost daily champion is the only thing live**; RL and LLM tracks are
validated-but-not-promoted because they haven't cleared the gate.

## 2. Subsystem map

- **Production spine** — the only money path. `alpaca_wrapper.py` (root) is the single Alpaca
  write surface; it imports `src/alpaca_singleton.py` which enforces the **one-live-writer lock**
  and the **death-spiral sell guard** at import. Deploys go *only* through
  `scripts/deploy_live_trader.sh` (registry: `xgb-daily-trader-live`, `trading-server`,
  `daily-rl-trader`). `src/trading_server/{server,client}.py` is an HTTP broker boundary.
  → consumes signals from the model tracks; gated by the eval harness.
- **Eval / promotion gate** — `scripts/eval_100d.py` runs the C++ binary-fill sim over rolling
  100-day unseen windows, worst slippage cell, median monthly ≥ 0.27. This is the judge every
  other subsystem answers to.
- **Market simulators** — *ground truth* is the **pufferlib C++ binary-fill sim**
  (`pufferlib_cpp_market_sim/`, also `c_market_sim/`, `market_sim_c/`). *Training-only soft sims*
  (`differentiable_market/`, `fast_market_sim/`, `binanceneural`) have lookahead bias — used for
  gradients, never trusted for promotion. Many legacy variants (`marketsimulator*`, `cppsimulator`,
  `frontiermarketsim`).
- **Forecasters** — `chronos2_trainer.py` fine-tunes Chronos2 LoRAs →
  `scripts/build_hourly_forecast_caches.py` writes parquet caches → `src/forecast_cache_lookup.py`
  reads them at trade time. Feeds RL/LLM features. (`cutechronos/`, `toto_features.py`,
  `chronos-forecasting/`.)
- **RL tracks** — see §3. Flagship `pufferlib_market/` (C/CUDA PPO).
- **XGBoost track** — `xgbnew/` (daily champion), `xgbnew_multiday/`, `boostbaseline/`. The
  currently-deployed strategy.
- **LLM-agent tracks** — `stockagent*` (~20 variants), `qwen_rl_trading/` + `trltraining/` (GRPO),
  `rl_trading_agent_binance/`.
- **Experiment orchestration** — `pufferlib_market/autoresearch_rl.py` drives hyperparameter
  sweeps; results → `*_leaderboard.csv`; `sweepresults/`, `hyperparam_*_results/`,
  `strategy_results/`; decision logs in `*prod.md`/`*progress*.md`/`*best.md`.
- **`src/` (239 files)** — clusters: brokers (`alpaca_*`, `binance_*`), trading_server, position
  sizing/Kelly (`advanced_position_sizing.py`), data/caching (`forecast_cache_lookup.py`,
  `*_data_utils.py`), config (`daily_stock_defaults.py`), monitoring/audits, risk guards
  (`price_guard`, `current_price_locked_execution`).

## 3. RL deep dive (comparative, all tracks)

There are **~12 parallel RL implementations** — a sign of aggressive search, not architecture.
They cluster into four lineages:

### A. PPO on the C/CUDA sim — the flagship
- `pufferlib_market/` is what the root `Makefile` builds (`setup.py build_ext`) and trains
  (`train.py`). Canonical PPO-clip loss; the update is at roughly
  `pufferlib_market/train.py:~1455–1540` *(line numbers [uncertain] — large file)*. Env is a
  **C binding** over a binary `.bin` market-data file (`export_data.py` packs Chronos2-forecast
  features). Discrete action = per-symbol position selection. Reward = clipped PnL with
  cash/drawdown/downside/trade-cost/smoothness penalties.
- Non-obvious engineering: online obs-normalization (Welford), BF16 forward / FP32 loss for
  numerics, optional **CUDA-graph replay of the PPO update**, cosine LR + entropy/clip annealing,
  Triton fused MLP/obs kernels, optional spectral reg. This is the most production-grade RL code.
- **Variants:** `gpu_trading_env/` (continuous Gaussian actions + bracket-order offsets, native
  CUDA env, `ppo_trainer.py:283–327`); `pufferlibtraining/algorithms/ppo_quantile.py`
  (distributional/quantile value head — **legacy**); `pufferlibtraining2/` (thin wrapper over
  upstream `pufferlib.pufferl`); `pufferlibtraining3/pufferrl.py` (hand-rolled PPO, experimental);
  `training/` (legacy monolith).

### B. Sharpness-Adjusted Proximal Policy (SAP) — active research, not deployed
- `sharpnessadjustedproximalpolicy/` couples PPO with a **SAM-style optimizer**
  (`sam_optimizer.py`): probes loss sharpness and modulates weight decay toward flat minima.
  Configurable objective (sortino/sharpe/return/log-return) + spectral reg, on the differentiable
  sim. `sharpnessadjustedproximalpolicy2/` is the older, superseded version. Leaderboards dated
  ~2026-03-29; results mixed, hasn't beaten the pufferlib baseline.

### C. LLM-RL (GRPO) — active research
- `trltraining/train_grpo.py` wraps HuggingFace TRL `GRPOTrainer` (+ optional vLLM).
  `qwen_rl_trading/` is the primary user: SFT warmstart → GRPO on Qwen2, reward from
  `reward.py::GRPORewardFn` (sortino-drawdown/sharpe/return) evaluating LLM-generated trading
  plans against market snapshots; prompts embed Chronos2 forecasts (`data_prompt.py`).

### D. Hybrid signal-calibration (not classical policy gradient)
- `rl_trading_agent_binance/train_directional.py:62–189` trains a small MLP `SignalCalibrator`
  by **backpropagating through the differentiable market sim** with `combined_sortino_pnl_loss`.
  Closest of the non-flagship tracks to actually executing (Binance), but see the caveat below.

**Data flow (all RL):** Chronos2 LoRA → forecast cache → `export_data.py` packs `.bin`/tensors →
env rollouts → PPO/GRPO update → checkpoint → `autoresearch_rl.py` sweep → leaderboard CSV →
(if it clears `eval_100d.py`) candidate for deploy.

> ⚠️ **Correction to flag:** an initial read labeled `pufferlib_market` and the Binance
> LLM/calibrator tracks as "LIVE." The deploy registry + `alpacaprod.md` say otherwise — the
> **daily-RL trader was stopped 2026-04-30** in favor of the XGB champion, and Binance is
> **paused**. Treat all RL "in production" claims as **research / validated-not-promoted**
> [corrected, high confidence].

### RL track comparison

| Track | Algorithm | Env | Training loop | Status |
|---|---|---|---|---|
| `pufferlib_market` | PPO + compile/CUDA graphs | C/CUDA binding | `train.py:~1455–1540` | Flagship; validated, not currently live |
| `gpu_trading_env` | PPO continuous Gaussian | CUDA kernel | `ppo_trainer.py:283–327` | Experimental |
| `pufferlibtraining` | PPO-Quantile | Python gym | `algorithms/ppo_quantile.py:44–80` | Legacy |
| `pufferlibtraining2` | wraps `pufferlib.pufferl` | Python gym | `trainer.py` | Infra wrapper |
| `pufferlibtraining3` | hand-rolled PPO | Python gym | `pufferrl.py` | Experimental |
| `sharpnessadjustedproximalpolicy` | PPO + SAM optimizer | differentiable sim | `trainer.py` / `sam_optimizer.py` | Active research |
| `sharpnessadjustedproximalpolicy2` | SAP (earlier) | differentiable sim | `trainer.py` | Legacy |
| `trltraining` | GRPO (TRL) | LLM | `train_grpo.py:101–150` | Active infra |
| `qwen_rl_trading` | GRPO + SFT warmstart | LLM (Qwen2) | `train_grpo.py:272–315` | Active research |
| `rl_trading_agent_binance` | signal calibration (MLP+backprop) | differentiable sim | `train_directional.py:62–189` | Active; live claim unverified |
| `training/` | mixed legacy PPO | Python gym | `ppo_trainer.py` | Legacy |
| `differentiable_market` | gradient descent (non-RL) | differentiable sim | `train.py` | Research |

## 4. Experiments deep dive (taxonomy)

Naming is the index. Decode it as `{model}{asset}{timeframe}{version}{strategy-modifier}`:
- **Version:** `v2…v9`, `v3timed`, `v41–v45` — monotonic improvement; highest = current
  (e.g. `neuraldailyv5` supersedes v2–v4).
- **Asset/timeframe infix:** `daily` (1x lev) / `hourly` / `crypto`|`binance` (2–3.4x) /
  `stocks` / `mixed23`|`mixed32` (stock+crypto blends) / `fdusd` (zero-fee stablecoin pairs).
- **Strategy modifiers:** `_maxdiff` (aggressive packing), `_entrytakeprofit`, `_twostage`,
  `_wide` (many symbols), `_leverage`, `_lora`, `_worksteal` (parallel scheduler).
- **Model markers:** `_neural`, `_xgb`, `_chronos2`, `_opus`/`_deepseek`/`_gpt`.

### Families

| Family | Purpose | Representative paths | Status |
|---|---|---|---|
| `neuraldaily*` / `neuralhourly*v5` | PPO on daily/hourly equity bars | `neuraldailyv5/`, `neuralhourlytradingv5/` | v5 active; v2–v4 legacy |
| `stockagent*` (~20) | LLM portfolio managers (GPT/Opus/Deepseek) | `stockagent/`, `stockagentdeepseek_*`, `stockagents.md` | research; `stockagent` primary |
| `pufferlib*` / `autoresearch*` | RL training + sweeps | `pufferlib_market/autoresearch_rl.py`, ~30 leaderboard CSVs | very active |
| `xgbnew*` | XGBoost directional classifier | `xgbnew/live_trader.py`, `xgbbest.md` | **daily live**; hourly data-blocked |
| `bags*` | Solana token trading | `bagsneural/`, `bagsreadme.md` | **paused** (env-gated) |
| `binance*`/`crypto*` | Margin shorts/longs, LoRA sweeps | `binanceprod.md`, `binanceleveragesui/` | paused, revalidating |
| `sharpness*`/`cvar*`/`sortino*` | Risk-objective experiments | `sharpnessadjustedproximalpolicy/` | research |
| `chronos*`/`toto*`/`cute*` | Forecasters (signal inputs) | `chronos2_trainer.py`, `sweep_configs.py` | active, feed everything |

### Lifecycle
Configured via in-dir `config.py` + argparse in `sweep_*.py`/`train_v*.py` + grids in
`sweep_configs.py`; launched via `run_*.py`/`launch_*.py`; results land in `*_leaderboard.csv`,
`sweepresults/`, `hyperparam_*_results/`, `strategy_results/`, JSON manifests (git hash + seed +
hardware), and wandb. **The markdown logs are the lab notebook**: `*prod.md` = live deploy state,
`*progress*.md` = chronological ops log, `*best.md` = champion frontier. `stockprogress.md`
defines a notable benchmark — train PPO on 12 equities in 300s on an A100, scored
`median_ann_return·(1+sortino)/(1+max_dd)` over 20 holdout windows.

**To reproduce a run:** find the family dir → its `config.py`/sweep script → run via the matching
`run_*`/`train_*` entrypoint → results append to that family's leaderboard CSV; validate with
`scripts/eval_100d.py` before considering deploy.

## 5. Signal vs. noise

**Worth your time (the load-bearing logic):**
1. `CLAUDE.md` + `AGENTS.md` + `alpacaprod.md` / `binanceprod.md` — the rules and current ground
   truth; read first.
2. `src/alpaca_singleton.py` + `alpaca_wrapper.py` — the entire production safety model
   (lock + death-spiral guard).
3. `scripts/deploy_live_trader.sh` + `scripts/eval_100d.py` — the only safe deploy path and the
   promotion gate.
4. `pufferlib_market/` (esp. `train.py`, `environment.py`, `autoresearch_rl.py`, `setup.py` +
   C sources) — flagship RL + sweep engine.
5. `xgbnew/live_trader.py` + `xgbbest.md` — the strategy that's actually live.
6. `src/forecast_cache_lookup.py` + `chronos2_trainer.py` — how forecasts reach trades.
7. `src/trading_server/{server,client}.py` — broker boundary.
8. `pufferlib_cpp_market_sim/` — ground-truth simulator.

**Safe to ignore (for understanding the system):**
- `.venv*/`, `chronos-forecasting/`, `modded-nanogpt/`, `nanochat`, `autoresearch*` upstream forks
  — vendored/reference deps.
- `sharpnessadjustedproximalpolicy2/`, `pufferlibtraining/`+`training/`, `neuraldailyv2..v4`,
  `binanceexp1`, `gstockagent`, `newnanoalpaca*` — superseded by their successors.
- `bags*` — explicitly paused; only relevant if reviving Solana trading.
- The hundreds of one-off `test_*_vs_chronos2.py`, `quick_*`, `profile_*`, `compare_*` root
  scripts — throwaway probes; noise unless chasing a specific number.
- Most `*progress*.md` beyond the latest of each family — historical narrative, not current state.

## 6. Open questions / gaps

- **Exact prod model config** — `alpacaprod.md` is consistent that XGB-daily is champion, but
  specifics (ensemble seeds, `top_n`/`allocation`, crypto weekend sleeve sizing) came from agent
  reads not independently re-verified line-by-line. [medium confidence]
- **`pufferlib_market/train.py` PPO line numbers** — cited `:1455–1540` for the loss; treat as
  approximate until opened. [uncertain]
- **Is anything besides XGB-daily touching real money now?** Binance is "paused" with 6 covered
  margin positions lingering; the Binance LLM/calibrator "live" claims are unverified and likely
  paper or stale. Worth a human confirm against running systemd units. [uncertain]
- **`deployments/live_trader_history.log` was empty** in this read — the audit trail of
  who-was-live isn't populated where expected; may live elsewhere or be gitignored.
- **SAP vs flagship** — whether sharpness-adjusted PPO actually beats `pufferlib_market` at scale
  is unresolved in the leaderboards. [uncertain]
