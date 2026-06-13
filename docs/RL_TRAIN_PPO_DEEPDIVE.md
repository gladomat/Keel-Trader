# `pufferlib_market/train.py` — PPO Loop, Pinned

> Exact mechanics of the flagship RL training loop: policy, env, rollout, GAE, advantage
> normalization (incl. the GSPO/group-relative variant), the PPO update, and the three CUDA
> acceleration paths. Companion to `RL_AUTORESEARCH_DEEPDIVE.md`. Generated 2026-06-13.
> All line numbers refer to `pufferlib_market/train.py` unless noted.

## TL;DR

It's **canonical PPO-clip** (single value head, GAE-λ, clipped surrogate + clipped/unclipped
value loss + entropy bonus) wrapped in a heavily performance-optimised harness. The novel/
non-textbook parts are: (1) a pluggable **advantage-normalization mode** including a
group-relative (GSPO-like) variant, (2) **three mutually-exclusive CUDA acceleration strategies**,
(3) **BF16-forward / FP32-loss** split for numerical safety, (4) an aggressive **per-minibatch
stability guard** that can skip updates / back off LR / abort, and (5) **production-realism
enforcement** (`decision_lag` floored) baked into training, not just eval.

`train(args)` spans lines 1076–2517. The loss is `_ppo_loss` (1456–1500); the update loop is
1730–2090.

## 1. Policies (lines 439–1031)

Five architectures behind `--arch`; default `mlp`:
- **`TradingPolicy`** (439) — MLP actor-critic, the default. Optional per-symbol input LayerNorm
  (each symbol's F features normalized independently, 550–558) and an optional **fused
  obs-norm + first-linear + ReLU** GPU path (570).
- **`ResidualTradingPolicy`** (667, `ResidualBlock` 652), **`TransformerTradingPolicy`** (749,
  with a `relu_sq`/ReLU² activation option), **`GRUTradingPolicy`** (854),
  **`DepthRecurrenceTradingPolicy`** (923).

Action space is **discrete** (a `Categorical` over `num_actions`); `get_action_and_value`
returns `(action, logprob, entropy, value)`. Short/sell actions can be masked via
`_mask_short_logits` (407) when `--disable-shorts`. Log-prob and entropy are computed **manually**
(softmax → log-softmax) rather than via `torch.distributions.Categorical`, to keep the graph
`fullgraph`-compilable and CUDA-graph-capturable (comment at 1475).

## 2. Environment & rollout (lines 1730–1807)

- Env is the **C/CUDA binding** (`pufferlib`), vectorized over `N = num_envs` (default 128):
  `binding.vec_step(vec_handle)` advances all envs; rewards/dones come back as numpy views over
  C buffers (zero-copy via `torch.from_numpy`, 1805–1806).
- Reward shaping is applied **inside the env** (config passed at construction, 1184–1194):
  `reward_scale`, `reward_clip`, `cash_penalty`, `trade_penalty`, `drawdown_penalty`,
  `downside_penalty`, `smooth_downside_penalty`(+temperature), `smoothness_penalty`,
  `fee_rate`, `fill_slippage_bps`. So the policy gradient never sees raw PnL — it sees the
  shaped, friction-inclusive reward. This is why the autoresearch reward knobs matter.
- **Observation normalization**: online **Welford** running mean/var (`RunningObsNorm`, 82–118),
  updated every step (`obs_norm.update`). Two consumption paths: CPU-normalize-then-transfer, or
  the fused on-device kernel (policy gets pushed Welford stats via `set_obs_norm_stats`, 1783).
- Rollout collects `T = rollout_len` (default 256) steps into preallocated buffers
  (`buf_obs/act/logprob/value/reward/done`). Two inference paths per step:
  - **CUDA-graph fast path** (1758–1773): replay a captured inference graph; only the action
    tensor round-trips to CPU for the C env.
  - **Eager fallback** (1774–1799): standard `policy.get_action_and_value` under
    `torch.inference_mode()`.

## 3. GAE (lines 1032–1061, 1809–1860)

Textbook GAE-λ. After the rollout, bootstrap `next_value = policy.get_value(next_obs)` (1817),
then:
```
delta_t   = r_t + γ·V_{t+1}·(1−done_t) − V_t
A_t       = delta_t + γ·λ·(1−done_t)·A_{t+1}     # reverse scan
returns   = A + V
```
Two implementations selected at runtime: a **Triton GPU kernel** `compute_gae_gpu` when
`HAS_TRITON_GAE and cuda` (1820), else a zero-alloc CPU scan `_compute_gae_cpu_inline` (1050)
using preallocated buffers. `γ=gamma` (0.99), `λ=gae_lambda` (0.95).

## 4. Advantage normalization (`pufferlib_market/advantage_utils.py::normalize_advantages`)

Called at 1846–1853 with `mode=args.advantage_norm`. This is the most research-distinctive knob.
Input is the `[T, N]` advantage tensor (T timesteps × N envs):

- **`global`** (default): standard batch z-score `(A − A.mean()) / (A.std + eps)`.
- **`per_env`**: z-score **each env's trajectory across time** independently
  (`mean/std` over `dim=0`). Rationale (from autoresearch comments): `global` can "collapse to
  NVDA-only" — per-env normalization prevents one high-variance symbol/env from dominating the
  gradient.
- **`group_relative`** (GSPO/GRPO-inspired, lines 47–74): the genuinely novel path.
  1. Start from `per_env` normalized advantages.
  2. Compute a **rollout-level score per env** = `rewards.sum(dim=0)` (total episode reward).
  3. Randomly **permute envs into groups** of `group_relative_size` (default 8/16).
  4. Within each group, z-score the scores, **clamp to ±`group_relative_clip`**, and form
     `weight = clamp(1 + group_relative_mix · rel, min=0.1)`.
  5. **Rescale** each env's advantages by its group weight, un-permute, renormalize globally.
  - If `group_relative_mix <= 0`, it short-circuits to just a global renorm of `per_env` (no
     grouping) — i.e. `mix` controls how strongly group-relative ranking reshapes advantages.

This imports the LLM-RL idea of *ranking trajectories within a group* into trading PPO: envs
that beat their randomly-assigned peer group get their advantages amplified, losers damped. It's
explicitly described in-code as "a conservative approximation," not true GSPO.

## 5. The PPO loss (`_ppo_loss`, lines 1456–1500)

```python
# BF16 autocast for the FORWARD pass only (disabled under cuda_graph_ppo); loss math in FP32
with autocast(bfloat16, enabled=_use_bf16 and not cuda_graph_ppo):
    logits, new_value = policy(obs)            # forward
logits = mask_short_logits(...)                # optional
log_probs_all = log_softmax(logits)            # manual, FP32
new_logprob   = gather(log_probs_all, actions)
entropy       = -(probs * log_probs_all).sum(-1)

log_ratio = (new_logprob - old_logprob).clamp(-10, 10)   # overflow guard
ratio     = log_ratio.exp()

pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * clamp(ratio, 1-clip_eps, 1+clip_eps)
pg_loss  = max(pg_loss1, pg_loss2).mean()                # canonical PPO-clip

if clip_vloss:                                            # optional value clipping
    v_clipped = old_values + clamp(new_value - old_values, -clip_eps, clip_eps)
    v_loss = 0.5 * max((new_value-returns)^2, (v_clipped-returns)^2).mean()
else:
    v_loss = 0.5 * ((new_value - returns)^2).mean()

ent_loss = entropy.mean()
loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss
```
Standard PPO. Notable details: log-ratio clamp at ±10 prevents `exp()` blow-ups; advantages are
**already normalized** before entering the loss (so there's no per-minibatch renorm here);
`clip_eps`, `vf_coef`, `ent_coef` are passed as **tensors** so they can be live-updated inside a
captured CUDA graph.

## 6. The update loop (lines 1865–2090)

```
for epoch in range(ppo_epochs):              # default 4
    shuffle = randperm(batch_size)           # batch_size = T*N
    for each minibatch (size = minibatch_size, default 2048):
        loss, pg, vl, el = _ppo_loss(minibatch...)
        loss.backward()
        # stability gate (see §7)
        clip_grad_norm_(policy.parameters(), max_grad_norm)   # default 0.5
        optimizer.step(); optimizer.zero_grad()
```
Two execution variants per minibatch:
- **CUDA-graph PPO path** (1900–1967): copy the minibatch into static tensors and `replay()` the
  captured update graph. ~10–20% faster; requires static shapes (hence fixed `minibatch_size`).
- **Eager path** (1967–2056): standard backward/step; also contains an optional second update on
  the same minibatch gated by ensemble entropy (1979).

`optimizer` is AdamW or **Muon** (`--optimizer muon`, with optional **NorMuon**
`--muon-norm-update` scaling the update norm to the param norm).

## 7. Stability machinery (the production-hardening that's easy to miss)

- **Per-minibatch stability classification** (`_classify_update_stability`, 168): inspects
  gradient norms each step. `grad_norm_warn_threshold` (50) logs; `grad_norm_skip_threshold`
  (1000) **skips the update entirely**; `unstable_update_patience` (3) consecutive unstable
  minibatches **aborts training** with a `RuntimeError` (1961/2050) — a dud config dies fast
  instead of burning the time budget.
- **LR backoff** (`_backoff_optimizer_lr`, 152): multiplies LR by `lr_backoff_factor` (0.5) down
  to `min_lr` (1e-6) on instability.
- **Entropy-collapse detection** (1651–1653): tracks entropy vs `log(num_actions)`; flags
  degenerate collapse to a single action.
- **Annealing** (120–136): `cosine_lr_with_warmup` (cosine LR with warmup floor) and
  `linear_anneal` for `ent_coef`/`clip_eps` when `--anneal-ent`/`--anneal-clip`. `--anneal-lr`
  is treated as near-mandatory (autoresearch notes `anneal_lr=False` collapses to hold-cash).
- **Val-neg early stopping** (1647–1649): stops if validation negative-return rate stays above a
  threshold for N consecutive evals — kills configs that look fine on training reward but lose
  out-of-sample.

## 8. Three CUDA acceleration strategies (mutually constrained)

1. **Inference CUDA graph** (`use_cuda_graph`, 1388–1444) — captures the rollout forward pass.
2. **PPO-update CUDA graph** (`--cuda-graph-ppo`, 1506–1605) — captures the backward+step;
   requires fixed minibatch shape and tensor-valued coefficients.
3. **`torch.compile(reduce-overhead)`** (1502–1538) — only when *not* using the manual PPO graph,
   because reduce-overhead captures its own internal graphs that conflict with manual capture.

`--no-cuda-graph` disables all of them (for shared-GPU compatibility); `--no-tf32` disables TF32
matmuls for max numerical fidelity. BF16 forward is disabled under `cuda_graph_ppo` (the graph
path keeps FP32 throughout).

## 9. Production-realism enforcement (lines 1064–1074)

`_validate_realism_args` runs first thing in `train()` and **forces**
`decision_lag` and `val_decision_lag` up to `PRODUCTION_DECISION_LAG` (≥2) via
`require_production_decision_lag`, unless `--allow-low-lag-diagnostics` is set. So you cannot
accidentally train/validate at lag 0 (which would leak lookahead) — the realism rule from
`CLAUDE.md` is enforced in code at the training entrypoint, not just in `eval_100d.py`.

## 10. How it maps back to the autoresearch knobs

Every `TrialConfig` field in `RL_AUTORESEARCH_DEEPDIVE.md` is a CLI flag consumed here:
- PPO core (`lr/clip_eps/ent_coef/gamma/gae_lambda/ppo_epochs/vf_coef/max_grad_norm`) → §5–§7.
- `advantage_norm` + `group_relative_*` → §4.
- Reward shaping (`*_penalty`, `reward_scale/clip`, `fee_rate`, `fill_slippage_bps`) → §2
  (applied in the C env).
- `obs_norm` → §2 Welford. `hidden_size/arch/optimizer/muon_norm_update` → §1.
- `use_bf16/cuda_graph_ppo/no_cuda_graph/no_tf32/num_envs/minibatch_size` → §8.
- Stability fields (`grad_norm_*`, `unstable_update_patience`, `lr_backoff_factor`, `min_lr`)
  → §7.

## 11. Verdict (keep / rebuild)

**Keep:** the loss is clean canonical PPO — no hidden cleverness to reverse-engineer. The
valuable, hard-to-rederive parts are (a) friction + decision-lag baked into *training*,
(b) the stability gate that fails duds fast, (c) Welford obs-norm with a fused kernel, and
(d) the `group_relative` advantage option as a genuine research lever.

**Rebuild / simplify:** the three-way CUDA-graph/compile/eager branching makes the update loop
hard to follow and is the most likely source of subtle correctness drift (e.g. coefficient
tensors that must be updated in-place for the captured graph). If rebuilding, pick **one**
acceleration path (torch.compile is the most maintainable) and delete the manual graph capture
unless the measured 10–20% is worth the complexity. The five policy classes are also mostly dead
weight — `mlp` is the default everywhere; keep one MLP + (optionally) one recurrent variant.
