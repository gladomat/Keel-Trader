"""One PPO training track — trains against the ONE C sim (issue #1), no second sim.

Offline trainer (needs torch; NOT imported by the stdlib test suite). It steps the
exact C fill engine through ``sim/keel_sim.TradingEnv`` to collect rollouts — there
is deliberately no parallel/soft Python reimplementation of fills (the
``BINANCENEURAL_SIM_DEEPDIVE.md`` cautionary tale). If a differentiable sim is ever
wanted, it must wrap this same arithmetic and ship a ``temperature->0 => binary``
parity test; that is explicitly deferred.

The trained checkpoint is served via ``models/rl/policy.load_torch_policy`` and
judged by the SAME Phase-3 gate every other candidate faces.

Usage:
    python -m models.rl.train --data sim/data/sample.bin --out artifacts/rl/ppo.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.rl.policy import compute_gae
from sim.keel_sim import MarketData, TradingEnv


class MLPPolicy(nn.Module):
    """Shared-trunk actor-critic: logits over discrete actions + a value head."""

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def collect_rollout(env: TradingEnv, net: MLPPolicy, steps: int):
    """Step the C sim for ``steps`` transitions under the current policy."""
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    for _ in range(steps):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = net(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        next_obs, reward, terminal = env.step(int(action.item()))

        obs_buf.append(obs)
        act_buf.append(int(action.item()))
        logp_buf.append(float(dist.log_prob(action).item()))
        rew_buf.append(float(reward))
        val_buf.append(float(value.item()))
        done_buf.append(1.0 if terminal else 0.0)

        obs = next_obs if not terminal else env.reset()
    return obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf


def ppo_update(net, optimizer, batch, *, clip=0.2, epochs=4, vf_coef=0.5,
               ent_coef=0.01, minibatch=256):
    obs, acts, old_logp, adv, returns = batch
    obs_t = torch.tensor(obs, dtype=torch.float32)
    act_t = torch.tensor(acts, dtype=torch.long)
    old_logp_t = torch.tensor(old_logp, dtype=torch.float32)
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(returns, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = obs_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch):
            idx = perm[start:start + minibatch]
            logits, value = net(obs_t[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_t[idx])
            ratio = torch.exp(logp - old_logp_t[idx])
            a = adv_t[idx]
            pol_loss = -torch.min(ratio * a,
                                  torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            vf_loss = ((value - ret_t[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pol_loss + vf_coef * vf_loss - ent_coef * ent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train(data_path: Path, out_path: Path, *, iterations: int = 100,
          rollout_steps: int = 2048, max_steps: int = 720, hidden: int = 64,
          lr: float = 3e-4, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    md = MarketData.load(data_path)
    try:
        # Production-realism env (matches the gate): decision_lag>=2, binary fills.
        env = TradingEnv(md, max_steps=max_steps, decision_lag=2,
                         fill_buffer_bps=5.0, fee_rate=0.001)
        net = MLPPolicy(env.obs_size, env.num_actions, hidden)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for it in range(iterations):
            obs, acts, logp, rew, val, done = collect_rollout(env, net, rollout_steps)
            adv, returns = compute_gae(rew, val, done)
            ppo_update(net, optimizer, (obs, acts, logp, adv, returns))
            if it % 10 == 0:
                print(f"iter {it}: mean_reward={sum(rew) / len(rew):+.5f}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": net.state_dict(),
            "obs_size": env.obs_size,
            "num_actions": env.num_actions,
            "hidden": hidden,
        }, out_path)
        print(f"saved PPO checkpoint -> {out_path}")
        env.free()
    finally:
        md.free()
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Train one PPO policy against the C sim")
    ap.add_argument("--data", default="sim/data/sample.bin")
    ap.add_argument("--out", default="artifacts/rl/ppo.pt")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--rollout-steps", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(Path(args.data), Path(args.out), iterations=args.iterations,
          rollout_steps=args.rollout_steps, max_steps=args.max_steps, seed=args.seed)


if __name__ == "__main__":
    main()
