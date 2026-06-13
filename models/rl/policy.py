"""RL policy serving + the policy -> Phase-3-gate adapter.

The whole point of the RL track is that a trained policy is judged by the SAME
out-of-sample gate (``research/eval.py``) every other candidate faces. A gate
policy is just a callable ``(obs, env) -> int``; this module turns an RL policy
into one via :func:`as_gate_policy`.

``LinearPolicy`` is a pure-stdlib argmax-over-(W·obs+b) policy — no torch — so the
RL->gate seam is end-to-end testable without the training stack. The trained torch
MLP is loaded by :func:`load_torch_policy` (lazy torch) and exposes the same
``act(obs) -> int`` interface, so swapping in the real policy changes nothing about
how the gate calls it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Protocol


class RLPolicy(Protocol):
    def act(self, obs: List[float]) -> int: ...


@dataclass
class LinearPolicy:
    """Pure argmax policy: action = argmax_a (sum_i W[a][i]*obs[i] + b[a]).

    ``weights`` is ``[num_actions][obs_size]``; ``bias`` is ``[num_actions]``.
    Deterministic and dependency-free — handy as a gate-testable stand-in and as a
    reference for the exported torch policy's forward pass.
    """
    weights: List[List[float]]
    bias: List[float]

    @property
    def num_actions(self) -> int:
        return len(self.weights)

    def logits(self, obs: List[float]) -> List[float]:
        out = []
        for a in range(len(self.weights)):
            row = self.weights[a]
            out.append(sum(row[i] * obs[i] for i in range(len(row))) + self.bias[a])
        return out

    def act(self, obs: List[float]) -> int:
        logits = self.logits(obs)
        best_i, best_v = 0, logits[0]
        for i in range(1, len(logits)):
            if logits[i] > best_v:
                best_i, best_v = i, logits[i]
        return best_i


def compute_gae(rewards, values, dones, gamma: float = 0.99, lam: float = 0.95):
    """Generalized advantage estimation (pure-python; bootstrap=0 at rollout tail).

    Lives here (torch-free) so the PPO update can use it and the stdlib tests can
    pin it without importing the training stack.
    """
    adv = [0.0] * len(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = [adv[t] + values[t] for t in range(len(rewards))]
    return adv, returns


def as_gate_policy(policy: RLPolicy) -> Callable[[List[float], object], int]:
    """Wrap an RLPolicy as a gate policy, clamping to the env's action space."""
    def _gate_policy(obs: List[float], env) -> int:
        action = policy.act(obs)
        if action < 0 or action >= env.num_actions:
            return 0  # invalid -> flat
        return action
    return _gate_policy


def load_torch_policy(checkpoint_path):  # pragma: no cover - needs torch
    """Load a trained torch MLP checkpoint and return an RLPolicy (argmax of logits)."""
    import torch

    from models.rl.train import MLPPolicy  # local import: train.py pulls torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    net = MLPPolicy(ckpt["obs_size"], ckpt["num_actions"], ckpt.get("hidden", 64))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    class _TorchPolicy:
        def act(self, obs: List[float]) -> int:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = net(x)
                return int(torch.argmax(logits, dim=-1).item())

    return _TorchPolicy()
