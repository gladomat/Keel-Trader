"""Trivial reference policies so the gate is testable with no trained model.

A *policy* is any callable ``policy(obs, env) -> int`` returning a valid action
index for the C sim (see ``sim/src/trading_env.c`` action decoding):

    0                       = go flat (close any position)
    1 .. side_block         = long actions  (symbol/alloc/level decoded)
    side_block+1 .. 2*sb    = short actions

With the default single allocation/level bin, action ``1`` is "long symbol 0".

These are deliberately dumb: a good gate must *reject* them. ``always_flat``
should score ~0 return (and so fail the promotion target), ``always_long`` takes
a single directional bet, and ``random_policy`` thrashes — all useful as
fail-fast / floor references.
"""
from __future__ import annotations

import random
from typing import Callable, List

Policy = Callable[[List[float], object], int]


def always_flat(obs: List[float], env) -> int:
    """Never take a position. Expected ~0 return; the gate must reject it."""
    return 0


def long_symbol(sym: int = 0) -> Policy:
    """Buy-and-hold a single symbol via its first long action.

    The long action for ``sym`` with the default bins is ``1 + sym``; we clamp to
    the env's action space so it stays valid for any bin configuration.
    """
    def _policy(obs: List[float], env) -> int:
        action = 1 + sym
        if action >= env.num_actions:
            action = 1  # fall back to the first long action
        return action
    return _policy


def always_long_0(obs: List[float], env) -> int:
    """Convenience: buy-and-hold symbol 0 (the canonical long reference)."""
    return long_symbol(0)(obs, env)


def random_policy(seed: int = 0) -> Policy:
    """Seeded uniform-random valid action — deterministic for a given seed."""
    rng = random.Random(seed)

    def _policy(obs: List[float], env) -> int:
        return rng.randrange(env.num_actions)

    return _policy
