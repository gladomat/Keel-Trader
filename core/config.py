"""Broker configuration — PAPER-FIRST by default.

keel replaces moray's uncommitted ``env_real.py`` secrets file. Credentials and
the paper/live flag come from the environment; nothing is hard-coded or committed.

Safety invariant (docs/REBUILD_HANDOFF.md §4, HARD RULE territory):
  - ``PAPER`` defaults to True. Live trading requires BOTH the paper flag flipped
    off AND, at the entry point, the explicit live-enable flag (gate enforced in
    the singleton + broker). A missing/empty env => paper.
  - There is intentionally NO live entry point wired yet. Porting a process that
    can win the live-writer lock is a deliberate, reviewed step (HARD RULE 2:
    exactly one live writer). Until then this module only configures paper.

K3 (#13) broker-neutral naming: the flags are read venue-neutrally. ``KEEL_PAPER``
/ ``KEEL_ALLOW_LIVE_TRADING`` are the primary names; the legacy ``ALP_PAPER`` /
``ALLOW_ALPACA_LIVE_TRADING`` names are still honoured as aliases so Kraken — and
any future venue — plug in without weakening the paper-first default.
"""
from __future__ import annotations

import os

# Paper flag: primary (broker-neutral) name first, legacy alias second.
PAPER_ENV_VARS = ("KEEL_PAPER", "ALP_PAPER")
# Explicit live-enable gate: primary (broker-neutral) name first, legacy second.
LIVE_ENABLE_ENV_VARS = ("KEEL_ALLOW_LIVE_TRADING", "ALLOW_ALPACA_LIVE_TRADING")


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_truthy_is_live(raw: str) -> bool:
    """A paper flag of 0/false/no/off => live; anything else => paper."""
    return str(raw).strip().lower() in {"0", "false", "no", "off"}


def _resolve_paper() -> bool:
    """True = paper (safe default). Only an explicit paper-flag=0/false flips it.

    The first paper env var that is actually set decides; unset => paper.
    """
    for name in PAPER_ENV_VARS:
        raw = os.environ.get(name)
        if raw is not None:
            return not _env_truthy_is_live(raw)
    return True  # default: paper


def live_trading_enabled() -> bool:
    """True iff ANY accepted live-enable env var is set truthy (alias support).

    Read live (not cached at import) so an entry point can set the gate before
    constructing the broker. Paper-first: with nothing set, this is False.
    """
    return any(_env_truthy(os.environ.get(n)) for n in LIVE_ENABLE_ENV_VARS)


PAPER: bool = _resolve_paper()

# Credentials are read lazily from env so importing this module never requires
# secrets. Live keys (ALP_KEY_ID_PROD / ALP_SECRET_KEY_PROD) vs paper keys
# (ALP_KEY_ID / ALP_SECRET_KEY) are selected by the caller based on PAPER.
ALP_KEY_ID = os.environ.get("ALP_KEY_ID", "")
ALP_SECRET_KEY = os.environ.get("ALP_SECRET_KEY", "")
ALP_KEY_ID_PROD = os.environ.get("ALP_KEY_ID_PROD", "")
ALP_SECRET_KEY_PROD = os.environ.get("ALP_SECRET_KEY_PROD", "")
