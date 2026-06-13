"""Broker configuration — PAPER-FIRST by default.

keel replaces moray's uncommitted ``env_real.py`` secrets file. Credentials and
the paper/live flag come from the environment; nothing is hard-coded or committed.

Safety invariant (docs/REBUILD_HANDOFF.md §4, HARD RULE territory):
  - ``PAPER`` defaults to True. Live trading requires BOTH ``ALP_PAPER=0`` (to flip
    PAPER off) AND, at the entry point, ``ALLOW_ALPACA_LIVE_TRADING=1`` (the
    explicit-enable gate enforced in the singleton). A missing/empty env => paper.
  - There is intentionally NO live entry point wired yet. Porting a process that
    can win the live-writer lock is a deliberate, reviewed step (HARD RULE 2:
    exactly one live writer). Until then this module only configures paper.
"""
from __future__ import annotations

import os


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_paper() -> bool:
    """True = paper (safe default). Only an explicit ALP_PAPER=0/false flips it."""
    raw = os.environ.get("ALP_PAPER")
    if raw is None:
        return True  # default: paper
    return not _env_truthy_is_live(raw)


def _env_truthy_is_live(raw: str) -> bool:
    """ALP_PAPER=0/false/no/off => live; anything else => paper."""
    return str(raw).strip().lower() in {"0", "false", "no", "off"}


PAPER: bool = _resolve_paper()

# Credentials are read lazily from env so importing this module never requires
# secrets. Live keys (ALP_KEY_ID_PROD / ALP_SECRET_KEY_PROD) vs paper keys
# (ALP_KEY_ID / ALP_SECRET_KEY) are selected by the caller based on PAPER.
ALP_KEY_ID = os.environ.get("ALP_KEY_ID", "")
ALP_SECRET_KEY = os.environ.get("ALP_SECRET_KEY", "")
ALP_KEY_ID_PROD = os.environ.get("ALP_KEY_ID_PROD", "")
ALP_SECRET_KEY_PROD = os.environ.get("ALP_SECRET_KEY_PROD", "")
