"""Resolve the strategy-state directory (account locks, buy memory, event logs).

Ported from moray's unified_orchestrator.state_paths, trimmed to what the safety
spine needs. State lives under ``strategy_state/`` at the repo root by default,
overridable with the ``STATE_DIR`` env var. This directory is git-ignored — it
holds live locks and buy-price memory and must never be committed.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def resolve_state_dir(state_dir: str | Path | None = None) -> Path:
    """Resolve the strategy state directory from an explicit path or environment."""
    raw_path = state_dir if state_dir is not None else os.environ.get("STATE_DIR", "strategy_state")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path
