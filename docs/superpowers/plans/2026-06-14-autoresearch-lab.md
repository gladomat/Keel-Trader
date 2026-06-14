# Autoresearch Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `research/lab/` — an agentic loop where an AI agent rewrites one mutable `strategy.py` each iteration, scored by the existing frozen out-of-sample gate, recorded to an append-only leaderboard, driven autonomously under a wall-clock budget.

**Architecture:** A single mutable file (`strategy.py`, the `build_policy` seam) is the only thing the agent edits. A frozen harness runs it through `research/eval.py` (train + unseen split) and `generalization_score`. A subprocess driver assembles context, invokes a configurable mutator command, runs each trial under a timeout, and appends results. Existing `research/autoresearch.py` is untouched.

**Tech Stack:** Python 3 stdlib only (ctypes sim, csv, importlib, subprocess, hashlib, json). No pytest — tests are plain-assert scripts run via `PYTHONPATH=. python3 tests/test_lab.py`, mirroring `tests/test_autoresearch.py`. Reuses `research.eval.evaluate`, `research.autoresearch.generalization_score`, `models.xgb.backtest`.

---

## File Structure

- Create `research/lab/__init__.py` — package marker.
- Create `research/lab/strategy.py` — THE mutable file; `build_policy(md, *, seed=0) -> Policy`; trial-0 = current baseline.
- Create `research/lab/harness.py` — frozen `run_one()` trial runner + `TrialResult` + hash helpers + `main()` for `make lab-trial`.
- Create `research/lab/leaderboard.py` — frozen append-only CSV writer + manifest + `archive_strategy()`.
- Create `research/lab/drive.py` — autonomous driver (context, mutator subprocess, budget, guardrails, keep-best).
- Create `research/lab/program.md` — human-authored research brief.
- Create `tests/test_lab.py` — stdlib-assert tests (stub mutator, no LLM).
- Modify `Makefile` — add `lab-trial`, `lab-drive`, `test-lab`; wire `test-lab` into `test`.

---

## Task 1: Package skeleton + mutable strategy.py (trial-0 baseline)

**Files:**
- Create: `research/lab/__init__.py`
- Create: `research/lab/strategy.py`
- Test: `tests/test_lab.py`

- [ ] **Step 1: Create the package marker**

Create `research/lab/__init__.py`:

```python
"""Autoresearch Lab: an agentic strategy-discovery loop.

The agent rewrites ONLY ``research/lab/strategy.py`` each iteration; everything
else here is the frozen scorer/harness. See ``program.md``.
"""
```

- [ ] **Step 2: Write the trial-0 mutable strategy**

Create `research/lab/strategy.py`:

```python
"""THE mutable file. The autoresearch agent rewrites this whole file each trial.

Contract (DO NOT CHANGE THE SIGNATURE):

    def build_policy(md: MarketData, *, seed: int = 0) -> Policy

``Policy = Callable[[obs, env], int]`` — the C sim calls it bar-by-bar with the
CURRENT bar's observation only (lookahead is impossible). Return a callable that
maps an observation to an action int: 0 = flat, 1 + sym = long symbol ``sym``.

Inside ``build_policy`` you may do anything: derive features from ``obs``,
accumulate rolling state across bars, implement any signal, sizing, entry/exit
or holding logic. You may NOT edit any other file.

Trial-0 ships the current repo baseline so the loop starts from a known number.
"""
from __future__ import annotations

from models.xgb.backtest import baseline_score_fn, make_strategy_policy
from research.policies import Policy
from sim.keel_sim import MarketData


def build_policy(md: MarketData, *, seed: int = 0) -> Policy:
    return make_strategy_policy(
        baseline_score_fn(), md.num_symbols, md.features_per_sym
    )
```

- [ ] **Step 3: Write the failing smoke test**

Create `tests/test_lab.py`:

```python
"""Autoresearch Lab tests. Stdlib-only (plain asserts, no pytest).

Run: PYTHONPATH=. python3 tests/test_lab.py   (or `make test-lab`)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from sim.keel_sim import MarketData
from sim.make_sample_data import make_sample


def test_baseline_strategy_builds_runnable_policy():
    import research.lab.strategy as strat
    from research.eval import evaluate

    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "lab.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        md = MarketData.load(data)
        try:
            policy = strat.build_policy(md, seed=0)
            v1 = evaluate(md=md, policy=policy, n_windows=4, window_steps=100, seed=3)
            v2 = evaluate(md=md, policy=policy, n_windows=4, window_steps=100, seed=3)
        finally:
            md.free()
        assert isinstance(v1.promote, bool)
        # Seeded gate is reproducible for a fixed policy.
        assert abs(v1.worst_cell_median_monthly - v2.worst_cell_median_monthly) < 1e-9
    print("ok test_baseline_strategy_builds_runnable_policy")


if __name__ == "__main__":
    test_baseline_strategy_builds_runnable_policy()
    print("all lab tests passed")
```

- [ ] **Step 4: Build the sim and run the test (expect PASS — baseline already exists)**

Run: `make build-sim && PYTHONPATH=. python3 tests/test_lab.py`
Expected: `ok test_baseline_strategy_builds_runnable_policy` then `all lab tests passed`.

(Note: `evaluate` is called with keyword args because its signature is
`evaluate(policy, md, *, ...)`; passing `md=` and `policy=` by keyword is valid.)

- [ ] **Step 5: Commit**

```bash
git add research/lab/__init__.py research/lab/strategy.py tests/test_lab.py
git commit -m "feat(lab): package skeleton + trial-0 baseline strategy seam"
```

---

## Task 2: Frozen trial harness (`run_one` + `TrialResult` + hashes)

**Files:**
- Create: `research/lab/harness.py`
- Test: `tests/test_lab.py`

- [ ] **Step 1: Write the harness**

Create `research/lab/harness.py`:

```python
"""Frozen trial harness: load the mutable strategy, score it through the ONE gate.

``run_one`` imports ``build_policy`` from an arbitrary strategy file path (so the
driver can point at a temp copy), runs the existing out-of-sample gate twice — an
in-sample split and the unseen split — and ranks by the existing
``generalization_score``. The agent never touches anything imported here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from research.autoresearch import generalization_score
from research.eval import evaluate
from research.policies import Policy
from sim.keel_sim import MarketData

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The frozen scorer. If any of these change, recorded rows are no longer
# comparable — we hash them into every row so tampering is detectable.
_FROZEN_PATHS = [
    _REPO_ROOT / "research" / "eval.py",
    _REPO_ROOT / "sim" / "include" / "trading_env.h",
    _REPO_ROOT / "sim" / "src" / "trading_env.c",
    _REPO_ROOT / "sim" / "src" / "keel_sim.c",
]


@dataclass
class TrialResult:
    strategy_hash: str
    frozen_hash: str
    seed: int
    gate_seed: int
    train_median_monthly: float
    test_median_monthly: float
    generalization_score: float
    gate_promoted: bool
    promoted: bool
    trial_status: str          # "ok" | "error" | "timeout" | "rejected_write"
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialResult":
        return cls(**d)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hash() -> str:
    h = hashlib.sha256()
    for p in _FROZEN_PATHS:
        h.update(p.read_bytes())
    return h.hexdigest()


def load_build_policy(strategy_path: Path):
    """Import ``build_policy`` from an arbitrary file path (uncached, fresh each call)."""
    strategy_path = Path(strategy_path)
    mod_name = f"_lab_strategy_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strategy from {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_policy"):
        raise AttributeError(f"{strategy_path} has no build_policy()")
    return module.build_policy


def run_one(strategy_path: Path, md: MarketData, *, seed: int = 0,
            gate_seed: int = 0, train_frac: float = 0.6, n_windows: int = 8,
            window_steps: int = 120, overfit_penalty: float = 1.0) -> TrialResult:
    """Score the strategy at ``strategy_path`` and return a typed result."""
    s_hash = sha256_file(strategy_path)
    f_hash = frozen_hash()
    build_policy = load_build_policy(strategy_path)
    policy: Policy = build_policy(md, seed=seed)

    train_v = evaluate(policy, md, n_windows=n_windows, window_steps=window_steps,
                       seed=gate_seed, fail_fast=False,
                       offset_lo_frac=0.0, offset_hi_frac=train_frac)
    test_v = evaluate(policy, md, n_windows=n_windows, window_steps=window_steps,
                      seed=gate_seed, offset_lo_frac=train_frac, offset_hi_frac=1.0)

    gen = generalization_score(train_v.worst_cell_median_monthly,
                               test_v.worst_cell_median_monthly, overfit_penalty)
    return TrialResult(
        strategy_hash=s_hash, frozen_hash=f_hash, seed=seed, gate_seed=gate_seed,
        train_median_monthly=train_v.worst_cell_median_monthly,
        test_median_monthly=test_v.worst_cell_median_monthly,
        generalization_score=gen, gate_promoted=test_v.promote,
        promoted=test_v.promote, trial_status="ok", reason=test_v.reason,
    )


def failed_result(strategy_path: Optional[Path], status: str, reason: str) -> TrialResult:
    """A non-ok trial (error/timeout/rejected_write) with sentinel metrics."""
    s_hash = sha256_file(strategy_path) if strategy_path and Path(strategy_path).exists() else ""
    return TrialResult(
        strategy_hash=s_hash, frozen_hash=frozen_hash(), seed=-1, gate_seed=-1,
        train_median_monthly=math.nan, test_median_monthly=math.nan,
        generalization_score=-math.inf, gate_promoted=False, promoted=False,
        trial_status=status, reason=reason,
    )
```

- [ ] **Step 2: Add failing tests for the harness**

Append to `tests/test_lab.py` (above the `__main__` block):

```python
def test_run_one_reproducible_and_typed():
    import research.lab.harness as H
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "lab.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        strat = Path("research/lab/strategy.py")
        md = MarketData.load(data)
        try:
            r1 = H.run_one(strat, md, seed=0, gate_seed=5, n_windows=6, window_steps=100)
            r2 = H.run_one(strat, md, seed=0, gate_seed=5, n_windows=6, window_steps=100)
        finally:
            md.free()
        assert r1.trial_status == "ok"
        assert len(r1.strategy_hash) == 64 and len(r1.frozen_hash) == 64
        assert abs(r1.generalization_score - r2.generalization_score) < 1e-9
        assert isinstance(r1.gate_promoted, bool)
    print("ok test_run_one_reproducible_and_typed")


def test_failed_result_is_not_promoted():
    import research.lab.harness as H
    r = H.failed_result(None, "timeout", "ran past budget")
    assert r.trial_status == "timeout"
    assert r.promoted is False and r.gate_promoted is False
    assert r.to_dict()["trial_status"] == "timeout"
    print("ok test_failed_result_is_not_promoted")
```

And add to the `__main__` block:

```python
    test_run_one_reproducible_and_typed()
    test_failed_result_is_not_promoted()
```

- [ ] **Step 3: Run tests (expect PASS)**

Run: `PYTHONPATH=. python3 tests/test_lab.py`
Expected: all three `ok ...` lines, then `all lab tests passed`.

- [ ] **Step 4: Commit**

```bash
git add research/lab/harness.py tests/test_lab.py
git commit -m "feat(lab): frozen trial harness (run_one, TrialResult, hashes)"
```

---

## Task 3: Append-only leaderboard + strategy archive

**Files:**
- Create: `research/lab/leaderboard.py`
- Test: `tests/test_lab.py`

- [ ] **Step 1: Write the leaderboard module**

Create `research/lab/leaderboard.py`:

```python
"""Append-only leaderboard for the lab. Never truncates; header written once.

Each row carries a reproducibility manifest (git hash, hardware, seeds) plus the
content hash of the exact ``strategy.py`` scored, and that file is archived to
``<archive_dir>/<strategy_hash>.py`` so every champion is byte-for-byte
recoverable.
"""
from __future__ import annotations

import csv
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

from research.lab.harness import TrialResult

LAB_LEADERBOARD_COLUMNS = [
    "timestamp", "git_hash", "hardware", "seed", "gate_seed",
    "strategy_hash", "frozen_hash", "train_median_monthly",
    "test_median_monthly", "generalization_score",
    "gate_promoted", "promoted", "trial_status", "reason",
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def archive_strategy(strategy_path: Path, strategy_hash: str,
                     archive_dir: Path) -> Path:
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{strategy_hash}.py"
    if not dest.exists():
        shutil.copyfile(strategy_path, dest)
    return dest


def append_row(path: Path, result: TrialResult, *,
               git_sha: Optional[str] = None,
               hardware: Optional[str] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    git_sha = git_sha if git_sha is not None else git_hash()
    hardware = hardware if hardware is not None else platform.platform()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LAB_LEADERBOARD_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": ts, "git_hash": git_sha, "hardware": hardware,
            "seed": result.seed, "gate_seed": result.gate_seed,
            "strategy_hash": result.strategy_hash, "frozen_hash": result.frozen_hash,
            "train_median_monthly": f"{result.train_median_monthly:.6f}",
            "test_median_monthly": f"{result.test_median_monthly:.6f}",
            "generalization_score": f"{result.generalization_score:.6f}",
            "gate_promoted": int(result.gate_promoted),
            "promoted": int(result.promoted),
            "trial_status": result.trial_status,
            "reason": result.reason,
        })
    return path


def read_rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def best_ok_row(rows: Sequence[dict]) -> Optional[dict]:
    """Highest generalization_score among status==ok rows (the champion)."""
    ok = [r for r in rows if r.get("trial_status") == "ok"]
    if not ok:
        return None
    return max(ok, key=lambda r: float(r["generalization_score"]))
```

- [ ] **Step 2: Add failing tests**

Append to `tests/test_lab.py`:

```python
def test_leaderboard_append_only_and_archive():
    import research.lab.harness as H
    import research.lab.leaderboard as LB
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "lab.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        strat = Path("research/lab/strategy.py")
        lb = Path(d) / "leaderboard.csv"
        arch = Path(d) / "strategies"
        md = MarketData.load(data)
        try:
            r = H.run_one(strat, md, seed=0, gate_seed=1, n_windows=6, window_steps=100)
        finally:
            md.free()
        LB.append_row(lb, r, git_sha="deadbeef", hardware="test")
        LB.append_row(lb, r, git_sha="deadbeef", hardware="test")
        LB.archive_strategy(strat, r.strategy_hash, arch)

        rows = LB.read_rows(lb)
        assert len(rows) == 2                                  # append-only
        assert set(rows[0].keys()) == set(LB.LAB_LEADERBOARD_COLUMNS)
        archived = arch / f"{r.strategy_hash}.py"
        assert archived.exists()
        # Archived copy is byte-identical -> re-hash matches the row.
        assert H.sha256_file(archived) == r.strategy_hash
        assert LB.best_ok_row(rows)["strategy_hash"] == r.strategy_hash
    print("ok test_leaderboard_append_only_and_archive")
```

Add to `__main__`:

```python
    test_leaderboard_append_only_and_archive()
```

- [ ] **Step 3: Run tests (expect PASS)**

Run: `PYTHONPATH=. python3 tests/test_lab.py`
Expected: new `ok test_leaderboard_append_only_and_archive` line included.

- [ ] **Step 4: Commit**

```bash
git add research/lab/leaderboard.py tests/test_lab.py
git commit -m "feat(lab): append-only leaderboard + strategy archive"
```

---

## Task 4: One-trial runner CLI (`make lab-trial`)

**Files:**
- Modify: `research/lab/harness.py` (add `main()` + arg parsing)
- Modify: `Makefile`
- Test: `tests/test_lab.py`

- [ ] **Step 1: Add `main()` to harness.py**

Append to `research/lab/harness.py`:

```python
def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run ONE lab trial on the current strategy")
    ap.add_argument("--data", default="sim/data/sample.bin")
    ap.add_argument("--strategy", default="research/lab/strategy.py")
    ap.add_argument("--leaderboard", default="artifacts/lab/leaderboard.csv")
    ap.add_argument("--archive-dir", default="artifacts/lab/strategies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window-steps", type=int, default=120)
    ap.add_argument("--json", action="store_true",
                    help="print result JSON and DO NOT append (driver mode)")
    args = ap.parse_args()

    md = MarketData.load(args.data)
    try:
        result = run_one(Path(args.strategy), md, seed=args.seed,
                         gate_seed=args.gate_seed, n_windows=args.windows,
                         window_steps=args.window_steps)
    finally:
        md.free()

    if args.json:
        print(json.dumps(result.to_dict()))
        return

    # Standalone mode: archive + append + human summary.
    from research.lab import leaderboard as LB
    LB.archive_strategy(Path(args.strategy), result.strategy_hash, Path(args.archive_dir))
    LB.append_row(Path(args.leaderboard), result)
    flag = "PROMOTE" if result.promoted else "reject "
    print(f"[lab] [{flag}] gen={result.generalization_score:+.4f} "
          f"(train={result.train_median_monthly:+.4f} "
          f"test={result.test_median_monthly:+.4f}) -> {args.leaderboard}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add Makefile targets**

In `Makefile`, add `lab-trial` and `lab-drive` to the `.PHONY` line, then add (after the `walkforward` target):

```makefile
LAB_DATA ?= $(KRAKEN_BIN)
TRIALS   ?= 50

lab-trial: build-sim ## run ONE lab trial on the current research/lab/strategy.py
	PYTHONPATH=. $(PYTHON) -m research.lab.harness --data $(LAB_DATA) \
		--leaderboard artifacts/lab/leaderboard.csv

test-lab: build-sim ## pin the lab harness/leaderboard/driver (stub mutator, no LLM)
	PYTHONPATH=. $(PYTHON) tests/test_lab.py
```

- [ ] **Step 3: Add failing test for the `--json` runner**

Append to `tests/test_lab.py`:

```python
def test_harness_cli_json_does_not_append():
    import json
    import os
    import subprocess
    import sys
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "lab.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        lb = Path(d) / "leaderboard.csv"
        env = dict(os.environ, PYTHONPATH=".")
        out = subprocess.check_output(
            [sys.executable, "-m", "research.lab.harness", "--json",
             "--data", str(data), "--strategy", "research/lab/strategy.py",
             "--leaderboard", str(lb), "--windows", "6", "--window-steps", "100",
             "--gate-seed", "2"],
            env=env, text=True,
        )
        result = json.loads(out.strip().splitlines()[-1])
        assert result["trial_status"] == "ok"
        assert not lb.exists()          # --json must NOT append
    print("ok test_harness_cli_json_does_not_append")
```

Add to `__main__`:

```python
    test_harness_cli_json_does_not_append()
```

- [ ] **Step 4: Run tests + smoke the target (expect PASS)**

Run: `PYTHONPATH=. python3 tests/test_lab.py`
Expected: `ok test_harness_cli_json_does_not_append` included.

- [ ] **Step 5: Commit**

```bash
git add research/lab/harness.py Makefile tests/test_lab.py
git commit -m "feat(lab): one-trial runner CLI + make lab-trial/test-lab"
```

---

## Task 5: Autonomous driver — core loop (stub mutator)

**Files:**
- Create: `research/lab/drive.py`
- Test: `tests/test_lab.py`

- [ ] **Step 1: Write the driver core**

Create `research/lab/drive.py`:

```python
"""Autonomous driver: loop trials, mutating ONLY research/lab/strategy.py.

Each iteration: assemble context (program.md + current strategy + recent
leaderboard + last result) -> run a configurable mutator command that rewrites
the strategy file -> run the trial in a subprocess under a wall-clock budget ->
append the result. Greedy keep-best: context surfaces the best-so-far champion.
The leaderboard stays append-only so no history is lost.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from research.lab.harness import TrialResult, failed_result
from research.lab import leaderboard as LB

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DriveConfig:
    strategy_path: Path
    leaderboard_path: Path
    archive_dir: Path
    program_path: Path
    data_path: Path
    mutator: str                      # shell command template (see run_mutator)
    trials: int = 50
    budget_seconds: float = 300.0
    base_seed: int = 0
    windows: int = 8
    window_steps: int = 120
    recent_rows: int = 12


def assemble_context(cfg: DriveConfig, last: Optional[TrialResult]) -> str:
    """Build the prompt bundle handed to the mutator (via the LAB_CONTEXT file)."""
    parts: list[str] = []
    if cfg.program_path.exists():
        parts.append("# program.md\n" + cfg.program_path.read_text())
    rows = LB.read_rows(cfg.leaderboard_path)
    if rows:
        recent = rows[-cfg.recent_rows:]
        parts.append("# recent leaderboard rows\n" + "\n".join(
            f"{r['trial_status']:<14} gen={r['generalization_score']} "
            f"test={r['test_median_monthly']} promoted={r['promoted']} "
            f"hash={r['strategy_hash'][:12]}" for r in recent))
        best = LB.best_ok_row(rows)
        if best:
            parts.append(f"# best-so-far champion\n"
                         f"generalization_score={best['generalization_score']} "
                         f"strategy_hash={best['strategy_hash']}")
    if last is not None:
        parts.append("# last trial result\n" + json.dumps(last.to_dict(), indent=2))
    parts.append("# current research/lab/strategy.py\n" + cfg.strategy_path.read_text())
    return "\n\n".join(parts)


def run_mutator(cfg: DriveConfig, context: str) -> None:
    """Run the configurable mutator. It must rewrite ONLY the strategy file.

    The context is written to a temp file; the command receives its path and the
    target strategy path via env vars LAB_CONTEXT_FILE and LAB_STRATEGY_FILE.
    Default production mutator (set by main()) shells out to a coding agent CLI.
    """
    ctx_file = cfg.strategy_path.parent / ".lab_context.txt"
    ctx_file.write_text(context)
    env = dict(os.environ,
               LAB_CONTEXT_FILE=str(ctx_file),
               LAB_STRATEGY_FILE=str(cfg.strategy_path),
               PYTHONPATH=str(_REPO_ROOT))
    subprocess.run(cfg.mutator, shell=True, env=env, cwd=str(_REPO_ROOT),
                   timeout=cfg.budget_seconds, check=True)


def run_trial_subprocess(cfg: DriveConfig, gate_seed: int) -> TrialResult:
    """Run one trial in a subprocess under the wall-clock budget."""
    env = dict(os.environ, PYTHONPATH=str(_REPO_ROOT))
    cmd = [sys.executable, "-m", "research.lab.harness", "--json",
           "--data", str(cfg.data_path), "--strategy", str(cfg.strategy_path),
           "--seed", str(cfg.base_seed), "--gate-seed", str(gate_seed),
           "--windows", str(cfg.windows), "--window-steps", str(cfg.window_steps)]
    try:
        out = subprocess.run(cmd, env=env, cwd=str(_REPO_ROOT), timeout=cfg.budget_seconds,
                             capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return failed_result(cfg.strategy_path, "timeout",
                             f"trial exceeded {cfg.budget_seconds}s budget")
    if out.returncode != 0:
        return failed_result(cfg.strategy_path, "error",
                             (out.stderr or "nonzero exit").strip()[:300])
    try:
        return TrialResult.from_dict(json.loads(out.stdout.strip().splitlines()[-1]))
    except Exception as e:  # malformed strategy output
        return failed_result(cfg.strategy_path, "error", f"unparseable trial output: {e}")


def drive(cfg: DriveConfig) -> list[TrialResult]:
    """Run cfg.trials iterations; return the results in order."""
    results: list[TrialResult] = []
    last: Optional[TrialResult] = None
    for i in range(cfg.trials):
        gate_seed = cfg.base_seed + i          # rotate so no single 'unseen' draw is memorized
        try:
            run_mutator(cfg, assemble_context(cfg, last))
        except subprocess.TimeoutExpired:
            res = failed_result(cfg.strategy_path, "timeout", "mutator exceeded budget")
        except subprocess.CalledProcessError as e:
            res = failed_result(cfg.strategy_path, "error", f"mutator failed: {e}")
        else:
            res = run_trial_subprocess(cfg, gate_seed)
        if res.trial_status == "ok":
            LB.archive_strategy(cfg.strategy_path, res.strategy_hash, cfg.archive_dir)
        LB.append_row(cfg.leaderboard_path, res)
        results.append(res)
        last = res
    return results
```

- [ ] **Step 2: Add failing test — multi-trial loop with a stub mutator**

Append to `tests/test_lab.py`:

```python
def _write_stub_mutator(dirpath: Path) -> Path:
    """A stub mutator script: writes a valid long-symbol strategy into the target."""
    script = dirpath / "stub_mutator.py"
    script.write_text(
        "import os\n"
        "target = os.environ['LAB_STRATEGY_FILE']\n"
        "open(target, 'w').write(\n"
        "    'from research.policies import long_symbol\\n'\n"
        "    'def build_policy(md, *, seed=0):\\n'\n"
        "    '    return long_symbol(0)\\n')\n"
    )
    return script


def test_driver_loop_appends_and_keeps_best():
    import sys
    import research.lab.drive as D
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        data = dd / "lab.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        strat = dd / "strategy.py"
        strat.write_text(
            "from research.policies import always_flat\n"
            "def build_policy(md, *, seed=0):\n"
            "    return always_flat\n"
        )
        program = dd / "program.md"
        program.write_text("# objective\nclear the gate net of friction\n")
        stub = _write_stub_mutator(dd)
        cfg = D.DriveConfig(
            strategy_path=strat, leaderboard_path=dd / "lb.csv",
            archive_dir=dd / "arch", program_path=program, data_path=data,
            mutator=f"{sys.executable} {stub}", trials=3, budget_seconds=60,
            windows=6, window_steps=100,
        )
        results = D.drive(cfg)
        rows = D.LB.read_rows(cfg.leaderboard_path)
        assert len(results) == 3 and len(rows) == 3          # append-only, one row/trial
        assert all(r.trial_status == "ok" for r in results)
        # Gate seed rotated across trials.
        assert [r.gate_seed for r in results] == [0, 1, 2]
        # Champion is archived and recoverable.
        best = D.LB.best_ok_row(rows)
        assert (cfg.archive_dir / f"{best['strategy_hash']}.py").exists()
    print("ok test_driver_loop_appends_and_keeps_best")
```

Add to `__main__`:

```python
    test_driver_loop_appends_and_keeps_best()
```

- [ ] **Step 3: Run tests (expect PASS)**

Run: `PYTHONPATH=. python3 tests/test_lab.py`
Expected: `ok test_driver_loop_appends_and_keeps_best` included.

- [ ] **Step 4: Commit**

```bash
git add research/lab/drive.py tests/test_lab.py
git commit -m "feat(lab): autonomous driver core loop + gate-seed rotation"
```

---

## Task 6: Driver guardrails — no-write-outside-strategy + budget timeout

**Files:**
- Modify: `research/lab/drive.py`
- Test: `tests/test_lab.py`

- [ ] **Step 1: Add the guard-tree snapshot/restore to drive.py**

In `research/lab/drive.py`, add a `guard_root` field to `DriveConfig` (default `None`):

```python
    guard_root: Optional[Path] = None    # dir whose files (except strategy) must not change
```

Add these helpers above `drive()`:

```python
def _tree_snapshot(root: Path) -> dict[str, bytes]:
    """Map of relative-path -> bytes for every file under ``root``."""
    root = Path(root)
    snap: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            snap[str(p.relative_to(root))] = p.read_bytes()
    return snap


def _enforce_only_strategy_changed(cfg: DriveConfig, before: dict[str, bytes]) -> Optional[str]:
    """Restore any non-strategy file the mutator touched; return a reason if it cheated."""
    after = _tree_snapshot(cfg.guard_root)
    allowed = str(cfg.strategy_path.resolve().relative_to(cfg.guard_root.resolve()))
    offenders: list[str] = []
    for rel in set(before) | set(after):
        if rel == allowed:
            continue
        if before.get(rel) != after.get(rel):
            offenders.append(rel)
            # Restore: rewrite original bytes, or delete a newly-created file.
            target = cfg.guard_root / rel
            if rel in before:
                target.write_bytes(before[rel])
            elif target.exists():
                target.unlink()
    if offenders:
        return "mutator wrote outside strategy.py: " + ", ".join(sorted(offenders))
    return None
```

Then in `drive()`, wrap the mutator call so the guard runs when `guard_root` is set:

```python
        try:
            before = _tree_snapshot(cfg.guard_root) if cfg.guard_root else None
            run_mutator(cfg, assemble_context(cfg, last))
            violation = _enforce_only_strategy_changed(cfg, before) if before is not None else None
        except subprocess.TimeoutExpired:
            res = failed_result(cfg.strategy_path, "timeout", "mutator exceeded budget")
        except subprocess.CalledProcessError as e:
            res = failed_result(cfg.strategy_path, "error", f"mutator failed: {e}")
        else:
            if violation:
                res = failed_result(cfg.strategy_path, "rejected_write", violation)
            else:
                res = run_trial_subprocess(cfg, gate_seed)
```

(Replace the previous `try/except/else` block from Task 5 with this expanded one. The `.lab_context.txt` scratch file is written into `strategy_path.parent`; if that dir is the guard root, exclude it by adding `if rel == ".lab_context.txt": continue` to the loop in `_enforce_only_strategy_changed`.)

- [ ] **Step 2: Add the guard exclusion for the scratch context file**

In `_enforce_only_strategy_changed`, inside the `for rel in ...` loop, after the `if rel == allowed: continue` line add:

```python
        if rel == ".lab_context.txt":
            continue
```

- [ ] **Step 3: Add failing tests — rejected write + timeout**

Append to `tests/test_lab.py`:

```python
def test_driver_rejects_writes_outside_strategy():
    import sys
    import research.lab.drive as D
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        data = dd / "lab.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        guard = dd / "lab"
        guard.mkdir()
        strat = guard / "strategy.py"
        strat.write_text(
            "from research.policies import always_flat\n"
            "def build_policy(md, *, seed=0):\n    return always_flat\n"
        )
        sentinel = guard / "frozen.py"
        sentinel.write_text("ORIGINAL\n")
        # Cheating mutator: edits the sentinel instead of the strategy.
        cheat = dd / "cheat.py"
        cheat.write_text(
            "import os\n"
            "open(os.path.join(os.path.dirname(os.environ['LAB_STRATEGY_FILE']),"
            " 'frozen.py'), 'w').write('HACKED\\n')\n"
        )
        cfg = D.DriveConfig(
            strategy_path=strat, leaderboard_path=dd / "lb.csv",
            archive_dir=dd / "arch", program_path=dd / "noprogram.md",
            data_path=data, mutator=f"{sys.executable} {cheat}", trials=1,
            budget_seconds=60, windows=6, window_steps=100, guard_root=guard,
        )
        results = D.drive(cfg)
        assert results[0].trial_status == "rejected_write"
        assert sentinel.read_text() == "ORIGINAL\n"      # restored
    print("ok test_driver_rejects_writes_outside_strategy")


def test_driver_kills_over_budget_trial():
    import sys
    import research.lab.drive as D
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        data = dd / "lab.bin"
        make_sample(data, num_symbols=2, num_timesteps=500, seed=7)
        strat = dd / "strategy.py"
        strat.write_text(
            "from research.policies import always_flat\n"
            "def build_policy(md, *, seed=0):\n    return always_flat\n"
        )
        # Mutator writes a strategy that sleeps past the budget in build_policy.
        slow = dd / "slow_mutator.py"
        slow.write_text(
            "import os\n"
            "open(os.environ['LAB_STRATEGY_FILE'], 'w').write(\n"
            "    'import time\\n'\n"
            "    'def build_policy(md, *, seed=0):\\n'\n"
            "    '    time.sleep(10)\\n'\n"
            "    '    from research.policies import always_flat\\n'\n"
            "    '    return always_flat\\n')\n"
        )
        cfg = D.DriveConfig(
            strategy_path=strat, leaderboard_path=dd / "lb.csv",
            archive_dir=dd / "arch", program_path=dd / "noprogram.md",
            data_path=data, mutator=f"{sys.executable} {slow}", trials=1,
            budget_seconds=1.0, windows=6, window_steps=100,
        )
        results = D.drive(cfg)
        assert results[0].trial_status == "timeout"
    print("ok test_driver_kills_over_budget_trial")
```

Add to `__main__`:

```python
    test_driver_rejects_writes_outside_strategy()
    test_driver_kills_over_budget_trial()
```

- [ ] **Step 4: Run tests (expect PASS)**

Run: `PYTHONPATH=. python3 tests/test_lab.py`
Expected: both new `ok ...` lines included. (The budget test takes ~1s.)

- [ ] **Step 5: Commit**

```bash
git add research/lab/drive.py tests/test_lab.py
git commit -m "feat(lab): driver guardrails (no-write-outside-strategy + budget kill)"
```

---

## Task 7: Driver CLI + default mutator + program.md + Makefile wiring

**Files:**
- Modify: `research/lab/drive.py` (add `main()`)
- Create: `research/lab/program.md`
- Modify: `Makefile` (add `lab-drive`; wire `test-lab` into `test`)
- Modify: `.gitignore`

- [ ] **Step 1: Add `main()` with the default `claude -p` mutator to drive.py**

Append to `research/lab/drive.py`:

```python
# Default mutator: a headless coding-agent CLI rewrites the strategy file from the
# context. Override with --mutator for any other agent CLI or a stub.
_DEFAULT_MUTATOR = (
    'claude -p "You are the autoresearch lab agent. Read $LAB_CONTEXT_FILE, then '
    'rewrite ONLY the file at $LAB_STRATEGY_FILE to try a better trading strategy. '
    'Keep the build_policy(md, *, seed=0) -> Policy contract. Edit no other file."'
)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run the autoresearch lab driver")
    ap.add_argument("--data", default="sim/data/kraken_market.bin")
    ap.add_argument("--strategy", default="research/lab/strategy.py")
    ap.add_argument("--leaderboard", default="artifacts/lab/leaderboard.csv")
    ap.add_argument("--archive-dir", default="artifacts/lab/strategies")
    ap.add_argument("--program", default="research/lab/program.md")
    ap.add_argument("--mutator", default=_DEFAULT_MUTATOR,
                    help="shell command that rewrites $LAB_STRATEGY_FILE from $LAB_CONTEXT_FILE")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--budget-seconds", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window-steps", type=int, default=120)
    args = ap.parse_args()

    cfg = DriveConfig(
        strategy_path=Path(args.strategy), leaderboard_path=Path(args.leaderboard),
        archive_dir=Path(args.archive_dir), program_path=Path(args.program),
        data_path=Path(args.data), mutator=args.mutator, trials=args.trials,
        budget_seconds=args.budget_seconds, base_seed=args.seed,
        windows=args.windows, window_steps=args.window_steps,
        guard_root=Path(args.strategy).resolve().parent,
    )
    results = drive(cfg)
    n_ok = sum(r.trial_status == "ok" for r in results)
    n_promoted = sum(r.promoted for r in results)
    print(f"[lab] ran {len(results)} trials ({n_ok} ok, {n_promoted} promoted) "
          f"-> {args.leaderboard}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write program.md**

Create `research/lab/program.md`:

```markdown
# Autoresearch Lab — research brief

## Objective
Find a trading strategy that **clears the out-of-sample gate** on the Kraken
hourly majors *net of ~52 bps round-trip taker friction*. The gate
(`research/eval.py`) runs unseen windows across a slippage matrix {0,10,20,30}bps
and promotes only if the **worst-cell median monthly return ≥ 0.10**. Train≫test
gaps are penalized by `generalization_score`. Honest failure is an acceptable,
loggable outcome.

## Your one move each iteration
Rewrite **only** `research/lab/strategy.py`. Keep the contract exactly:

    def build_policy(md: MarketData, *, seed: int = 0) -> Policy

`Policy = (obs, env) -> int`. The C sim calls it bar-by-bar with the CURRENT
bar's observation only — you cannot see the future, so do not try. Obs layout is
`[num_symbols * features_per_sym | account | position]`; action `0` = flat,
`1 + sym` = long symbol `sym` (single position at a time).

## What you may change
Anything inside `build_policy`: feature transforms over `obs`, rolling state you
accumulate across bars, the signal, sizing, entry/exit and holding logic. You may
load a trained model if one exists. You may NOT edit any other file — doing so
fails the trial (`rejected_write`).

## Scar tissue (read before repeating known dead ends)
- Resizing a zero-edge signal stays zero-edge. Tuning sizing knobs alone has
  never cleared the gate — change the *signal*.
- 26 bps taker fee per leg is fatal to high-turnover hourly churn; either find a
  signal strong enough to pay it, or hold longer / trade less.
- A strategy that's great on average but loses in many windows is crushed by the
  worst-cell median — favor robustness over peak return.
```

- [ ] **Step 3: Wire Makefile**

In `Makefile`: add `lab-drive` to `.PHONY`; append `test-lab` to the `test:` aggregate target's dependency list; and add the `lab-drive` recipe:

```makefile
lab-drive: build-sim ## run the autonomous lab driver (TRIALS=N, needs a coding-agent CLI for the default mutator)
	PYTHONPATH=. $(PYTHON) -m research.lab.drive --data $(LAB_DATA) --trials $(TRIALS) \
		--leaderboard artifacts/lab/leaderboard.csv
```

The `test:` line becomes (append `test-lab`):

```makefile
test: test-fill test-safety test-sim test-features test-gate test-strategy test-forecast test-backtest test-rl test-autoresearch test-kraken-paper test-kraken-executor test-lab ## run all golden fixtures
```

- [ ] **Step 4: Ignore generated artifacts**

In `.gitignore`, add:

```
artifacts/lab/
research/lab/.lab_context.txt
```

- [ ] **Step 5: Run the full lab test + the aggregate (expect PASS)**

Run: `PYTHONPATH=. python3 tests/test_lab.py && make test-lab`
Expected: all `ok ...` lines, `all lab tests passed`.

- [ ] **Step 6: Commit**

```bash
git add research/lab/drive.py research/lab/program.md Makefile .gitignore
git commit -m "feat(lab): driver CLI + default mutator + program.md + make wiring"
```

---

## Task 8: Full-suite verification + README pointer

**Files:**
- Modify: `sim/README.md` (one-paragraph pointer) — or `docs/REPO_MAP.md` if that is the canonical index.

- [ ] **Step 1: Run the whole test aggregate**

Run: `make test`
Expected: every fixture passes, ending with the lab tests. If `data-kraken`
artifacts are absent, `lab-trial`/`lab-drive` are NOT part of `make test` (only
`test-lab`, which uses synthetic `make_sample` data), so the suite is self-contained.

- [ ] **Step 2: Add a short pointer to the repo docs**

Append to `docs/REPO_MAP.md` (under the research section) a single paragraph:

```markdown
### research/lab/ — agentic autoresearch loop
Adapts karpathy/autoresearch: the agent rewrites ONE mutable file
(`research/lab/strategy.py`, the `build_policy` seam) each iteration; the frozen
harness scores it through the existing OOS gate and appends to an append-only
leaderboard. Drive it with `make lab-drive TRIALS=N` (autonomous, needs a
coding-agent CLI) or `make lab-trial` (one trial on the current strategy). The
human-steered brief lives in `research/lab/program.md`. Distinct from the
non-agentic grid-sweep `research/autoresearch.py`. See
`docs/superpowers/specs/2026-06-14-autoresearch-lab-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/REPO_MAP.md
git commit -m "docs: point REPO_MAP at the research/lab autoresearch loop"
```

---

## Self-Review

**Spec coverage:**
- §2 honesty contract (frozen vs mutable) → Task 1 (strategy seam), Task 2 (`frozen_hash`), Task 6 (no-write guard). ✓
- §3.1 mutable `strategy.py` `build_policy` seam → Task 1. ✓
- §3.2 `run_trial` reuse of `evaluate` + `generalization_score` → Task 2 (`run_one`). ✓
- §3.3 append-only leaderboard + manifest (`strategy_hash`, `frozen_hash`, `gate_seed`, `trial_status`) + archive → Task 3. ✓
- §3.4 driver: context, configurable mutator, subprocess budget, only-writes-strategy, gate-seed rotation, keep-best, stub mode → Tasks 5–7. ✓
- §3.5 `program.md` → Task 7. ✓
- §4 layout + Makefile targets → Tasks 4, 7. ✓
- §5 tests (frozen seam, stub loop, archive/repro, no-write, budget) → Tasks 1–6. ✓
- §6 non-goals: existing `autoresearch.py` untouched (no task modifies it); no optimizer; sequential; offline. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `TrialResult` fields defined in Task 2 are used identically by `leaderboard.append_row` (Task 3), `harness.main --json` (Task 4), and `drive` (Tasks 5–6). `DriveConfig` defined in Task 5, extended with `guard_root` in Task 6, consumed by `main()` in Task 7 — all field names match. `evaluate` is always called with the real signature `evaluate(policy, md, *, ...)`. `generalization_score` imported from `research.autoresearch`. ✓

**Note for the implementer:** the C sim must be built (`make build-sim`) before any test runs; Task 1 Step 4 does this and the Makefile lab targets depend on `build-sim`.
