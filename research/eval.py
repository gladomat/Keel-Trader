"""The out-of-sample gate: the single trusted judge of whether a policy promotes.

Ports the load-bearing logic of moray's ``scripts/eval_100d.py`` (keel invariant:
the gate is the ONE arbiter, and it judges through the ONE C fill engine — never a
Python reimplementation of fills). For a policy it:

  * runs N random *unseen* windows through the C sim (``sim/keel_sim``);
  * sweeps a **slippage matrix {0,5,10,20} bps and reports the worst cell** — a
    policy only promotes if it survives the harshest slippage;
  * uses production-realism defaults: ``decision_lag>=2``, binary fills, 26bps
    Kraken-taker fee, 5bps fill buffer, 6h max hold;
  * **fails fast** on bad policies: bails a cell early when a window breaches the
    max-drawdown limit (0.20) or when the cell's median-monthly target has become
    arithmetically unreachable;
  * promotes only if the worst-cell **median monthly return >= 0.27**.

"Unseen" is the caller's contract: pass a held-out ``.bin`` (or restrict the
offset range via ``offset_lo_frac``/``offset_hi_frac``) so windows never overlap
the policy's training data. Offsets are seeded for reproducibility.

Monthly return is the window's total return scaled to a 730h month
(``8760/12``); with the default 720-step window that scaling is ~1.0, i.e. each
window is roughly one trading month.
"""
from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Sequence

from sim.keel_sim import MarketData, TradingEnv
from research.policies import Policy

# --- gate constants (the promotion contract) -------------------------------
DEFAULT_SLIPPAGES_BPS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
PROMOTION_TARGET_MEDIAN_MONTHLY = 0.27
FAIL_FAST_MAX_DD = 0.20
DEFAULT_WINDOW_STEPS = 720          # ~1 month at hourly bars
BARS_PER_MONTH = 730.0             # 8760 / 12

# --- production-realism env defaults ---------------------------------------
DEFAULT_DECISION_LAG = 2
# K3 (#13): Kraken spot taker fee ~26 bps per leg (the equity 10 bps was
# optimistic for crypto). Applied per leg by the C sim, so the gate prices a
# round trip at ~52 bps of friction before slippage — no rosy fills.
DEFAULT_FEE_RATE = 0.0026           # 26 bps (Kraken taker)
DEFAULT_FILL_BUFFER_BPS = 5.0
DEFAULT_FILL_PROBABILITY = 1.0      # binary fills (all-or-nothing in the C sim)
DEFAULT_MAX_HOLD_HOURS = 6


@dataclass
class WindowResult:
    offset: int
    total_return: float
    monthly_return: float
    max_drawdown: float
    num_trades: float


@dataclass
class CellResult:
    slippage_bps: float
    median_monthly: float
    windows_run: int
    failed_fast: bool
    fail_reason: str
    results: List[WindowResult] = field(default_factory=list)


@dataclass
class Verdict:
    promote: bool
    worst_cell_median_monthly: float
    promotion_target: float
    failed_fast: bool
    reason: str
    n_windows: int
    window_steps: int
    cells: List[CellResult] = field(default_factory=list)

    def summary(self) -> str:
        head = "PROMOTE" if self.promote else "REJECT"
        lines = [
            f"[gate] {head}: worst-cell median monthly = "
            f"{self.worst_cell_median_monthly:+.4f} (target {self.promotion_target:+.2f})",
            f"[gate] {self.reason}",
        ]
        for c in self.cells:
            tag = " FAIL-FAST" if c.failed_fast else ""
            lines.append(
                f"[gate]   slip={c.slippage_bps:>4.0f}bps  "
                f"median_monthly={c.median_monthly:+.4f}  "
                f"windows={c.windows_run}{tag}"
                + (f"  ({c.fail_reason})" if c.fail_reason else "")
            )
        return "\n".join(lines)


def generate_offsets(md: MarketData, n_windows: int, window_steps: int, *,
                     seed: int = 0, offset_lo_frac: float = 0.0,
                     offset_hi_frac: float = 1.0) -> List[int]:
    """Seeded random episode start offsets within the held-out range."""
    max_offset = md.num_timesteps - window_steps - 1
    if max_offset <= 0:
        raise ValueError(
            f"window_steps={window_steps} too large for {md.num_timesteps} timesteps"
        )
    lo = max(0, int(max_offset * offset_lo_frac))
    hi = min(max_offset, int(max_offset * offset_hi_frac))
    if hi <= lo:
        lo, hi = 0, max_offset
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n_windows)]


def run_window(policy: Policy, md: MarketData, offset: int, window_steps: int,
               slippage_bps: float, *, decision_lag: int = DEFAULT_DECISION_LAG,
               fee_rate: float = DEFAULT_FEE_RATE,
               fill_buffer_bps: float = DEFAULT_FILL_BUFFER_BPS,
               fill_probability: float = DEFAULT_FILL_PROBABILITY,
               max_hold_hours: int = DEFAULT_MAX_HOLD_HOURS) -> WindowResult:
    """Run one episode of ``policy`` through the C sim and return its metrics."""
    env = TradingEnv(
        md,
        max_steps=window_steps,
        forced_offset=offset,
        decision_lag=decision_lag,
        fee_rate=fee_rate,
        fill_slippage_bps=slippage_bps,
        fill_buffer_bps=fill_buffer_bps,
        fill_probability=fill_probability,
        max_hold_hours=max_hold_hours,
    )
    try:
        obs = env.reset()
        # window_steps decisions are enough; a small margin covers decision_lag.
        for _ in range(window_steps + decision_lag + 2):
            action = policy(obs, env)
            obs, _reward, terminal = env.step(action)
            if terminal:
                break
        log = env.log
        total_return = log["total_return"]
        max_dd = abs(log["max_drawdown"])
        monthly = total_return * (BARS_PER_MONTH / window_steps)
        return WindowResult(offset, total_return, monthly, max_dd, log["num_trades"])
    finally:
        env.free()


def _windows_needed_ge(n: int) -> int:
    """How many windows must be >= target for the median to be >= target."""
    return n - (n // 2)  # ceil(n/2)


def _run_cell(policy: Policy, md: MarketData, offsets: Sequence[int],
              window_steps: int, slippage_bps: float, *, promotion_target: float,
              fail_fast: bool, fail_fast_max_dd: float, env_kwargs: dict) -> CellResult:
    n = len(offsets)
    needed_ge = _windows_needed_ge(n)
    results: List[WindowResult] = []
    ge_count = 0
    failed_fast = False
    fail_reason = ""

    for i, off in enumerate(offsets):
        wr = run_window(policy, md, off, window_steps, slippage_bps, **env_kwargs)
        results.append(wr)
        if wr.monthly_return >= promotion_target:
            ge_count += 1

        if fail_fast:
            if wr.max_drawdown > fail_fast_max_dd:
                failed_fast = True
                fail_reason = (
                    f"window {i} max_dd {wr.max_drawdown:.4f} > limit {fail_fast_max_dd}"
                )
                break
            remaining = n - (i + 1)
            if ge_count + remaining < needed_ge:
                failed_fast = True
                fail_reason = (
                    f"median-monthly unreachable: {ge_count}/{needed_ge} windows "
                    f">= target after {i + 1}/{n} (only {remaining} left)"
                )
                break

    median_monthly = (
        statistics.median([r.monthly_return for r in results]) if results else 0.0
    )
    return CellResult(slippage_bps, median_monthly, len(results),
                      failed_fast, fail_reason, results)


def evaluate(policy: Policy, md: MarketData, *, n_windows: int = 20,
             window_steps: int = DEFAULT_WINDOW_STEPS,
             slippages_bps: Sequence[float] = DEFAULT_SLIPPAGES_BPS,
             promotion_target: float = PROMOTION_TARGET_MEDIAN_MONTHLY,
             fail_fast: bool = True, fail_fast_max_dd: float = FAIL_FAST_MAX_DD,
             seed: int = 0, offset_lo_frac: float = 0.0,
             offset_hi_frac: float = 1.0, **env_kwargs) -> Verdict:
    """Judge ``policy`` and return a promotion :class:`Verdict`.

    Cells run worst-realism-last is unnecessary; we run the given slippage order
    and short-circuit on the first failing cell (fail-fast across cells too). The
    *worst* cell median still drives the reported verdict.
    """
    offsets = generate_offsets(md, n_windows, window_steps, seed=seed,
                               offset_lo_frac=offset_lo_frac,
                               offset_hi_frac=offset_hi_frac)

    cells: List[CellResult] = []
    worst_median = float("inf")
    overall_failed_fast = False
    promote = True
    reason = f"all slippage cells median monthly >= target {promotion_target:+.2f}"

    for slip in slippages_bps:
        cell = _run_cell(policy, md, offsets, window_steps, slip,
                         promotion_target=promotion_target, fail_fast=fail_fast,
                         fail_fast_max_dd=fail_fast_max_dd, env_kwargs=env_kwargs)
        cells.append(cell)
        worst_median = min(worst_median, cell.median_monthly)

        if cell.failed_fast:
            overall_failed_fast = True
            promote = False
            reason = f"fail-fast at slippage {slip:.0f}bps: {cell.fail_reason}"
            break
        if cell.median_monthly < promotion_target:
            promote = False
            reason = (
                f"slippage {slip:.0f}bps median monthly {cell.median_monthly:+.4f} "
                f"< target {promotion_target:+.2f}"
            )
            break

    if worst_median == float("inf"):
        worst_median = 0.0

    return Verdict(promote, worst_median, promotion_target, overall_failed_fast,
                   reason, n_windows, window_steps, cells)


def main():
    ap = argparse.ArgumentParser(description="Run the out-of-sample gate on a reference policy")
    ap.add_argument("--data", default="sim/data/sample.bin", help="MKTD .bin path")
    ap.add_argument("--policy", default="long", choices=["flat", "long", "random"])
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--window-steps", type=int, default=DEFAULT_WINDOW_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fail-fast", action="store_true")
    ap.add_argument("--offset-lo-frac", type=float, default=0.0)
    ap.add_argument("--offset-hi-frac", type=float, default=1.0)
    args = ap.parse_args()

    from research import policies as P
    policy = {
        "flat": P.always_flat,
        "long": P.always_long_0,
        "random": P.random_policy(args.seed),
    }[args.policy]

    md = MarketData.load(args.data)
    try:
        verdict = evaluate(
            policy, md, n_windows=args.windows, window_steps=args.window_steps,
            seed=args.seed, fail_fast=not args.no_fail_fast,
            offset_lo_frac=args.offset_lo_frac, offset_hi_frac=args.offset_hi_frac,
        )
    finally:
        md.free()
    print(verdict.summary())
    raise SystemExit(0 if verdict.promote else 1)


if __name__ == "__main__":
    main()
