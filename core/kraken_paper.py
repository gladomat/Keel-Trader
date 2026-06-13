"""Live-data paper trading against Kraken (K5, #15) — NEVER a live writer.

The forward test the user requires before any live work: trade on **real-time
Kraken prices** but simulate every fill locally through the ONE C engine. Zero
orders are sent to Kraken.

Safety guarantees (mirrors ``core.paper_runner`` + the broker boundary):

  * asserts ``core.config.PAPER`` at construction and refuses otherwise;
  * routes every buy AND sell through the single guarded write surface
    ``core.broker.Broker`` with a ``PaperExecutor`` — so the death-spiral guard
    runs on the exact path the live path will, but nothing touches the network;
  * has NO ``enforce_live_singleton`` / ``acquire_*_lock`` / ``go_live`` /
    ``create_order`` method — it is structurally incapable of winning the
    live-writer lock or placing a real order (HARD RULE 2);
  * simulates fills through the SAME C fill arithmetic the gate uses
    (``sim.keel_sim.roundtrip_cost``) — no second/soft Python fill model
    (the BINANCENEURAL cautionary tale).

The live feed (``core.kraken_feed.KrakenPublicFeed``) is offline tooling; the
local fill-simulation + guard wiring here is exercised by stdlib tests with
synthetic snapshots (no network).
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from core import config
from core.broker import Broker, PaperExecutor
from core.kraken_feed import Bar
from forecast.technical import technical_features
from models.xgb.strategy import StrategyConfig, SymbolSignal, decide
from research.eval import DEFAULT_FEE_RATE, DEFAULT_FILL_BUFFER_BPS
from sim.keel_sim import fill_price

logger = logging.getLogger("keel.kraken_paper")

# Score function seam: (symbol, recent_bars) -> (score, volatility).
ScoreFn = Callable[[str, List[Bar]], tuple]


class LiveWriteAttemptError(RuntimeError):
    """Raised if anything tries to make this paper trader act as a live writer."""


@dataclass
class PaperFill:
    """One simulated fill recorded to the paper ledger."""
    ts: float
    symbol: str
    side: str            # "buy" or "sell"
    qty: float
    decision_price: float   # the live price the decision was made at
    fill_price: float       # the C-engine-resolved fill price
    realized_pnl: float = 0.0  # only set on the closing (sell) leg


def momentum_score(symbol: str, bars: List[Bar]) -> tuple:
    """Default signal: 24h momentum as conviction, ATR% as the risk proxy.

    Derived from the ONE technical-feature definition (``forecast.technical``) so
    the paper loop runs without a trained model. A champion ``score_fn`` (XGB /
    forecast) can be injected to replace it.
    """
    if len(bars) < 2:
        return 0.0, 1e-4
    o = [b.open for b in bars]
    h = [b.high for b in bars]
    low = [b.low for b in bars]
    c = [b.close for b in bars]
    feats = technical_features(o, h, low, c)[-1]
    # technical block order: [return_1h, return_24h, volatility_24h, ma_delta_24h,
    #   ma_delta_72h, atr_pct_24h, trend_72h, drawdown_72h]
    score = feats[1]                 # return_24h as conviction
    volatility = max(feats[5], 1e-4)  # atr_pct_24h as the inverse-vol risk proxy
    return score, volatility


@dataclass
class KrakenPaperTrader:
    """Paper-only live loop. Construct, then ``step(snapshot)`` each tick.

    ``snapshot`` is ``{symbol: {"bar": Bar, "price": float}}`` (what
    ``KrakenPublicFeed.snapshot()`` returns). Fills are simulated through the C
    engine and appended to a JSONL ledger; running realized PnL is tracked.
    """

    score_fn: ScoreFn = momentum_score
    strategy_cfg: StrategyConfig = field(default_factory=StrategyConfig)
    broker: Optional[Broker] = None
    ledger_path: Optional[Path] = None
    equity: float = 10_000.0
    slippage_bps: float = 10.0
    fill_buffer_bps: float = DEFAULT_FILL_BUFFER_BPS
    fee_rate: float = DEFAULT_FEE_RATE   # per-leg taker fee (Kraken ~26 bps, K3)
    history: int = 128

    _bars: Dict[str, Deque] = field(default_factory=lambda: defaultdict(deque))
    _held: Dict[str, dict] = field(default_factory=dict)  # sym -> {qty, entry_fill}
    realized_pnl: float = 0.0
    fills: List[PaperFill] = field(default_factory=list)

    def __post_init__(self):
        # Fail closed: a paper trader must never come up in live mode.
        if not config.PAPER:
            raise LiveWriteAttemptError(
                "KrakenPaperTrader refuses to run with PAPER=False; it is "
                "paper-only and must never become a live writer (K5)."
            )
        if self.broker is None:
            # The single guarded write surface, paper executor — no network.
            self.broker = Broker(executor=PaperExecutor())
        if self.ledger_path is None:
            from core.state_paths import resolve_state_dir
            self.ledger_path = resolve_state_dir(None) / "kraken_paper" / "ledger.jsonl"
        self.ledger_path = Path(self.ledger_path)

    # No enforce_live_singleton / acquire_live_lock / go_live / create_order
    # method exists here BY DESIGN — see the module docstring.

    def step(self, snapshot: dict) -> List[PaperFill]:
        """One decision cycle: live prices -> strategy -> guard -> C-engine fill."""
        for sym, snap in snapshot.items():
            self._bars[sym].append(snap["bar"])
            while len(self._bars[sym]) > self.history:
                self._bars[sym].popleft()

        signals: List[SymbolSignal] = []
        for sym in snapshot:
            bars = list(self._bars[sym])
            if not bars:
                continue
            score, vol = self.score_fn(sym, bars)
            signals.append(SymbolSignal(sym, score, vol))

        targets = {p.symbol: p.weight for p in decide(signals, self.strategy_cfg)}
        events: List[PaperFill] = []

        # Exit names we no longer want — guarded sells (refusal raises and stops).
        for sym in list(self._held):
            if sym in targets:
                continue
            snap = snapshot.get(sym)
            if snap is None:
                continue
            ev = self._simulate_exit(sym, snap["bar"], self._exec_price(snap))
            if ev is not None:
                events.append(ev)

        # Enter desired names — guarded buys (records the buy price for the guard).
        for sym, weight in targets.items():
            if sym in self._held:
                continue
            snap = snapshot[sym]
            price = self._exec_price(snap)
            if price <= 0:
                continue
            qty = weight * self.equity / price
            ev = self._simulate_entry(sym, qty, snap["bar"], price)
            if ev is not None:
                events.append(ev)

        self._persist(events)
        self.fills.extend(events)
        return events

    @staticmethod
    def _exec_price(snap: dict) -> float:
        """The price a decision transacts at: the just-CLOSED bar's close.

        The C fill engine is a limit-within-bar model, so the executable
        reference is the bar we fill against (matches the gate's fill-off-the-bar
        semantics + decision lag), not the mid-forming-hour live ticker. The live
        ``price`` is kept in the snapshot only as a freshness signal.
        """
        return float(snap["bar"].close)

    # --- C-engine fill simulation (the ONE fill model) ---------------------- #
    def _fill(self, bar: Bar, target: float, is_buy: bool) -> float:
        """C-engine fill at ``target`` against the bar; 0.0 if it cannot fill."""
        return fill_price(bar.open, bar.high, bar.low, bar.close, target,
                          self.fill_buffer_bps, self.slippage_bps, is_buy)

    def _simulate_entry(self, sym: str, qty: float, bar: Bar,
                        price: float) -> Optional[PaperFill]:
        entry_fill = self._fill(bar, price, is_buy=True)
        if entry_fill <= 0.0:
            logger.info("[kraken-paper] no entry fill for %s @ %.4f (gapped/slipped "
                        "out of bar) — skipping", sym, price)
            return None
        # Guard FIRST (records the buy price for the death-spiral floor).
        self.broker.submit_order(sym, "buy", qty, price)
        self._held[sym] = {"qty": qty, "entry_fill": entry_fill}
        return PaperFill(time.time(), sym, "buy", qty, price, entry_fill)

    def _simulate_exit(self, sym: str, bar: Bar,
                       price: float) -> Optional[PaperFill]:
        sell_fill = self._fill(bar, price, is_buy=False)
        if sell_fill <= 0.0:
            logger.info("[kraken-paper] no exit fill for %s @ %.4f — staying held",
                        sym, price)
            return None
        qty = self._held[sym]["qty"]
        # SAME guarded write surface the live path uses; a death-spiral sell raises.
        self.broker.submit_order(sym, "sell", qty, price)
        held = self._held.pop(sym)
        entry_fill = held["entry_fill"]
        # Slippage is already in entry_fill/sell_fill (C engine); apply the taker
        # fee on both legs (the same DEFAULT_FEE_RATE the gate prices).
        fee = self.fee_rate * qty * (entry_fill + sell_fill)
        net = qty * (sell_fill - entry_fill) - fee
        self.realized_pnl += net
        return PaperFill(time.time(), sym, "sell", qty, price, sell_fill, net)

    def _persist(self, events: List[PaperFill]) -> None:
        if not events:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(asdict(ev), sort_keys=True) + "\n")

    def summary(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "open_positions": {s: r["qty"] for s, r in self._held.items()},
            "n_fills": len(self.fills),
            "ledger": str(self.ledger_path),
        }


def run_loop(feed, trader: KrakenPaperTrader, *, ticks: int,
             sleep_s: float = 3600.0):  # pragma: no cover - network demo harness
    """Pull ``ticks`` live snapshots and step the paper trader (offline demo).

    Not a live entry point — it places no real orders. Intended to run over a
    meaningful window so the paper ledger can be reviewed before any live decision.
    """
    for i in range(ticks):
        snapshot = feed.snapshot()
        events = trader.step(snapshot)
        logger.info("[kraken-paper] tick %d: %d fills, realized_pnl=%.2f",
                    i, len(events), trader.realized_pnl)
        if i < ticks - 1:
            time.sleep(sleep_s)
    return trader.summary()


def main():  # pragma: no cover - offline demo, needs ccxt/network
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Live-data Kraken PAPER trading (no real orders)")
    ap.add_argument("--ticks", type=int, default=24)
    ap.add_argument("--sleep", type=float, default=3600.0)
    ap.add_argument("--ledger", default=None)
    args = ap.parse_args()

    from core.kraken_feed import KrakenPublicFeed

    feed = KrakenPublicFeed()
    trader = KrakenPaperTrader(
        ledger_path=Path(args.ledger) if args.ledger else None)
    summary = run_loop(feed, trader, ticks=args.ticks, sleep_s=args.sleep)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
