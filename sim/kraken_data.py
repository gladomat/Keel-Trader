"""Kraken hourly OHLCV -> MKTD ``.bin`` (issue #11, K1 — first Kraken slice).

The offline data adapter that turns real Kraken hourly OHLCV into the repo's
single ``.bin`` MKTD format, so the gate / backtest / training run on real crypto
instead of the synthetic ``sim/make_sample_data.py`` sample.

Boundaries (keel guardrails):
  - **Offline tooling.** Uses the network + ``ccxt`` (not installed in the test
    env). It is NOT imported by ``make test`` — the stdlib suite keeps using
    ``sim/make_sample_data.py``. Verify edits here with ``python3 -m py_compile``.
  - **Public, no-auth endpoints only** (K1-K5 safety boundary). This module never
    touches private/keyed Kraken endpoints; no API keys are read or required.
  - Writes through the ONE shared packer ``sim/binpack.write_market_bin`` so the
    byte layout matches every other producer/consumer of the ``.bin``.

Feature block: the prices block is the real OHLCV; the feature block is written as
**zeros** (a documented placeholder, ``FEATURES_PER_SYM`` wide in the exact
``forecast.features.FEATURE_SPEC`` order). K2 (#12) replaces these zeros with the
real Chronos2 forecast + technical features. Until then the gate/backtest run on
price action with neutral features — honest, not silently fabricated.

Run:   make data-kraken
   or: PYTHONPATH=. python3 sim/kraken_data.py --output sim/data/kraken_market.bin
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# The ONE feature spec (count + ordering) and the ONE byte packer.
from forecast.features import FEATURES_PER_SYM
from sim.binpack import PRICE_FEATURES, read_header, write_market_bin

# Locked universe (see Muninn "keel_trader Kraken build decisions"): the 5 USD
# majors, quote = USD, hourly bars. Kraken's XBT normalises to BTC/USD via ccxt.
KRAKEN_USD_MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LTC/USD"]

TIMEFRAME = "1h"
MS_PER_HOUR = 3_600_000
VERSION = 1

# Map our USD pairs onto Binance's USDT pairs for the optional deep-history
# backfill (Binance has no native USD spot; USDT ~= USD for OHLCV shape).
_BINANCE_BACKFILL = {
    "BTC/USD": "BTC/USDT",
    "ETH/USD": "ETH/USDT",
    "SOL/USD": "SOL/USDT",
    "XRP/USD": "XRP/USDT",
    "LTC/USD": "LTC/USDT",
}

# One bar = (timestamp_ms, open, high, low, close, volume). Symbol series are
# dicts {ts_ms: [o, h, l, c, v]} so merging sources / dedup is just dict update.
Bar = list


# --------------------------------------------------------------------------- #
# Fetching                                                                     #
# --------------------------------------------------------------------------- #
def _now_ms() -> int:
    return int(time.time() * 1000)


def fetch_ohlcv_paged(exchange, symbol: str, since_ms: int,
                      until_ms: int | None = None, limit: int = 720) -> dict:
    """Page through ``exchange.fetch_ohlcv`` to assemble deep history.

    Kraken REST only returns its most recent ~720 bars per call regardless of
    ``since``; we still page so the loop works for any ccxt exchange (e.g. the
    Binance backfill below genuinely walks back years). Returns {ts_ms: [o,h,l,c,v]}.
    """
    until_ms = until_ms or _now_ms()
    out: dict[int, Bar] = {}
    cursor = since_ms
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=cursor, limit=limit)
        if not batch:
            break
        for ts, o, h, l, c, v in batch:
            if ts <= until_ms:
                out[int(ts)] = [float(o), float(h), float(l), float(c), float(v)]
        last_ts = int(batch[-1][0])
        if last_ts < cursor + MS_PER_HOUR:
            # No forward progress (exchange clamped to its recent window) — stop.
            break
        cursor = last_ts + MS_PER_HOUR
        if exchange.enableRateLimit:
            # ccxt sleeps internally on calls, but be explicit between pages.
            time.sleep(exchange.rateLimit / 1000.0)
    return out


def fetch_kraken(symbols: list[str], since_ms: int) -> dict:
    """Fetch recent hourly OHLCV from Kraken (public endpoint) for each symbol."""
    import ccxt

    ex = ccxt.kraken({"enableRateLimit": True})
    series: dict[str, dict] = {}
    for sym in symbols:
        print(f"  [kraken] fetching {sym} since ms={since_ms} ...")
        bars = fetch_ohlcv_paged(ex, sym, since_ms)
        print(f"  [kraken] {sym}: {len(bars)} bars")
        series[sym] = bars
    return series


def fetch_binance_backfill(symbols: list[str], since_ms: int) -> dict:
    """Deep-history backfill via Binance USDT pairs (ccxt walks back years).

    Used for lookback beyond Kraken REST's ~720-bar window. Keyed back to our
    USD symbol names so the merge with Kraken bars is by-symbol.
    """
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    series: dict[str, dict] = {}
    for sym in symbols:
        b_sym = _BINANCE_BACKFILL.get(sym)
        if b_sym is None:
            print(f"  [binance] no backfill mapping for {sym}, skipping")
            continue
        print(f"  [binance] backfill {sym} via {b_sym} since ms={since_ms} ...")
        bars = fetch_ohlcv_paged(ex, b_sym, since_ms)
        print(f"  [binance] {sym}: {len(bars)} bars")
        series[sym] = bars
    return series


def load_kraken_csv(symbol: str, csv_dir: Path) -> dict:
    """Load a Kraken OHLCVT CSV dump for deep history.

    Kraken publishes downloadable OHLCVT CSVs (https://support.kraken.com/ ->
    "Historical OHLCVT") with columns: timestamp(s), open, high, low, close,
    volume, trades — no header. File name convention here: ``<BASE><QUOTE>_60.csv``
    for the 60-minute (hourly) set, e.g. ``XBTUSD_60.csv`` / ``ETHUSD_60.csv``.
    """
    base, quote = symbol.split("/")
    kraken_base = "XBT" if base == "BTC" else base  # Kraken dumps use XBT for BTC
    fname = f"{kraken_base}{quote}_60.csv"
    path = csv_dir / fname
    if not path.exists():
        print(f"  [csv] no dump for {symbol} at {path}, skipping")
        return {}
    out: dict[int, Bar] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 6:
                continue
            ts_s, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
            out[int(float(ts_s)) * 1000] = [float(o), float(h), float(l), float(c), float(v)]
    print(f"  [csv] {symbol}: {len(out)} bars from {path.name}")
    return out


def merge_series(*sources: dict) -> dict:
    """Merge per-symbol series; later sources win on overlapping timestamps.

    Call order = oldest/least-authoritative first, so e.g.
    ``merge_series(backfill, kraken)`` keeps Kraken's bar for any hour both cover.
    """
    out: dict[str, dict] = {}
    for src in sources:
        for sym, bars in src.items():
            out.setdefault(sym, {}).update(bars)
    return out


# --------------------------------------------------------------------------- #
# Alignment                                                                    #
# --------------------------------------------------------------------------- #
def align_to_grid(series: dict, symbols: list[str]) -> tuple[list[str], list[int], np.ndarray]:
    """Align all symbols to one regular hourly grid (intersection + forward-fill).

    The grid spans [max-of-per-symbol-first-bar .. min-of-per-symbol-last-bar] so
    every symbol genuinely covers the whole range (drop-to-intersection at the
    edges). Interior holes are forward-filled into a flat bar (o=h=l=c=last close,
    volume=0) so the C sim sees a continuous hourly series. Returns
    (kept_symbols, grid_ts_ms, prices[T][S][5]).
    """
    present = [s for s in symbols if series.get(s)]
    if not present:
        raise ValueError("no symbol has any bars to align")

    first = max(min(series[s]) for s in present)
    last = min(max(series[s]) for s in present)
    if last < first:
        raise ValueError(
            f"symbols do not overlap in time (latest-first={first}, earliest-last={last}); "
            "fetch a wider window or add a backfill source"
        )

    grid = list(range(first, last + MS_PER_HOUR, MS_PER_HOUR))
    n_t, n_s = len(grid), len(present)
    prices = np.zeros((n_t, n_s, PRICE_FEATURES), dtype=np.float32)

    for si, sym in enumerate(present):
        bars = series[sym]
        last_bar: Bar | None = None
        for ti, ts in enumerate(grid):
            bar = bars.get(ts)
            if bar is None and last_bar is not None:
                c = last_bar[3]
                bar = [c, c, c, c, 0.0]  # flat forward-fill, no fake volume
            if bar is None:
                # No prior bar to carry (only at the very start) — leave zeros.
                continue
            prices[ti, si, :] = bar
            last_bar = bar

    return present, grid, prices


# --------------------------------------------------------------------------- #
# Write                                                                        #
# --------------------------------------------------------------------------- #
def build_bin(series: dict, symbols: list[str], output: Path) -> Path:
    kept, grid, prices = align_to_grid(series, symbols)
    n_t, n_s = len(grid), len(kept)

    # Feature block = zeros placeholder until K2 (#12) fills the real FEATURE_SPEC
    # values. Shape/order match the spec so the .bin is already K2-ready.
    features = np.zeros((n_t, n_s, FEATURES_PER_SYM), dtype=np.float32)

    write_market_bin(
        output, kept, features, prices,
        num_timesteps=n_t, features_per_sym=FEATURES_PER_SYM, version=VERSION,
    )
    size = output.stat().st_size
    print(f"  wrote {output} ({size:,} bytes): {n_s} symbols x {n_t} hourly bars")
    print(f"  symbols: {kept}")
    print(f"  features: ZEROS placeholder ({FEATURES_PER_SYM}/sym) — filled by K2 (#12)")
    return output


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _since_ms(arg: str | None, default_days: int) -> int:
    """Parse --since (ISO date 'YYYY-MM-DD' or epoch ms) into epoch ms."""
    if arg is None:
        return _now_ms() - default_days * 24 * MS_PER_HOUR
    if arg.isdigit():
        return int(arg)
    import datetime as dt

    d = dt.datetime.strptime(arg, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch Kraken hourly OHLCV -> MKTD .bin (K1)")
    ap.add_argument("--symbols", default=",".join(KRAKEN_USD_MAJORS),
                    help="Comma-separated pairs (default: the 5 locked USD majors)")
    ap.add_argument("--output", default="sim/data/kraken_market.bin",
                    help="Output .bin (git-ignored)")
    ap.add_argument("--since", default=None,
                    help="Start as YYYY-MM-DD or epoch ms (default: --days back)")
    ap.add_argument("--days", type=int, default=29,
                    help="Lookback in days when --since omitted (Kraken REST ~720 hourly bars)")
    ap.add_argument("--backfill", choices=["none", "binance", "kraken-csv"], default="none",
                    help="Deep-history source beyond Kraken REST's ~720-bar window")
    ap.add_argument("--backfill-dir", default="sim/data/kraken_csv",
                    help="Directory of Kraken OHLCVT CSV dumps (--backfill kraken-csv)")
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    since_ms = _since_ms(args.since, args.days)
    output = Path(args.output)

    print(f"K1 Kraken ingestion: {len(symbols)} symbols, backfill={args.backfill}")

    sources: list[dict] = []
    if args.backfill == "binance":
        sources.append(fetch_binance_backfill(symbols, since_ms))
    elif args.backfill == "kraken-csv":
        csv_dir = Path(args.backfill_dir)
        sources.append({s: load_kraken_csv(s, csv_dir) for s in symbols})

    # Kraken REST last so its authoritative recent bars win on overlap.
    sources.append(fetch_kraken(symbols, since_ms))

    series = merge_series(*sources)
    build_bin(series, symbols, output)

    # Sanity: the file we just wrote must parse through the shared reader.
    hdr = read_header(output)
    print(f"  verified header: {hdr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
