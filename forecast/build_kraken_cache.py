"""Build the Chronos-2 forecast cache on real Kraken bars (offline, K2 follow-up).

The driver that turns historical Kraken OHLCV into the ``cache_root/h{H}/{sym}.parquet``
forecast cache the ``.bin`` join reads. It runs Chronos-2 inference (MPS/bf16 on
Apple Silicon) — heavy + offline, never imported by ``make test``.

Pipeline:
  fetch hourly OHLCV (Kraken public; optional Binance deep-history backfill)
    -> per-symbol time-ascending series (epoch seconds)
    -> forecast.chronos.Chronos2LoRAForecaster.forecast_rows (leakage-safe)
    -> forecast.build_cache.build_cache -> parquet (assert_no_leakage on write)

Then join into the .bin:
  python3 sim/kraken_data.py --output sim/data/kraken_market.bin \
                             --forecast-cache forecast/cache/kraken

Run:  make build-cache-kraken
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forecast.build_cache import assert_features_in_spec, write_cache
from forecast.chronos import Chronos2LoRAForecaster
from sim.kraken_data import (
    KRAKEN_USD_MAJORS,
    _since_ms,
    fetch_binance_backfill,
    fetch_kraken,
    merge_series,
)


def series_by_symbol(merged: dict) -> dict:
    """{sym: {ts_ms:[o,h,l,c,v]}} -> {sym: {symbol,timestamp(s),close,high,low}}."""
    out: dict[str, dict] = {}
    for sym, bars in merged.items():
        if not bars:
            continue
        ts_sorted = sorted(bars)
        out[sym] = {
            "symbol": sym,
            "timestamp": [t // 1000 for t in ts_sorted],   # ms -> epoch seconds
            "close": [bars[t][3] for t in ts_sorted],
            "high": [bars[t][1] for t in ts_sorted],
            "low": [bars[t][2] for t in ts_sorted],
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Chronos-2 forecast cache on Kraken bars")
    ap.add_argument("--symbols", default=",".join(KRAKEN_USD_MAJORS))
    ap.add_argument("--cache-root", default="forecast/cache/kraken")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD or epoch ms")
    ap.add_argument("--days", type=int, default=120,
                    help="Lookback when --since omitted (use --backfill for deep history)")
    ap.add_argument("--backfill", choices=["none", "binance"], default="binance",
                    help="Deep-history source beyond Kraken REST's ~720-bar window")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--horizons", default="1,24")
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    since_ms = _since_ms(args.since, args.days)

    sources = []
    if args.backfill == "binance":
        sources.append(fetch_binance_backfill(symbols, since_ms))
    sources.append(fetch_kraken(symbols, since_ms))
    merged = merge_series(*sources)

    series = series_by_symbol(merged)
    for sym, s in series.items():
        print(f"  {sym}: {len(s['close'])} bars "
              f"(anchors with ctx>={args.context_length}: "
              f"{max(0, len(s['close']) - args.context_length + 1)})")

    forecaster = Chronos2LoRAForecaster()
    forecaster.load_base()
    print(f"  chronos loaded on device={forecaster._device}")

    assert_features_in_spec()  # forecast names still land in the ONE spec
    cache_root = Path(args.cache_root)
    written = []
    for sym, s in series.items():
        for h in horizons:
            rows = forecaster.forecast_rows(s, horizon=h,
                                            context_length=args.context_length)
            if not rows:
                print(f"  {sym} h{h}: no anchors (need >= {args.context_length} bars) — skip")
                continue
            path = write_cache(cache_root, sym, h, rows)  # asserts no leakage
            written.append(path)
            print(f"  {sym} h{h}: {len(rows)} rows -> {path}")
    print(f"wrote {len(written)} parquet files under {cache_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
