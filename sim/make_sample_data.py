"""Generate a small, deterministic synthetic ``.bin`` so tests/backtests run
without the full moray data tree.

Writes the MKTD v1 format documented in ``sim/README.md`` (header + symbol table +
float32 features + float32 OHLCV), using only the stdlib so ``make test`` needs no
numpy/pandas. Large real ``.bin`` files stay git-ignored; this regenerator is the
committed source of a standalone sample (``make data``).

The price path is a seeded gentle up-drift with mean-reverting noise, so some
windows reward going long and others punish it — enough to exercise the gate's
accept/reject and fail-fast branches without a trained model.
"""
from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path

MAGIC = b"MKTD"
VERSION = 1
PRICE_FEATURES = 5  # OHLCV

# Imported lazily by tests; the canonical feature count lives in forecast.features.
DEFAULT_FEATURES_PER_SYM = 16


def _gen_symbol(rng: random.Random, n: int, base: float, drift: float):
    """Return (ohlcv_rows, feature_rows) for one symbol."""
    prices = []
    p = base
    for _ in range(n):
        # mean-reverting noise around an up-drifting level
        shock = rng.gauss(0.0, 0.012)
        p = p * (1.0 + drift + shock)
        if p < 1.0:
            p = 1.0
        prices.append(p)

    ohlcv = []
    feats = []
    for i, close in enumerate(prices):
        prev = prices[i - 1] if i > 0 else close
        hi = max(prev, close) * (1.0 + abs(rng.gauss(0.0, 0.004)))
        lo = min(prev, close) * (1.0 - abs(rng.gauss(0.0, 0.004)))
        op = prev
        vol = 1000.0 + rng.random() * 500.0
        ohlcv.append((op, hi, lo, close, vol))

        ret1 = (close - prev) / prev if prev else 0.0
        f = [0.0] * DEFAULT_FEATURES_PER_SYM
        f[8] = max(-0.5, min(0.5, ret1))          # return_1h
        f[10] = abs(ret1)                          # volatility proxy
        feats.append(f)
    return ohlcv, feats


def make_sample(output: Path, num_symbols: int = 2, num_timesteps: int = 1500,
                features_per_sym: int = DEFAULT_FEATURES_PER_SYM, seed: int = 7) -> Path:
    rng = random.Random(seed)
    syms = [f"SYM{i}" for i in range(num_symbols)]
    per_sym = []
    for i in range(num_symbols):
        per_sym.append(_gen_symbol(rng, num_timesteps, base=100.0 + 10.0 * i,
                                   drift=0.0002 + 0.0001 * i))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as fh:
        fh.write(struct.pack(
            "<4sIIIII40s", MAGIC, VERSION, num_symbols, num_timesteps,
            features_per_sym, PRICE_FEATURES, b"\x00" * 40,
        ))
        for s in syms:
            fh.write(s.encode("ascii")[:15].ljust(16, b"\x00"))
        # feature data: [T][S][F]
        for t in range(num_timesteps):
            for s in range(num_symbols):
                row = per_sym[s][1][t]
                fh.write(struct.pack(f"<{features_per_sym}f", *row))
        # price data: [T][S][5]
        for t in range(num_timesteps):
            for s in range(num_symbols):
                fh.write(struct.pack("<5f", *per_sym[s][0][t]))

    size = output.stat().st_size
    print(f"wrote {output} ({size:,} bytes): {num_symbols} symbols x {num_timesteps} timesteps")
    return output


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic sample .bin")
    ap.add_argument("--output", default="sim/data/sample.bin")
    ap.add_argument("--symbols", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    make_sample(Path(args.output), num_symbols=args.symbols,
                num_timesteps=args.timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
