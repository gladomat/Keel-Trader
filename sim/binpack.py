"""The ONE writer/reader for the MKTD ``.bin`` format (see sim/README.md).

Shared by ``sim/export_data.py`` (real pandas-built arrays) and the tests
(stdlib-built fixtures) so there is a single packing definition. The feature
ordering is *not* defined here — callers pass features already ordered per
``forecast.features.FEATURE_SPEC``; this module only does the byte layout.

Stdlib-only. Accepts numpy arrays (fast ``.tobytes()`` path) or nested Python
sequences (``features[t][s][f]`` / ``prices[t][s][p]``).
"""
from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"MKTD"
PRICE_FEATURES = 5  # OHLCV
HEADER_SIZE = 64
SYM_NAME_LEN = 16


def _is_ndarray(x) -> bool:
    return type(x).__module__ == "numpy" and type(x).__name__ == "ndarray"


def write_market_bin(path, symbol_names, features, prices, *,
                     num_timesteps: int, features_per_sym: int, version: int = 1) -> Path:
    """Write the MKTD file. ``features``/``prices`` are [T][S][F]/[T][S][5]."""
    path = Path(path)
    num_symbols = len(symbol_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack(
            "<4sIIIII40s", MAGIC, version, num_symbols, num_timesteps,
            features_per_sym, PRICE_FEATURES, b"\x00" * 40,
        ))
        for name in symbol_names:
            fh.write(name.encode("ascii")[:SYM_NAME_LEN - 1].ljust(SYM_NAME_LEN, b"\x00"))

        if _is_ndarray(features):
            import numpy as np
            fh.write(np.ascontiguousarray(features, dtype=np.float32).tobytes())
            fh.write(np.ascontiguousarray(prices, dtype=np.float32).tobytes())
        else:
            for t in range(num_timesteps):
                for s in range(num_symbols):
                    fh.write(struct.pack(f"<{features_per_sym}f", *features[t][s]))
            for t in range(num_timesteps):
                for s in range(num_symbols):
                    fh.write(struct.pack(f"<{PRICE_FEATURES}f", *prices[t][s]))
    return path


def read_header(path) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_SIZE)
    magic, version, ns, nt, fps, pf, _ = struct.unpack("<4sIIIII40s", raw)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r} in {path}")
    return {"version": version, "num_symbols": ns, "num_timesteps": nt,
            "features_per_sym": fps, "price_features": pf}


def _read_block(path, offset, nt, ns, per):
    count = nt * ns * per
    with open(path, "rb") as fh:
        fh.seek(offset)
        flat = struct.unpack(f"<{count}f", fh.read(count * 4))
    out = []
    i = 0
    for _ in range(nt):
        row = []
        for _ in range(ns):
            row.append(list(flat[i:i + per]))
            i += per
        out.append(row)
    return out


def read_features(path):
    """Pure-Python read-back of the feature block as nested lists [T][S][F]."""
    hdr = read_header(path)
    ns, nt, fps = hdr["num_symbols"], hdr["num_timesteps"], hdr["features_per_sym"]
    offset = HEADER_SIZE + ns * SYM_NAME_LEN
    return _read_block(path, offset, nt, ns, fps)


def read_prices(path):
    """Pure-Python read-back of the OHLCV price block as nested lists [T][S][5]."""
    hdr = read_header(path)
    ns, nt, fps, pf = (hdr["num_symbols"], hdr["num_timesteps"],
                       hdr["features_per_sym"], hdr["price_features"])
    offset = HEADER_SIZE + ns * SYM_NAME_LEN + nt * ns * fps * 4
    return _read_block(path, offset, nt, ns, pf)


def read_symbols(path):
    """Read the symbol-name table as a list of ASCII strings."""
    hdr = read_header(path)
    ns = hdr["num_symbols"]
    with open(path, "rb") as fh:
        fh.seek(HEADER_SIZE)
        names = []
        for _ in range(ns):
            raw = fh.read(SYM_NAME_LEN)
            names.append(raw.split(b"\x00", 1)[0].decode("ascii"))
    return names
