# The MKTD `.bin` market-data format

`sim/binpack.py` is the **one** writer/reader for this format. Both the real
exporter (`sim/export_data.py`, pandas-built) and the tests/sample generator
(`sim/make_sample_data.py`, stdlib-built) go through it, so there is a single
byte-layout definition. The C loader (`sim/src/trading_env.c` via
`keel_market_data_load`) reads the same bytes.

The feature *ordering* is **not** defined here — it lives in
`forecast/features.py` (`FEATURE_SPEC`). Callers pass features already ordered
per that spec; this format only fixes the byte layout. The spec is append-only:
never reorder or repurpose an index within a version or an old `.bin` silently
means something else.

## Layout

All multi-byte numbers are little-endian. Floats are IEEE-754 `float32`.

### Header — 64 bytes (`<4sIIIII40s`)

| Field              | Type      | Bytes | Notes                              |
|--------------------|-----------|-------|------------------------------------|
| `magic`            | `char[4]` | 4     | `"MKTD"`                           |
| `version`          | `uint32`  | 4     | currently `1`                      |
| `num_symbols`      | `uint32`  | 4     | `S`                                |
| `num_timesteps`    | `uint32`  | 4     | `T`                                |
| `features_per_sym` | `uint32`  | 4     | `F` (16 for `FEATURE_SPEC` v1)     |
| `price_features`   | `uint32`  | 4     | always `5` (OHLCV)                 |
| `padding`          | `byte[40]`| 40    | zero-filled, reserved              |

### Symbol table — `S * 16` bytes

`num_symbols` entries, each a 16-byte null-padded ASCII name (names are
truncated to 15 chars + a guaranteed NUL terminator).

### Feature block — `T * S * F` float32

Row-major `[t][s][f]`: for each timestep, for each symbol, `F` feature values
in `FEATURE_SPEC` order.

### Price block — `T * S * 5` float32

Row-major `[t][s][p]`: for each timestep, for each symbol, the 5 OHLCV values
`(open, high, low, close, volume)`.

## Total size

```
64 + S*16 + T*S*F*4 + T*S*5*4  bytes
```

## Generating data

- `make data` — regenerate the committed deterministic sample
  (`sim/data/sample.bin`) via `sim/make_sample_data.py` (stdlib only, seeded).
- `python -m sim.export_data --symbols ... --forecast-cache-root ...` — build a
  real `.bin` from forecast parquet + hourly OHLCV CSVs (needs numpy/pandas).

Large real `.bin` files are git-ignored; the committed sample's *source* is the
regenerator script, not the bytes.
