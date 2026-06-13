# Kraken crypto research findings (does it make money?)

Consolidated, honest record of the Kraken-crypto research (June 2026). The
pipeline works end-to-end; **no robust tradeable edge was found** in the 5 USD
majors. This doc is the one place to read *what we tried and why it's a dead end*
so nobody re-runs it blind. Per-experiment detail is in `git log` and the Muninn
vault; the numbers below are reproduced from the runs.

## TL;DR

- The full chain is real and validated: **data → Chronos-2 forecasts → 16-feature
  `.bin` → train → out-of-sample gate → autoresearch → walk-forward → paper loop.**
- Across **hourly + daily**, **taker + maker** fees, **long / long-short /
  cross-sectional-vol**, **120d / 365d / ~4yr**, every candidate that passed a
  single train/test split **failed walk-forward**. The gate/walk-forward correctly
  refused every loser (the moray replay-vs-holdout lesson, enforced in code).
- Two independent reasons it fails, by cadence:
  - **Hourly → friction.** 26 bps Kraken taker on hourly turnover is fatal: a
    ~−0.05 gross result becomes ~−0.89 net (~0.8/mo of pure cost).
  - **Daily → signal + no breadth.** With ~10-day holds taker is a non-issue, yet
    the best signal (momentum) is still net-losing. 5 highly-correlated majors give
    no cross-sectional breadth; a single-position long/short ≈ a leveraged bet on
    one coin.

## Evidence

### XGB champions (leakage-safe temporal split, `--train-frac 0.7`)
| model | data | train_acc | gate (unseen) |
|-------|------|-----------|---------------|
| XGB | 120d + forecasts | 0.787 | REJECT −0.78/mo (overfit) |
| XGB | 365d technical | 0.647 | REJECT −0.75/mo |

### Signal sweep (`research/feature_search.py`) — every feature × sign × config
- Taker fees, hourly: **0 of 512** candidates (long + long-short × 120d + 365d)
  had positive out-of-sample return. Flat beats every signal.
- Gross/net diagnostic: friction is the dominant loss term hourly; the gross
  signal is ~nil (only `chronos_close_delta_h24` at 72h hold is barely positive,
  +0.007).

### The maker-fee "edge" that wasn't (`feature_search` + `research/walkforward.py`)
- Maker (10 bps) + 72h hold, single 60/40 split: 8/128 "promote" — best was
  **short the highest-vol major** (`atr_pct_24h` sign −1), +0.14–0.16/mo.
- **Walk-forward killed it**: across 8 consecutive OOS folds it is positive in
  **1/8** (median −0.11/mo). The single split had landed on the one lucky fold.

### Daily (`--timeframe 1d`, ~4yr, walk-forward at taker)
| signal | folds positive | median |
|--------|----------------|--------|
| return_24h momentum (long strong / short weak) | 3/8 | net-neg |
| trend_72h | 2/8 | net-neg |
| ma_delta_24h | 1/8 | net-neg |
| xsvol (long low-vol / short high-vol) | 1/8 | net-neg |

Daily beats hourly (momentum 3/8 vs 1/8) but stays net-losing; on daily the
blocker is **signal/breadth**, not friction.

## Root cause (the ceiling)

A real cross-sectional edge (moray's verified money came from ranking **846
stocks**) needs **many symbols** to diversify idiosyncratic risk **and** a
**multi-position book**. keel's C sim is **single-position by invariant**
(`position_sym` = one symbol long or short; no multi-asset book), so a diversified
long/short basket is not expressible. This is architectural, not a tuning problem.

## What it would take (both, not either)

1. **Breadth** — dozens+ of symbols for a genuine cross-sectional rank. The data
   adapter now supports any ccxt pairs at `1h/4h/1d` (`sim/kraken_data.py
   --timeframe`).
2. **A multi-position portfolio sim** — extend the ONE C engine to a multi-asset
   book (long/short basket), preserving the single-fill-engine invariant. No
   second/soft Python fill model (the BINANCENEURAL cautionary tale).

Until both exist, the 5-pair single-position crypto track has no robust edge at
any cadence/cost model we tested. That is a valid, money-saving conclusion.

## Research tools (offline; not in `make test`)

| tool | what | run |
|------|------|-----|
| `research/feature_search.py` | sweep signal feature × sign × config through the gate (train/test split), long / long-short, `--fee-rate/--slip-bps/--max-hold` | `make feature-search` |
| `research/walkforward.py` | evaluate a signal across N consecutive OOS folds (regime test); `xsvol` cross-sectional policy | `make walkforward` |
| `forecast/build_kraken_cache.py` | Chronos-2 zero-shot forecast cache (MPS) | `make build-cache-kraken` |
| `forecast/finetune_kraken.py` | optional LoRA fine-tune + held-out MAE | `make finetune-kraken` |
| `models/xgb/train.py` | XGB champion, leakage-safe `--train-frac` | `make train-kraken` |

Forecast/fine-tune detail + the crypto gate calibration are in
[`KRAKEN_CALIBRATION.md`](KRAKEN_CALIBRATION.md).
