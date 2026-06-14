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

## Residual analysis (`research/residual_analysis.py`, `make residual`)

A cost-free *measurement* pass (Information Coefficient / market-neutral residual /
vol clustering) asking the upstream question the gate can't: is there **any** linear
predictability in the raw series, decoupled from the 26 bps + slippage + binary-fill
frictions? Run on all three `.bin`s. Three results:

1. **Forecast columns were zero in the files the verdict was written on.** Spec
   indices 0-7 (the 8 Chronos features) are **constant 0** in `kraken_deep.bin` and
   `kraken_daily.bin` — they were built without `--forecast-cache`
   (`sim/kraken_data.py` leaves forecasts as "honest zeros" by design). Only
   `kraken_market.bin` (~4 months, ~2880 h) carries real forecasts, and it is **too
   short for the walk-forward regime test** that actually kills candidates. So the
   long-history sweeps / walk-forward (`fs_deep*.csv`, the 365d/4yr runs) judged the
   technical half of the spec only; the Chronos forecasts were never present where
   the killing happens. **Now closed (2026-06-14):** rebuilt the Chronos cache over
   the full 365d (`forecast/cache/kraken_deep`, 8249 rows/sym/horizon), rebuilt
   `kraken_deep.bin` with `--forecast-cache` (forecast cols 0-7 non-zero), and re-ran
   `feature-search` long + long-short (`artifacts/fs_deep_fc.csv`): **256 candidates,
   0 positive OOS, 0 promoted.** The 128 forecast-feature candidates are flat-zero
   (degenerate — a near-constant confidence signal takes no position) or net-negative;
   the best long-short is still the `atr_pct_24h` short at −0.47/mo. So the forecasts
   are now **verified to add no tradeable edge on long history**, not merely untested.
2. **Direction is genuinely dead** (confirms the headline). Pooled time-series IC
   ≈ 0; multi-feature OLS OOS R² ≤ 0; next-bar sign hit-rate 0.496–0.521 (base 0.50);
   return autocorrelation within ±0.04. No directional timing edge exists pre-cost.
3. **The one robust, universal residual is variance, not drift.** `|r|` lag-1
   autocorrelation is **+0.21–0.25 on every file and cadence**; R²(|r₊₁| ~
   `atr_pct_24h`) ≈ 0.06–0.10. GARCH-grade, regime-stable, sign-independent volatility
   clustering. Not a directional money-pump, but it is exactly the structure a
   **vol-target / position-sizing risk overlay** needs — which serves keel's actual
   mandate (low-drawdown, smooth-Sortino) even with zero alpha. The cross-sectional
   vol rank-IC is large (t up to −18) but **survives market-neutralization unchanged**
   and is the same low-vol effect walk-forward already flagged as regime luck (1/8
   folds) — big IC, not robustly tradeable single-position.

**Done (2026-06-14):** the long-history re-run above closes the forecast gap for
hourly `feature-search` — no edge. The faint 4-month `forecast_confidence` /
`chronos_*_delta` ICs did not survive contact with the full-year gate. Remaining
loose ends if ever revisited: rebuild `kraken_daily` the same way and re-run
`walkforward` (the regime test) on the forecast-populated deep — but given 0/256
positive OOS hourly, expect the same verdict.

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
| `research/residual_analysis.py` | cost-free predictability map: pooled/XS-rank IC, market-neutral residual IC, return autocorr, vol clustering, multi-feature OOS R² | `make residual` |
| `forecast/build_kraken_cache.py` | Chronos-2 zero-shot forecast cache (MPS) | `make build-cache-kraken` |
| `forecast/finetune_kraken.py` | optional LoRA fine-tune + held-out MAE | `make finetune-kraken` |
| `models/xgb/train.py` | XGB champion, leakage-safe `--train-frac` | `make train-kraken` |

Forecast/fine-tune detail + the crypto gate calibration are in
[`KRAKEN_CALIBRATION.md`](KRAKEN_CALIBRATION.md).
