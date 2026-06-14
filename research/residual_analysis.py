"""Residual analysis: how much *predictable structure* is actually in the data?

The gate / feature_search / walkforward all measure **tradeable** edge — signal
net of the C sim's frictions (26 bps taker, 5 bps slip, binary fills, lag>=2).
They all said no. This tool asks the upstream question, decoupled from cost:

    is there ANY linear predictability in the raw series, and if so where —
    in the level (market beta), the cross-section (market-neutral residual),
    the autocorrelation, or only in the variance (vol clustering)?

That tells us whether the dead edge is a *signal* problem (nothing to trade) or a
*friction/breadth* problem (real predictability the single-position sim can't
harvest). Pure measurement — no sim, no trading, no frictions.

Probes (per horizon h in {1,24,72,240} bars):
  1. Pooled IC      corr(feature_t , fwd_ret_{t->t+h})      over all sym,t
  2. XS rank IC     mean_t spearman( feature , fwd_ret ) across the S symbols
  3. Market-neutral residual: r~_s = r_s - mean_s r  (strip the basket);
     re-run 1+2 on residuals -> is there cross-sectional alpha at all?
  4. Autocorr of returns (market basket & residual): momentum / mean-reversion
  5. Vol clustering: autocorr of |r| and R^2 of predicting |r_{t+1}| from
     realized-vol / atr feature -> a *variance* edge even if direction is dead

Run:  .venv/bin/python -m research.residual_analysis --data sim/data/kraken_deep.bin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from forecast.features import FEATURE_NAMES
from sim.binpack import read_header, read_symbols

CLOSE = 3  # OHLCV index of close


def load(path: str):
    """Return feats [T,S,F] and close [T,S] as float64 arrays (mmap-free)."""
    hdr = read_header(path)
    ns, nt, fps, pf = (hdr["num_symbols"], hdr["num_timesteps"],
                       hdr["features_per_sym"], hdr["price_features"])
    from sim.binpack import HEADER_SIZE, SYM_NAME_LEN
    base = HEADER_SIZE + ns * SYM_NAME_LEN
    raw = np.fromfile(path, dtype="<f4")
    fblock = raw[base // 4: base // 4 + nt * ns * fps].reshape(nt, ns, fps)
    poff = base // 4 + nt * ns * fps
    pblock = raw[poff: poff + nt * ns * pf].reshape(nt, ns, pf)
    return fblock.astype(np.float64), pblock[:, :, CLOSE].astype(np.float64)


def fwd_logret(close: np.ndarray, h: int) -> np.ndarray:
    """[T,S] forward log return over h bars; last h rows are nan."""
    out = np.full_like(close, np.nan)
    out[:-h] = np.log(close[h:] / close[:-h])
    return out


def _safe_pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.pearsonr(x[m], y[m])
    return r, p, int(m.sum())


def pooled_ic(feats, fwd, fi):
    x = feats[:, :, fi].ravel()
    y = fwd.ravel()
    return _safe_pearson(x, y)


def xs_rank_ic(feats, fwd, fi):
    """Mean over t of cross-sectional Spearman corr across symbols (needs S>=3)."""
    T, S, _ = feats.shape
    ics = []
    for t in range(T):
        x, y = feats[t, :, fi], fwd[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
            continue
        rho, _ = stats.spearmanr(x[m], y[m])
        if np.isfinite(rho):
            ics.append(rho)
    if len(ics) < 30:
        return np.nan, np.nan, len(ics)
    ics = np.array(ics)
    # IC t-stat (IR * sqrt(n)) — significance of a nonzero mean IC
    t_stat = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
    return ics.mean(), t_stat, len(ics)


def market_neutralize(ret: np.ndarray) -> np.ndarray:
    """Strip the equal-weight basket return at each t -> residual [T,S]."""
    mkt = np.nanmean(ret, axis=1, keepdims=True)
    return ret - mkt


def autocorr(x: np.ndarray, lags) -> dict:
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x * x)
    out = {}
    for k in lags:
        if k >= n:
            out[k] = np.nan
            continue
        out[k] = np.sum(x[:-k] * x[k:]) / denom
    return out


def vol_clustering(close, atr_feats, fi_atr):
    """|r_{t+1}| predictability: autocorr of |r| + R^2 from realized & atr vol."""
    r = np.full_like(close, np.nan)
    r[1:] = np.log(close[1:] / close[:-1])
    absr = np.abs(r)
    # pool symbols for autocorr of |r| (lag1) — vol clustering signature
    ac1 = []
    for s in range(close.shape[1]):
        a = absr[:, s]
        ac1.append(autocorr(a, [1])[1])
    ac1 = np.nanmean(ac1)
    # predict |r_{t+1}| from |r_t| (realized) and from atr_pct_24h feature
    x_real = absr[:-1].ravel()
    x_atr = atr_feats[:-1, :, fi_atr].ravel()
    y = absr[1:].ravel()
    r_real, _, _ = _safe_pearson(x_real, y)
    r_atr, _, _ = _safe_pearson(x_atr, y)
    return ac1, r_real ** 2 if np.isfinite(r_real) else np.nan, \
        r_atr ** 2 if np.isfinite(r_atr) else np.nan


def multifeat_oos_r2(feats, fwd, train_frac=0.7):
    """OLS forward-ret ~ all features, OOS R^2 (sign predictability ceiling)."""
    T, S, F = feats.shape
    X = feats.reshape(T * S, F)
    y = fwd.reshape(T * S)
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[m], y[m]
    n = len(y)
    ntr = int(n * train_frac)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    # standardize on train
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Xtr = np.c_[np.ones(ntr), (Xtr - mu) / sd]
    Xte = np.c_[np.ones(len(yte)), (Xte - mu) / sd]
    # ridge (tiny) for stability
    lam = 1e-3
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    beta = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ beta
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2_oos = 1 - ss_res / ss_tot
    # directional hit-rate (sign agreement), baseline 0.5
    hit = np.mean(np.sign(pred) == np.sign(yte))
    return r2_oos, hit, len(yte)


def run(path: str, horizons=(1, 24, 72, 240)):
    feats, close = load(path)
    syms = read_symbols(path)
    T, S, F = feats.shape
    fi_atr = FEATURE_NAMES.index("atr_pct_24h")
    print(f"\n{'='*78}\n{path}  T={T} S={S} F={F}  syms={syms}\n{'='*78}")

    for h in horizons:
        fwd = fwd_logret(close, h)
        res = market_neutralize(fwd)
        print(f"\n--- horizon h={h} bars ---")
        print(f"{'feature':<26} {'pooledIC':>9} {'p':>7} | "
              f"{'xs_rankIC':>9} {'t':>6} | {'resid_pooledIC':>14} {'resid_xsIC':>10}")
        rows = []
        for fi, fn in enumerate(FEATURE_NAMES):
            pic, pp, _ = pooled_ic(feats, fwd, fi)
            xic, xt, _ = xs_rank_ic(feats, fwd, fi)
            rpic, _, _ = pooled_ic(feats, res, fi)
            rxic, rxt, _ = xs_rank_ic(feats, res, fi)
            rows.append((fn, pic, pp, xic, xt, rpic, rxic, rxt))
        # sort by |xs_rankIC| to surface the strongest cross-sectional signal
        for fn, pic, pp, xic, xt, rpic, rxic, rxt in sorted(
                rows, key=lambda r: -abs(r[3]) if np.isfinite(r[3]) else 0):
            print(f"{fn:<26} {pic:>+9.4f} {pp:>7.3f} | {xic:>+9.4f} {xt:>+6.1f} | "
                  f"{rpic:>+14.4f} {rxic:>+10.4f}")

    # autocorr of basket & residual 1-bar returns
    r1 = fwd_logret(close, 1)
    mkt = np.nanmean(r1, axis=1)
    resid1 = market_neutralize(r1)
    lags = [1, 2, 3, 5, 24]
    print(f"\n--- return autocorrelation (1-bar) ---")
    print(f"market basket : " + "  ".join(f"lag{k}={autocorr(mkt,[k])[k]:+.3f}" for k in lags))
    # per-symbol residual autocorr, averaged
    racs = {k: [] for k in lags}
    for s in range(S):
        ac = autocorr(resid1[:, s], lags)
        for k in lags:
            racs[k].append(ac[k])
    print(f"resid (avg)   : " + "  ".join(f"lag{k}={np.nanmean(racs[k]):+.3f}" for k in lags))

    # vol clustering
    ac1, r2_real, r2_atr = vol_clustering(close, feats, fi_atr)
    print(f"\n--- volatility clustering (|r_t+1|) ---")
    print(f"autocorr |r| lag1 (avg sym) : {ac1:+.3f}")
    print(f"R^2  |r_t+1| ~ |r_t|        : {r2_real:.4f}")
    print(f"R^2  |r_t+1| ~ atr_pct_24h  : {r2_atr:.4f}")

    # multi-feature OOS R^2 on 24-bar fwd ret
    for h in (1, 24):
        fwd = fwd_logret(close, h)
        r2, hit, n = multifeat_oos_r2(feats, fwd)
        print(f"\n--- multi-feature OLS, fwd h={h} (OOS) ---")
        print(f"OOS R^2 = {r2:+.5f}   sign hit-rate = {hit:.4f} (base 0.5)   n_test={n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="sim/data/kraken_deep.bin")
    ap.add_argument("--horizons", default="1,24,72,240")
    args = ap.parse_args()
    run(args.data, tuple(int(x) for x in args.horizons.split(",")))


if __name__ == "__main__":
    main()
