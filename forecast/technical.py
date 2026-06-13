"""The ONE pure-stdlib technical-feature computation (FEATURE_SPEC indices 8-15).

K2 (#12) needs the technical features computed directly from Kraken OHLCV bars and
written into the ``.bin`` — and the guard the issue asks for ("pure guards remain
in ``make test``") means this computation must be unit-pinnable WITHOUT pandas.

So this module is the canonical, stdlib-only definition of the 8 technical
features. It mirrors the formulas in ``sim/export_data.compute_features`` (the
legacy pandas path that builds the equity ``.bin``) bar-for-bar, but operates on
plain Python sequences so the test suite can pin it with no numpy/pandas. The
offline Kraken builder (``sim/kraken_data``) calls into here, and a parity test
(skipped when pandas is absent) keeps the two implementations from drifting.

Ordering is owned by ``forecast.features.FEATURE_SPEC`` — ``technical_features``
returns a row in exactly ``FEATURE_SPEC.names_of_kind("technical")`` order, and
``assert_technical_in_spec`` fails loudly if that order ever changes.
"""
from __future__ import annotations

import math
from typing import List, Sequence

from forecast.features import FEATURE_SPEC

# The technical feature names this module produces, in spec order (indices 8-15).
TECHNICAL_FEATURES = tuple(FEATURE_SPEC.names_of_kind("technical"))

_EPS = 1e-8
_VOL_FALLBACK = 0.01  # pandas .fillna(0.01) on volatility when std is undefined


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _pct_change(series: Sequence[float], n: int) -> List[float]:
    """``pandas.Series.pct_change(n).fillna(0)`` — first ``n`` entries are 0."""
    out = [0.0] * len(series)
    for i in range(n, len(series)):
        prev = series[i - n]
        out[i] = (series[i] - prev) / prev if abs(prev) > _EPS else 0.0
    return out


def _rolling_mean(series: Sequence[float], w: int) -> List[float]:
    """``rolling(w, min_periods=1).mean()`` — trailing mean over up to ``w`` values."""
    out = [0.0] * len(series)
    acc = 0.0
    from collections import deque

    window: deque = deque()
    for i, v in enumerate(series):
        window.append(v)
        acc += v
        if len(window) > w:
            acc -= window.popleft()
        out[i] = acc / len(window)
    return out


def _rolling_max(series: Sequence[float], w: int) -> List[float]:
    """``rolling(w, min_periods=1).max()`` — trailing max over up to ``w`` values."""
    out = [0.0] * len(series)
    for i in range(len(series)):
        lo = max(0, i - w + 1)
        out[i] = max(series[lo : i + 1])
    return out


def _rolling_std_of_returns(closes: Sequence[float], w: int) -> List[float]:
    """``close.pct_change(1).rolling(w, min_periods=1).std()`` then ``fillna(0.01)``.

    The 1-bar return series has a NaN at index 0 (pandas pct_change). Sample std
    (ddof=1) over the non-NaN returns in the trailing ``w`` window; when fewer than
    two valid returns are in view the std is undefined and pandas fills 0.01.
    """
    n = len(closes)
    out = [_VOL_FALLBACK] * n
    # returns[i] valid only for i >= 1 (index-0 return is NaN in pandas).
    for i in range(n):
        lo = max(1, i - w + 1)
        vals = []
        for j in range(lo, i + 1):
            prev = closes[j - 1]
            vals.append((closes[j] - prev) / prev if abs(prev) > _EPS else 0.0)
        if len(vals) < 2:
            out[i] = _VOL_FALLBACK
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)  # ddof=1
        out[i] = math.sqrt(var)
    return out


def _true_range(highs, lows, closes) -> List[float]:
    """ATR true range; index 0 uses (high-low) only (prev close is NaN in pandas)."""
    n = len(closes)
    tr = [0.0] * n
    for i in range(n):
        hl = highs[i] - lows[i]
        if i == 0:
            tr[i] = hl
        else:
            pc = closes[i - 1]
            tr[i] = max(hl, abs(highs[i] - pc), abs(lows[i] - pc))
    return tr


def technical_features(opens: Sequence[float], highs: Sequence[float],
                       lows: Sequence[float], closes: Sequence[float]) -> List[List[float]]:
    """Compute the 8 technical features per bar from one symbol's OHLC series.

    Returns ``[T][8]`` rows in ``FEATURE_SPEC.names_of_kind("technical")`` order:
    return_1h, return_24h, volatility_24h, ma_delta_24h, ma_delta_72h,
    atr_pct_24h, trend_72h, drawdown_72h. Pure stdlib; mirrors the pandas
    formulas in ``sim/export_data.compute_features``.
    """
    n = len(closes)
    if not (len(opens) == len(highs) == len(lows) == n):
        raise ValueError("OHLC series must be equal length")
    if n == 0:
        return []

    return_1h = [_clip(x, -0.5, 0.5) for x in _pct_change(closes, 1)]
    return_24h = [_clip(x, -1.0, 1.0) for x in _pct_change(closes, 24)]
    volatility_24h = _rolling_std_of_returns(closes, 24)

    ma24 = _rolling_mean(closes, 24)
    ma72 = _rolling_mean(closes, 72)
    ma_delta_24h = [_clip((closes[i] - ma24[i]) / max(ma24[i], _EPS), -0.5, 0.5)
                    for i in range(n)]
    ma_delta_72h = [_clip((closes[i] - ma72[i]) / max(ma72[i], _EPS), -0.5, 0.5)
                    for i in range(n)]

    tr = _true_range(highs, lows, closes)
    atr24 = _rolling_mean(tr, 24)
    atr_pct_24h = [_clip(atr24[i] / max(closes[i], _EPS), 0.0, 0.5) for i in range(n)]

    trend_72h = [_clip(x, -1.0, 1.0) for x in _pct_change(closes, 72)]

    rmax72 = _rolling_max(closes, 72)
    drawdown_72h = [_clip((closes[i] - rmax72[i]) / max(rmax72[i], _EPS), -1.0, 0.0)
                    for i in range(n)]

    return [
        [return_1h[i], return_24h[i], volatility_24h[i], ma_delta_24h[i],
         ma_delta_72h[i], atr_pct_24h[i], trend_72h[i], drawdown_72h[i]]
        for i in range(n)
    ]


def assert_technical_in_spec() -> List[str]:
    """Fail loudly if the technical feature names/order drift out of the ONE spec."""
    names = FEATURE_SPEC.names_of_kind("technical")
    if list(TECHNICAL_FEATURES) != names:
        raise ValueError(
            f"technical features drifted from FEATURE_SPEC {FEATURE_SPEC.version}: "
            f"module={list(TECHNICAL_FEATURES)} spec={names}"
        )
    return names
