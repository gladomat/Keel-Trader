"""The locked Kraken trading universe + bar cadence (stdlib-only).

Single source for the constants both the offline data adapter (``sim/kraken_data``,
which pulls in numpy) and the live paper feed (``core/kraken_feed``, which must
stay stdlib-importable for ``make test``) need. Keeping them here means neither
module has to import the other, and nothing drags numpy into the test path.

Decisions are LOCKED (see Muninn "keel_trader Kraken build decisions"): the 5 USD
majors, quote = USD, hourly bars. Kraken's XBT normalises to BTC/USD via ccxt.
"""
from __future__ import annotations

KRAKEN_USD_MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LTC/USD"]
TIMEFRAME = "1h"
MS_PER_HOUR = 3_600_000

# ms per bar for each supported ccxt timeframe (grid stepping / paging cadence).
TIMEFRAME_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def ms_per_bar(timeframe: str = TIMEFRAME) -> int:
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        raise ValueError(f"unsupported timeframe {timeframe!r}; add it to TIMEFRAME_MS")
