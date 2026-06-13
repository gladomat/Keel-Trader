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
