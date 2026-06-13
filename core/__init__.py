"""keel_trader core: the safety spine + broker boundary.

The crown jewel ported from moray. Three fail-closed gates protect real money:
  1. explicit-enable  (ALLOW_ALPACA_LIVE_TRADING=1) — paper-first by default
  2. single-writer fcntl lock per account
  3. per-sell death-spiral guard (time-aware tolerance)

See docs/LIVE_TRADER_DEEPDIVE.md for the full design.
"""
