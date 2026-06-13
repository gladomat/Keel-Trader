# Production — live trading state of truth

This file is the human-maintained record of **what is running live**, the
**deploy commands**, and the **marketsim/gate scores** behind the current
champion. Update it in the same commit that changes the live state.

> HARD RULE 2: exactly one live writer per Alpaca account. There is currently
> **no live entry point wired** — the system is paper-first by default
> (`core/config.py`: `PAPER=True` unless `ALP_PAPER=0`). Cutover is gated by
> `ops/deploy_live_trader.sh` and the 6-step checklist below.

## Current running state

| Field | Value |
|-------|-------|
| Live writer unit | _none — paper-only_ |
| Alpaca account name | `alpaca_live_writer` (lock: `strategy_state/account_locks/alpaca_live_writer.lock`) |
| Champion | _none promoted to live_ |
| Gate verdict (median monthly, worst slip cell) | _n/a_ |
| Paper run window | _n/a_ |
| Last deploy | see `ops/deploy_history.log` (git-ignored, append-only) |

When a champion is promoted, fill the table above and record its scores under
[Marketsim / gate scores](#marketsim--gate-scores).

### Kraken venue (K-track)

The active venue is **Kraken spot** (5 USD majors, hourly; see
`docs/KRAKEN_CALIBRATION.md`). The live-writer account/lock identity is reused
unchanged (`alpaca_live_writer`); only the executor differs.

- **K6 live executor is STAGED but DISABLED.** `core/kraken_executor.py` exists
  and is fail-closed (construction requires `allow_live=True` **and**
  `KEEL_ALLOW_LIVE_TRADING=1`), but it is **not wired**: `LIVE_WRITER_UNITS` is
  still empty, no live entry point imports it, and no Kraken API keys are on the
  box. Enabling it is a **HITL-only** step gated on a clean K5 paper run — do the
  6-step checklist below in the same reviewed commit that wires it.
- **K5 paper trading** runs with `make paper-kraken` (public data, no orders) and
  writes a paper ledger under `strategy_state/kraken_paper/`. Review a meaningful
  window before any live decision.

## Deploy commands

The deploy script is **dry-run by default** and fails closed on the live path.

```bash
# Verify the lock handshake only — starts/stops nothing. Safe anytime.
ops/deploy_live_trader.sh
ops/deploy_live_trader.sh --dry-run keel-live-trader.service

# Gated live cutover. Refused unless EVERY checklist gate below is satisfied.
# Env gates belong in the supervised systemd unit, NEVER set ad hoc on a shell.
ops/deploy_live_trader.sh --live keel-live-trader.service
```

What `--live` enforces before it touches anything (all must pass, else it dies):

- the unit is registered in `LIVE_WRITER_UNITS` (in `deploy_live_trader.sh`);
- `ALLOW_ALPACA_LIVE_TRADING=1` and `ALP_PAPER=0` are present;
- operator attestations `KEEL_GATE_CLEARED=1`, `KEEL_PAPER_CLEAN=1`, and (if the
  account is shared) `KEEL_ACCOUNT_LOCK_SHARED_OK=1`.

After (re)starting the unit it verifies **lock-holder PID == supervisor MainPID**
and appends the result to `ops/deploy_history.log`. If the handshake fails it
exits non-zero — the live writer is *not* considered verified.

## Live-cutover checklist (do not skip)

1. **Gate cleared.** Champion cleared the Phase-3 gate on unseen data
   (median monthly ≥ 0.27 at the worst slippage cell). Attest: `KEEL_GATE_CLEARED=1`.
2. **Paper clean.** Paper run is clean over a meaningful window.
   Attest: `KEEL_PAPER_CLEAN=1`.
3. **Registered in the same commit.** The live entry point imports the singleton
   (`core.alpaca_singleton.enforce_live_singleton`) and is added to
   `LIVE_WRITER_UNITS` **in the same commit** that wires it.
4. **Shared account reconciled.** If sharing an Alpaca account with another
   system, both use the same lock path + account name. Attest:
   `KEEL_ACCOUNT_LOCK_SHARED_OK=1`.
5. **Env gates in the unit only.** `ALLOW_ALPACA_LIVE_TRADING=1` + `ALP_PAPER=0`
   are set only in the supervised unit file, never ad hoc on a shell.
6. **Handshake OK.** `deploy_live_trader.sh --live …` reports OK (lock-holder PID
   matches the supervisor PID) before you walk away.

## Marketsim / gate scores

Record each promoted champion's evidence here (append, never overwrite):

| Date | Champion / config | Gate median monthly | Worst slip cell | Notes |
|------|-------------------|---------------------|-----------------|-------|
| _—_  | _—_               | _—_                 | _—_             | none promoted yet |
