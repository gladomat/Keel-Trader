#!/usr/bin/env bash
# deploy_live_trader.sh — the gated live-cutover handshake (issue #10).
#
# Ports moray's deploy: a LIVE_WRITER_UNITS registry, stop-others, a
# lock-holder-PID == supervisor-PID verification, and an append-only history
# log. It is DRY-RUN BY DEFAULT and fails closed: an actual live cutover
# requires the explicit --live flag AND every item of the 6-step checklist
# (HARD RULE 2: exactly one live writer; no ad-hoc live enablement).
#
# Usage:
#   ops/deploy_live_trader.sh                 # dry-run: verify the handshake only
#   ops/deploy_live_trader.sh --dry-run UNIT  # dry-run a specific unit
#   ops/deploy_live_trader.sh --live UNIT     # gated cutover (all checks enforced)
#
# This script NEVER flips a paper system to live on its own — it only verifies
# that a properly-supervised, already-gated unit holds the single writer lock.
set -euo pipefail

# ---------------------------------------------------------------------------
# Registry: the ONLY systemd units permitted to win the live-writer lock.
# A live entry point must be added here IN THE SAME COMMIT that wires it
# (checklist step 3). It is intentionally EMPTY: no live entry point exists
# yet, so there is nothing to cut over to. Adding a name here is a reviewed act.
# ---------------------------------------------------------------------------
LIVE_WRITER_UNITS=(
  # "keel-live-trader.service"   # <- uncomment only when the gated live unit lands
)

ACCOUNT_NAME="${KEEL_ALPACA_ACCOUNT:-alpaca_live_writer}"
STATE_DIR="${STATE_DIR:-strategy_state}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORY_LOG="${KEEL_DEPLOY_HISTORY:-$REPO_ROOT/ops/deploy_history.log}"

log()  { printf '[deploy] %s\n' "$*" >&2; }
die()  { printf '[deploy] FATAL: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# Resolve the absolute lock-file path the singleton would use for this account.
lock_file_path() {
  local sd="$STATE_DIR"
  case "$sd" in
    /*) : ;;                       # already absolute
    *)  sd="$REPO_ROOT/$sd" ;;     # repo-relative (matches state_paths.py)
  esac
  printf '%s/account_locks/%s.lock' "$sd" "$ACCOUNT_NAME"
}

# Read the PID recorded in the lock file (empty if no holder / unreadable).
lock_holder_pid() {
  local lf; lf="$(lock_file_path)"
  [[ -s "$lf" ]] || { printf ''; return 0; }
  python3 - "$lf" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as fh:
        raw = fh.read().strip()
    print(json.loads(raw).get("pid", "") if raw else "")
except Exception:
    print("")
PY
}

is_registered_unit() {
  local unit="$1"
  for u in "${LIVE_WRITER_UNITS[@]:-}"; do
    [[ "$u" == "$unit" ]] && return 0
  done
  return 1
}

# Verify the lock-holder PID matches the PID the supervisor reports for UNIT.
# Returns 0 on a clean handshake, non-zero otherwise. Pure verification — it
# changes nothing.
verify_lock_handshake() {
  local unit="$1"
  local lf holder_pid sup_pid
  lf="$(lock_file_path)"
  holder_pid="$(lock_holder_pid)"

  log "lock file:        $lf"
  log "lock-holder PID:  ${holder_pid:-<none>}"

  if [[ -z "$holder_pid" ]]; then
    log "no live writer currently holds the lock (paper-safe state)."
    return 1
  fi

  if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl not available here — cannot read supervisor PID for '$unit'."
    return 1
  fi
  sup_pid="$(systemctl show -p MainPID --value "$unit" 2>/dev/null || true)"
  log "supervisor PID:   ${sup_pid:-<none>} (unit=$unit)"

  if [[ -z "$sup_pid" || "$sup_pid" == "0" ]]; then
    log "unit '$unit' has no running MainPID."
    return 1
  fi
  if [[ "$holder_pid" != "$sup_pid" ]]; then
    log "MISMATCH: lock-holder PID ($holder_pid) != supervisor PID ($sup_pid)."
    return 1
  fi
  log "OK: lock-holder PID == supervisor PID ($holder_pid)."
  return 0
}

# Stop every OTHER registered live unit so only the target can hold the lock.
stop_other_units() {
  local keep="$1" did=0
  for u in "${LIVE_WRITER_UNITS[@]:-}"; do
    [[ -z "$u" || "$u" == "$keep" ]] && continue
    if [[ "$DRY_RUN" == "1" ]]; then
      log "[dry-run] would stop other live unit: $u"
    else
      log "stopping other live unit: $u"
      systemctl stop "$u" || die "failed to stop $u"
    fi
    did=1
  done
  [[ "$did" == "0" ]] && log "no other registered live units to stop."
}

append_history() {
  local verb="$1" unit="$2" result="$3"
  mkdir -p "$(dirname "$HISTORY_LOG")"
  printf '%s\t%s\tunit=%s\taccount=%s\tholder_pid=%s\tresult=%s\toperator=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$verb" "$unit" "$ACCOUNT_NAME" \
    "$(lock_holder_pid)" "$result" "${USER:-unknown}" >>"$HISTORY_LOG"
  log "history appended -> $HISTORY_LOG"
}

# The 6-step live-cutover checklist (issue #10). Enforced before any --live run.
enforce_checklist() {
  local unit="$1" failed=0
  log "enforcing live-cutover checklist for unit '$unit':"

  # 3 + no ad-hoc: the unit must be a registered live writer.
  if is_registered_unit "$unit"; then
    log "  [ok] (3) unit is registered in LIVE_WRITER_UNITS"
  else
    log "  [NO] (3) unit '$unit' is NOT in LIVE_WRITER_UNITS — add it in the same"
    log "          commit that wires the gated live entry point."
    failed=1
  fi

  # 5: env gates must be present (and only ever in the supervised unit, not here).
  if [[ "${ALLOW_ALPACA_LIVE_TRADING:-}" == "1" ]]; then
    log "  [ok] (5a) ALLOW_ALPACA_LIVE_TRADING=1"
  else
    log "  [NO] (5a) ALLOW_ALPACA_LIVE_TRADING must be 1 (set in the unit only)"
    failed=1
  fi
  if [[ "${ALP_PAPER:-}" == "0" ]]; then
    log "  [ok] (5b) ALP_PAPER=0"
  else
    log "  [NO] (5b) ALP_PAPER must be 0 (set in the unit only)"
    failed=1
  fi

  # 1, 2, 4, 6: human-confirmed gates the script cannot prove on its own.
  if [[ "${KEEL_GATE_CLEARED:-}" == "1" ]]; then
    log "  [ok] (1) champion cleared the Phase-3 gate (operator-attested)"
  else
    log "  [NO] (1) set KEEL_GATE_CLEARED=1 to attest the gate pass (median>=0.27)"
    failed=1
  fi
  if [[ "${KEEL_PAPER_CLEAN:-}" == "1" ]]; then
    log "  [ok] (2) paper run clean over a meaningful window (operator-attested)"
  else
    log "  [NO] (2) set KEEL_PAPER_CLEAN=1 to attest a clean paper window"
    failed=1
  fi
  if [[ "${KEEL_ACCOUNT_LOCK_SHARED_OK:-}" == "1" ]]; then
    log "  [ok] (4) shared-account lock path/name reconciled (operator-attested)"
  else
    log "  [NO] (4) set KEEL_ACCOUNT_LOCK_SHARED_OK=1 if sharing an Alpaca account"
    failed=1
  fi

  return "$failed"
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
MODE="dry-run"
DRY_RUN="1"
UNIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)    usage 0 ;;
    --dry-run)    MODE="dry-run"; DRY_RUN="1"; shift ;;
    --live)       MODE="live";    DRY_RUN="0"; shift ;;
    --*)          die "unknown flag: $1" ;;
    *)            UNIT="$1"; shift ;;
  esac
done

if [[ -z "$UNIT" ]]; then
  if [[ "${#LIVE_WRITER_UNITS[@]}" -gt 0 ]]; then
    UNIT="${LIVE_WRITER_UNITS[0]}"
  else
    UNIT="<none-registered>"
  fi
fi

log "mode=$MODE account=$ACCOUNT_NAME unit=$UNIT"

if [[ "$MODE" == "dry-run" ]]; then
  log "DRY-RUN: verifying the lock handshake only. Nothing is started or stopped."
  if verify_lock_handshake "$UNIT"; then
    append_history "dry-run" "$UNIT" "handshake-ok"
    log "dry-run handshake OK."
    exit 0
  else
    append_history "dry-run" "$UNIT" "handshake-incomplete"
    log "dry-run handshake incomplete (expected in a paper-safe state)."
    exit 0
  fi
fi

# --- live path: fail closed unless EVERY gate passes -----------------------
log "LIVE cutover requested — enforcing all gates before any action."

if [[ "${#LIVE_WRITER_UNITS[@]}" -eq 0 ]]; then
  append_history "live" "$UNIT" "refused-empty-registry"
  die "LIVE_WRITER_UNITS is empty — no gated live entry point exists. Refusing."
fi

if ! enforce_checklist "$UNIT"; then
  append_history "live" "$UNIT" "refused-checklist"
  die "live-cutover checklist incomplete — refusing live enablement."
fi

stop_other_units "$UNIT"

log "starting/ensuring supervised unit: $UNIT"
systemctl restart "$UNIT" || die "failed to (re)start $UNIT"

# Give the unit a moment to acquire the lock, then verify the handshake.
for _ in 1 2 3 4 5; do
  verify_lock_handshake "$UNIT" && break || sleep 1
done

if verify_lock_handshake "$UNIT"; then
  append_history "live" "$UNIT" "cutover-ok"
  log "LIVE cutover verified: $UNIT holds the single writer lock."
  exit 0
fi

append_history "live" "$UNIT" "cutover-FAILED"
die "post-deploy handshake FAILED — lock-holder PID does not match $UNIT. " \
    "Investigate immediately; the live writer is NOT verified."
