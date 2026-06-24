#!/usr/bin/env bash
# ============================================================================
#  stop_exhibition.sh  --  clean shutdown for the exhibition kiosk
#
#  Stops, in order: the supervisor loops, the CS:Source dedicated server, and
#  (optionally) the game client. Safe to run from any terminal.
#
#  Usage:
#    ./stop_exhibition.sh           stop server + supervisors, leave client open
#    ./stop_exhibition.sh --all     also close the CS:Source game client
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
LOGS="$REPO/logs"
RUNFLAG="$LOGS/exhibition.run"
PIDFILE="$LOGS/exhibition.pids"

KILL_CLIENT=0
[ "${1:-}" = "--all" ] && KILL_CLIENT=1

log() { echo "[exhibition] $*"; }

# 1. Drop the run flag so the supervisor loops exit instead of restarting.
rm -f "$RUNFLAG"
log "Cleared run flag (supervisors will stop restarting)."

# 2. Kill the supervisor processes recorded at launch.
if [ -f "$PIDFILE" ]; then
    while IFS='=' read -r name pid; do
        [ -n "${pid:-}" ] || continue
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            log "Signalled $name (pid $pid)."
        fi
    done < "$PIDFILE"
fi

# 3. Kill the dedicated server.
if pgrep -f "srcds_linux" >/dev/null 2>&1; then
    pkill -f "srcds_linux" 2>/dev/null || true
    log "Stopped CS:Source dedicated server (srcds_linux)."
else
    log "No srcds_linux process found."
fi

# 4. Optionally close the game client.
if [ "$KILL_CLIENT" = "1" ]; then
    if pkill -f "hl2.*-game cstrike" 2>/dev/null || pkill -x "hl2_linux" 2>/dev/null; then
        log "Closed CS:Source game client."
    else
        log "No CS:Source client process found."
    fi
else
    log "Left the game client running (use --all to close it too)."
fi

# 5. Tidy.
rm -f "$PIDFILE"
log "Shutdown complete."
