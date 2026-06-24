#!/usr/bin/env bash
# ============================================================================
#  start_exhibition.sh  --  one-command exhibition kiosk launcher
#
#  Brings up the unattended "walk up and play" kiosk:
#    1. deploys the kiosk plugin + configs + MOTD into the (gitignored) server
#    2. runs the CS:Source dedicated server under a self-healing supervisor
#    3. auto-launches the CS:Source game client and connects it to the server
#
#  No telemetry / recording is involved (exhibition is for SHOW, not data).
#  The only aimbot used is the SM-native in-engine controller.
#
#  Leave this terminal open. To stop: run ./stop_exhibition.sh (any terminal)
#  or press Ctrl+C here.
# ============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
#  Operator-tunable settings
# ----------------------------------------------------------------------------
MAP="de_dust2"
SERVER_IP="127.0.0.1"        # single box: client connects to localhost
SERVER_PORT=27015            # srcds game port (UDP)
AUTO_LAUNCH_CLIENT=1         # 1 = also start + connect the game client
CSS_APPID=240                # CS:Source Steam app id
BOT_QUOTA=5
CLIENT_WAIT_TIMEOUT=180      # max seconds to wait for the server to be joinable
                             # before launching the client anyway

# ----------------------------------------------------------------------------
#  Paths
# ----------------------------------------------------------------------------
REPO="$(cd "$(dirname "$0")" && pwd)"
INSTALL="$REPO/cssource_server"
CSTRIKE="$INSTALL/cstrike"
SM_SCRIPTING="$CSTRIKE/addons/sourcemod/scripting"
SM_PLUGINS="$CSTRIKE/addons/sourcemod/plugins"
LOGS="$REPO/logs"
RUNFLAG="$LOGS/exhibition.run"
PIDFILE="$LOGS/exhibition.pids"

mkdir -p "$LOGS"
: > "$PIDFILE"

log()  { echo "[exhibition] $*"; }
fail() { echo "[exhibition] ERROR: $*" >&2; exit 1; }

[ -d "$INSTALL" ]  || fail "CS:Source install not found at $INSTALL (install via SteamCMD; see SETUP.md)."
[ -x "$INSTALL/srcds_run" ] || fail "srcds_run not found/executable at $INSTALL/srcds_run."

# ----------------------------------------------------------------------------
#  1. Deploy: compile the kiosk plugin and copy configs / MOTD into the install
# ----------------------------------------------------------------------------
deploy() {
    log "Deploying kiosk plugin + configs into $CSTRIKE ..."

    # Compile cs_aim_kiosk.sp fresh if the compiler is available; otherwise fall
    # back to the committed .smx so the kiosk still works on a bare machine.
    if [ -x "$SM_SCRIPTING/spcomp" ]; then
        if "$SM_SCRIPTING/spcomp" "$REPO/sourcemod/cs_aim_kiosk.sp" \
                -o"$SM_PLUGINS/cs_aim_kiosk.smx" -i"$SM_SCRIPTING/include" \
                > "$LOGS/plugin_compile.log" 2>&1; then
            log "Compiled cs_aim_kiosk.smx (see logs/plugin_compile.log)."
        else
            log "WARN: kiosk plugin compile failed; using committed .smx. See logs/plugin_compile.log."
            cp "$REPO/sourcemod/cs_aim_kiosk.smx" "$SM_PLUGINS/cs_aim_kiosk.smx"
        fi
    else
        log "spcomp not found; using committed cs_aim_kiosk.smx."
        cp "$REPO/sourcemod/cs_aim_kiosk.smx" "$SM_PLUGINS/cs_aim_kiosk.smx"
    fi

    # Ensure the live aim controller is present (the only aimbot used here).
    cp "$REPO/sourcemod/cs_aim_live_controller.smx" "$SM_PLUGINS/cs_aim_live_controller.smx"

    # Server cfg + MOTD (plain text is what actually shows; HTML kept as backup).
    cp "$REPO/exhibition/cfg/exhibition_server.cfg" "$CSTRIKE/cfg/exhibition_server.cfg"
    cp "$REPO/exhibition/motd_text.txt"             "$CSTRIKE/motd_text.txt"
    cp "$REPO/exhibition/motd.txt"                  "$CSTRIKE/motd.txt"

    log "Deploy complete."
}

# ----------------------------------------------------------------------------
#  2. Find the CS:Source CLIENT install and deploy its autoexec
# ----------------------------------------------------------------------------
find_client_cfg_dir() {
    local candidates=(
        "$HOME/.steam/steam/steamapps/common/Counter-Strike Source/cstrike/cfg"
        "$HOME/.local/share/Steam/steamapps/common/Counter-Strike Source/cstrike/cfg"
        "$HOME/.steam/root/steamapps/common/Counter-Strike Source/cstrike/cfg"
    )
    for d in "${candidates[@]}"; do
        if [ -d "$d" ]; then echo "$d"; return 0; fi
    done
    return 1
}

deploy_client_cfg() {
    local cfgdir
    if cfgdir="$(find_client_cfg_dir)"; then
        cp "$REPO/exhibition/cfg/autoexec_exhibition.cfg" "$cfgdir/autoexec_exhibition.cfg"
        log "Client autoexec deployed to: $cfgdir"
        return 0
    fi
    log "WARN: could not auto-locate the CS:Source client cfg dir."
    log "      Copy exhibition/cfg/autoexec_exhibition.cfg into your client's cstrike/cfg/ by hand."
    return 1
}

# ----------------------------------------------------------------------------
#  3. Supervised CS:Source dedicated server (auto-restart on crash)
# ----------------------------------------------------------------------------
supervise_server() {
    # IMPORTANT: srcds with `-console` HANGS at SteamAPI_Init when it has no
    # controlling terminal (i.e. when backgrounded by a supervisor like this).
    # `script` allocates a pseudo-TTY for it, which is what lets it finish Steam
    # init and load the map. tmux/screen would also work but aren't installed
    # here; `script` is part of util-linux and is always present.
    if ! command -v script >/dev/null 2>&1; then
        log "ERROR: 'script' (util-linux) not found; cannot give srcds a TTY. Install util-linux."
        return 1
    fi

    # Single-line command run inside the pty. srcds_run self-cd's to its own dir,
    # but we cd explicitly so relative paths (steam_appid.txt) resolve too.
    local srv_cmd="cd '$INSTALL' && exec ./srcds_run -game cstrike -console -insecure \
-tickrate 100 +sv_lan 1 +map '$MAP' +bot_quota $BOT_QUOTA \
+sv_maxcmdrate 128 +sv_maxupdaterate 128 +sv_mincmdrate 100 +sv_minupdaterate 100 \
+sv_maxrate 1000000 +sv_minrate 100000 +exec exhibition_server.cfg"

    while [ -f "$RUNFLAG" ]; do
        echo "===== $(date '+%F %T')  srcds starting (pty via script) =====" >> "$LOGS/srcds.log"
        # -q quiet, -a append (keep the markers above), -f flush, -c run command.
        script -q -a -f -c "$srv_cmd" "$LOGS/srcds.log" >/dev/null 2>&1 || true
        [ -f "$RUNFLAG" ] || break
        echo "===== $(date '+%F %T')  srcds exited; restarting in 3s =====" >> "$LOGS/srcds.log"
        sleep 3
    done
}

# ----------------------------------------------------------------------------
#  4. Supervised game client (single box). Relaunch if the visitor closes it.
# ----------------------------------------------------------------------------
client_running() {
    pgrep -f "hl2.*-game cstrike" >/dev/null 2>&1 || pgrep -x "hl2_linux" >/dev/null 2>&1
}

# Readiness probe. An empty CS:Source server hibernates and won't answer an A2S
# query, so we can't use a network ping. Instead we wait for srcds to print its
# boot-complete markers (Steam ID assigned / tickrate set / VAC line), which
# only appear once Steam init and the map load have finished.
server_ready() {
    pgrep -x srcds_linux >/dev/null 2>&1 || return 1
    tail -n 120 "$LOGS/srcds.log" 2>/dev/null \
        | grep -qE "Assigned anonymous gameserver Steam ID|VAC secure mode disabled|setting tickrate to"
}

supervise_client() {
    command -v steam >/dev/null 2>&1 || { log "WARN: steam not on PATH; cannot auto-launch client."; return; }

    # Wait until the server actually ANSWERS (not just until the port is bound).
    # srcds can take 30-90s to finish Steam init + load the map; launching the
    # client before then leaves it stuck at the menu ("Connection failed after
    # 4 retries"), which is exactly what happens with a naive fixed sleep.
    local waited=0
    while [ -f "$RUNFLAG" ] && ! server_ready; do
        sleep 3; waited=$((waited + 3))
        if [ "$waited" -ge "$CLIENT_WAIT_TIMEOUT" ]; then
            log "WARN: server still not answering after ${waited}s. Launching the client"
            log "      anyway; if it sits at the menu, the server hasn't finished booting —"
            log "      check logs/srcds.log, then reconnect with: connect $SERVER_IP"
            break
        fi
    done
    [ -f "$RUNFLAG" ] || return
    server_ready && log "Server is answering on $SERVER_IP:$SERVER_PORT after ~${waited}s; launching client."

    while [ -f "$RUNFLAG" ]; do
        if ! client_running; then
            echo "===== $(date '+%F %T')  launching CS:S client =====" >> "$LOGS/client.log"
            steam -applaunch "$CSS_APPID" -novid -console \
                +exec autoexec_exhibition.cfg +connect "$SERVER_IP" \
                >> "$LOGS/client.log" 2>&1 &
            # Give Steam time to spin the game up before re-checking.
            sleep 30
        fi
        sleep 10
    done
}

# ----------------------------------------------------------------------------
#  Shutdown handling
# ----------------------------------------------------------------------------
cleanup() {
    log "Shutting down ..."
    rm -f "$RUNFLAG"
    pkill -f "srcds_linux" 2>/dev/null || true
    wait 2>/dev/null || true
    log "Stopped."
}
trap cleanup INT TERM

# ----------------------------------------------------------------------------
#  Go
# ----------------------------------------------------------------------------
touch "$RUNFLAG"
deploy
deploy_client_cfg || true

log "Starting supervised CS:Source server (logs/srcds.log) ..."
supervise_server &
SRV_SUP=$!
echo "server_supervisor=$SRV_SUP" >> "$PIDFILE"

if [ "$AUTO_LAUNCH_CLIENT" = "1" ]; then
    log "Auto-launching game client and connecting to $SERVER_IP (logs/client.log) ..."
    supervise_client &
    CLI_SUP=$!
    echo "client_supervisor=$CLI_SUP" >> "$PIDFILE"
fi

log "Kiosk is up. Map=$MAP  bots=$BOT_QUOTA  client_autolaunch=$AUTO_LAUNCH_CLIENT"
log "Leave this window open. Stop with ./stop_exhibition.sh or Ctrl+C."
log "Logs: $LOGS/srcds.log  |  $LOGS/client.log  |  $LOGS/plugin_compile.log"

# Wait on the supervisors. Ctrl+C triggers cleanup() via the trap.
wait
