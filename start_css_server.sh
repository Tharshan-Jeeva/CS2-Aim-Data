#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/cssource_server"

exec ./srcds_run -game cstrike -console -insecure -tickrate 100 \
    +sv_lan 1 +map de_dust2 +bot_quota 5 \
    +sv_maxcmdrate 128 +sv_maxupdaterate 128 \
    +sv_mincmdrate 100 +sv_minupdaterate 100 \
    +sv_maxrate 1000000 +sv_minrate 100000 \
    "$@"
