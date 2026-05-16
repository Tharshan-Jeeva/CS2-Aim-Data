#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/cssource_server"

exec ./srcds_run -game cstrike -console -insecure -tickrate 100 +sv_lan 1 +map de_dust2 +bot_quota 5 "$@"
