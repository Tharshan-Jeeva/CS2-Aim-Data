#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/cssource_server"

# Give srcds_linux the highest scheduler priority we can without requiring
# root. `nice -n -5` requires root or a raised RLIMIT_NICE; we fall back to
# the default niceness if the kernel refuses, but warn so the researcher
# knows to investigate if sessions are still running slow.
#
# Why: when the CS:Source client and the dedicated server are on the same
# machine, the client tends to dominate CPU. The server then ticks at
# whatever rate it can manage — captured `timestamp_server` still advances
# 0.01 s per tick (so the audit reports 100 Hz), but the WALL CLOCK rate
# can drop to 20-30 Hz, producing a session that looks like "1 minute of
# play" when the participant played for 5. This is the root cause of the
# short P02-P06 sessions; see the orchestrator's realtime_ratio detector.
NICE_PREFIX=()
if command -v nice >/dev/null 2>&1; then
    NICE_PREFIX=(nice -n -5)
fi

# Run; if nice fails (no permission to raise priority) retry without it.
if ! "${NICE_PREFIX[@]}" ./srcds_run -game cstrike -console -insecure -tickrate 100 \
    +sv_lan 1 +map de_dust2 +bot_quota 5 \
    +sv_maxcmdrate 128 +sv_maxupdaterate 128 \
    +sv_mincmdrate 100 +sv_minupdaterate 100 \
    +sv_maxrate 1000000 +sv_minrate 100000 \
    "$@" 2>&1 | tee /tmp/srcds_launch.log; then
    if grep -q "cannot set niceness" /tmp/srcds_launch.log 2>/dev/null; then
        echo "[start_css_server] nice -n -5 was refused; falling back to default priority."
        echo "[start_css_server] If sessions run slow, run as root or grant CAP_SYS_NICE."
        exec ./srcds_run -game cstrike -console -insecure -tickrate 100 \
            +sv_lan 1 +map de_dust2 +bot_quota 5 \
            +sv_maxcmdrate 128 +sv_maxupdaterate 128 \
            +sv_mincmdrate 100 +sv_minupdaterate 100 \
            +sv_maxrate 1000000 +sv_minrate 100000 \
            "$@"
    fi
fi
