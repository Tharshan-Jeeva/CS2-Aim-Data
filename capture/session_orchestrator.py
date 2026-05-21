import argparse
import json
import os
import signal
import sys
import threading
import time
import queue
from pathlib import Path

from capture.telemetry_server import create_app, get_events, save_events, run_server
from capture.bot_aim_generator import BotAimGenerator, load_config

BOT_LABELS = {"bot_raw", "bot_smooth", "bot_humanised_low",
              "bot_humanised_med", "bot_humanised_high"}
SM_NATIVE_LABELS = {"sm_native_raw", "sm_native_smooth",
                    "sm_native_humanised_med", "sm_native_humanised_high"}
PROFILES_DIR = Path(__file__).parent / "bot_profiles"

# Default per-condition duration. Matches the protocol's 5-min requirement.
# The timer starts from the FIRST TICK, not orchestrator launch, so setup
# time (server boot, joining, console commands) does not eat into the session.
DEFAULT_DURATION_S = 300.0

# If ticks stop arriving for this many seconds after the first tick has been
# seen, warn the researcher. Ticks can legitimately pause when the player is
# dead between rounds, so this is a SOFT warning and never aborts the session.
TICK_LOSS_WARN_S = 10.0

# Heartbeats fire every 0.5 s in the SourceMod plugin regardless of whether the
# player is alive. If heartbeats stop, the participant disconnected, the server
# was killed, or Flask is no longer reachable — the session is dead.
HEARTBEAT_LOSS_WARN_S = 5.0
HEARTBEAT_LOSS_ABORT_S = 30.0

# How often to print progress lines.
PROGRESS_INTERVAL_S = 10.0


def build_session_name(player_id: str, label: str) -> str:
    return f"{player_id}_{label}_{int(time.time())}"


def is_bot_session(label: str) -> bool:
    return label in BOT_LABELS


def is_sm_native_session(label: str) -> bool:
    return label in SM_NATIVE_LABELS


def native_mode_from_label(label: str) -> str:
    if label == "sm_native_raw":
        return "raw"
    if label == "sm_native_smooth":
        return "smooth"
    if label == "sm_native_humanised_med":
        return "humanised"
    if label == "sm_native_humanised_high":
        return "humanised_high"
    raise ValueError(f"Unknown SM-native label: {label}")


def resolve_bot_profile(label: str) -> str:
    path = PROFILES_DIR / f"{label}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Bot profile not found: {path}")
    return str(path)


def count_ticks(app) -> int:
    # Walk the live event list; cheap even at 100 Hz × 5 min (~30 k events).
    return sum(1 for e in get_events(app) if e.get("type") == "tick")


def count_heartbeats(app) -> int:
    return sum(1 for e in get_events(app) if e.get("type") == "heartbeat")


def generate_manifest(session_name: str, events_path: str,
                      diagnostics: dict | None = None) -> dict:
    """Write a quick-check summary next to the events file.

    The flags it emits are the same ones the analysis pipeline uses as
    exclusion criteria, so a researcher sees data-quality problems
    immediately after Ctrl+C instead of three weeks later in the audit.

    `diagnostics` records orchestrator-side timing (start/stop in wall-clock
    Unix epoch, stop_reason). Compare these against the in-game
    `timestamp_server` to see whether the data stream died early (plugin /
    participant gone) or the orchestrator itself was killed early
    (Ctrl+C in the wrong window).
    """
    path = Path(events_path)
    with open(path) as f:
        events = json.load(f)

    ticks = [e for e in events if e.get("type") == "tick"]
    heartbeats = [e for e in events if e.get("type") == "heartbeat"]
    fires = [e for e in events if e.get("type") == "weapon_fire"]
    kills = [e for e in events if e.get("type") == "kill"]
    rounds = [e for e in events if e.get("type") == "round_start"]

    if ticks:
        ts = [float(t["timestamp_server"]) for t in ticks]
        duration_s = ts[-1] - ts[0]
        mean_hz = len(ticks) / duration_s if duration_s > 0 else 0.0
    else:
        duration_s = 0.0
        mean_hz = 0.0

    flags: list[str] = []
    if duration_s < 270:
        flags.append("duration_below_270s")
    # Note: the manifest's `estimated_hz` is intentionally naive — it averages
    # ticks across the full game-time span, including between-round dead time
    # when the player isn't alive. That makes it always look ~85-90 Hz for a
    # healthy session. The audit's `--active-only` mode is the authoritative
    # Hz check (strips dead time → reports the true ~100 Hz). We only flag
    # here if Hz is catastrophically low — a real signal of capture failure
    # rather than a normal artefact of between-round downtime.
    if mean_hz and mean_hz < 60:
        flags.append("mean_hz_below_60")
    if len(fires) < 10:
        flags.append("fewer_than_10_fires")
    if not ticks:
        flags.append("no_ticks_received")

    # Cross-check: did the orchestrator outlive the data stream?
    # If yes, the source of the cut-off is in CS:Source / SourceMod / the
    # participant's client — NOT the orchestrator. If no, it's the
    # orchestrator (Ctrl+C in wrong window, terminal closed, etc.).
    if diagnostics and ticks:
        orch_runtime_after_last_tick = (
            diagnostics["stop_unix_ts"] - diagnostics["last_event_unix_ts"]
        )
        if orch_runtime_after_last_tick > 5.0:
            flags.append("data_stream_died_before_orchestrator")
        elif orch_runtime_after_last_tick < 1.0 and diagnostics["stop_reason"] != "duration_reached":
            flags.append("orchestrator_killed_while_streaming")

    # Wall-clock vs game-time ratio.
    #
    # The plugin sends one tick per server frame, with each tick stamped at
    # GetGameTime() + 0.01s. If the server is CPU-starved and only manages,
    # say, 28 frames per real second, the captured `timestamp_server` values
    # will look like a clean 100 Hz stream (because each frame still advances
    # the clock by 0.01 s) — but the actual session length will be ~28% of
    # wall clock. The audit cannot see this, because it measures inter-tick
    # intervals using the *server's* clock.
    #
    # Detect it by comparing wall-clock event-arrival span against game-time
    # tick span. A healthy server runs at ~100% real-time; <90% is a hard
    # red flag that the server is falling behind.
    if diagnostics and ticks and diagnostics.get("first_tick_unix_ts"):
        wall_span = diagnostics["last_event_unix_ts"] - diagnostics["first_tick_unix_ts"]
        if wall_span > 5.0:
            realtime_ratio = duration_s / wall_span
            manifest_extra_realtime = round(realtime_ratio, 3)
            if realtime_ratio < 0.90:
                flags.append("server_slower_than_realtime")
        else:
            manifest_extra_realtime = None
    else:
        manifest_extra_realtime = None

    manifest = {
        "session_name": session_name,
        "events_file": path.name,
        "tick_count": len(ticks),
        "heartbeat_count": len(heartbeats),
        "duration_s": round(duration_s, 1),
        "estimated_hz": round(mean_hz, 2),
        "realtime_ratio": manifest_extra_realtime,
        "weapon_fires": len(fires),
        "kills": len(kills),
        "rounds": len(rounds),
        "flags": flags,
    }
    if diagnostics:
        manifest["diagnostics"] = diagnostics

    manifest_path = path.with_name(path.stem.replace("_events", "_manifest") + ".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Session] Manifest: {manifest_path.name}")
    print(f"[Session]   duration={duration_s:.1f}s  ticks={len(ticks)}  "
          f"~{mean_hz:.1f}Hz  fires={len(fires)}  rounds={len(rounds)}")
    if flags:
        print(f"[Session] ⚠ FLAGS: {', '.join(flags)}")
    else:
        print(f"[Session] ✓ no flags")
    return manifest


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run a single capture session. Auto-stops after --duration "
                    "seconds from the first tick (no Ctrl+C required).")
    ap.add_argument("--participant", "-p",
        help="Participant ID (skips the interactive prompt).")
    ap.add_argument("--label", "-l",
        help="Condition label (skips the interactive prompt).")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
        help=f"Per-condition session duration in seconds, measured from the "
             f"first tick (default: {DEFAULT_DURATION_S:.0f}s = 5 min). "
             f"The protocol requires >=270s.")
    ap.add_argument("--no-auto-stop", action="store_true",
        help="Disable auto-stop and revert to Ctrl+C-only behaviour. "
             "Not recommended; only included for backwards compatibility.")
    return ap.parse_args(argv)


def run_session(argv=None):
    args = parse_args(argv)

    player_id = args.participant or input("Enter participant ID (e.g. P01): ").strip()
    if not player_id:
        print("Error: participant ID required.")
        sys.exit(1)

    if args.label:
        label = args.label
    else:
        print("\nLabels: human, bot_raw, bot_smooth, bot_humanised_low, "
              "bot_humanised_med, bot_humanised_high, "
              "sm_native_raw, sm_native_smooth, "
              "sm_native_humanised_med, sm_native_humanised_high")
        label = input("Enter label: ").strip()
    if not label:
        print("Error: label required.")
        sys.exit(1)

    session_name = build_session_name(player_id, label)

    os.makedirs("sessions", exist_ok=True)
    os.makedirs("demos", exist_ok=True)

    auto_stop = not args.no_auto_stop

    print(f"\n[Session] {session_name}")
    print(f"[Session] Label: {label}")
    print(f"[Session] Duration: {args.duration:.0f}s ({args.duration/60:.1f} min)")
    print(f"[Session] Auto-stop: {'ENABLED — timer starts on first tick' if auto_stop else 'DISABLED (Ctrl+C only)'}")
    print(f"[Session] Bot session: {is_bot_session(label)}")
    print(f"[Session] SM-native session: {is_sm_native_session(label)}")

    # maxsize=1: live bot control always reads the freshest tick only.
    # Full telemetry history is still saved by telemetry_server regardless.
    # A one-item queue prevents a stale tick backlog from causing aim lag / overshoot.
    bot_queue = queue.Queue(maxsize=1) if is_bot_session(label) else None
    stop_event = threading.Event()
    stop_reason = {"reason": "unknown"}

    app = create_app(session_name=session_name, bot_queue=bot_queue)
    server_thread = threading.Thread(
        target=run_server, args=(app,), daemon=True)
    server_thread.start()
    time.sleep(0.5)
    print("[Session] Telemetry server running on port 3000")

    bot_thread = None
    if is_bot_session(label):
        config_path = resolve_bot_profile(label)
        config = load_config(config_path)
        generator = BotAimGenerator(config)
        bot_thread = threading.Thread(
            target=generator.run, args=(bot_queue, stop_event), daemon=True)
        bot_thread.start()
        print(f"[Session] Bot aim generator started (mode={config['mode']})")
        print("[Session] In CS:Source console run:")
        print("          sm_override_active 1")
        print("          sm_override_me")

    if is_sm_native_session(label):
        native_mode = native_mode_from_label(label)
        print("[Session] SourceMod-native aim mode selected.")
        print("[Session] Python BotAimGenerator is NOT started for this label.")
        print("[Session] In CS:Source console run:")
        print(f"          sm_nativeaim_mode {native_mode}")
        print("          sm_nativeaim_active 0")
        print('          alias +nativeaim "sm_nativeaim_active 1"')
        print('          alias -nativeaim "sm_nativeaim_active 0"')
        print("          bind mouse4 +nativeaim")
        print("          sm_nativeaim_status")
        print("          sm_telemetry_me")

    print("\n" + "=" * 50)
    print("In CS:Source console, run (in this order):")
    # cl_cmdrate caps the OnPlayerRunCmd hook in cs_aim_telemetry.sp.
    # Default cl_cmdrate is ~66 → telemetry caps at ~66 Hz instead of 100.
    print("  cl_cmdrate 100")
    print("  cl_updaterate 100")
    print("  rate 1000000")
    print(f"  record {session_name}")
    print("  sm_telemetry_me")
    print("=" * 50)
    if auto_stop:
        print(f"\nThe session will auto-stop {args.duration:.0f}s after the first tick.")
        print("Ctrl+C in THIS window is still available as an emergency stop.\n")
    else:
        print("\nPlay your session. Press Ctrl+C in THIS window when done.\n")

    def signal_handler(sig, frame):
        if not stop_event.is_set():
            print("\n[Session] Stop signal received...")
            stop_reason["reason"] = "ctrl_c"
            stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # Main wait loop.
    #
    # The duration timer is anchored to the FIRST TICK so that setup time
    # (server boot, joining the server, entering console commands) does not
    # consume the session window. This was the structural bug that caused
    # P02 / P03 / P04 sessions to be 1-3 min instead of 5.
    #
    # There are TWO watchdogs:
    #   * tick watchdog (warns):  ticks can legitimately pause when the
    #     participant dies between rounds, so we only warn — never abort.
    #   * heartbeat watchdog (aborts): the SourceMod plugin sends a heartbeat
    #     every 0.5 s regardless of whether the player is alive. If heartbeats
    #     stop arriving, the participant has disconnected, the server is gone,
    #     or the orchestrator's Flask receiver is no longer reachable. In all
    #     three cases the session is dead — there is nothing to wait for.
    monot_now = time.monotonic
    loop_start_mono = monot_now()
    loop_start_wall = time.time()
    first_tick_mono = None
    first_tick_wall = None
    deadline = None
    last_seen_ticks = 0
    last_seen_heartbeats = 0
    last_tick_change_mono = loop_start_mono
    last_heartbeat_change_mono = loop_start_mono
    last_event_wall = loop_start_wall  # wall-clock of most recent ANY event
    last_progress_t = loop_start_mono - PROGRESS_INTERVAL_S  # force immediate
    warned_no_first_tick = False
    warned_tick_loss = False
    warned_heartbeat_loss = False

    try:
        while not stop_event.is_set():
            now = monot_now()
            elapsed = now - loop_start_mono

            current_ticks = count_ticks(app)
            current_heartbeats = count_heartbeats(app)

            # First-tick detection — this is what starts the duration clock.
            if first_tick_mono is None and current_ticks > 0:
                first_tick_mono = now
                first_tick_wall = time.time()
                if auto_stop:
                    deadline = first_tick_mono + args.duration
                print(f"\n[Session] First tick received at t={elapsed:.0f}s. "
                      f"{'Timer started, auto-stop in ' + str(int(args.duration)) + 's.' if auto_stop else 'Streaming.'}\n")
                last_tick_change_mono = now
                last_heartbeat_change_mono = now

            # Tick-flow watchdog (warn-only).
            if current_ticks > last_seen_ticks:
                last_seen_ticks = current_ticks
                last_tick_change_mono = now
                last_event_wall = time.time()
                warned_tick_loss = False
            elif first_tick_mono is not None:
                idle_s = now - last_tick_change_mono
                if idle_s >= TICK_LOSS_WARN_S and not warned_tick_loss:
                    print(f"\n[Session] ⚠ no ticks for {idle_s:.0f}s "
                          f"(player may be dead between rounds — checking heartbeats next).\n")
                    warned_tick_loss = True

            # Heartbeat watchdog (abort if stream is truly dead).
            if current_heartbeats > last_seen_heartbeats:
                last_seen_heartbeats = current_heartbeats
                last_heartbeat_change_mono = now
                last_event_wall = time.time()
                warned_heartbeat_loss = False
            elif first_tick_mono is not None:
                hb_idle_s = now - last_heartbeat_change_mono
                if hb_idle_s >= HEARTBEAT_LOSS_WARN_S and not warned_heartbeat_loss:
                    print(f"\n[Session] ⚠ WARNING: no heartbeats for "
                          f"{hb_idle_s:.0f}s — the plugin/server side is "
                          f"silent. Likely causes:\n"
                          f"             - CS:Source dedicated server was killed "
                          f"(Ctrl+C in the wrong terminal?)\n"
                          f"             - participant's game crashed / they "
                          f"alt-tabbed away long enough to disconnect\n"
                          f"             - the orchestrator's Flask receiver is "
                          f"no longer accepting requests\n"
                          f"           Will auto-abort at {HEARTBEAT_LOSS_ABORT_S:.0f}s "
                          f"of silence to avoid hiding the failure.\n")
                    warned_heartbeat_loss = True
                if hb_idle_s >= HEARTBEAT_LOSS_ABORT_S:
                    print(f"\n[Session] Heartbeat silence exceeded "
                          f"{HEARTBEAT_LOSS_ABORT_S:.0f}s — aborting. "
                          f"Investigate before re-recording.")
                    stop_reason["reason"] = "heartbeat_lost"
                    stop_event.set()
                    break

            # Auto-stop on deadline.
            if deadline is not None and now >= deadline:
                print(f"\n[Session] Duration reached "
                      f"({args.duration:.0f}s of ticks). Stopping cleanly.")
                stop_reason["reason"] = "duration_reached"
                stop_event.set()
                break

            # Periodic progress line.
            if now - last_progress_t >= PROGRESS_INTERVAL_S:
                if first_tick_mono is None:
                    print(f"[Session] t={elapsed:.0f}s — "
                          f"waiting for first tick "
                          f"(heartbeats received: {current_heartbeats}). "
                          f"Did you run sm_telemetry_me in the game console?")
                    if elapsed >= 60 and not warned_no_first_tick:
                        print("[Session] ⚠ Still no ticks after 60s. "
                              "Check: server running? plugin loaded? "
                              "player in-game and alive?")
                        warned_no_first_tick = True
                else:
                    session_elapsed = now - first_tick_mono
                    remaining = max(0.0, args.duration - session_elapsed) if auto_stop else float('inf')
                    rem_str = f"remaining={remaining:.0f}s" if auto_stop else "no auto-stop"
                    print(f"[Session] session t={session_elapsed:.0f}s  "
                          f"{rem_str}  ticks={current_ticks}  hb={current_heartbeats}")
                last_progress_t = now

            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_reason["reason"] = "ctrl_c"
        stop_event.set()

    # Final save + manifest. This block must run for every exit path.
    stop_wall = time.time()
    events_path = save_events(app)
    diagnostics = {
        "stop_reason": stop_reason["reason"],
        "auto_stop_enabled": auto_stop,
        "requested_duration_s": args.duration if auto_stop else None,
        "start_unix_ts": round(loop_start_wall, 3),
        "first_tick_unix_ts": round(first_tick_wall, 3) if first_tick_wall else None,
        "last_event_unix_ts": round(last_event_wall, 3),
        "stop_unix_ts": round(stop_wall, 3),
        "orchestrator_runtime_s": round(stop_wall - loop_start_wall, 1),
        "orchestrator_seconds_after_last_event": round(stop_wall - last_event_wall, 1),
    }
    try:
        generate_manifest(session_name, events_path, diagnostics=diagnostics)
    except Exception as exc:
        print(f"[Session] Manifest generation failed: {exc}")

    print(f"\n[Session] Done. Stop reason: {stop_reason['reason']}")
    print(f"[Session] Orchestrator ran for {diagnostics['orchestrator_runtime_s']}s; "
          f"last event was {diagnostics['orchestrator_seconds_after_last_event']}s "
          f"before stop.")
    if diagnostics['orchestrator_seconds_after_last_event'] > 5:
        print("[Session] → The data stream died BEFORE the orchestrator stopped — "
              "look at CS:Source / SourceMod / the participant's game.")
    elif stop_reason["reason"] == "ctrl_c" and diagnostics['orchestrator_seconds_after_last_event'] < 1:
        print("[Session] → The orchestrator was Ctrl+C'd while data was still "
              "streaming. Was the Ctrl+C in the right window?")

    # If the server was running slower than real time, the data IS captured
    # cleanly but the recorded gameplay is short. Tell the researcher LOUDLY.
    if first_tick_wall is not None:
        wall_span_final = last_event_wall - first_tick_wall
        if wall_span_final > 5.0:
            ticks_final = count_ticks(app)
            actual_hz = ticks_final / wall_span_final if wall_span_final else 0.0
            # Reload events to compute game-time span (cheap; same as manifest).
            try:
                with open(events_path) as f:
                    ev = json.load(f)
                tks = [e for e in ev if e.get("type") == "tick"]
                game_span = (float(tks[-1]["timestamp_server"]) -
                             float(tks[0]["timestamp_server"])) if len(tks) >= 2 else 0.0
            except Exception:
                game_span = 0.0
            if wall_span_final and game_span:
                ratio = game_span / wall_span_final
                if ratio < 0.90:
                    print()
                    print("=" * 60)
                    print(f"[Session] ⚠⚠⚠ SERVER RAN AT {ratio*100:.0f}% REAL-TIME SPEED ⚠⚠⚠")
                    print(f"[Session] Wall clock window:  {wall_span_final:.1f}s "
                          f"({wall_span_final/60:.1f} min)")
                    print(f"[Session] Captured gameplay: {game_span:.1f}s "
                          f"({game_span/60:.2f} min)")
                    print(f"[Session] Actual tick rate:  {actual_hz:.1f} Hz "
                          f"(target: 100 Hz)")
                    print()
                    print(f"[Session] CS:Source dedicated server is CPU-starved. "
                          f"The audit cannot see this because timestamp_server "
                          f"still advances cleanly at 100 Hz inside the stream.")
                    print(f"[Session] Fixes (try in order):")
                    print(f"[Session]   1. fps_max 60 in the game client (frees CPU)")
                    print(f"[Session]   2. renice -n -10 the srcds_linux process "
                          f"(higher scheduler priority)")
                    print(f"[Session]   3. Reduce bot_quota or bot_difficulty")
                    print(f"[Session]   4. Run the server on a separate machine")
                    print("=" * 60)
    print(f"[Session] Files saved to sessions/")
    print(f"[Session] Don't forget: run 'stop' in CS:Source console")
    print(f"[Session] Move the .dem file to demos/{session_name}.dem")


if __name__ == "__main__":
    run_session()
