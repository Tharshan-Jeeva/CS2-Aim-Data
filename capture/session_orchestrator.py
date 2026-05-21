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
# seen, warn the researcher. The most common cause is the participant having
# accidentally killed the CS:Source server in another terminal.
TICK_LOSS_WARN_S = 10.0

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


def generate_manifest(session_name: str, events_path: str) -> dict:
    """Write a quick-check summary next to the events file.

    The flags it emits are the same ones the analysis pipeline uses as
    exclusion criteria, so a researcher sees data-quality problems
    immediately after Ctrl+C instead of three weeks later in the audit.
    """
    path = Path(events_path)
    with open(path) as f:
        events = json.load(f)

    ticks = [e for e in events if e.get("type") == "tick"]
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
    if mean_hz and mean_hz < 95:
        flags.append("mean_hz_below_95")
    if len(fires) < 10:
        flags.append("fewer_than_10_fires")
    if not ticks:
        flags.append("no_ticks_received")

    manifest = {
        "session_name": session_name,
        "events_file": path.name,
        "tick_count": len(ticks),
        "duration_s": round(duration_s, 1),
        "estimated_hz": round(mean_hz, 2),
        "weapon_fires": len(fires),
        "kills": len(kills),
        "rounds": len(rounds),
        "flags": flags,
    }

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
    monot_now = time.monotonic
    loop_start = monot_now()
    first_tick_t = None
    deadline = None
    last_seen_ticks = 0
    last_tick_change_t = loop_start
    last_progress_t = loop_start - PROGRESS_INTERVAL_S  # force immediate
    warned_no_first_tick = False
    warned_tick_loss = False

    try:
        while not stop_event.is_set():
            now = monot_now()
            elapsed = now - loop_start

            current_ticks = count_ticks(app)

            # First-tick detection — this is what starts the duration clock.
            if first_tick_t is None and current_ticks > 0:
                first_tick_t = now
                if auto_stop:
                    deadline = first_tick_t + args.duration
                print(f"\n[Session] First tick received at t={elapsed:.0f}s. "
                      f"{'Timer started, auto-stop in ' + str(int(args.duration)) + 's.' if auto_stop else 'Streaming.'}\n")
                last_tick_change_t = now

            # Tick-flow watchdog.
            if current_ticks > last_seen_ticks:
                last_seen_ticks = current_ticks
                last_tick_change_t = now
                warned_tick_loss = False
            elif first_tick_t is not None:
                idle_s = now - last_tick_change_t
                if idle_s >= TICK_LOSS_WARN_S and not warned_tick_loss:
                    print(f"\n[Session] ⚠ WARNING: no ticks received for "
                          f"{idle_s:.0f}s.\n"
                          f"           Possible causes:\n"
                          f"             - CS:Source server was Ctrl+C'd in "
                          f"another terminal\n"
                          f"             - participant disconnected from the server\n"
                          f"             - player is dead between rounds (resumes shortly)\n")
                    warned_tick_loss = True

            # Auto-stop on deadline.
            if deadline is not None and now >= deadline:
                print(f"\n[Session] Duration reached "
                      f"({args.duration:.0f}s of ticks). Stopping cleanly.")
                stop_reason["reason"] = "duration_reached"
                stop_event.set()
                break

            # Periodic progress line.
            if now - last_progress_t >= PROGRESS_INTERVAL_S:
                if first_tick_t is None:
                    print(f"[Session] t={elapsed:.0f}s — "
                          f"waiting for first tick. "
                          f"Did you run sm_telemetry_me in the game console?")
                    if elapsed >= 60 and not warned_no_first_tick:
                        print("[Session] ⚠ Still no ticks after 60s. "
                              "Check: server running? plugin loaded? "
                              "player in-game and alive?")
                        warned_no_first_tick = True
                else:
                    session_elapsed = now - first_tick_t
                    remaining = max(0.0, args.duration - session_elapsed) if auto_stop else float('inf')
                    rem_str = f"remaining={remaining:.0f}s" if auto_stop else "no auto-stop"
                    print(f"[Session] session t={session_elapsed:.0f}s  "
                          f"{rem_str}  ticks={current_ticks}")
                last_progress_t = now

            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_reason["reason"] = "ctrl_c"
        stop_event.set()

    # Final save + manifest. This block must run for every exit path.
    events_path = save_events(app)
    try:
        generate_manifest(session_name, events_path)
    except Exception as exc:
        print(f"[Session] Manifest generation failed: {exc}")

    print(f"\n[Session] Done. Stop reason: {stop_reason['reason']}")
    print(f"[Session] Files saved to sessions/")
    print(f"[Session] Don't forget: run 'stop' in CS:Source console")
    print(f"[Session] Move the .dem file to demos/{session_name}.dem")


if __name__ == "__main__":
    run_session()
