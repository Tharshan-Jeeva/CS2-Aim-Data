import os
import signal
import sys
import threading
import time
import queue
from pathlib import Path

from capture.telemetry_server import create_app, save_events, run_server
from capture.bot_aim_generator import BotAimGenerator, load_config

BOT_LABELS = {"bot_raw", "bot_smooth", "bot_humanised_low",
              "bot_humanised_med", "bot_humanised_high"}
SM_NATIVE_LABELS = {"sm_native_raw", "sm_native_smooth",
                    "sm_native_humanised_med", "sm_native_humanised_high"}
PROFILES_DIR = Path(__file__).parent / "bot_profiles"


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


def run_session():
    player_id = input("Enter participant ID (e.g. P01): ").strip()
    if not player_id:
        print("Error: participant ID required.")
        sys.exit(1)

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

    print(f"\n[Session] {session_name}")
    print(f"[Session] Label: {label}")
    print(f"[Session] Bot session: {is_bot_session(label)}")
    print(f"[Session] SM-native session: {is_sm_native_session(label)}")

    # maxsize=1: live bot control always reads the freshest tick only.
    # Full telemetry history is still saved by telemetry_server regardless.
    # A one-item queue prevents a stale tick backlog from causing aim lag / overshoot.
    bot_queue = queue.Queue(maxsize=1) if is_bot_session(label) else None
    stop_event = threading.Event()

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
    print("\nPlay your session. Press Ctrl+C when done.\n")

    def signal_handler(sig, frame):
        print("\n[Session] Stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    save_events(app)

    print(f"\n[Session] Done. Files saved to sessions/")
    print(f"[Session] Don't forget: run 'stop' in CS:Source console")
    print(f"[Session] Move the .dem file to demos/{session_name}.dem")


if __name__ == "__main__":
    run_session()
