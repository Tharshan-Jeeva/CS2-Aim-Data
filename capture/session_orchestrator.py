import os
import signal
import sys
import threading
import time
import queue
from pathlib import Path

from capture.telemetry_server import create_app, save_events, run_server
from capture.evdev_keyboard_capture import EvdevKeyboardCapture
from capture.bot_aim_generator import BotAimGenerator, load_config

BOT_LABELS = {"bot_raw", "bot_smooth", "bot_humanised_low",
              "bot_humanised_med", "bot_humanised_high"}
PROFILES_DIR = Path(__file__).parent / "bot_profiles"


def build_session_name(player_id: str, label: str) -> str:
    return f"{player_id}_{label}_{int(time.time())}"


def is_bot_session(label: str) -> bool:
    return label in BOT_LABELS


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
          "bot_humanised_med, bot_humanised_high")
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

    bot_queue = queue.Queue() if is_bot_session(label) else None
    stop_event = threading.Event()

    app = create_app(session_name=session_name, bot_queue=bot_queue)
    server_thread = threading.Thread(
        target=run_server, args=(app,), daemon=True)
    server_thread.start()
    time.sleep(0.5)
    print("[Session] Telemetry server running on port 3000")

    kbd = EvdevKeyboardCapture(
        output_file=f"sessions/{session_name}_keyboard",
        cs_focus_only=True)
    kbd.start_capture(session_name)
    print("[Session] Keyboard capture started")

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
        print("          sm_aim_override_active 1")
        print("          sm_override_target <your_userid>")

    print("\n" + "=" * 50)
    print("In CS:Source console, run:")
    print(f"  record {session_name}")
    print(f"  sm_telemetry_target <your_userid>")
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

    kbd.stop_capture()
    save_events(app)

    print(f"\n[Session] Done. Files saved to sessions/")
    print(f"[Session] Don't forget: run 'stop' in CS:Source console")
    print(f"[Session] Move the .dem file to demos/{session_name}.dem")


if __name__ == "__main__":
    run_session()
