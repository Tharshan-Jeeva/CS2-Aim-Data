# session_orchestrator.py
import threading
import time
import os
import keyboard  # pip install keyboard
import subprocess
from gsi_server import run_gsi_server, set_session, save_events
from raw_mouse_capture import RawMouseCapture

def run_session():
    # Generate unique session name
    session_name = f"session_{int(time.time())}"
    player_id = input("Enter player ID (e.g. player_01): ").strip()
    label = input("Enter label (human / bot_smooth / bot_humanised): ").strip()
    full_name = f"{player_id}_{label}_{session_name}"

    os.makedirs("sessions", exist_ok=True)
    os.makedirs("demos", exist_ok=True)

    print(f"\n[Session] Name: {full_name}")
    print("[Session] Setting up GSI server...")

    # Start GSI server in background thread
    set_session(full_name)
    gsi_thread = threading.Thread(target=run_gsi_server, daemon=True)
    gsi_thread.start()
    time.sleep(1)  # Let Flask start
    print("[Session] GSI server running on port 3000")

    # Start raw mouse capture
    capture = RawMouseCapture(f"sessions/{full_name}_mouse")
    capture.start_capture(full_name)
    print("[Session] Raw mouse capture started")

    # Print CS2 commands to run
    print("\n" + "="*50)
    print("NOW IN CS2, run these console commands:")
    print(f'  record {full_name}')
    print("="*50)
    print("\nPlay your session. Press F10 when done.\n")

    # Wait for F10 to end session
    keyboard.wait("f10")

    print("\n[Session] Stopping...")

    # Stop mouse capture
    capture.stop_capture()

    # Save GSI events
    save_events()

    # Remind to stop demo
    print("\n" + "="*50)
    print("IN CS2 CONSOLE, run:")
    print("  stop")
    print(f"Then move the demo file to: demos/{full_name}.dem")
    print("="*50)

    print(f"\n[Session] Complete. Files saved with prefix: {full_name}")
    print("[Session] Run parse/demo_parser.py next to extract tick data.")

if __name__ == "__main__":
    run_session()