import csv
import os
import subprocess
import threading
import time

from evdev import InputDevice, ecodes, list_devices

LINUX_TO_VK = {
    ecodes.KEY_W:          0x57,
    ecodes.KEY_A:          0x41,
    ecodes.KEY_S:          0x53,
    ecodes.KEY_D:          0x44,
    ecodes.KEY_SPACE:      0x20,
    ecodes.KEY_LEFTSHIFT:  0xA0,
    ecodes.KEY_RIGHTSHIFT: 0xA1,
    ecodes.KEY_LEFTCTRL:   0xA2,
    ecodes.KEY_RIGHTCTRL:  0xA3,
    ecodes.KEY_E:          0x45,
    ecodes.KEY_R:          0x52,
    ecodes.KEY_1:          0x31,
    ecodes.KEY_2:          0x32,
    ecodes.KEY_3:          0x33,
    ecodes.KEY_4:          0x34,
    ecodes.KEY_5:          0x35,
    ecodes.KEY_TAB:        0x09,
    ecodes.KEY_ESC:        0x1B,
}

CS_WINDOW_SUBSTRING = "Counter-Strike"


class EvdevKeyboardCapture:

    def __init__(self, output_file: str, device_path: str = None,
                 cs_focus_only: bool = True):
        self.output_file = output_file
        self.device_path = device_path
        self.cs_focus_only = cs_focus_only
        self.running = False
        self.events = []
        self.start_time_ns = 0
        self._cs_focused = not cs_focus_only

    def _find_keyboard_device(self) -> str:
        if self.device_path:
            return self.device_path
        devices = [InputDevice(path) for path in list_devices()]
        for dev in devices:
            caps = dev.capabilities(verbose=False)
            if ecodes.EV_KEY in caps:
                keys = caps[ecodes.EV_KEY]
                if ecodes.KEY_W in keys and ecodes.KEY_A in keys:
                    return dev.path
        raise RuntimeError("No keyboard device found. Ensure user is in 'input' group.")

    def _poll_focus(self):
        while self.running:
            try:
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=1)
                self._cs_focused = CS_WINDOW_SUBSTRING in result.stdout
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                self._cs_focused = False
            time.sleep(0.1)

    def _process_event(self, code, scan_code, value, timestamp_ns):
        if not self._cs_focused:
            return
        if value == 2:
            return
        if code not in LINUX_TO_VK:
            return

        elapsed_us = (timestamp_ns - self.start_time_ns) // 1000
        self.events.append({
            "timestamp_us": timestamp_ns // 1000,
            "elapsed_us": elapsed_us,
            "vk_code": LINUX_TO_VK[code],
            "scan_code": scan_code,
            "extended": 0,
            "event_type": "down" if value == 1 else "up",
        })

    def _read_loop(self):
        dev = InputDevice(self._find_keyboard_device())
        for event in dev.read_loop():
            if not self.running:
                break
            if event.type == ecodes.EV_KEY:
                ts_ns = event.sec * 1_000_000_000 + event.usec * 1000
                self._process_event(event.code, event.code, event.value, ts_ns)

    def start_capture(self, session_label: str = "session"):
        self.session_label = session_label
        self.running = True
        self.events = []
        self.start_time_ns = time.time_ns()

        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

        if self.cs_focus_only:
            self._focus_thread = threading.Thread(target=self._poll_focus, daemon=True)
            self._focus_thread.start()

        print(f"[KbdCapture] Started (cs_focus_only={self.cs_focus_only})")

    def stop_capture(self):
        self.running = False
        time.sleep(0.2)
        self._flush_csv()

    def _flush_csv(self):
        filename = f"{self.output_file}_{self.session_label}.csv"
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp_us", "elapsed_us",
                "vk_code", "scan_code", "extended", "event_type",
            ])
            writer.writeheader()
            writer.writerows(self.events)
        print(f"[KbdCapture] Stopped. {len(self.events)} events -> {filename}")
