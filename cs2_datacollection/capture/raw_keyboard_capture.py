"""
raw_keyboard_capture.py
────────────────────────
Captures low-level keyboard input via the Windows SetWindowsHookExW API
(WH_KEYBOARD_LL). Shares the QueryPerformanceCounter clock with
raw_mouse_capture so keystroke timestamps can be merged against mouse
events in the same microsecond timeline.

Scope
─────
Events are only recorded while CS2 has the foreground window. This is a
deliberate ethical scoping: keystrokes the participant types in other
applications (passwords, messages, browsers) are never recorded.

Notes on Windows specifics
──────────────────────────
• LL hooks must be installed on a thread that pumps messages — we run
  one inside a daemon thread.
• LL hooks have an implicit ~300 ms timeout on Windows; the callback
  must be very cheap. We append to a list and return immediately.
• If CS2 is launched as administrator, this process must also be
  elevated for the hook to see CS2's keystrokes.
• Auto-repeat is not flagged by the OS in LL hook data, so we track
  which VK codes are currently held and drop subsequent down-events
  for keys already marked down.
"""

import ctypes
import ctypes.wintypes as wintypes
import time
import csv
import threading
from ctypes import (
    Structure, WINFUNCTYPE,
    c_int, c_uint,
    c_ssize_t, c_size_t,
)

# ── Pointer-sized types (64-bit correctness) ───────────────────────────
LPARAM  = c_ssize_t
WPARAM  = c_size_t
LRESULT = c_ssize_t

# ── Constants ──────────────────────────────────────────────────────────
WH_KEYBOARD_LL   = 13

WM_KEYDOWN       = 0x0100
WM_KEYUP         = 0x0101
WM_SYSKEYDOWN    = 0x0104
WM_SYSKEYUP      = 0x0105
WM_QUIT          = 0x0012

PM_REMOVE        = 0x0001

LLKHF_EXTENDED   = 0x01
LLKHF_INJECTED   = 0x10
LLKHF_ALTDOWN    = 0x20
LLKHF_UP         = 0x80

# Window title substring used to detect CS2 / CS:GO foreground focus.
CS2_WINDOW_TITLE_SUBSTRING = "Counter-Strike"


# ── KBDLLHOOKSTRUCT ────────────────────────────────────────────────────
class KBDLLHOOKSTRUCT(Structure):
    _fields_ = [
        ("vkCode",      wintypes.DWORD),
        ("scanCode",    wintypes.DWORD),
        ("flags",       wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", c_size_t),
    ]


# ── LowLevelKeyboardProc callback type ────────────────────────────────
LowLevelKeyboardProc = WINFUNCTYPE(LRESULT, c_int, WPARAM, LPARAM)


# ── Declare Win32 signatures ──────────────────────────────────────────
def _declare_win32():
    k32 = ctypes.windll.kernel32
    u32 = ctypes.windll.user32

    k32.GetModuleHandleW.restype  = wintypes.HMODULE
    k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

    k32.GetCurrentThreadId.restype  = wintypes.DWORD
    k32.GetCurrentThreadId.argtypes = []

    k32.QueryPerformanceFrequency.restype  = wintypes.BOOL
    k32.QueryPerformanceFrequency.argtypes = [
        ctypes.POINTER(wintypes.LARGE_INTEGER)
    ]
    k32.QueryPerformanceCounter.restype    = wintypes.BOOL
    k32.QueryPerformanceCounter.argtypes   = [
        ctypes.POINTER(wintypes.LARGE_INTEGER)
    ]

    u32.SetWindowsHookExW.restype  = c_ssize_t
    u32.SetWindowsHookExW.argtypes = [
        c_int,                   # idHook
        LowLevelKeyboardProc,    # lpfn
        wintypes.HMODULE,        # hmod
        wintypes.DWORD,          # dwThreadId (0 = global)
    ]

    u32.UnhookWindowsHookEx.restype  = wintypes.BOOL
    u32.UnhookWindowsHookEx.argtypes = [c_ssize_t]

    u32.CallNextHookEx.restype  = LRESULT
    u32.CallNextHookEx.argtypes = [c_ssize_t, c_int, WPARAM, LPARAM]

    u32.PeekMessageW.restype  = wintypes.BOOL
    u32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND, c_uint, c_uint, c_uint,
    ]
    u32.TranslateMessage.restype  = wintypes.BOOL
    u32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]

    u32.DispatchMessageW.restype  = LRESULT
    u32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]

    u32.PostThreadMessageW.restype  = wintypes.BOOL
    u32.PostThreadMessageW.argtypes = [
        wintypes.DWORD, c_uint, WPARAM, LPARAM,
    ]

    u32.GetForegroundWindow.restype  = wintypes.HWND
    u32.GetForegroundWindow.argtypes = []

    u32.GetWindowTextW.restype  = c_int
    u32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, c_int]


_declare_win32()


# ── Main capture class ────────────────────────────────────────────────
class RawKeyboardCapture:

    def __init__(self, output_file="raw_keyboard_data",
                 cs2_focus_only: bool = True):
        self.output_file     = output_file
        self.cs2_focus_only  = cs2_focus_only
        self.running         = False
        self.events          = []
        self.start_time      = None

        # Keys currently held, used to drop OS-generated auto-repeat downs
        self._keys_down      = set()

        # Updated by the foreground-poll thread. If focus-only mode is off
        # we simply leave this True so every event is recorded.
        self._cs2_focused    = not cs2_focus_only

        self._hook           = None
        self._hook_thread_id = None

        self._k32            = ctypes.windll.kernel32
        self._u32            = ctypes.windll.user32

        freq = wintypes.LARGE_INTEGER()
        self._k32.QueryPerformanceFrequency(ctypes.byref(freq))
        self._perf_freq = freq.value

    # ── Timestamp (matches raw_mouse_capture._now_us) ────────────────
    def _now_us(self):
        ctr = wintypes.LARGE_INTEGER()
        self._k32.QueryPerformanceCounter(ctypes.byref(ctr))
        return (ctr.value * 1_000_000) // self._perf_freq

    # ── Foreground-window poller ─────────────────────────────────────
    def _poll_foreground(self):
        buf = ctypes.create_unicode_buffer(256)
        while self.running:
            if self.cs2_focus_only:
                hwnd = self._u32.GetForegroundWindow()
                if hwnd:
                    self._u32.GetWindowTextW(hwnd, buf, 256)
                    self._cs2_focused = (
                        CS2_WINDOW_TITLE_SUBSTRING in buf.value
                    )
                else:
                    self._cs2_focused = False
            time.sleep(0.1)

    # ── Hook procedure ───────────────────────────────────────────────
    def _make_hook_proc(self):
        u32 = self._u32

        def proc(nCode, wParam, lParam):
            # nCode < 0 → must pass straight through without processing
            if nCode < 0:
                return u32.CallNextHookEx(0, nCode, wParam, lParam)

            if self._cs2_focused:
                kb    = KBDLLHOOKSTRUCT.from_address(lParam)
                vk    = kb.vkCode
                sc    = kb.scanCode
                flags = kb.flags

                is_down = wParam in (WM_KEYDOWN, WM_SYSKEYDOWN)
                is_up   = wParam in (WM_KEYUP,   WM_SYSKEYUP)

                record = False
                if is_down:
                    # Suppress OS auto-repeat: only keep the first down
                    if vk not in self._keys_down:
                        self._keys_down.add(vk)
                        record = True
                elif is_up:
                    self._keys_down.discard(vk)
                    record = True

                # Skip synthetic / injected events (SendInput etc.)
                if flags & LLKHF_INJECTED:
                    record = False

                if record:
                    ts = self._now_us()
                    self.events.append({
                        "timestamp_us": ts,
                        "elapsed_us":   ts - self.start_time,
                        "vk_code":      vk,
                        "scan_code":    sc,
                        "extended":     int(bool(flags & LLKHF_EXTENDED)),
                        "event_type":   "down" if is_down else "up",
                    })

            return u32.CallNextHookEx(0, nCode, wParam, lParam)

        # Keep a typed reference alive on the caller — returned wrapper
        # must not be GC'd while the hook is active.
        return LowLevelKeyboardProc(proc)

    # ── Message loop (hooks require one on the installing thread) ────
    def _hook_loop(self):
        u32 = self._u32
        self._hook_thread_id = self._k32.GetCurrentThreadId()
        self._hook_proc_cb   = self._make_hook_proc()
        hinstance            = self._k32.GetModuleHandleW(None)

        self._hook = u32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._hook_proc_cb, hinstance, 0
        )
        if not self._hook:
            raise RuntimeError(
                f"SetWindowsHookExW failed "
                f"(error {ctypes.GetLastError()})"
            )

        msg = wintypes.MSG()
        while self.running:
            if u32.PeekMessageW(
                ctypes.byref(msg), None, 0, 0, PM_REMOVE
            ):
                if msg.message == WM_QUIT:
                    break
                u32.TranslateMessage(ctypes.byref(msg))
                u32.DispatchMessageW(ctypes.byref(msg))
            else:
                time.sleep(0.0005)

        u32.UnhookWindowsHookEx(self._hook)
        self._hook = None

    # ── Public API ───────────────────────────────────────────────────
    def start_capture(self, session_label="session"):
        self.session_label = session_label
        self.running       = True
        self.start_time    = self._now_us()
        self.events        = []
        self._keys_down    = set()

        self._thread = threading.Thread(
            target=self._hook_loop,
            daemon=True,
            name="RawKeyboardThread",
        )
        self._thread.start()

        self._focus_thread = threading.Thread(
            target=self._poll_foreground,
            daemon=True,
            name="ForegroundPollThread",
        )
        self._focus_thread.start()

        time.sleep(0.3)
        print(
            f"[KbdCapture] Started. Session: {session_label} "
            f"(cs2_focus_only={self.cs2_focus_only})"
        )

    def stop_capture(self):
        self.running = False
        # Wake PeekMessage so the loop notices self.running == False
        if self._hook_thread_id:
            self._u32.PostThreadMessageW(
                self._hook_thread_id, WM_QUIT, 0, 0
            )
        if hasattr(self, "_thread"):
            self._thread.join(timeout=3.0)
        self._flush_to_csv()

    def _flush_to_csv(self):
        filename = f"{self.output_file}_{self.session_label}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp_us", "elapsed_us",
                "vk_code", "scan_code", "extended", "event_type",
            ])
            writer.writeheader()
            writer.writerows(self.events)
        print(
            f"[KbdCapture] Stopped. "
            f"{len(self.events)} events saved to {filename}"
        )
