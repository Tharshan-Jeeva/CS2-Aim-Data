"""
raw_mouse_capture.py
────────────────────
Captures raw mouse input via the Windows Raw Input API.
All Win32 function signatures are fully declared so ctypes
never has to guess — this is the only safe approach on 64-bit Windows
where LPARAM/WPARAM/HANDLE are 8 bytes, not 4.
"""

import ctypes
import ctypes.wintypes as wintypes
import time
import csv
import threading
from ctypes import (
    Structure, WINFUNCTYPE,
    c_int, c_uint, c_long, c_ulong,
    c_ushort, c_wchar_p, c_void_p,
    c_ssize_t, c_size_t,
)

# ── Pointer-sized types (critical for 64-bit correctness) ───────────────────
# On 64-bit Windows: LPARAM = LONG_PTR = 8 bytes, WPARAM = UINT_PTR = 8 bytes
# ctypes.wintypes defines them as c_long (4 bytes) — DO NOT use them directly.
LPARAM  = c_ssize_t   # signed 64-bit
WPARAM  = c_size_t    # unsigned 64-bit
LRESULT = c_ssize_t   # signed 64-bit return from WndProc / DefWindowProc

# ── Constants ────────────────────────────────────────────────────────────────
RIM_TYPEMOUSE   = 0
RID_INPUT       = 0x10000003
RIDEV_INPUTSINK = 0x00000100
WM_INPUT        = 0x00FF
PM_REMOVE       = 0x0001

# ── WNDPROC callback type (must use our 64-bit LPARAM / WPARAM) ─────────────
WNDPROCTYPE = WINFUNCTYPE(
    LRESULT,            # return type
    wintypes.HWND,      # hWnd
    c_uint,             # uMsg
    WPARAM,             # wParam
    LPARAM,             # lParam
)

# ── WNDCLASSW (defined manually — not in ctypes.wintypes) ───────────────────
class WNDCLASS(Structure):
    _fields_ = [
        ("style",         c_uint),
        ("lpfnWndProc",   WNDPROCTYPE),
        ("cbClsExtra",    c_int),
        ("cbWndExtra",    c_int),
        ("hInstance",     wintypes.HMODULE),
        ("hIcon",         wintypes.HANDLE),
        ("hCursor",       wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName",  c_wchar_p),
        ("lpszClassName", c_wchar_p),
    ]

# ── Raw input structures ─────────────────────────────────────────────────────
class RAWMOUSE(Structure):
    _fields_ = [
        ("usFlags",            c_ushort),
        ("usButtonFlags",      c_ushort),
        ("usButtonData",       c_ushort),
        ("ulRawButtons",       c_ulong),
        ("lLastX",             c_long),   # raw delta X
        ("lLastY",             c_long),   # raw delta Y
        ("ulExtraInformation", c_ulong),
    ]

class RAWINPUTHEADER(Structure):
    _fields_ = [
        ("dwType",  c_uint),
        ("dwSize",  c_uint),
        ("hDevice", wintypes.HANDLE),
        ("wParam",  WPARAM),
    ]

class RAWINPUT(Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("mouse",  RAWMOUSE),
    ]

class RAWINPUTDEVICE(Structure):
    _fields_ = [
        ("usUsagePage", c_ushort),
        ("usUsage",     c_ushort),
        ("dwFlags",     c_uint),
        ("hwndTarget",  wintypes.HWND),
    ]


# ── Declare ALL Win32 signatures before first use ────────────────────────────
def _declare_win32():
    k32 = ctypes.windll.kernel32
    u32 = ctypes.windll.user32

    # kernel32
    k32.GetModuleHandleW.restype  = wintypes.HMODULE
    k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

    k32.QueryPerformanceFrequency.restype  = wintypes.BOOL
    k32.QueryPerformanceFrequency.argtypes = [
        ctypes.POINTER(wintypes.LARGE_INTEGER)
    ]
    k32.QueryPerformanceCounter.restype  = wintypes.BOOL
    k32.QueryPerformanceCounter.argtypes = [
        ctypes.POINTER(wintypes.LARGE_INTEGER)
    ]

    # user32
    u32.RegisterClassW.restype  = c_ushort
    u32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]

    u32.UnregisterClassW.restype  = wintypes.BOOL
    u32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HMODULE]

    u32.CreateWindowExW.restype  = wintypes.HWND
    u32.CreateWindowExW.argtypes = [
        wintypes.DWORD,    # dwExStyle
        wintypes.LPCWSTR,  # lpClassName
        wintypes.LPCWSTR,  # lpWindowName
        wintypes.DWORD,    # dwStyle
        c_int,             # X
        c_int,             # Y
        c_int,             # nWidth
        c_int,             # nHeight
        wintypes.HWND,     # hWndParent
        wintypes.HANDLE,   # hMenu
        wintypes.HMODULE,  # hInstance
        c_void_p,          # lpParam
    ]

    u32.DestroyWindow.restype  = wintypes.BOOL
    u32.DestroyWindow.argtypes = [wintypes.HWND]

    # DefWindowProcW — lparam MUST be LRESULT-sized (c_ssize_t)
    u32.DefWindowProcW.restype  = LRESULT
    u32.DefWindowProcW.argtypes = [
        wintypes.HWND,
        c_uint,
        WPARAM,
        LPARAM,
    ]

    u32.PeekMessageW.restype  = wintypes.BOOL
    u32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        c_uint,
        c_uint,
        c_uint,
    ]

    u32.TranslateMessage.restype  = wintypes.BOOL
    u32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]

    u32.DispatchMessageW.restype  = LRESULT
    u32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]

    u32.RegisterRawInputDevices.restype  = wintypes.BOOL
    u32.RegisterRawInputDevices.argtypes = [
        ctypes.POINTER(RAWINPUTDEVICE),
        c_uint,
        c_uint,
    ]

    u32.GetRawInputData.restype  = c_uint
    u32.GetRawInputData.argtypes = [
        wintypes.HANDLE,            # hRawInput
        c_uint,                     # uiCommand
        c_void_p,                   # pData
        ctypes.POINTER(c_uint),     # pcbSize
        c_uint,                     # cbSizeHeader
    ]


_declare_win32()


# ── Main capture class ───────────────────────────────────────────────────────
class RawMouseCapture:

    def __init__(self, output_file="raw_mouse_data"):
        self.output_file   = output_file
        self.running       = False
        self.events        = []
        self.start_time    = None
        self._k32          = ctypes.windll.kernel32
        self._u32          = ctypes.windll.user32

        freq = wintypes.LARGE_INTEGER()
        self._k32.QueryPerformanceFrequency(ctypes.byref(freq))
        self._perf_freq = freq.value

    # ── Timestamp ─────────────────────────────────────────────────────────
    def _now_us(self):
        ctr = wintypes.LARGE_INTEGER()
        self._k32.QueryPerformanceCounter(ctypes.byref(ctr))
        return (ctr.value * 1_000_000) // self._perf_freq

    # ── Window procedure ──────────────────────────────────────────────────
    def _make_wnd_proc(self):
        u32 = self._u32

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                self._process_raw_input(lparam)
            return u32.DefWindowProcW(hwnd, msg, wparam, lparam)

        # Wrap in the correctly-typed WINFUNCTYPE — keeps 64-bit values intact
        return WNDPROCTYPE(wnd_proc)

    # ── Raw input processing ──────────────────────────────────────────────
    def _process_raw_input(self, lparam):
        u32      = self._u32
        dwSize   = c_uint(0)
        # lparam IS the HRAWINPUT handle — cast it to HANDLE
        hRawInput = wintypes.HANDLE(lparam)

        u32.GetRawInputData(
            hRawInput, RID_INPUT, None,
            ctypes.byref(dwSize),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if dwSize.value == 0:
            return

        buf = ctypes.create_string_buffer(dwSize.value)
        ret = u32.GetRawInputData(
            hRawInput, RID_INPUT, buf,
            ctypes.byref(dwSize),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if ret == 0:
            return

        raw = RAWINPUT.from_buffer_copy(buf)

        if raw.header.dwType == RIM_TYPEMOUSE:
            ts  = self._now_us()
            dx  = raw.mouse.lLastX
            dy  = raw.mouse.lLastY
            btn = raw.mouse.usButtonFlags

            if dx != 0 or dy != 0 or btn != 0:
                self.events.append({
                    "timestamp_us": ts,
                    "elapsed_us":   ts - self.start_time,
                    "dx":           dx,
                    "dy":           dy,
                    "button_flags": btn,
                })

    # ── Message loop ──────────────────────────────────────────────────────
    def _message_loop(self):
        u32        = self._u32
        CLASS_NAME = f"RawInputCapture_{id(self)}"

        # Must stay alive on self — Python GC will free a local reference
        self._wnd_proc_cb = self._make_wnd_proc()
        hinstance = self._k32.GetModuleHandleW(None)

        wc               = WNDCLASS()
        wc.lpfnWndProc   = self._wnd_proc_cb
        wc.lpszClassName = CLASS_NAME
        wc.hInstance     = hinstance

        atom = u32.RegisterClassW(ctypes.byref(wc))
        if not atom:
            raise RuntimeError(
                f"RegisterClassW failed (error {ctypes.GetLastError()})"
            )

        hwnd = u32.CreateWindowExW(
            0, CLASS_NAME, CLASS_NAME,
            0, 0, 0, 0, 0,
            None, None, hinstance, None,
        )
        if not hwnd:
            u32.UnregisterClassW(CLASS_NAME, hinstance)
            raise RuntimeError(
                f"CreateWindowExW failed (error {ctypes.GetLastError()})"
            )

        rid             = RAWINPUTDEVICE()
        rid.usUsagePage = 0x01
        rid.usUsage     = 0x02
        rid.dwFlags     = RIDEV_INPUTSINK
        rid.hwndTarget  = hwnd

        ok = u32.RegisterRawInputDevices(
            ctypes.byref(rid), 1, ctypes.sizeof(rid)
        )
        if not ok:
            u32.DestroyWindow(hwnd)
            u32.UnregisterClassW(CLASS_NAME, hinstance)
            raise RuntimeError(
                f"RegisterRawInputDevices failed (error {ctypes.GetLastError()})"
            )

        msg = wintypes.MSG()
        while self.running:
            if u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                u32.TranslateMessage(ctypes.byref(msg))
                u32.DispatchMessageW(ctypes.byref(msg))
            else:
                time.sleep(0.0005)

        u32.DestroyWindow(hwnd)
        u32.UnregisterClassW(CLASS_NAME, hinstance)

    # ── Public API ────────────────────────────────────────────────────────
    def start_capture(self, session_label="session"):
        self.session_label = session_label
        self.running       = True
        self.start_time    = self._now_us()
        self.events        = []

        self._thread = threading.Thread(
            target=self._message_loop,
            daemon=True,
            name="RawMouseThread",
        )
        self._thread.start()
        time.sleep(0.3)
        print(f"[RawCapture] Started. Session: {session_label}")

    def stop_capture(self):
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=3.0)
        self._flush_to_csv()

    def _flush_to_csv(self):
        filename = f"{self.output_file}_{self.session_label}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp_us", "elapsed_us", "dx", "dy", "button_flags"
            ])
            writer.writeheader()
            writer.writerows(self.events)
        print(
            f"[RawCapture] Stopped. "
            f"{len(self.events)} events saved to {filename}"
        )