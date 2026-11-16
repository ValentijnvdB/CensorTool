"""
The window system for Windows 10/11 (see base.py), with ctypes bindings to user32, gdi32, dwmapi and kernel32.

NOT TESTED ON WINDOWS YET: written on Linux, only the platform-independent parts are covered by tests.

- Capturing: Windows Graphics Capture through the optional 'windows-capture' package (fast, captures covered
  windows); without it, or when it fails, PrintWindow (slower; some GPU-rendered programs come out black).
- Following windows and noticing new ones: WinEvent hooks (out of context, delivered through the message queue
  of the thread that made them, which therefore pumps messages while it waits), with polling as a safety net.
- Stacking: the overlay is made an owned window of its target (GWLP_HWNDPARENT). Windows keeps owned windows
  just above their owner, also when the owner is activated, minimised or moved to another virtual desktop.
- Picking: low-level mouse and keyboard hooks catch the click (and swallow it) or Escape / right click.
- Programs are identified by the name of their executable without extension (vlc.exe -> 'vlc').

Window areas are the window's visible frame (DWMWA_EXTENDED_FRAME_BOUNDS), without the invisible resize borders.
All coordinates are physical pixels: Qt 6 makes the process per-monitor DPI aware, and overlay mode turns off Qt's
own scaling.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes as wt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger

from .base import (Area, Display, FocusTracker, Grabber, TargetState, Tracker, WindowChanges, WindowInfo,
                   WindowMonitor, WindowSystem, WindowSystemError, window_at)

# user32 / dwmapi constants
GW_OWNER = 4
GA_ROOT = 2
GWL_EXSTYLE = -20
GWLP_HWNDPARENT = -8
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
DWM_CLOAKED_SHELL = 0x2  # on another virtual desktop: still an application window, but not shown
PW_RENDERFULLCONTENT = 0x2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
OBJID_WINDOW = 0
WINEVENT_OUTOFCONTEXT = 0x0
WINEVENT_SKIPOWNPROCESS = 0x2
EVENT_SYSTEM_MINIMIZESTART, EVENT_SYSTEM_MINIMIZEEND = 0x0016, 0x0017
EVENT_OBJECT_CREATE, EVENT_OBJECT_DESTROY, EVENT_OBJECT_SHOW, EVENT_OBJECT_HIDE = 0x8000, 0x8001, 0x8002, 0x8003
EVENT_OBJECT_LOCATIONCHANGE, EVENT_OBJECT_NAMECHANGE = 0x800B, 0x800C
EVENT_OBJECT_CLOAKED, EVENT_OBJECT_UNCLOAKED = 0x8017, 0x8018
WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
WM_LBUTTONDOWN, WM_LBUTTONUP, WM_RBUTTONDOWN, WM_RBUTTONUP = 0x0201, 0x0202, 0x0204, 0x0205
VK_ESCAPE = 0x1B
PM_REMOVE = 0x1
QS_ALLINPUT = 0x04FF
BI_RGB, DIB_RGB_COLORS = 0, 0

# window classes of the taskbars and the desktop: never censored automatically
PANEL_CLASSES = {'Shell_TrayWnd', 'Shell_SecondaryTrayWnd', 'Progman', 'WorkerW'}

LRESULT = ctypes.c_ssize_t


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [('pt', wt.POINT), ('mouseData', wt.DWORD), ('flags', wt.DWORD), ('time', wt.DWORD),
                ('dwExtraInfo', ctypes.c_size_t)]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [('vkCode', wt.DWORD), ('scanCode', wt.DWORD), ('flags', wt.DWORD), ('time', wt.DWORD),
                ('dwExtraInfo', ctypes.c_size_t)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [('biSize', wt.DWORD), ('biWidth', wt.LONG), ('biHeight', wt.LONG), ('biPlanes', wt.WORD),
                ('biBitCount', wt.WORD), ('biCompression', wt.DWORD), ('biSizeImage', wt.DWORD),
                ('biXPelsPerMeter', wt.LONG), ('biYPelsPerMeter', wt.LONG), ('biClrUsed', wt.DWORD),
                ('biClrImportant', wt.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', wt.DWORD * 3)]


class WinError(WindowSystemError):
    pass


class WgcUnavailable(WinError):
    """Windows Graphics Capture can't be used at all (the package is missing or too old)."""


def session_problem() -> str | None:
    if sys.platform != 'win32':
        return "The Windows window system only works on Windows."
    if sys.getwindowsversion().build < 10240:
        return "Overlay mode needs Windows 10 or newer."
    return None


# ------------------------------------------------------------------ library loading

_api_lock = threading.Lock()
_api: SimpleNamespace | None = None


def api() -> SimpleNamespace:
    """The Windows libraries with the functions used declared (once). Only works on Windows."""
    global _api
    with _api_lock:
        if _api is not None:
            return _api
        if sys.platform != 'win32':
            raise WinError("The Windows API is only available on Windows.")
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
        dwmapi = ctypes.WinDLL('dwmapi', use_last_error=True)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        P = ctypes.POINTER

        def declare(lib, name, restype, *argtypes):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = list(argtypes)
            return fn

        WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        WINEVENTPROC = ctypes.WINFUNCTYPE(None, wt.HANDLE, wt.DWORD, wt.HWND, wt.LONG, wt.LONG, wt.DWORD, wt.DWORD)
        HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wt.WPARAM, wt.LPARAM)

        declare(user32, 'EnumWindows', wt.BOOL, WNDENUMPROC, wt.LPARAM)
        declare(user32, 'IsWindow', wt.BOOL, wt.HWND)
        declare(user32, 'IsWindowVisible', wt.BOOL, wt.HWND)
        declare(user32, 'IsIconic', wt.BOOL, wt.HWND)
        declare(user32, 'GetWindowTextLengthW', ctypes.c_int, wt.HWND)
        declare(user32, 'GetWindowTextW', ctypes.c_int, wt.HWND, wt.LPWSTR, ctypes.c_int)
        declare(user32, 'GetClassNameW', ctypes.c_int, wt.HWND, wt.LPWSTR, ctypes.c_int)
        declare(user32, 'GetWindowThreadProcessId', wt.DWORD, wt.HWND, P(wt.DWORD))
        declare(user32, 'GetWindow', wt.HWND, wt.HWND, wt.UINT)
        declare(user32, 'GetAncestor', wt.HWND, wt.HWND, wt.UINT)
        get_long = getattr(user32, 'GetWindowLongPtrW', None) or user32.GetWindowLongW  # 32-bit Python
        set_long = getattr(user32, 'SetWindowLongPtrW', None) or user32.SetWindowLongW
        get_long.restype, get_long.argtypes = ctypes.c_ssize_t, [wt.HWND, ctypes.c_int]
        set_long.restype, set_long.argtypes = ctypes.c_ssize_t, [wt.HWND, ctypes.c_int, ctypes.c_ssize_t]
        declare(user32, 'GetWindowRect', wt.BOOL, wt.HWND, P(wt.RECT))
        declare(user32, 'GetForegroundWindow', wt.HWND)
        declare(user32, 'PrintWindow', wt.BOOL, wt.HWND, wt.HDC, wt.UINT)
        declare(user32, 'GetDC', wt.HDC, wt.HWND)
        declare(user32, 'ReleaseDC', ctypes.c_int, wt.HWND, wt.HDC)
        declare(user32, 'SetWinEventHook', wt.HANDLE, wt.DWORD, wt.DWORD, wt.HMODULE, WINEVENTPROC, wt.DWORD,
                wt.DWORD, wt.DWORD)
        declare(user32, 'UnhookWinEvent', wt.BOOL, wt.HANDLE)
        declare(user32, 'SetWindowsHookExW', wt.HHOOK, ctypes.c_int, HOOKPROC, wt.HINSTANCE, wt.DWORD)
        declare(user32, 'UnhookWindowsHookEx', wt.BOOL, wt.HHOOK)
        declare(user32, 'CallNextHookEx', LRESULT, wt.HHOOK, ctypes.c_int, wt.WPARAM, wt.LPARAM)
        declare(user32, 'PeekMessageW', wt.BOOL, P(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT)
        declare(user32, 'TranslateMessage', wt.BOOL, P(wt.MSG))
        declare(user32, 'DispatchMessageW', LRESULT, P(wt.MSG))
        declare(user32, 'MsgWaitForMultipleObjects', wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.BOOL, wt.DWORD,
                wt.DWORD)
        declare(gdi32, 'CreateCompatibleDC', wt.HDC, wt.HDC)
        declare(gdi32, 'DeleteDC', wt.BOOL, wt.HDC)
        declare(gdi32, 'CreateDIBSection', wt.HBITMAP, wt.HDC, P(BITMAPINFO), wt.UINT, P(ctypes.c_void_p),
                wt.HANDLE, wt.DWORD)
        declare(gdi32, 'SelectObject', wt.HGDIOBJ, wt.HDC, wt.HGDIOBJ)
        declare(gdi32, 'DeleteObject', wt.BOOL, wt.HGDIOBJ)
        declare(dwmapi, 'DwmGetWindowAttribute', ctypes.c_long, wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD)
        declare(kernel32, 'OpenProcess', wt.HANDLE, wt.DWORD, wt.BOOL, wt.DWORD)
        declare(kernel32, 'CloseHandle', wt.BOOL, wt.HANDLE)
        declare(kernel32, 'QueryFullProcessImageNameW', wt.BOOL, wt.HANDLE, wt.DWORD, wt.LPWSTR, P(wt.DWORD))
        declare(kernel32, 'GetModuleHandleW', wt.HMODULE, wt.LPCWSTR)

        _api = SimpleNamespace(user32=user32, gdi32=gdi32, dwmapi=dwmapi, kernel32=kernel32,
                               get_window_long=get_long, set_window_long=set_long,
                               WNDENUMPROC=WNDENUMPROC, WINEVENTPROC=WINEVENTPROC, HOOKPROC=HOOKPROC)
        return _api


# ------------------------------------------------------------------ helpers

def _int(handle) -> int:
    """A handle as an int (ctypes returns None for NULL)."""
    return int(handle or 0)


def is_app_window_style(exstyle: int, owned: bool, cloaked: int, width: int, height: int) -> bool:
    """
    Whether a visible top-level window is an application window, from its extended style, whether it has an
    owner, its DWMWA_CLOAKED value and its size. Tool windows (palettes, our overlays) are not, unless they ask for a
    taskbar button; windows cloaked by their app (hidden UWP frames) are not; windows on another virtual desktop
    (cloaked by the shell) are. Owned windows (dialogs) are.
    """
    if exstyle & WS_EX_TOOLWINDOW and not exstyle & WS_EX_APPWINDOW:
        return False
    if cloaked and not cloaked & DWM_CLOAKED_SHELL:
        return False
    return width > 1 and height > 1


def exe_stem(path: str) -> str:
    """The program name of an executable path: C:\\Program Files\\VideoLAN\\VLC\\vlc.exe -> 'vlc'."""
    return Path(path.replace('\\', '/')).stem.lower()


def _cloaked(a, hwnd: int) -> int:
    value = wt.DWORD()
    if a.dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value)) != 0:
        return 0
    return value.value


def _window_rect(a, hwnd: int) -> wt.RECT | None:
    rect = wt.RECT()
    return rect if a.user32.GetWindowRect(hwnd, ctypes.byref(rect)) else None


def _frame_rect(a, hwnd: int) -> wt.RECT | None:
    """The visible frame, without the invisible resize borders (falls back to the window rectangle)."""
    rect = wt.RECT()
    if a.dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect),
                                      ctypes.sizeof(rect)) == 0:
        return rect
    return _window_rect(a, hwnd)


def _area(rect: wt.RECT) -> Area:
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def _is_shown(a, hwnd: int) -> bool:
    """Visible on the screen now: not hidden, not minimised, not on another virtual desktop."""
    return bool(a.user32.IsWindowVisible(hwnd)) and not a.user32.IsIconic(hwnd) and not _cloaked(a, hwnd)


def _is_app_window(a, hwnd: int) -> bool:
    u = a.user32
    if not u.IsWindow(hwnd) or not u.IsWindowVisible(hwnd) or _int(u.GetAncestor(hwnd, GA_ROOT)) != hwnd:
        return False
    rect = _window_rect(a, hwnd)
    if rect is None:
        return False
    return is_app_window_style(a.get_window_long(hwnd, GWL_EXSTYLE), bool(u.GetWindow(hwnd, GW_OWNER)),
                               _cloaked(a, hwnd), rect.right - rect.left, rect.bottom - rect.top)


def _top_level_windows(a) -> list[int]:
    """All top-level windows, top to bottom."""
    windows = []

    @a.WNDENUMPROC
    def collect(hwnd, _lparam):
        windows.append(_int(hwnd))
        return True

    a.user32.EnumWindows(collect, 0)
    return windows


def _pump_messages(a, timeout: float):
    """Wait at most timeout seconds for messages to this thread and dispatch them (WinEvent and hook callbacks)."""
    a.user32.MsgWaitForMultipleObjects(0, None, False, max(0, int(timeout * 1000)), QS_ALLINPUT)
    msg = wt.MSG()
    while a.user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
        a.user32.TranslateMessage(ctypes.byref(msg))
        a.user32.DispatchMessageW(ctypes.byref(msg))


def _process_name(a, pid: int) -> str | None:
    handle = a.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None  # e.g. an elevated process
    try:
        size = wt.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not a.kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return exe_stem(buffer.value)
    finally:
        a.kernel32.CloseHandle(handle)


# ------------------------------------------------------------------ queries

class WinDisplay(Display):

    def __init__(self):
        self.a = api()

    def close(self):
        pass

    def window_list(self, stacking: bool = False) -> list[int]:
        # Windows doesn't tell the creation order; both orders are bottom to top
        return [w for w in reversed(_top_level_windows(self.a)) if _is_app_window(self.a, w)]

    def window_info(self, win: int) -> WindowInfo:
        u = self.a.user32
        pid = wt.DWORD()
        u.GetWindowThreadProcessId(win, ctypes.byref(pid))
        name = _process_name(self.a, pid.value) if pid.value else None
        class_name = ctypes.create_unicode_buffer(256)
        u.GetClassNameW(win, class_name, 256)
        return WindowInfo(window=win, pid=pid.value or None, app_ids=(name,) if name else (),
                          normal=not u.GetWindow(win, GW_OWNER), panel=class_name.value in PANEL_CLASSES)

    def title(self, win: int) -> str:
        u = self.a.user32
        length = u.GetWindowTextLengthW(win)
        if length <= 0:
            return ''
        buffer = ctypes.create_unicode_buffer(length + 1)
        u.GetWindowTextW(win, buffer, length + 1)
        return buffer.value

    def is_app_window(self, win: int) -> bool:
        return _is_app_window(self.a, win)

    def visible_area(self, win: int) -> Area | None:
        if not self.a.user32.IsWindow(win) or not _is_shown(self.a, win):
            return None
        rect = _frame_rect(self.a, win)
        return _area(rect) if rect is not None else None

    def application_window(self, win: int) -> int:
        if not self.a.user32.IsWindow(win):
            raise ValueError(f"There is no window with id {win:#x}.")
        return _int(self.a.user32.GetAncestor(win, GA_ROOT)) or win

    def active_window(self) -> int | None:
        hwnd = _int(self.a.user32.GetForegroundWindow())
        if hwnd:
            hwnd = _int(self.a.user32.GetAncestor(hwnd, GA_ROOT)) or hwnd
        return hwnd or None


class WinMonitor(WindowMonitor):
    """
    WinEvent hooks for windows being created, shown, hidden, destroyed, cloaked or (with follow_titles) renamed.
    The events arrive through this thread's message queue, which wait() pumps.
    """

    def __init__(self, follow_titles: bool = False):
        self.a = api()
        self.display = WinDisplay()
        self._changes = WindowChanges()
        self._top = set(_top_level_windows(self.a))  # top-level windows seen, to tell which destroyed ones count
        self._callback = self.a.WINEVENTPROC(self._on_event)  # kept: the hooks call it
        ranges = [(EVENT_OBJECT_CREATE, EVENT_OBJECT_HIDE), (EVENT_OBJECT_CLOAKED, EVENT_OBJECT_UNCLOAKED)]
        if follow_titles:
            ranges.append((EVENT_OBJECT_NAMECHANGE, EVENT_OBJECT_NAMECHANGE))
        flags = WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS
        self._hooks = [self.a.user32.SetWinEventHook(lo, hi, None, self._callback, 0, 0, flags) for lo, hi in ranges]
        if not all(self._hooks):
            logger.warning("Could not follow all window events; new windows are noticed by polling.")

    def _on_event(self, _hook, event, hwnd, id_object, id_child, _thread, _time):
        try:
            hwnd = _int(hwnd)
            if not hwnd or id_object != OBJID_WINDOW or id_child != 0:
                return
            if event == EVENT_OBJECT_DESTROY:
                if hwnd in self._top:
                    self._top.discard(hwnd)
                    self._changes.list_changed = True
                    self._changes.windows.add(hwnd)
                return
            if _int(self.a.user32.GetAncestor(hwnd, GA_ROOT)) != hwnd:
                return  # a child window (a control)
            self._top.add(hwnd)
            self._changes.windows.add(hwnd)
            if event != EVENT_OBJECT_NAMECHANGE:
                self._changes.list_changed = True
        except Exception as e:  # never let an exception escape into Windows
            logger.debug(f"Window event failed: {e}")

    def wait(self, timeout: float) -> WindowChanges:
        has_changes = self._changes.list_changed or self._changes.windows
        _pump_messages(self.a, 0 if has_changes else timeout)
        changes, self._changes = self._changes, WindowChanges()
        return changes

    def close(self):
        for hook in self._hooks:
            if hook:
                self.a.user32.UnhookWinEvent(hook)
        self._hooks = []


# ------------------------------------------------------------------ capture

class PrintWindowGrabber(Grabber):
    """Captures a window with PrintWindow (PW_RENDERFULLCONTENT) into a DIB section. Use from one thread only."""

    def __init__(self, hwnd: int):
        self.a = api()
        self.window = hwnd
        self.area = (0, 0, 0, 0)
        self.visible = False
        self.failures = 0
        self._dc = None
        self._bitmap = None
        self._old = None
        self._bits = ctypes.c_void_p()
        self._size = (0, 0)  # of the bitmap: the whole window rectangle
        self._offset = (0, 0)  # of the visible frame inside it

    def _free(self):
        g = self.a.gdi32
        if self._dc:
            if self._old:
                g.SelectObject(self._dc, self._old)
            g.DeleteDC(self._dc)
        if self._bitmap:
            g.DeleteObject(self._bitmap)
        self._dc = self._bitmap = self._old = None
        self._size = (0, 0)

    def _make(self, width: int, height: int):
        self._free()
        screen = self.a.user32.GetDC(None)
        self._dc = self.a.gdi32.CreateCompatibleDC(screen)
        self.a.user32.ReleaseDC(None, screen)
        info = BITMAPINFO()
        h = info.bmiHeader
        h.biSize, h.biWidth, h.biHeight = ctypes.sizeof(BITMAPINFOHEADER), width, -height  # negative: top-down
        h.biPlanes, h.biBitCount, h.biCompression = 1, 32, BI_RGB
        self._bitmap = self.a.gdi32.CreateDIBSection(self._dc, ctypes.byref(info), DIB_RGB_COLORS,
                                                     ctypes.byref(self._bits), None, 0)
        if not self._bitmap or not self._bits.value:
            self._free()
            raise WinError("Could not create a bitmap to capture the window.")
        self._old = self.a.gdi32.SelectObject(self._dc, self._bitmap)
        self._size = (width, height)

    def refresh(self) -> bool:
        if not self.a.user32.IsWindow(self.window):
            self.visible = False
            return False
        self.visible = _is_shown(self.a, self.window)
        if not self.visible:
            return True
        outer, frame = _window_rect(self.a, self.window), _frame_rect(self.a, self.window)
        if outer is None or frame is None:
            self.visible = False
            return True
        self.area = _area(frame)
        self._offset = (frame.left - outer.left, frame.top - outer.top)
        size = (outer.right - outer.left, outer.bottom - outer.top)
        if size != self._size and size[0] > 0 and size[1] > 0:
            self._make(*size)
        return True

    def grab(self) -> np.ndarray | None:
        if not self.visible or not self._dc:
            return None
        if not self.a.user32.PrintWindow(self.window, self._dc, PW_RENDERFULLCONTENT):
            self.failures += 1
            return None
        self.failures = 0
        width, height = self._size
        buffer = (ctypes.c_ubyte * (width * height * 4)).from_address(self._bits.value)
        image = np.frombuffer(buffer, np.uint8).reshape(height, width, 4)
        return crop(image, self._offset, self.area[2:])

    def close(self):
        self._free()


def crop(image: np.ndarray, offset: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    """A copy of the part of image at offset (x, y) with size (w, h), clipped to the image."""
    x, y = max(0, offset[0]), max(0, offset[1])
    w, h = size
    return image[y:y + h, x:x + w].copy()


class WgcGrabber(Grabber):
    """
    Captures a window with Windows Graphics Capture (the 'windows-capture' package). Frames arrive in the package's
    own thread whenever the window changes; grab returns the newest one.

    :raises WindowSystemError: when the package is missing or too old, or capturing can't start.
    """

    FIRST_FRAME_TIMEOUT = 1.0

    def __init__(self, hwnd: int):
        try:
            from windows_capture import WindowsCapture
        except ImportError:
            raise WgcUnavailable("The 'windows-capture' package is not installed.") from None
        self.a = api()
        self.window = hwnd
        self.area = (0, 0, 0, 0)
        self.visible = False
        self.failures = 0
        self.ended = False  # the capture stopped (e.g. the window was closed)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._arrived = threading.Event()
        self._closing = False
        self._control = None

        problem = None
        for draw_border in (False, None):  # hiding the yellow border needs Windows 11; else keep the default
            try:
                self._control = self._start(WindowsCapture, hwnd, draw_border)
                return
            except TypeError:
                raise WgcUnavailable("The 'windows-capture' package is too old (no window_hwnd); "
                               "pip install -U windows-capture") from None
            except Exception as e:
                problem = e
        raise WinError(f"Windows Graphics Capture can't capture window {hwnd:#x}: {problem}")

    def _start(self, capture_class, hwnd: int, draw_border: bool | None):
        capture = capture_class(cursor_capture=False, draw_border=draw_border, window_hwnd=hwnd)

        @capture.event
        def on_frame_arrived(frame, capture_control):
            if self._closing:
                capture_control.stop()
                return
            data = np.array(frame.frame_buffer, copy=True)  # the buffer is only valid during this call
            with self._lock:
                self._frame = data
            self._arrived.set()

        @capture.event
        def on_closed():
            self.ended = True
            self._arrived.set()

        return capture.start_free_threaded()

    def refresh(self) -> bool:
        if not self.a.user32.IsWindow(self.window):
            self.visible = False
            return False
        self.visible = _is_shown(self.a, self.window)
        frame = _frame_rect(self.a, self.window)
        if frame is not None:
            self.area = _area(frame)
        return True

    def grab(self) -> np.ndarray | None:
        if not self.visible:
            return None
        if self._frame is None:
            self._arrived.wait(self.FIRST_FRAME_TIMEOUT)
        with self._lock:
            frame = self._frame
        if frame is None or self.ended:
            self.failures += 1
            return None
        self.failures = 0
        if frame.shape[2] == 3:
            frame = np.dstack([frame, np.full(frame.shape[:2], 255, np.uint8)])
        return crop(frame, (0, 0), self.area[2:])  # the frame is the visible frame; crop in case it is larger

    def close(self):
        self._closing = True
        if self._control is not None:
            try:
                self._control.stop()
            except Exception:
                pass


class WindowsGrabber(Grabber):
    """Windows Graphics Capture when it works, else PrintWindow (also when it keeps failing later on)."""

    MAX_WGC_FAILURES = 10
    _wgc_unavailable: str | None = None  # why WGC can't be used in this process, once known

    def __init__(self, hwnd: int):
        self.window = hwnd
        self._inner: Grabber | None = None
        if WindowsGrabber._wgc_unavailable is None:
            try:
                self._inner = WgcGrabber(hwnd)
            except WgcUnavailable as e:
                WindowsGrabber._wgc_unavailable = str(e)
                logger.warning(f"Capturing windows with PrintWindow (slower; some programs come out black): {e}")
            except WindowSystemError as e:
                logger.warning(f"Capturing window {hwnd:#x} with PrintWindow: {e}")
        if self._inner is None:
            self._inner = PrintWindowGrabber(hwnd)

    @property
    def area(self):
        return self._inner.area

    @property
    def visible(self):
        return self._inner.visible

    @property
    def failures(self):
        return self._inner.failures

    def refresh(self) -> bool:
        return self._inner.refresh()

    def grab(self) -> np.ndarray | None:
        frame = self._inner.grab()
        if frame is None and isinstance(self._inner, WgcGrabber) and self._inner.failures >= self.MAX_WGC_FAILURES:
            logger.warning(f"Windows Graphics Capture keeps failing for window {self.window:#x}, "
                           f"using PrintWindow for it.")
            self._inner.close()
            self._inner = PrintWindowGrabber(self.window)
            self._inner.refresh()
        return frame

    def close(self):
        self._inner.close()


# ------------------------------------------------------------------ following windows

class WinTracker(threading.Thread, Tracker):
    """Follows a window with WinEvent hooks on its process (moves, resizes, minimising, cloaking), plus polling."""

    POLL_INTERVAL = 0.25

    def __init__(self, hwnd: int, on_change=None):
        super().__init__(name=f'win-tracker-{hwnd:#x}', daemon=True)
        self.window = hwnd
        self.state = TargetState()
        self._on_change = on_change
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._changed = False

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    def wait_ready(self, timeout: float = 2):
        self._ready.wait(timeout)

    def run(self):
        a = api()
        hooks = []
        try:
            pid = wt.DWORD()
            a.user32.GetWindowThreadProcessId(self.window, ctypes.byref(pid))

            def on_event(_hook, _event, hwnd, id_object, _id_child, _thread, _time):
                if _int(hwnd) == self.window and id_object == OBJID_WINDOW:
                    self._changed = True

            callback = a.WINEVENTPROC(on_event)
            for lo, hi in ((EVENT_SYSTEM_MINIMIZESTART, EVENT_SYSTEM_MINIMIZEEND),
                           (EVENT_OBJECT_DESTROY, EVENT_OBJECT_LOCATIONCHANGE),
                           (EVENT_OBJECT_CLOAKED, EVENT_OBJECT_UNCLOAKED)):
                hooks.append(a.user32.SetWinEventHook(lo, hi, None, callback, pid.value, 0, WINEVENT_OUTOFCONTEXT))
            self._update(a)
            self._ready.set()
            last = time.monotonic()
            while not self._stop_event.is_set() and self.state.alive:
                _pump_messages(a, 0.05)
                now = time.monotonic()
                if self._changed or now - last > self.POLL_INTERVAL:
                    self._changed = False
                    self._update(a)
                    last = now
        except Exception as e:
            logger.error(f"Following window {self.window:#x} failed: {e}")
            self.state = TargetState(version=self.state.version + 1, alive=False)
        finally:
            for hook in hooks:
                if hook:
                    a.user32.UnhookWinEvent(hook)
            self._ready.set()

    def _update(self, a):
        if not a.user32.IsWindow(self.window):
            new = TargetState(alive=False)
        elif not _is_shown(a, self.window):
            new = TargetState(alive=True, visible=False, area=self.state.area)
        else:
            rect = _frame_rect(a, self.window)
            new = TargetState(alive=True, visible=rect is not None, area=_area(rect) if rect else self.state.area)
        old = self.state
        if (new.alive, new.visible, new.area) != (old.alive, old.visible, old.area):
            self.state = TargetState(version=old.version + 1, alive=new.alive, visible=new.visible, area=new.area)
            if self._on_change is not None:
                self._on_change()


class WinFocusTracker(threading.Thread, FocusTracker):
    """Follows the foreground window by polling (cheap: one call every 0.1 s)."""

    def __init__(self):
        super().__init__(name='win-focus', daemon=True)
        self.active: int | None = None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    def run(self):
        try:
            a = api()
            while not self._stop_event.is_set():
                hwnd = _int(a.user32.GetForegroundWindow())
                if hwnd:
                    hwnd = _int(a.user32.GetAncestor(hwnd, GA_ROOT)) or hwnd
                self.active = hwnd or None
                self._stop_event.wait(0.1)
        except Exception as e:
            logger.error(f"Following the focused window failed: {e}")


# ------------------------------------------------------------------ picking a window

def pick_window(exclude: set[int] = frozenset(), cancel: threading.Event | None = None) -> int | None:
    """Let the user click a window; the click is swallowed. Right click or Escape cancels. See WindowSystem."""
    a = api()
    u = a.user32
    state = {'click': None, 'cancelled': False, 'swallow_up': None, 'done': False}

    def on_mouse(code, wparam, lparam):
        if code == 0 and not state['done']:
            if state['swallow_up'] is not None:
                if wparam == state['swallow_up']:
                    state['done'] = True
                return 1
            if wparam == WM_LBUTTONDOWN:
                info = ctypes.cast(ctypes.c_void_p(lparam), ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                state['click'] = (info.pt.x, info.pt.y)
                state['swallow_up'] = WM_LBUTTONUP
                return 1
            if wparam == WM_RBUTTONDOWN:
                state['cancelled'] = True
                state['swallow_up'] = WM_RBUTTONUP
                return 1
        return u.CallNextHookEx(None, code, wparam, lparam)

    def on_key(code, wparam, lparam):
        if code == 0 and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
            info = ctypes.cast(ctypes.c_void_p(lparam), ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if info.vkCode == VK_ESCAPE:
                state['cancelled'] = state['done'] = True
                return 1
        return u.CallNextHookEx(None, code, wparam, lparam)

    mouse_proc, key_proc = a.HOOKPROC(on_mouse), a.HOOKPROC(on_key)  # kept alive while the hooks exist
    module = a.kernel32.GetModuleHandleW(None)
    mouse_hook = u.SetWindowsHookExW(WH_MOUSE_LL, mouse_proc, module, 0)
    if not mouse_hook:
        raise WinError("Could not catch the mouse to select a window.")
    key_hook = u.SetWindowsHookExW(WH_KEYBOARD_LL, key_proc, module, 0)
    try:
        clicked_at = None
        while not state['done']:
            if cancel is not None and cancel.is_set():
                return None
            _pump_messages(a, 0.05)
            if state['click'] is not None and clicked_at is None:
                clicked_at = time.monotonic()
            if clicked_at is not None and time.monotonic() - clicked_at > 1:
                break  # the release never came
    finally:
        u.UnhookWindowsHookEx(mouse_hook)
        if key_hook:
            u.UnhookWindowsHookEx(key_hook)

    if state['cancelled'] or state['click'] is None:
        return None
    display = WinDisplay()
    stack = [(w, area) for w in display.window_list() if (area := display.visible_area(w)) is not None]
    return window_at(stack, *state['click'], exclude=set(exclude))


# ------------------------------------------------------------------ the window system

class WindowsWindowSystem(WindowSystem):
    name = 'windows'

    def open_display(self) -> WinDisplay:
        return WinDisplay()

    def monitor(self, follow_titles: bool = False) -> WinMonitor:
        return WinMonitor(follow_titles)

    def grabber(self, win: int) -> WindowsGrabber:
        return WindowsGrabber(win)

    def tracker(self, win: int, on_change=None) -> WinTracker:
        tracker = WinTracker(win, on_change=on_change)
        tracker.start()
        tracker.wait_ready()
        return tracker

    def focus_tracker(self) -> WinFocusTracker:
        tracker = WinFocusTracker()
        tracker.start()
        return tracker

    def pick_window(self, exclude: set[int] = frozenset(), cancel: threading.Event | None = None) -> int | None:
        return pick_window(exclude, cancel)

    def place_above(self, overlay, win: int) -> object:
        # an owned window: Windows keeps it above its owner (and hides it with it)
        api().set_window_long(int(overlay.winId()), GWLP_HWNDPARENT, win)
        return win

    def overlay_shown(self, overlay, win: int):
        a = api()
        hwnd = int(overlay.winId())
        if a.get_window_long(hwnd, GWLP_HWNDPARENT) != win:  # Qt may reset the owner when it shows the window
            a.set_window_long(hwnd, GWLP_HWNDPARENT, win)
