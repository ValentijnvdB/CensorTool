"""
Minimal ctypes bindings to Xlib, XComposite and XShm for overlay mode (X11 only).

- WindowGrabber: captures one window, also while other windows (such as the overlay) cover it.
- TargetTracker: follows the position, size and visibility of a window using X events.
- pick_window: lets the user click a window.

The X libraries are loaded on first use, so importing this module works on any platform.
Every thread that talks to the X server opens its own connection (Display).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import select
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

Window = Drawable = Pixmap = Atom = Cursor = Time = ctypes.c_ulong

ZPixmap = 2
ALL_PLANES = 0xFFFFFFFFFFFFFFFF
IPC_PRIVATE, IPC_CREAT, IPC_RMID = 0, 0o1000, 0
COMPOSITE_REDIRECT_AUTOMATIC = 0

KEY_PRESS, BUTTON_PRESS, BUTTON_RELEASE, DESTROY_NOTIFY = 2, 4, 5, 17
KEY_PRESS_MASK = 1 << 0
BUTTON_PRESS_MASK, BUTTON_RELEASE_MASK = 1 << 2, 1 << 3
STRUCTURE_NOTIFY_MASK, SUBSTRUCTURE_NOTIFY_MASK = 1 << 17, 1 << 19
PROPERTY_CHANGE_MASK = 1 << 22
CREATE_NOTIFY = 16
DESTROY_NOTIFY = 17
PROPERTY_NOTIFY = 28
IS_VIEWABLE = 2
INPUT_OUTPUT = 1
GRAB_MODE_ASYNC = 1
GRAB_SUCCESS = 0
XC_CROSSHAIR = 34
XK_ESCAPE = 0xFF1B


class XImage(ctypes.Structure):
    _fields_ = [('width', ctypes.c_int), ('height', ctypes.c_int), ('xoffset', ctypes.c_int),
                ('format', ctypes.c_int), ('data', ctypes.c_void_p), ('byte_order', ctypes.c_int),
                ('bitmap_unit', ctypes.c_int), ('bitmap_bit_order', ctypes.c_int), ('bitmap_pad', ctypes.c_int),
                ('depth', ctypes.c_int), ('bytes_per_line', ctypes.c_int), ('bits_per_pixel', ctypes.c_int),
                ('red_mask', ctypes.c_ulong), ('green_mask', ctypes.c_ulong), ('blue_mask', ctypes.c_ulong)]


class XWindowAttributes(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int), ('width', ctypes.c_int), ('height', ctypes.c_int),
                ('border_width', ctypes.c_int), ('depth', ctypes.c_int), ('visual', ctypes.c_void_p),
                ('root', Window), ('class_', ctypes.c_int), ('bit_gravity', ctypes.c_int),
                ('win_gravity', ctypes.c_int), ('backing_store', ctypes.c_int), ('backing_planes', ctypes.c_ulong),
                ('backing_pixel', ctypes.c_ulong), ('save_under', ctypes.c_int), ('colormap', ctypes.c_ulong),
                ('map_installed', ctypes.c_int), ('map_state', ctypes.c_int), ('all_event_masks', ctypes.c_long),
                ('your_event_mask', ctypes.c_long), ('do_not_propagate_mask', ctypes.c_long),
                ('override_redirect', ctypes.c_int), ('screen', ctypes.c_void_p)]


class XShmSegmentInfo(ctypes.Structure):
    _fields_ = [('shmseg', ctypes.c_ulong), ('shmid', ctypes.c_int), ('shmaddr', ctypes.c_void_p),
                ('readOnly', ctypes.c_int)]


class XButtonEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', Window), ('root', Window), ('subwindow', Window),
                ('time', Time), ('x', ctypes.c_int), ('y', ctypes.c_int), ('x_root', ctypes.c_int),
                ('y_root', ctypes.c_int), ('state', ctypes.c_uint), ('button', ctypes.c_uint),
                ('same_screen', ctypes.c_int)]


class XKeyEvent(ctypes.Structure):
    _fields_ = XButtonEvent._fields_[:-2] + [('keycode', ctypes.c_uint), ('same_screen', ctypes.c_int)]


class XAnyEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', Window)]


class XCreateWindowEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('parent', Window), ('window', Window), ('x', ctypes.c_int),
                ('y', ctypes.c_int), ('width', ctypes.c_int), ('height', ctypes.c_int),
                ('border_width', ctypes.c_int), ('override_redirect', ctypes.c_int)]


class XPropertyEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong), ('send_event', ctypes.c_int),
                ('display', ctypes.c_void_p), ('window', Window), ('atom', Atom), ('time', Time),
                ('state', ctypes.c_int)]


class XEvent(ctypes.Union):
    _fields_ = [('type', ctypes.c_int), ('any', XAnyEvent), ('button', XButtonEvent), ('key', XKeyEvent),
                ('create', XCreateWindowEvent), ('property', XPropertyEvent), ('pad', ctypes.c_long * 24)]


class X11Error(RuntimeError):
    pass


# ------------------------------------------------------------------ library loading

_lock = threading.Lock()
_libs = None
_x_err = threading.local()
_ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


@_ERROR_HANDLER_TYPE
def _on_x_error(_dpy, _event):
    # Xlib's default handler exits the process. X errors are expected here (e.g. a window that is destroyed while
    # it is being captured), so they are only counted. The handler runs in the thread that made the failing call.
    _x_err.count = getattr(_x_err, 'count', 0) + 1
    return 0


def _error_count(reset: bool = False) -> int:
    n = getattr(_x_err, 'count', 0)
    if reset:
        _x_err.count = 0
    return n


def session_problem() -> str | None:
    """Why overlay mode can't run in this session, or None if it can."""
    if not sys.platform.startswith('linux'):
        return f"Overlay mode only works on Linux with X11, not on {sys.platform}."
    if os.environ.get('XDG_SESSION_TYPE') == 'wayland' or os.environ.get('WAYLAND_DISPLAY'):
        return "Overlay mode needs an X11 session; Wayland is not supported."
    if not os.environ.get('DISPLAY'):
        return "Overlay mode needs an X11 display, but DISPLAY is not set."
    try:
        libs()
    except OSError as e:
        return f"Overlay mode needs the X11 libraries (libX11, libXext, libXcomposite): {e}"
    return None


def libs():
    """Load the X libraries and declare the functions used (once)."""
    global _libs
    with _lock:
        if _libs is not None:
            return _libs

        def load(name: str):
            path = ctypes.util.find_library(name)
            if path is None:
                raise OSError(f"lib{name} not found")
            return ctypes.CDLL(path)

        x, xext, xcomp = load('X11'), load('Xext'), load('Xcomposite')
        libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
        vp, ulong, c_int, c_uint = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_uint
        P = ctypes.POINTER

        def declare(lib, name, restype, *argtypes):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = list(argtypes)

        declare(x, 'XOpenDisplay', vp, ctypes.c_char_p)
        declare(x, 'XCloseDisplay', c_int, vp)
        declare(x, 'XConnectionNumber', c_int, vp)
        declare(x, 'XDefaultRootWindow', Window, vp)
        declare(x, 'XGetWindowAttributes', c_int, vp, Window, P(XWindowAttributes))
        declare(x, 'XTranslateCoordinates', c_int, vp, Window, Window, c_int, c_int, P(c_int), P(c_int), P(Window))
        declare(x, 'XQueryTree', c_int, vp, Window, P(Window), P(Window), P(P(Window)), P(c_uint))
        declare(x, 'XFree', c_int, vp)
        declare(x, 'XGetImage', P(XImage), vp, Drawable, c_int, c_int, c_uint, c_uint, ulong, c_int)
        declare(x, 'XDestroyImage', c_int, P(XImage))
        declare(x, 'XFreePixmap', c_int, vp, Pixmap)
        declare(x, 'XSync', c_int, vp, c_int)
        declare(x, 'XFlush', c_int, vp)
        declare(x, 'XSelectInput', c_int, vp, Window, ctypes.c_long)
        declare(x, 'XPending', c_int, vp)
        declare(x, 'XNextEvent', c_int, vp, P(XEvent))
        declare(x, 'XInternAtom', Atom, vp, ctypes.c_char_p, c_int)
        declare(x, 'XGetWindowProperty', c_int, vp, Window, Atom, ctypes.c_long, ctypes.c_long, c_int, Atom,
                P(Atom), P(c_int), P(ulong), P(ulong), P(vp))
        declare(x, 'XCreateFontCursor', Cursor, vp, c_uint)
        declare(x, 'XFreeCursor', c_int, vp, Cursor)
        declare(x, 'XGrabPointer', c_int, vp, Window, c_int, c_uint, c_int, c_int, Window, Cursor, Time)
        declare(x, 'XUngrabPointer', c_int, vp, Time)
        declare(x, 'XGrabKeyboard', c_int, vp, Window, c_int, c_int, c_int, Time)
        declare(x, 'XUngrabKeyboard', c_int, vp, Time)
        declare(x, 'XKeysymToKeycode', ctypes.c_ubyte, vp, ulong)
        declare(x, 'XSetErrorHandler', vp, _ERROR_HANDLER_TYPE)
        declare(xext, 'XShmQueryExtension', c_int, vp)
        declare(xext, 'XShmCreateImage', P(XImage), vp, vp, c_uint, c_int, ctypes.c_char_p, P(XShmSegmentInfo),
                c_uint, c_uint)
        declare(xext, 'XShmAttach', c_int, vp, P(XShmSegmentInfo))
        declare(xext, 'XShmDetach', c_int, vp, P(XShmSegmentInfo))
        declare(xext, 'XShmGetImage', c_int, vp, Drawable, P(XImage), c_int, c_int, ulong)
        declare(xcomp, 'XCompositeQueryExtension', c_int, vp, P(c_int), P(c_int))
        declare(xcomp, 'XCompositeRedirectWindow', None, vp, Window, c_int)
        declare(xcomp, 'XCompositeNameWindowPixmap', Pixmap, vp, Window)
        declare(libc, 'shmget', c_int, c_int, ctypes.c_size_t, c_int)
        declare(libc, 'shmat', vp, c_int, vp, c_int)
        declare(libc, 'shmdt', c_int, vp)
        declare(libc, 'shmctl', c_int, c_int, c_int, vp)

        x.XSetErrorHandler(_on_x_error)
        _libs = (x, xext, xcomp, libc)
        return _libs


def install_error_handler():
    """(Re)install the X error handler, e.g. after a GUI toolkit replaced it."""
    libs()[0].XSetErrorHandler(_on_x_error)


# ------------------------------------------------------------------ connection and helpers

class Connection:
    """One X connection with the helpers used here. Use it from one thread only."""

    def __init__(self):
        self.x, self.xext, self.xcomp, self.libc = libs()
        self.dpy = self.x.XOpenDisplay(None)
        if not self.dpy:
            raise X11Error("Could not connect to the X server.")
        self.root = self.x.XDefaultRootWindow(self.dpy)
        self._atoms: dict[str, int] = {}

    def close(self):
        if self.dpy:
            self.x.XCloseDisplay(self.dpy)
            self.dpy = None

    def atom(self, name: str) -> int:
        if name not in self._atoms:
            self._atoms[name] = self.x.XInternAtom(self.dpy, name.encode(), 0)
        return self._atoms[name]

    def query_tree(self, win: int) -> tuple[int, int, list[int]]:
        """(root, parent, children from bottom to top), or (0, 0, []) when win does not exist."""
        root, parent = Window(), Window()
        children, n = ctypes.POINTER(Window)(), ctypes.c_uint()
        if not self.x.XQueryTree(self.dpy, win, ctypes.byref(root), ctypes.byref(parent), ctypes.byref(children),
                                 ctypes.byref(n)):
            return 0, 0, []
        kids = [children[i] for i in range(n.value)]
        if children:
            self.x.XFree(children)
        return root.value, parent.value, kids

    def toplevel(self, win: int) -> int:
        """The ancestor of win that is a child of the root window (its WM frame), or win itself."""
        while True:
            root, parent, _ = self.query_tree(win)
            if not parent or parent == root:
                return win
            win = parent

    def root_position(self, win: int) -> tuple[int, int]:
        x, y, child = ctypes.c_int(), ctypes.c_int(), Window()
        self.x.XTranslateCoordinates(self.dpy, win, self.root, 0, 0, ctypes.byref(x), ctypes.byref(y),
                                     ctypes.byref(child))
        return x.value, y.value

    def attributes(self, win: int) -> XWindowAttributes | None:
        attrs = XWindowAttributes()
        _error_count(reset=True)
        ok = self.x.XGetWindowAttributes(self.dpy, win, ctypes.byref(attrs))
        return attrs if ok and not _error_count() else None

    def cardinals(self, win: int, name: str, n: int) -> list[int] | None:
        """The first n values of a 32-bit property, [] if it has another format, None if it is not set."""
        actual_type, actual_format = Atom(), ctypes.c_int()
        nitems, after, prop = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_void_p()
        status = self.x.XGetWindowProperty(self.dpy, win, self.atom(name), 0, n, 0, 0, ctypes.byref(actual_type),
                                           ctypes.byref(actual_format), ctypes.byref(nitems), ctypes.byref(after),
                                           ctypes.byref(prop))
        if status != 0 or not prop.value:
            return None
        values: list[int] = []
        if actual_format.value == 32 and nitems.value >= n:
            values = list(ctypes.cast(prop, ctypes.POINTER(ctypes.c_long))[:n])  # format 32 is returned as longs
        self.x.XFree(prop)
        return values

    def strings(self, win: int, name: str) -> list[str] | None:
        """An 8-bit property split at NUL bytes (e.g. WM_CLASS), or None if it is not set."""
        actual_type, actual_format = Atom(), ctypes.c_int()
        nitems, after, prop = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_void_p()
        status = self.x.XGetWindowProperty(self.dpy, win, self.atom(name), 0, 1024, 0, 0, ctypes.byref(actual_type),
                                           ctypes.byref(actual_format), ctypes.byref(nitems), ctypes.byref(after),
                                           ctypes.byref(prop))
        if status != 0 or not prop.value:
            return None
        values = []
        if actual_format.value == 8:
            data = ctypes.string_at(prop, nitems.value)
            values = [v.decode('utf-8', 'replace') for v in data.split(b'\0') if v]
        self.x.XFree(prop)
        return values

    def client_list(self, name: str = '_NET_CLIENT_LIST') -> list[int]:
        """
        The application windows the window manager manages: oldest first (_NET_CLIENT_LIST) or bottom to top
        (_NET_CLIENT_LIST_STACKING).
        """
        actual_type, actual_format = Atom(), ctypes.c_int()
        nitems, after, prop = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_void_p()
        status = self.x.XGetWindowProperty(self.dpy, self.root, self.atom(name), 0, 65536, 0, 0,
                                           ctypes.byref(actual_type), ctypes.byref(actual_format),
                                           ctypes.byref(nitems), ctypes.byref(after), ctypes.byref(prop))
        if status != 0 or not prop.value:
            return []
        windows = []
        if actual_format.value == 32:
            windows = list(ctypes.cast(prop, ctypes.POINTER(ctypes.c_ulong))[:nitems.value])
        self.x.XFree(prop)
        return windows

    def window_info(self, win: int) -> 'WindowInfo':
        """The properties used to tell which program a window belongs to."""
        pid = self.cardinals(win, '_NET_WM_PID', 1)
        types = self.cardinals(win, '_NET_WM_WINDOW_TYPE', 1)
        return WindowInfo(
            window=win,
            pid=pid[0] if pid else None,
            wm_class=tuple(self.strings(win, 'WM_CLASS') or ()),
            normal=not types or types[0] == self.atom('_NET_WM_WINDOW_TYPE_NORMAL'),
        )

    def title(self, win: int) -> str:
        """The window's title (_NET_WM_NAME, else WM_NAME), '' if it has none."""
        for name in ('_NET_WM_NAME', 'WM_NAME'):
            values = self.strings(win, name)
            if values:
                return values[0]
        return ''

    def window_type(self, win: int) -> str | None:
        """The window's first _NET_WM_WINDOW_TYPE without its prefix (e.g. 'NORMAL', 'DIALOG', 'DOCK'), or None."""
        types = self.cardinals(win, '_NET_WM_WINDOW_TYPE', 1)
        if not types:
            return None
        for name in ('NORMAL', 'DIALOG', 'DOCK', 'DESKTOP', 'UTILITY', 'SPLASH', 'TOOLBAR', 'MENU',
                     'NOTIFICATION'):
            if types[0] == self.atom(f'_NET_WM_WINDOW_TYPE_{name}'):
                return name
        return 'OTHER'

    def active_window(self) -> int | None:
        """The focused application window (_NET_ACTIVE_WINDOW), or None."""
        value = self.cardinals(self.root, '_NET_ACTIVE_WINDOW', 1)
        return value[0] if value and value[0] else None

    def client_of(self, win: int) -> int | None:
        """The application window (with WM_STATE) of a toplevel: win itself or one of its children."""
        if self.cardinals(win, 'WM_STATE', 1) is not None:
            return win
        for child in self.query_tree(win)[2]:
            if self.cardinals(child, 'WM_STATE', 1) is not None:
                return child
        return None

    def visible_client_area(self, client: int, attrs: XWindowAttributes) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        """
        ((x, y, w, h) on the root window, (x, y) inside the client) of the part of the client that is drawn:
        without the transparent shadow margins some applications draw themselves (_GTK_FRAME_EXTENTS).
        """
        x, y = self.root_position(client)
        return visible_area(x, y, attrs.width, attrs.height, self.cardinals(client, '_GTK_FRAME_EXTENTS', 4))


@dataclass(frozen=True)
class WindowInfo:
    window: int
    pid: int | None  # _NET_WM_PID: the process that made the window (not reliable for Flatpak/Snap apps)
    wm_class: tuple[str, ...]  # WM_CLASS: (instance, class), e.g. ('celluloid', 'Celluloid')
    normal: bool  # a normal window, not a dialog, splash screen, toolbar, ...


def visible_area(x: int, y: int, width: int, height: int, gtk_extents: list[int] | None
                 ) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Remove the shadow margins (left, right, top, bottom) from a window area. See visible_client_area."""
    left = right = top = bottom = 0
    if gtk_extents:
        left, right, top, bottom = (max(0, v) for v in gtk_extents)
    w, h = width - left - right, height - top - bottom
    if w <= 0 or h <= 0:  # nonsense extents: use the whole window
        return (x, y, width, height), (0, 0)
    return (x + left, y + top, w, h), (left, top)


# ------------------------------------------------------------------ capture

class WindowGrabber:
    """
    Captures the visible area of one window (see Connection.visible_client_area), also when it is covered.

    method 'pixmap' reads the window's off-screen buffer that the compositor (XComposite) keeps for its frame,
    'window' reads the client window directly. 'pixmap' is used when available; when it fails, the grabber
    switches to 'window'. XShm (shared memory) is used when available.
    Use from one thread only.
    """

    def __init__(self, client: int, method: str = 'pixmap'):
        self.conn = Connection()
        self.client = client
        major, minor = ctypes.c_int(), ctypes.c_int()
        has_composite = bool(self.conn.xcomp.XCompositeQueryExtension(self.conn.dpy, ctypes.byref(major),
                                                                      ctypes.byref(minor)))
        if method == 'pixmap' and not has_composite:
            logger.warning("The X server has no Composite extension, capturing the window directly.")
            method = 'window'
        self.method = method
        self.use_shm = bool(self.conn.xext.XShmQueryExtension(self.conn.dpy))
        self.frame = 0
        self.pixmap = 0
        self._shm_img = None
        self._shm_info = None
        self._buffer_key = None  # (drawable, size, depth) the buffers were made for
        self.area = (0, 0, 0, 0)  # visible area on the root window
        self._source = (0, 0)  # top left of the visible area inside the drawable
        self.visible = False
        self.failures = 0

    def close(self):
        self._free_buffers()
        self.conn.close()

    def _free_buffers(self):
        c = self.conn
        if self._shm_img is not None:
            c.xext.XShmDetach(c.dpy, ctypes.byref(self._shm_info))
            c.x.XSync(c.dpy, 0)
            c.libc.shmdt(self._shm_info.shmaddr)
            self._shm_img.contents.data = None
            c.x.XDestroyImage(self._shm_img)
            self._shm_img = None
        if self.pixmap:
            c.x.XFreePixmap(c.dpy, self.pixmap)
            self.pixmap = 0
        self._buffer_key = None

    def refresh(self) -> bool:
        """
        Update the position and size of the window. Returns whether it exists.
        Afterwards, self.visible tells whether it is shown (not minimised, on the current workspace).
        """
        c = self.conn
        attrs = c.attributes(self.client)
        if attrs is None:
            self.visible = False
            self._free_buffers()
            return False
        self.visible = attrs.map_state == IS_VIEWABLE
        if not self.visible:
            self._free_buffers()  # the frame's pixmap is replaced when the window is shown again
            return True

        self.area, crop = c.visible_client_area(self.client, attrs)
        size = self.area[2:]
        if self.method == 'pixmap':
            frame = c.toplevel(self.client)
            if frame != self.frame:
                self._free_buffers()
                self.frame = frame
                c.xcomp.XCompositeRedirectWindow(c.dpy, frame, COMPOSITE_REDIRECT_AUTOMATIC)
            fx, fy = c.root_position(frame)
            self._source = (self.area[0] - fx, self.area[1] - fy)
            frame_attrs = c.attributes(frame)
            if frame_attrs is None:
                return True
            key = (frame, (frame_attrs.width, frame_attrs.height), size)
            if key != self._buffer_key:
                self._free_buffers()
                self.pixmap = c.xcomp.XCompositeNameWindowPixmap(c.dpy, frame)
                self._make_shm(frame_attrs, size)
                self._buffer_key = key
        else:
            self._source = crop
            key = (self.client, size)
            if key != self._buffer_key:
                self._free_buffers()
                self._make_shm(attrs, size)
                self._buffer_key = key
        return True

    def _make_shm(self, attrs: XWindowAttributes, size: tuple[int, int]):
        if not self.use_shm:
            return
        c = self.conn
        info = XShmSegmentInfo()
        img = c.xext.XShmCreateImage(c.dpy, attrs.visual, attrs.depth, ZPixmap, None, ctypes.byref(info), *size)
        if not img:
            self.use_shm = False
            return
        info.shmid = c.libc.shmget(IPC_PRIVATE, img.contents.bytes_per_line * img.contents.height,
                                   IPC_CREAT | 0o600)
        info.shmaddr = c.libc.shmat(info.shmid, None, 0)
        img.contents.data = info.shmaddr
        info.readOnly = 0
        c.xext.XShmAttach(c.dpy, ctypes.byref(info))
        c.x.XSync(c.dpy, 0)
        c.libc.shmctl(info.shmid, IPC_RMID, None)  # removed once both sides detach
        self._shm_img, self._shm_info = img, info

    def grab(self) -> np.ndarray | None:
        """A BGRA copy of the visible area, or None when it could not be captured (call refresh and try again)."""
        if not self.visible or self._buffer_key is None:
            return None
        c = self.conn
        drawable = self.pixmap if self.method == 'pixmap' else self.client
        (x, y), (w, h) = self._source, self.area[2:]
        _error_count(reset=True)
        if self._shm_img is not None:
            ok = c.xext.XShmGetImage(c.dpy, drawable, self._shm_img, x, y, ALL_PLANES)
            c.x.XSync(c.dpy, 0)
            frame = self._to_array(self._shm_img.contents, w, h) if ok and not _error_count() else None
        else:
            img = c.x.XGetImage(c.dpy, drawable, x, y, w, h, ALL_PLANES, ZPixmap)
            c.x.XSync(c.dpy, 0)
            frame = self._to_array(img.contents, w, h) if img and not _error_count() else None
            if img:
                c.x.XDestroyImage(img)

        if frame is not None:
            self.failures = 0
            return frame
        self.failures += 1
        self._buffer_key = None  # make the buffers again on the next refresh
        if self.method == 'pixmap' and self.failures >= 10:
            logger.warning("Capturing the window through the compositor keeps failing, capturing it directly.")
            self._free_buffers()
            self.method = 'window'
            self.failures = 0
        return None

    @staticmethod
    def _to_array(img: XImage, w: int, h: int) -> np.ndarray | None:
        if img.bits_per_pixel != 32 or not img.data:
            return None
        buf = (ctypes.c_ubyte * (img.bytes_per_line * h)).from_address(img.data)
        return np.frombuffer(buf, np.uint8).reshape(h, img.bytes_per_line // 4, 4)[:, :w].copy()


# ------------------------------------------------------------------ following a window

@dataclass(frozen=True)
class TargetState:
    version: int = 0
    alive: bool = True  # the window exists
    visible: bool = False  # shown: not minimised and on the current workspace
    area: tuple[int, int, int, int] = (0, 0, 0, 0)  # visible area on the root window (see visible_client_area)


class TargetTracker(threading.Thread):
    """
    Follows the visible area of a window using X events, for placing the overlay. Read .state from any thread.
    """

    def __init__(self, client: int, on_change=None):
        super().__init__(name=f'x11-tracker-{client:#x}', daemon=True)
        self.client = client
        self.state = TargetState()
        self._on_change = on_change
        self._stop_event = threading.Event()
        self._ready = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    def wait_ready(self, timeout: float = 2):
        self._ready.wait(timeout)

    def run(self):
        conn = Connection()
        try:
            x = conn.x
            x.XSelectInput(conn.dpy, conn.root, SUBSTRUCTURE_NOTIFY_MASK)
            x.XSelectInput(conn.dpy, self.client, STRUCTURE_NOTIFY_MASK | PROPERTY_CHANGE_MASK)
            x.XFlush(conn.dpy)
            fd = x.XConnectionNumber(conn.dpy)
            event = XEvent()
            self._update(conn)
            self._ready.set()
            last = time.monotonic()
            while not self._stop_event.is_set() and self.state.alive:
                select.select([fd], [], [], 0.1)
                n = 0
                while x.XPending(conn.dpy):
                    x.XNextEvent(conn.dpy, ctypes.byref(event))
                    n += 1
                now = time.monotonic()
                if n or now - last > 0.5:  # also poll now and then, in case an event was missed
                    self._update(conn)
                    last = now
        except Exception as e:
            logger.error(f"Following window {self.client:#x} failed: {e}")
            self.state = TargetState(version=self.state.version + 1, alive=False)
        finally:
            self._ready.set()
            conn.close()

    def _update(self, conn: Connection):
        attrs = conn.attributes(self.client)
        if attrs is None:
            new = TargetState(alive=False)
        elif attrs.map_state != IS_VIEWABLE:
            new = TargetState(alive=True, visible=False, area=self.state.area)
        else:
            new = TargetState(alive=True, visible=True, area=conn.visible_client_area(self.client, attrs)[0])
        old = self.state
        if (new.alive, new.visible, new.area) != (old.alive, old.visible, old.area):
            self.state = TargetState(version=old.version + 1, alive=new.alive, visible=new.visible, area=new.area)
            if self._on_change is not None:
                self._on_change()


class ActiveWindowTracker(threading.Thread):
    """Follows which window has the focus. Read .active from any thread."""

    def __init__(self):
        super().__init__(name='x11-active-window', daemon=True)
        self.active: int | None = None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    def run(self):
        try:
            conn = Connection()
        except X11Error as e:
            logger.error(f"Following the focused window failed: {e}")
            return
        try:
            x = conn.x
            x.XSelectInput(conn.dpy, conn.root, PROPERTY_CHANGE_MASK)
            x.XFlush(conn.dpy)
            fd = x.XConnectionNumber(conn.dpy)
            event = XEvent()
            last = 0.0
            while not self._stop_event.is_set():
                n = 0
                while x.XPending(conn.dpy):
                    x.XNextEvent(conn.dpy, ctypes.byref(event))
                    n += 1
                now = time.monotonic()
                if n or now - last > 0.5:
                    self.active = conn.active_window()
                    last = now
                select.select([fd], [], [], 0.1)
        except Exception as e:
            logger.error(f"Following the focused window failed: {e}")
        finally:
            conn.close()


# ------------------------------------------------------------------ picking a window

def window_at(stack: list[tuple[int, tuple[int, int, int, int]]], x: int, y: int, exclude: set[int]) -> int | None:
    """The topmost window of stack (bottom to top, (window, (x, y, w, h))) that contains (x, y)."""
    for win, (wx, wy, ww, wh) in reversed(stack):
        if win not in exclude and wx <= x < wx + ww and wy <= y < wy + wh:
            return win
    return None


def pick_window(exclude: set[int] = frozenset(), cancel: threading.Event | None = None) -> int | None:
    """
    Let the user click a window (crosshair cursor). Right click or Escape cancels.

    :param exclude: windows that can't be picked (the overlay's own windows); clicks go to the window below them.
    :param cancel: set it to stop waiting for a click.
    :return: the id of the application window that was clicked, or None.
    """
    conn = Connection()
    x = conn.x
    cursor = x.XCreateFontCursor(conn.dpy, XC_CROSSHAIR)
    try:
        # a menu that was just closed (e.g. the tray menu) can still hold a grab for a moment
        for _ in range(20):
            status = x.XGrabPointer(conn.dpy, conn.root, 0, BUTTON_PRESS_MASK | BUTTON_RELEASE_MASK,
                                    GRAB_MODE_ASYNC, GRAB_MODE_ASYNC, 0, cursor, 0)
            if status == GRAB_SUCCESS:
                break
            time.sleep(0.05)
        else:
            raise X11Error("Could not grab the mouse to select a window.")
        keyboard = x.XGrabKeyboard(conn.dpy, conn.root, 0, GRAB_MODE_ASYNC, GRAB_MODE_ASYNC, 0) == GRAB_SUCCESS
        escape = x.XKeysymToKeycode(conn.dpy, XK_ESCAPE)
        x.XFlush(conn.dpy)

        fd = x.XConnectionNumber(conn.dpy)
        event = XEvent()
        clicked = None
        try:
            while clicked is None:
                if cancel is not None and cancel.is_set():
                    return None
                if not x.XPending(conn.dpy):
                    select.select([fd], [], [], 0.1)
                    continue
                x.XNextEvent(conn.dpy, ctypes.byref(event))
                if event.type == KEY_PRESS and event.key.keycode == escape:
                    return None
                if event.type == BUTTON_PRESS:
                    if event.button.button != 1:
                        return None
                    clicked = (event.button.x_root, event.button.y_root)
            # swallow the release, so the click doesn't also reach the window
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if x.XPending(conn.dpy):
                    x.XNextEvent(conn.dpy, ctypes.byref(event))
                    if event.type == BUTTON_RELEASE:
                        break
                else:
                    select.select([fd], [], [], 0.05)
        finally:
            x.XUngrabPointer(conn.dpy, 0)
            if keyboard:
                x.XUngrabKeyboard(conn.dpy, 0)
            x.XFlush(conn.dpy)

        excluded = {conn.toplevel(w) for w in exclude}
        stack = []
        for win in conn.query_tree(conn.root)[2]:
            attrs = conn.attributes(win)
            if (attrs is None or attrs.map_state != IS_VIEWABLE or attrs.class_ != INPUT_OUTPUT
                    or attrs.override_redirect):
                continue
            stack.append((win, (attrs.x, attrs.y, attrs.width + 2 * attrs.border_width,
                                attrs.height + 2 * attrs.border_width)))
        toplevel = window_at(stack, *clicked, exclude=excluded)
        return conn.client_of(toplevel) if toplevel is not None else None
    finally:
        x.XFreeCursor(conn.dpy, cursor)
        conn.close()
