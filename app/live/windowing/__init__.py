"""
The window system overlay mode works with: capturing, following and picking windows, and noticing new ones.
See base.py for what a platform has to provide: x11.py implements it for Linux with X11, windows.py for Windows
10/11 (not tested on Windows yet).
"""
from __future__ import annotations

import sys
import threading

from .base import (Area, Display, FocusTracker, Grabber, TargetState, Tracker, WindowChanges, WindowInfo,
                   WindowMonitor, WindowSystem, WindowSystemError)

__all__ = ['Area', 'Display', 'FocusTracker', 'Grabber', 'TargetState', 'Tracker', 'WindowChanges', 'WindowInfo',
           'WindowMonitor', 'WindowSystem', 'WindowSystemError', 'get', 'session_problem']

_lock = threading.Lock()
_system: WindowSystem | None = None


def session_problem() -> str | None:
    """Why overlay mode can't run in this session, or None if it can."""
    if sys.platform.startswith('linux'):
        from . import x11
        return x11.session_problem()
    if sys.platform == 'win32':
        from . import windows
        return windows.session_problem()
    return f"Overlay mode works on Linux with X11 and on Windows, not on {sys.platform} (yet)."


def get() -> WindowSystem:
    """
    The window system of this session.

    :raises WindowSystemError: if overlay mode can't run in this session (see session_problem).
    """
    global _system
    with _lock:
        if _system is None:
            problem = session_problem()
            if problem:
                raise WindowSystemError(problem)
            if sys.platform == 'win32':
                from .windows import WindowsWindowSystem
                _system = WindowsWindowSystem()
            else:
                from .x11 import X11WindowSystem
                _system = X11WindowSystem()
        return _system
