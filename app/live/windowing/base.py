"""
What overlay mode needs from the window system, independent of the platform. x11.py implements it for Linux
with X11; another platform (Windows, macOS, a Wayland compositor) would add a module that implements the same
classes and is selected in __init__.py.

Windows are identified by an int id (X11: the application window, not the window manager's frame). Areas are
(x, y, width, height) in screen pixels.

Threads: a Display, WindowMonitor or Grabber is used by one thread only (X11: each has its own connection).
Trackers run in a thread of their own and can be read from any thread.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Area = tuple[int, int, int, int]


class WindowSystemError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowInfo:
    window: int
    pid: int | None  # the process that made the window (X11: not reliable for Flatpak/Snap apps)
    app_ids: tuple[str, ...]  # names that tell the program, matched by launch.class_matches (X11: WM_CLASS)
    normal: bool  # a normal window, not a dialog, splash screen, toolbar, ...
    panel: bool = False  # a panel or the desktop: never censored automatically


@dataclass(frozen=True)
class TargetState:
    version: int = 0
    alive: bool = True  # the window exists
    visible: bool = False  # shown: not minimised and on the current workspace
    area: Area = (0, 0, 0, 0)  # the part of the window that is drawn, on the screen


@dataclass
class WindowChanges:
    """What a WindowMonitor saw change."""
    list_changed: bool = False  # windows were added to or removed from the window list
    windows: set[int] = field(default_factory=set)  # windows whose identity, title or managed state changed


class Display(ABC):
    """Queries about windows. Use from one thread only."""

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def window_list(self, stacking: bool = False) -> list[int]:
        """The application windows: oldest first, or with stacking bottom to top."""

    @abstractmethod
    def window_info(self, win: int) -> WindowInfo:
        ...

    @abstractmethod
    def title(self, win: int) -> str:
        """The window's title, '' if it has none."""

    @abstractmethod
    def is_app_window(self, win: int) -> bool:
        """Whether win is an application window the window manager manages (shown or minimised)."""

    @abstractmethod
    def visible_area(self, win: int) -> Area | None:
        """The part of the window that is drawn, or None when it is not shown."""

    @abstractmethod
    def application_window(self, win: int) -> int:
        """
        The application window for a window id the user gave (e.g. a window manager frame).

        :raises ValueError: if there is no such window.
        """


class WindowMonitor(ABC):
    """Waits for windows to appear, disappear or change. Use from one thread only."""

    display: Display  # for queries, from the same thread

    @abstractmethod
    def wait(self, timeout: float) -> WindowChanges:
        """Wait at most timeout seconds for changes and return them (empty when there were none)."""

    @abstractmethod
    def close(self):
        ...


class Grabber(ABC):
    """Captures the drawn part of one window, also while other windows cover it. Use from one thread only."""

    window: int
    area: Area = (0, 0, 0, 0)  # where the window was at the last refresh
    visible: bool = False  # whether it was shown at the last refresh
    failures: int = 0  # grabs that failed since the last one that worked

    @abstractmethod
    def refresh(self) -> bool:
        """Update the window's position, size and visibility. Returns whether it exists."""

    @abstractmethod
    def grab(self) -> np.ndarray | None:
        """A BGRA copy of the drawn part, or None when it could not be captured (refresh and try again)."""

    @abstractmethod
    def close(self):
        ...


class Tracker(ABC):
    """Follows where a window is and whether it is shown. Read .state from any thread."""

    window: int
    state: TargetState

    @abstractmethod
    def stop(self):
        ...


class FocusTracker(ABC):
    """Follows which application window has the focus. Read .active from any thread."""

    active: int | None

    @abstractmethod
    def stop(self):
        ...


class WindowSystem(ABC):
    """The window system of this session. One per process, see windowing.get()."""

    name: str

    def prepare_qt_environment(self, environ: dict[str, str]):
        """Settings for Qt that must be made before the QApplication is created."""

    def qt_started(self):
        """Called after the QApplication was created."""

    @abstractmethod
    def open_display(self) -> Display:
        ...

    @abstractmethod
    def monitor(self, follow_titles: bool = False) -> WindowMonitor:
        """
        :param follow_titles: also report windows whose title changed (otherwise only new windows, windows that
            become application windows and changes to the window list).
        """

    @abstractmethod
    def grabber(self, win: int) -> Grabber:
        ...

    @abstractmethod
    def tracker(self, win: int, on_change: Callable[[], None] | None = None) -> Tracker:
        """
        Start following a window. on_change is called (from the tracker's thread) when its state changed. The
        state is up to date when this returns.
        """

    @abstractmethod
    def focus_tracker(self) -> FocusTracker:
        """Start following the focus."""

    @abstractmethod
    def pick_window(self, exclude: set[int] = frozenset(), cancel: threading.Event | None = None) -> int | None:
        """
        Let the user click a window. Right click or Escape cancels.

        :param exclude: windows that can't be picked (the overlays); clicks go to the window below them.
        :param cancel: set it to stop waiting for a click.
        :return: the application window that was clicked, or None.
        """

    @abstractmethod
    def place_above(self, overlay, win: int) -> object:
        """
        Keep a Qt window (QWindow, created but not shown yet) just above win: above it, below the windows that are
        above it, also when either moves in the stacking order.

        :return: an object to keep as long as the overlay exists.
        """
