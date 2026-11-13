"""
Shows the censored frames of each censored (target) window in an overlay that lies exactly over it.

The overlay ignores mouse and keyboard input, so clicks and scrolling go to the target window below it. It is a
transient of its target, so the window manager keeps it directly above the target: other windows (also other
censored windows with their overlays), menus and panels still appear on top of it.

Needs PySide6 and an X11 session. Create it in the Qt (main) thread; push_frame and poll_command can be called
from any thread. Frames are pushed with the target's window id as source.
"""
import threading
from collections import deque
from typing import Callable

import numpy as np
from loguru import logger
from PySide6.QtCore import QObject, QRect, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QWindow
from PySide6.QtWidgets import QWidget

from .generic import Command, OutputDevice
from .. import x11


class OverlayWindow(QWidget):
    """The overlay window. Paints the newest frame stretched over itself, or a placeholder until there is one."""

    PLACEHOLDER_COLOR = QColor(32, 32, 32)
    CENSORING_TEXT = 'CensorTool: censoring...'

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowTransparentForInput
                         | Qt.WindowDoesNotAcceptFocus | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setWindowTitle('CensorTool overlay')
        self.frame: np.ndarray | None = None
        self.placeholder_text = self.CENSORING_TEXT

    def paintEvent(self, _event):
        painter = QPainter(self)
        frame = self.frame
        if frame is None:
            painter.fillRect(self.rect(), self.PLACEHOLDER_COLOR)
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont('Sans', 14))
            painter.drawText(self.rect().adjusted(8, 8, -8, -8), Qt.AlignCenter | Qt.TextWordWrap,
                             self.placeholder_text)
            return
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        # stretched when the target was resized and no frame of the new size is censored yet
        painter.drawImage(self.rect(), image)


class _Bridge(QObject):
    """Runs callbacks in the Qt thread when its signals are emitted from other threads."""
    frame_ready = Signal(object)  # client (object: X ids don't fit a signed 32-bit int)
    target_changed = Signal(object)

    def __init__(self, on_frame_ready: Callable[[int], None], on_target_changed: Callable[[int], None]):
        super().__init__()
        self._on_frame_ready = on_frame_ready
        self._on_target_changed = on_target_changed
        self.frame_ready.connect(self._frame_ready, Qt.QueuedConnection)
        self.target_changed.connect(self._target_changed, Qt.QueuedConnection)

    @Slot(object)
    def _frame_ready(self, client):
        self._on_frame_ready(client)

    @Slot(object)
    def _target_changed(self, client):
        self._on_target_changed(client)


class _Target:
    """The overlay of one censored window."""

    def __init__(self, client: int, since: float, on_change: Callable[[], None]):
        self.client = client
        self.accept_since = since  # frames captured before this belong to an earlier time this window was censored
        self.newest: np.ndarray | None = None  # pushed, not painted yet
        self.window = OverlayWindow()
        self.window.winId()  # create the native window, so it can be made a transient before it is shown
        self.parent_qwindow = QWindow.fromWinId(client)
        self.window.windowHandle().setTransientParent(self.parent_qwindow)
        self.applied_version = -1
        self.tracker = x11.TargetTracker(client, on_change=on_change)
        self.tracker.start()
        self.tracker.wait_ready()

    def close(self):
        self.tracker.stop()
        self.window.hide()


class OverlayDevice(OutputDevice):

    def __init__(self, on_target_closed: Callable[[int], None] | None = None):
        """
        :param on_target_closed: called in the Qt thread with the window id when a censored window no longer
            exists. Its overlay is removed already.
        """
        self._on_target_closed = on_target_closed
        self._bridge = _Bridge(self._show_newest_frame, self._follow_target)
        self._lock = threading.Lock()  # guards _targets' frames and the dict itself
        self._targets: dict[int, _Target] = {}
        self._commands: deque[Command] = deque()
        self._closed = False

    @property
    def targets(self) -> list[int]:
        """The censored windows, in the order they were added."""
        with self._lock:
            return list(self._targets)

    # ---------------------------------------------------------------- any thread

    def push_frame(self, frame: np.ndarray, timestamp: float | None = None, source=None):
        with self._lock:
            target = self._targets.get(source)
            if target is None or self._closed or (timestamp is not None and timestamp < target.accept_since):
                return
            target.newest = np.ascontiguousarray(frame)
        self._bridge.frame_ready.emit(source)

    def add_command(self, command: Command):
        self._commands.append(command)

    def poll_command(self) -> Command | None:
        try:
            return self._commands.popleft()
        except IndexError:
            return None

    # ---------------------------------------------------------------- Qt thread

    def add_target(self, client: int, since: float | None = None, placeholder: str | None = None):
        """
        Cover a window. A placeholder is shown until a frame captured at or after since (time.monotonic()) arrives,
        so frames of an earlier time this window was censored are not shown.

        :param placeholder: the placeholder's text (default: OverlayWindow.CENSORING_TEXT). A window whose frames
            are never pushed stays hidden behind it.
        """
        if client in self._targets:
            return
        target = _Target(client, since if since is not None else float('-inf'),
                         on_change=lambda: self._bridge.target_changed.emit(client))
        target.window.placeholder_text = placeholder or OverlayWindow.CENSORING_TEXT
        with self._lock:
            self._targets[client] = target
        self._follow_target(client)

    def set_placeholder(self, client: int, text: str | None = None, since: float | None = None):
        """Change the placeholder text of a covered window (and when its frames are accepted from, see add_target)."""
        target = self._targets.get(client)
        if target is None:
            return
        with self._lock:
            if since is not None:
                target.accept_since = since
        target.window.placeholder_text = text or OverlayWindow.CENSORING_TEXT
        target.window.update()

    def remove_target(self, client: int):
        with self._lock:
            target = self._targets.pop(client, None)
        if target is not None:
            target.close()

    def overlay_window_ids(self) -> set[int]:
        return {int(t.window.winId()) for t in self._targets.values()}

    def any_visible(self) -> bool:
        return any(t.window.isVisible() for t in self._targets.values())

    def _show_newest_frame(self, client: int):
        with self._lock:
            target = self._targets.get(client)
            if target is None:
                return
            frame, target.newest = target.newest, None
        if frame is not None:
            target.window.frame = frame
            target.window.update()

    def _follow_target(self, client: int):
        target = self._targets.get(client)
        if target is None:
            return
        state = target.tracker.state
        if state.version == target.applied_version:
            return
        target.applied_version = state.version
        window = target.window
        if not state.alive:
            logger.info(f"The censored window {client:#x} was closed.")
            self.remove_target(client)
            if self._on_target_closed is not None:
                self._on_target_closed(client)
            return
        if not state.visible or state.area[2] <= 0 or state.area[3] <= 0:
            window.hide()
            return
        rect = QRect(*state.area)
        if window.geometry() != rect:
            window.setGeometry(rect)
        if not window.isVisible():
            window.show()
            window.setGeometry(rect)  # the window manager may have placed it elsewhere when mapping it

    def close(self):
        with self._lock:
            self._closed = True  # ignore frames that are still on their way
            targets, self._targets = list(self._targets.values()), {}
        for target in targets:
            target.close()
