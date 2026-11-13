"""
Shows the censored frames in a window that lies exactly over the censored (target) window.

The overlay ignores mouse and keyboard input, so clicks and scrolling go to the target window below it. It is a
transient of the target, so the window manager keeps it directly above the target: other windows, menus and
panels still appear on top of it.

Needs PySide6 and an X11 session. Create it in the Qt (main) thread; push_frame and poll_command can be called
from any thread.
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

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowTransparentForInput
                         | Qt.WindowDoesNotAcceptFocus | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setWindowTitle('CensorTool overlay')
        self.frame: np.ndarray | None = None

    def paintEvent(self, _event):
        painter = QPainter(self)
        frame = self.frame
        if frame is None:
            painter.fillRect(self.rect(), self.PLACEHOLDER_COLOR)
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont('Sans', 14))
            painter.drawText(self.rect(), Qt.AlignCenter, 'CensorTool: censoring...')
            return
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        # stretched when the target was resized and no frame of the new size is censored yet
        painter.drawImage(self.rect(), image)


class _Bridge(QObject):
    """Runs callbacks in the Qt thread when its signals are emitted from other threads."""
    frame_ready = Signal()
    target_changed = Signal()

    def __init__(self, on_frame_ready: Callable[[], None], on_target_changed: Callable[[], None]):
        super().__init__()
        self._on_frame_ready = on_frame_ready
        self._on_target_changed = on_target_changed
        self.frame_ready.connect(self._frame_ready, Qt.QueuedConnection)
        self.target_changed.connect(self._target_changed, Qt.QueuedConnection)

    @Slot()
    def _frame_ready(self):
        self._on_frame_ready()

    @Slot()
    def _target_changed(self):
        self._on_target_changed()


class OverlayDevice(OutputDevice):

    def __init__(self, client: int, on_target_closed: Callable[[], None] | None = None):
        """
        :param client: the window to cover.
        :param on_target_closed: called in the Qt thread when the window no longer exists.
        """
        self.window = OverlayWindow()
        self.window.winId()  # create the native window, so it can be made a transient before it is shown
        self._on_target_closed = on_target_closed
        self._bridge = _Bridge(self._show_newest_frame, self._follow_target)

        self._lock = threading.Lock()
        self._newest: np.ndarray | None = None
        self._accept_since = float('-inf')  # frames captured before this belong to a previous target
        self._commands: deque[Command] = deque()

        self.client = 0
        self._tracker: x11.TargetTracker | None = None
        self._parent_qwindow: QWindow | None = None
        self._applied_version = -1
        self.set_target(client)

    # ---------------------------------------------------------------- any thread

    def push_frame(self, frame: np.ndarray, timestamp: float | None = None):
        with self._lock:
            if timestamp is not None and timestamp < self._accept_since:
                return
            self._newest = np.ascontiguousarray(frame)
        self._bridge.frame_ready.emit()

    def add_command(self, command: Command):
        self._commands.append(command)

    def poll_command(self) -> Command | None:
        try:
            return self._commands.popleft()
        except IndexError:
            return None

    # ---------------------------------------------------------------- Qt thread

    def set_target(self, client: int, since: float | None = None):
        """
        Cover another window. Until a frame captured at or after since (time.monotonic()) arrives, a placeholder
        is shown instead of frames of the previous window.
        """
        if self._tracker is not None:
            self._tracker.stop()
        with self._lock:
            self._newest = None
            self._accept_since = since if since is not None else float('-inf')
        self.window.frame = None
        self.window.hide()  # the window manager reads the transient parent when the window is shown
        self.client = client
        self._parent_qwindow = QWindow.fromWinId(client)
        self.window.windowHandle().setTransientParent(self._parent_qwindow)

        self._applied_version = -1
        self._tracker = x11.TargetTracker(client, on_change=self._bridge.target_changed.emit)
        self._tracker.start()
        self._tracker.wait_ready()
        self._follow_target()

    def overlay_window_ids(self) -> set[int]:
        return {int(self.window.winId())}

    def _show_newest_frame(self):
        with self._lock:
            frame, self._newest = self._newest, None
        if frame is not None:
            self.window.frame = frame
            self.window.update()

    def _follow_target(self):
        tracker = self._tracker
        if tracker is None:
            return
        state = tracker.state
        if state.version == self._applied_version:
            return
        self._applied_version = state.version
        if not state.alive:
            self.window.hide()
            logger.info(f"The censored window {tracker.client:#x} was closed.")
            if self._on_target_closed is not None:
                self._on_target_closed()
            return
        if not state.visible or state.area[2] <= 0 or state.area[3] <= 0:
            self.window.hide()
            return
        rect = QRect(*state.area)
        if self.window.geometry() != rect:
            self.window.setGeometry(rect)
        if not self.window.isVisible():
            self.window.show()
            self.window.setGeometry(rect)  # the window manager may have placed it elsewhere when mapping it

    def close(self):
        if self._tracker is not None:
            self._tracker.stop()
            self._tracker = None
        with self._lock:
            self._accept_since = float('inf')  # ignore frames that are still on their way
            self._newest = None
        self.window.hide()
