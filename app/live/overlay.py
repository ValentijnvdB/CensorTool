"""
Overlay mode: censor a window in place.

The target window is captured (also while covered), censored by the live loop and shown in an overlay exactly over
it. The overlay ignores input, so the target can be used as usual. A tray icon can select another window, reload
the configs and quit. Linux with X11 only; needs PySide6 (requirements-overlay.txt).

Threads: Qt runs in the main thread, the live loop (app/live/quick.py) in a worker thread, the live loop captures
in its own capture thread and window picking runs in a thread of its own.

Only one window for now. For more windows, the app would keep one capture device and overlay per target and
the live loop would have to tell which target each censored frame belongs to.
"""
from __future__ import annotations

import os
import signal
import threading
import time
import traceback
from pathlib import Path

from loguru import logger

from . import x11

QT_MISSING = "Overlay mode needs PySide6. Install it with: pip install -r requirements-overlay.txt"


def _prepare_qt_environment():
    """
    Settings that must be made before the QApplication is created.

    - Qt works in X pixels (no high-DPI scaling): the overlay must cover the target exactly, and fractional scale
      factors would leave a 1 px gap of uncensored window.
    - opencv-python points QT_QPA_PLATFORM_PLUGIN_PATH to its own Qt 5 plugins, which PySide6 (Qt 6) can't use.
    """
    os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    for name in ('QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS', 'QT_AUTO_SCREEN_SCALE_FACTOR'):
        os.environ.pop(name, None)
    for name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if 'cv2' in os.environ.get(name, ''):
            os.environ.pop(name)


def _resolve_target(window_id: int) -> int:
    """The application window for a window id (a WM frame id is turned into the window inside it)."""
    conn = x11.Connection()
    try:
        if conn.attributes(window_id) is None:
            raise ValueError(f"There is no window with id {window_id:#x}.")
        client = conn.client_of(conn.toplevel(window_id))
        return window_id if client is None or window_id == client else client
    finally:
        conn.close()


def start_overlay_censor(window_id: int | None = None, censor_config: Path | str | None = None):
    """
    Start overlay mode.

    :param window_id: the X11 id of the window to censor. If None, the user clicks it.
    :param censor_config: censor config file, None for the profile's.
    """
    problem = x11.session_problem()
    if problem:
        raise RuntimeError(problem)
    try:
        import PySide6  # noqa: F401
    except ImportError:
        raise RuntimeError(QT_MISSING) from None

    _prepare_qt_environment()
    if isinstance(censor_config, str):
        censor_config = Path(censor_config)

    if window_id is None:
        print("Click the window to censor (right click or Escape cancels).")
        window_id = x11.pick_window()
        if window_id is None:
            logger.info("No window selected.")
            return
    target = _resolve_target(window_id)
    logger.info(f"Censoring window {target:#x}.")

    OverlayApp(target, censor_config).run()


class OverlayApp:

    def __init__(self, target: int, censor_config: Path | None):
        from PySide6.QtWidgets import QApplication

        from .input_devices.window_capture_device import WindowCaptureDevice
        from .output_devices.overlay_device import OverlayDevice

        self.qt = QApplication.instance() or QApplication(['CensorTool'])
        self.qt.setQuitOnLastWindowClosed(False)
        x11.install_error_handler()  # in case Qt replaced it
        for screen in self.qt.screens():
            if screen.devicePixelRatio() != 1:
                logger.warning(f"Screen {screen.name()} has a device pixel ratio of {screen.devicePixelRatio()}; "
                               f"the overlay may not line up with the window.")

        self.censor_config = censor_config
        self.stop_event = threading.Event()
        self._picking = threading.Lock()
        self._error: BaseException | None = None

        # the overlay is shown before anything is captured, so the window is never visible uncensored
        self.overlay = OverlayDevice(target, on_target_closed=self.quit)
        self.capture = WindowCaptureDevice(target)
        self._bridge = self._make_bridge()
        self.tray = self._make_tray()

    def _make_bridge(self):
        from PySide6.QtCore import QObject, Qt, Signal, Slot

        app = self

        class Bridge(QObject):
            picked = Signal(int)
            quit_requested = Signal()

            def __init__(self):
                super().__init__()
                self.picked.connect(self._picked, Qt.QueuedConnection)
                self.quit_requested.connect(self._quit, Qt.QueuedConnection)

            @Slot(int)
            def _picked(self, client: int):
                app._switch_target(client)

            @Slot()
            def _quit(self):
                app.quit()

        return Bridge()

    def _make_tray(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
        from PySide6.QtWidgets import QMenu, QSystemTrayIcon

        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("No system tray found: stop overlay mode with Ctrl+C in the terminal.")
            return None

        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(168, 93, 253))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.setBrush(QColor(20, 20, 20))
        painter.drawRect(14, 26, 36, 12)  # a censor bar
        painter.end()

        menu = QMenu()
        actions = [('Select another window', self._start_pick),
                   ('Reload configs', lambda: self.overlay.add_command('reload')),
                   (None, None),
                   ('Quit', self.quit)]
        self._actions = []
        for text, callback in actions:
            if text is None:
                menu.addSeparator()
                continue
            action = QAction(text, menu)
            action.triggered.connect(callback)
            menu.addAction(action)
            self._actions.append(action)
        self._menu = menu

        tray = QSystemTrayIcon(QIcon(pixmap))
        tray.setToolTip('CensorTool overlay')
        tray.setContextMenu(menu)
        tray.activated.connect(lambda reason: menu.popup(tray.geometry().center())
                               if reason == QSystemTrayIcon.Trigger else None)
        tray.show()
        return tray

    # ---------------------------------------------------------------- selecting another window

    def _start_pick(self):
        if not self._picking.acquire(blocking=False):
            return  # already picking
        logger.info("Click the window to censor (right click or Escape cancels).")

        def pick():
            try:
                client = x11.pick_window(exclude=self.overlay.overlay_window_ids(), cancel=self.stop_event)
                if client is not None:
                    self._bridge.picked.emit(client)
                else:
                    logger.info("No window selected, keeping the current one.")
            except Exception as e:
                logger.error(f"Selecting a window failed: {e}")
            finally:
                self._picking.release()

        threading.Thread(target=pick, name='overlay-pick', daemon=True).start()

    def _switch_target(self, client: int):
        if client == self.overlay.client:
            return
        logger.info(f"Censoring window {client:#x}.")
        # the capture device switches first, so frames captured after `since` are of the new window
        self.capture.set_target(client)
        since = time.monotonic()
        self.overlay.set_target(client, since=since)

    # ---------------------------------------------------------------- running

    def _run_loop(self):
        from .quick import live_censor_loop
        try:
            live_censor_loop(self.stop_event, self.capture, self.overlay, self.censor_config)
        except BaseException as e:
            self._error = e
            logger.error(f"Overlay censoring stopped: {e}")
            logger.debug(traceback.format_exc())
        finally:
            self._bridge.quit_requested.emit()

    def quit(self):
        self.stop_event.set()
        self.qt.quit()

    def run(self):
        from PySide6.QtCore import QTimer

        worker = threading.Thread(target=self._run_loop, name='overlay-live-loop', daemon=True)
        worker.start()

        # Ctrl+C in the terminal: Python only handles signals while it runs, so wake it up regularly
        previous_handler = signal.signal(signal.SIGINT, lambda *_: self.quit())
        wake = QTimer()
        wake.timeout.connect(lambda: None)
        wake.start(200)
        try:
            self.qt.exec()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            self.stop_event.set()
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning("The live loop did not stop in time.")
            if self.tray is not None:
                self.tray.hide()
            self.overlay.close()
            self.capture.close()
        if self._error is not None and not isinstance(self._error, KeyboardInterrupt):
            raise self._error
