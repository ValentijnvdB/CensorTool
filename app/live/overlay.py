"""
Overlay mode: censor a window in place.

The target window is captured (also while covered), censored by the live loop and shown in an overlay exactly over
it. The overlay ignores input, so the target can be used as usual. A tray icon can select another window, reload
the configs and quit. Linux with X11 only; needs PySide6 (requirements-overlay.txt).

The target is clicked, given by id, or the window of a program that is started for it (see launch.py). While
such a program starts, an optional curtain covers the screens, so its window is never visible uncensored.

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

# the environment before _prepare_qt_environment changed it: programs started for overlay mode get this one
_original_environment: dict[str, str] | None = None


def _prepare_qt_environment():
    """
    Settings that must be made before the QApplication is created.

    - Qt works in X pixels (no high-DPI scaling): the overlay must cover the target exactly, and fractional scale
      factors would leave a 1 px gap of uncensored window.
    - opencv-python points QT_QPA_PLATFORM_PLUGIN_PATH to its own Qt 5 plugins, which PySide6 (Qt 6) can't use.

    A started program gets the environment from before these changes (a Qt program would be affected by them).
    """
    global _original_environment
    if _original_environment is None:
        _original_environment = dict(os.environ)
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


# seconds a started program gets to open its window
LAUNCH_TIMEOUT = 30


def start_overlay_censor(window_id: int | None = None, censor_config: Path | str | None = None,
                         command: list[str] | None = None, curtain: bool = False):
    """
    Start overlay mode.

    :param window_id: the X11 id of the window to censor. If None (and no command), the user clicks it.
    :param censor_config: censor config file, None for the profile's.
    :param command: start this program and censor its window. The program keeps running when overlay mode stops.
    :param curtain: with command: cover the screens until the program's window is covered by the overlay.
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

    from PySide6.QtWidgets import QApplication
    qt = QApplication.instance() or QApplication(['CensorTool'])
    qt.setQuitOnLastWindowClosed(False)

    curtain_widget = None
    try:
        if command:
            if curtain:
                curtain_widget = Curtain(f"Starting {Path(command[0]).name}...")
            target, covers = _launch(command, qt, cover_existing=curtain_widget is None)
            curtain_widget = curtain_widget or covers
        elif window_id is None:
            print("Click the window to censor (right click or Escape cancels).")
            window_id = x11.pick_window()
            if window_id is None:
                logger.info("No window selected.")
                return
            target = _resolve_target(window_id)
        else:
            target = _resolve_target(window_id)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        if curtain_widget is not None:
            curtain_widget.close()
        return
    except BaseException:
        if curtain_widget is not None:
            curtain_widget.close()
        raise

    logger.info(f"Censoring window {target:#x}.")
    OverlayApp(target, censor_config, curtain=curtain_widget).run()


def _launch(command: list[str], qt, cover_existing: bool = True) -> tuple[int, Curtain | None]:
    """
    Start the program and return its window.

    A single-instance program hands its arguments to the instance that is already running, which shows them right
    away; that is only noticed when the started program exits. So the program's open windows are covered before it
    is started (cover_existing). The covers are returned when such a window turned out to be the target, to be
    removed once the overlay covers it; otherwise they are removed here.

    :return: (window, covers or None)
    """
    from ..config import CONFIG
    from . import launch

    conn = x11.Connection()
    covers = None
    try:
        before = conn.client_list()
        if cover_existing:
            covers = _cover_windows_of(conn, before, command)
        proc = launch.start_program(command, show_output=CONFIG.debug, env=_original_environment)
        logger.info(f"Started {command[0]} (pid {proc.pid}), waiting for its window...")
        try:
            target, how = launch.find_program_window(conn, proc, command, set(before), timeout=LAUNCH_TIMEOUT,
                                                     tick=qt.processEvents)
        except launch.LaunchError as e:
            raise RuntimeError(str(e)) from None
    except BaseException:
        if covers is not None:
            covers.close()
        raise
    finally:
        conn.close()
    launch.describe_match(how, command)
    if how != 'class-existing' and covers is not None:
        covers.close()
        covers = None
    return target, covers


def _cover_windows_of(conn: x11.Connection, windows: list[int], command: list[str]) -> Curtain | None:
    """Cover the visible normal windows that belong to the program of command (by WM_CLASS)."""
    from . import launch

    names = launch.program_names(command)
    areas = []
    for win in windows:
        info = conn.window_info(win)
        if not info.normal or not launch.class_matches(names, info.wm_class):
            continue
        attrs = conn.attributes(win)
        if attrs is not None and attrs.map_state == x11.IS_VIEWABLE:
            areas.append(conn.visible_client_area(win, attrs)[0])
    if not areas:
        return None
    logger.debug(f"Covering {len(areas)} open window(s) of {command[0]} while it starts.")
    return Curtain(f"Starting {Path(command[0]).name}...", areas)


class Curtain:
    """
    Opaque windows over all screens (or the given areas), above everything else, that let clicks through. They hide
    the screens while a program starts, until its window is covered by the overlay.
    """

    def __init__(self, text: str, areas: list[tuple[int, int, int, int]] | None = None):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QFont, QPainter
        from PySide6.QtWidgets import QApplication, QWidget

        class Cover(QWidget):
            def paintEvent(self, _event):
                painter = QPainter(self)
                painter.fillRect(self.rect(), QColor(24, 24, 24))
                painter.setPen(QColor(200, 200, 200))
                painter.setFont(QFont('Sans', 16))
                painter.drawText(self.rect(), Qt.AlignCenter, text)

        flags = (Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.X11BypassWindowManagerHint
                 | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus | Qt.Tool)
        self.covers = []
        if areas is None:
            areas = [screen.geometry().getRect() for screen in QApplication.instance().screens()]
        for x, y, w, h in areas:
            cover = Cover(None, flags)
            cover.setAttribute(Qt.WA_ShowWithoutActivating)
            cover.setGeometry(x, y, w, h)
            cover.show()
            self.covers.append(cover)
        # make sure the covers are on screen before anything else happens
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)

    def close(self):
        for cover in self.covers:
            cover.close()
        self.covers = []


class OverlayApp:

    # the curtain is removed this long after the overlay is shown, so the compositor has drawn the overlay
    CURTAIN_DELAY_MS = 150
    # a curtain is removed after this long anyway (e.g. the program started minimised: nothing to cover)
    CURTAIN_TIMEOUT_S = 10

    def __init__(self, target: int, censor_config: Path | None, curtain: Curtain | None = None):
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
        self.curtain = curtain
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

    def _close_curtain(self):
        if self.curtain is not None:
            self.curtain.close()
            self.curtain = None

    def quit(self):
        self.stop_event.set()
        self.qt.quit()

    def run(self):
        from PySide6.QtCore import QTimer

        worker = threading.Thread(target=self._run_loop, name='overlay-live-loop', daemon=True)
        worker.start()

        curtain_timer = None
        if self.curtain is not None:
            started = time.monotonic()

            def lift_curtain():
                if self.curtain is None:
                    return
                if self.overlay.window.isVisible() or time.monotonic() - started > self.CURTAIN_TIMEOUT_S:
                    curtain_timer.stop()
                    QTimer.singleShot(self.CURTAIN_DELAY_MS, self._close_curtain)

            curtain_timer = QTimer()
            curtain_timer.timeout.connect(lift_curtain)
            curtain_timer.start(20)

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
            self._close_curtain()
            if self.tray is not None:
                self.tray.hide()
            self.overlay.close()
            self.capture.close()
        if self._error is not None and not isinstance(self._error, KeyboardInterrupt):
            raise self._error
