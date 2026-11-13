"""
Overlay mode: censor windows in place.

Each censored (target) window is captured (also while covered), censored by the live loop and shown in an overlay
exactly over it. The overlays ignore input, so the windows can be used as usual. A tray icon can add windows, stop
censoring one, reload the configs and quit. Linux with X11 only; needs PySide6 (requirements-overlay.txt).

The first windows are clicked, given by id, or the window of a program that is started for it (see launch.py);
windows that program opens later are added automatically. While such a program starts, an optional curtain
covers the screens, so its window is never visible uncensored.

All windows share one live loop and pipeline: they take turns, and the focused window gets the most turns
(quick.TurnScheduler). At most live_max_windows (config.yml) windows are censored.

Threads: Qt runs in the main thread, the live loop (app/live/quick.py) in a worker thread, the live loop captures
each window in a capture thread of its own; window picking, following the focus and watching for a started
program's windows run in threads of their own.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from . import x11
from .quick import Sources

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


def start_overlay_censor(window_ids: list[int] | None = None, censor_config: Path | str | None = None,
                         command: list[str] | None = None, curtain: bool = False):
    """
    Start overlay mode.

    :param window_ids: the X11 ids of the windows to censor. If None or empty (and no command), the user clicks one.
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
    started = None
    try:
        if command:
            if curtain:
                curtain_widget = Curtain(f"Starting {Path(command[0]).name}...")
            started = _launch(command, qt, cover_existing=curtain_widget is None)
            curtain_widget = curtain_widget or started.covers
            targets = [started.window]
        elif not window_ids:
            print("Click the window to censor (right click or Escape cancels).")
            window_id = x11.pick_window()
            if window_id is None:
                logger.info("No window selected.")
                targets = []  # keeps running in the tray (it stops right away when there is no tray)
            else:
                targets = [_resolve_target(window_id)]
        else:
            targets = list(dict.fromkeys(_resolve_target(w) for w in window_ids))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        if curtain_widget is not None:
            curtain_widget.close()
        return
    except BaseException:
        if curtain_widget is not None:
            curtain_widget.close()
        raise

    # started with --window or a command: stop when those windows are closed. Picked: keep running in the tray.
    quit_when_empty = bool(command or window_ids)
    OverlayApp(targets, censor_config, curtain=curtain_widget, started=started,
               quit_when_empty=quit_when_empty).run()


@dataclass
class StartedProgram:
    """A program started for overlay mode (python main.py overlay -- <command>)."""
    command: list[str]
    proc: subprocess.Popen
    window: int  # its window that was found
    how: str  # how it was found, see launch.Match
    before: set[int]  # windows that existed before it was started
    covers: Curtain | None  # covers over its windows that existed already (see _launch), or None


def _launch(command: list[str], qt, cover_existing: bool = True) -> StartedProgram:
    """
    Start the program and return its window.

    A single-instance program hands its arguments to the instance that is already running, which shows them right
    away; that is only noticed when the started program exits. So the program's open windows are covered before it
    is started (cover_existing). The covers are returned when such a window turned out to be the target, to be
    removed once the overlay covers it; otherwise they are removed here.
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
    return StartedProgram(command, proc, target, how, set(before), covers)


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


class WindowSources(Sources):
    """The censored windows as sources of the live loop, by window id. The focused one has priority."""

    def __init__(self):
        from .input_devices.window_capture_device import WindowCaptureDevice
        self._make_device = WindowCaptureDevice
        self._lock = threading.Lock()
        self._devices: dict[int, object] = {}
        self._removed: list = []  # devices the live loop may still use; closed at the end
        self.focus = x11.ActiveWindowTracker()
        self.focus.start()

    def add(self, client: int):
        with self._lock:
            if client not in self._devices:
                self._devices[client] = self._make_device(client)

    def remove(self, client: int):
        """Stop capturing the window. The live loop stops its capture thread, which releases what it used."""
        with self._lock:
            device = self._devices.pop(client, None)
            if device is not None:
                self._removed.append(device)

    def devices(self) -> dict:
        with self._lock:
            return dict(self._devices)

    def priority(self) -> int | None:
        return self.focus.active

    def close(self):
        self.focus.stop()
        with self._lock:
            devices = list(self._devices.values()) + self._removed
            self._devices, self._removed = {}, []
        for device in devices:
            device.close()


class OverlayApp:

    # the curtain is removed this long after an overlay is shown, so the compositor has drawn the overlay
    CURTAIN_DELAY_MS = 150
    # a curtain is removed after this long anyway (e.g. the program started minimised: nothing to cover)
    CURTAIN_TIMEOUT_S = 10

    def __init__(self, targets: list[int], censor_config: Path | None, curtain: Curtain | None = None,
                 started: StartedProgram | None = None, quit_when_empty: bool = True):
        """
        :param targets: the windows to censor first.
        :param started: the program that was started for overlay mode: its new windows are censored too.
        :param quit_when_empty: stop when no censored window is left (else keep running in the tray).
        """
        from PySide6.QtWidgets import QApplication

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
        self.quit_when_empty = quit_when_empty
        self.stop_event = threading.Event()
        self._picking = threading.Lock()
        self._error: BaseException | None = None

        self.overlay = OverlayDevice(on_target_closed=self._target_closed)
        self.sources = WindowSources()
        self._bridge = self._make_bridge()
        self.tray = self._make_tray()
        for target in targets:
            self.add_window(target)

        self.watcher = None
        if started is not None:
            from .launch import ProgramWindowWatcher
            self._program_name = Path(started.command[0]).name
            self.watcher = ProgramWindowWatcher(started.proc, started.command, started.how, started.window,
                                                known=started.before, on_new=self._bridge.discovered.emit)
            self.watcher.start()

    def _make_bridge(self):
        from PySide6.QtCore import QObject, Qt, Signal, Slot

        app = self

        class Bridge(QObject):
            picked = Signal(object)  # window ids don't fit a signed 32-bit int
            discovered = Signal(object)
            quit_requested = Signal()

            def __init__(self):
                super().__init__()
                self.picked.connect(self._picked, Qt.QueuedConnection)
                self.discovered.connect(self._discovered, Qt.QueuedConnection)
                self.quit_requested.connect(self._quit, Qt.QueuedConnection)

            @Slot(object)
            def _picked(self, client):
                app.add_window(client, picked=True)

            @Slot(object)
            def _discovered(self, client):
                logger.info(f"{app._program_name} opened window {client:#x}.")
                app.add_window(client)

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
        self._actions = []

        def add_action(target_menu, text, callback):
            action = QAction(text, target_menu)
            action.triggered.connect(callback)
            target_menu.addAction(action)
            self._actions.append(action)

        add_action(menu, 'Add window...', self._start_pick)
        self._stop_menu = menu.addMenu('Stop censoring')
        self._stop_menu.aboutToShow.connect(self._fill_stop_menu)
        add_action(menu, 'Reload configs', lambda: self.overlay.add_command('reload'))
        menu.addSeparator()
        add_action(menu, 'Quit', self.quit)
        self._menu = menu

        tray = QSystemTrayIcon(QIcon(pixmap))
        tray.setToolTip('CensorTool overlay')
        tray.setContextMenu(menu)
        tray.activated.connect(lambda reason: menu.popup(tray.geometry().center())
                               if reason == QSystemTrayIcon.Trigger else None)
        tray.show()
        return tray

    def _fill_stop_menu(self):
        from PySide6.QtGui import QAction

        self._stop_menu.clear()
        targets = self.overlay.targets
        if not targets:
            action = QAction('(no windows)', self._stop_menu)
            action.setEnabled(False)
            self._stop_menu.addAction(action)
            return
        conn = x11.Connection()
        try:
            titles = {t: conn.title(t) for t in targets}
        finally:
            conn.close()
        for target in targets:
            title = titles[target] or 'untitled'
            if len(title) > 60:
                title = title[:57] + '...'
            action = QAction(f"{title} ({target:#x})", self._stop_menu)
            action.triggered.connect(lambda _checked=False, t=target: self.remove_window(t))
            self._stop_menu.addAction(action)

    # ---------------------------------------------------------------- adding and removing windows

    def add_window(self, client: int, picked: bool = False):
        from ..config import CONFIG

        if client in self.overlay.targets:
            if picked:
                logger.info(f"Window {client:#x} is censored already.")
            return
        limit = CONFIG.live.max_windows
        if limit and len(self.overlay.targets) >= limit:
            logger.warning(f"Not censoring window {client:#x}: {limit} windows are censored already "
                           f"(live_max_windows in config.yml).")
            return
        logger.info(f"Censoring window {client:#x}.")
        # the overlay is shown before anything is captured, so the window is never visible uncensored. Frames
        # captured before now are of an earlier time this window was censored.
        self.overlay.add_target(client, since=time.monotonic())
        self.sources.add(client)

    def remove_window(self, client: int):
        logger.info(f"Stopped censoring window {client:#x}.")
        self.sources.remove(client)
        self.overlay.remove_target(client)
        self._check_empty()

    def _target_closed(self, client: int):
        self.sources.remove(client)
        self._check_empty()

    def _check_empty(self):
        if self.overlay.targets:
            return
        if self.quit_when_empty or self.tray is None:
            self.quit()
        else:
            logger.info("No windows are censored; add one from the tray icon.")

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
                    logger.info("No window selected.")
            except Exception as e:
                logger.error(f"Selecting a window failed: {e}")
            finally:
                self._picking.release()

        threading.Thread(target=pick, name='overlay-pick', daemon=True).start()

    # ---------------------------------------------------------------- running

    def _run_loop(self):
        from .quick import live_censor_loop
        try:
            live_censor_loop(self.stop_event, self.sources, self.overlay, self.censor_config)
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

        if not self.overlay.targets:  # e.g. the windows were closed meanwhile
            self._check_empty()
            if self.stop_event.is_set():
                self._cleanup()
                return

        worker = threading.Thread(target=self._run_loop, name='overlay-live-loop', daemon=True)
        worker.start()

        curtain_timer = None
        if self.curtain is not None:
            started = time.monotonic()

            def lift_curtain():
                if self.curtain is None:
                    return
                if self.overlay.any_visible() or time.monotonic() - started > self.CURTAIN_TIMEOUT_S:
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
            self._cleanup()
        if self._error is not None and not isinstance(self._error, KeyboardInterrupt):
            raise self._error

    def _cleanup(self):
        if self.watcher is not None:
            self.watcher.stop()
        self._close_curtain()
        if self.tray is not None:
            self.tray.hide()
        self.overlay.close()
        self.sources.close()
