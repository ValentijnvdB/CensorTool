"""
Start a program and find the window it opens, for `python main.py overlay -- <command>`.

A window belongs to the program when its process (X11: _NET_WM_PID) is the started process or one of its
children. Programs that don't report it correctly (Flatpak, Snap) are matched by their window class (app_ids, X11:
WM_CLASS) against the program's name. Single-instance programs (Celluloid, Firefox, ...) hand their arguments to an
instance that is already running and exit; their existing window is then used, also matched by window class.

Windows the program opens later (dialogs, more windows) are found the same way by ProgramWindowWatcher. Auto mode
(auto.py) uses the same class matching.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Literal

from loguru import logger

from . import windowing
from .windowing import WindowInfo
from .windowing.processes import parent_map, process_tree

Match = Literal['pid', 'class-new', 'class-existing']

# words in front of the actual program, e.g. "flatpak run io.github.celluloid_player.Celluloid"
WRAPPERS = {'flatpak', 'snap', 'run', 'env', 'nice', 'nohup', 'setsid', 'gtk-launch', 'gtk4-launch', 'exec'}

# how long a program that exited (handed its arguments to a running instance) gets to show a window of its own
EXITED_GRACE = 2.0


class LaunchError(RuntimeError):
    pass


def program_names(command: list[str]) -> set[str]:
    """Lowercase names the program's windows may have as window class: the program, after any wrapper commands."""
    names = set()
    for token in command:
        if token.startswith('-') or '=' in token:  # options and environment assignments (env FOO=1 prog)
            continue
        name = Path(token).name.lower()
        if not name:
            continue
        names.add(name)
        if name not in WRAPPERS:
            break
    return names - WRAPPERS


def _name_variants(name: str) -> set[str]:
    """A name and its last dotted part: io.github.celluloid_player.Celluloid -> also celluloid."""
    name = name.lower()
    return {name, name.rsplit('.', 1)[-1]}


def class_matches(names: Iterable[str], app_ids: Iterable[str]) -> bool:
    """Whether a window's app_ids (X11: WM_CLASS, (instance, class)) belong to a program with one of these names."""
    wanted = set().union(*(_name_variants(n) for n in names)) if names else set()
    return any(_name_variants(c) & wanted for c in app_ids)


def choose_window(new: list[WindowInfo], pids: set[int], names: set[str]) -> tuple[int, Match] | None:
    """
    The window of the program among windows that appeared after it was started (oldest first): one made by its
    processes, else one whose window class matches. Normal windows go before dialogs, splash screens etc.
    """
    for normal_only in (True, False):
        for info in new:
            if info.pid in pids and (info.normal or not normal_only):
                return info.window, 'pid'
    for normal_only in (True, False):
        for info in new:
            if class_matches(names, info.app_ids) and (info.normal or not normal_only):
                return info.window, 'class-new'
    return None


def choose_existing_window(existing: list[WindowInfo], names: set[str]) -> int | None:
    """The topmost (last) normal window of the program among windows that were open before it was started."""
    for info in reversed(existing):
        if info.normal and class_matches(names, info.app_ids):
            return info.window
    return None


def start_program(command: list[str], show_output: bool = False,
                  env: dict[str, str] | None = None) -> subprocess.Popen:
    """
    Start the program in its own session, so Ctrl+C in the terminal and quitting overlay mode leave it running.

    :param env: its environment, None for this process's.
    """
    output = None if show_output else subprocess.DEVNULL
    try:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                start_new_session=True, env=env)
    except FileNotFoundError:
        raise LaunchError(f"Program not found: {command[0]}") from None
    except PermissionError:
        raise LaunchError(f"Program can't be run: {command[0]}") from None


def find_program_window(monitor: windowing.WindowMonitor, proc: subprocess.Popen, command: list[str],
                        before: set[int], timeout: float, tick: Callable[[], None] | None = None) -> tuple[int, Match]:
    """
    Wait for the window of a started program.

    :param monitor: a window monitor (of the calling thread).
    :param proc: the started program.
    :param before: the windows (Display.window_list) that existed before the program was started.
    :param timeout: seconds to wait.
    :param tick: called regularly while waiting (e.g. to keep a GUI responsive).
    :return: (window, how it was matched).
    :raises LaunchError: the program exited without a window of its own or of a running instance, or timed out.
    """
    names = program_names(command)
    display = monitor.display
    infos: dict[int, WindowInfo] = {}

    def info(win: int) -> WindowInfo:
        cached = infos.get(win)
        if cached is not None:
            return cached
        result = display.window_info(win)
        if result.pid is not None or result.app_ids:  # not complete yet otherwise: read it again next time
            infos[win] = result
        return result

    deadline = time.monotonic() + timeout
    exited_at = None
    while True:
        if tick is not None:
            tick()
        new = [info(w) for w in display.window_list() if w not in before]
        found = choose_window(new, process_tree(proc.pid, parent_map()), names)
        if found is not None:
            return found

        now = time.monotonic()
        if exited_at is None and proc.poll() is not None:
            exited_at = now
        if exited_at is not None and now - exited_at >= EXITED_GRACE:
            stacking = display.window_list(stacking=True)
            existing = choose_existing_window([info(w) for w in stacking if w in before], names)
            if existing is not None:
                return existing, 'class-existing'
            raise LaunchError(f"{command[0]} exited (code {proc.returncode}) without opening a window.")
        if now >= deadline:
            raise LaunchError(f"{command[0]} did not open a window within {timeout:.0f} s.")
        monitor.wait(0.02)


class ProgramWindowWatcher(threading.Thread):
    """
    Reports the windows a started program opens after its first one: windows of its processes, or (when its first
    window was recognised by its window class) windows with that class. on_new is called from this thread.
    """

    # a window whose process and window class are still not set after this long is ignored
    INCOMPLETE_GRACE = 2.0

    def __init__(self, proc: subprocess.Popen, command: list[str], how: Match, first_window: int, known: set[int],
                 on_new: Callable[[int], None]):
        """
        :param first_window: the program's window that was found by find_program_window.
        :param known: windows that are not the program's new windows (those that existed before it was started).
        """
        super().__init__(name='program-window-watcher', daemon=True)
        self._proc = proc
        self._names = program_names(command)
        self._by_class = how != 'pid'
        self._first_window = first_window
        self._known = set(known) | {first_window}
        self._on_new = on_new
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    def belongs(self, info: WindowInfo, pids: set[int]) -> bool:
        return info.pid in pids or (self._by_class and class_matches(self._names, info.app_ids))

    def run(self):
        try:
            monitor = windowing.get().monitor()
        except windowing.WindowSystemError as e:
            logger.error(f"Watching for new windows failed: {e}")
            return
        try:
            display = monitor.display
            roots = {self._proc.pid}  # processes whose descendants' windows belong to the program
            first_pid = display.window_info(self._first_window).pid
            if first_pid is not None:
                roots.add(first_pid)
            first_seen: dict[int, float] = {}
            last = 0.0
            while not self._stop_event.is_set():
                changes = monitor.wait(0.1)
                now = time.monotonic()
                if changes.list_changed or changes.windows or now - last > 0.5:
                    last = now
                    self._check(display, roots, first_seen, now)
        except Exception as e:
            logger.error(f"Watching for new windows failed: {e}")
        finally:
            monitor.close()

    def _check(self, display: windowing.Display, roots: set[int], first_seen: dict[int, float], now: float):
        new = [w for w in display.window_list() if w not in self._known]
        if not new:
            return
        parents = parent_map()
        pids = set().union(*(process_tree(r, parents) for r in roots))
        for win in new:
            info = display.window_info(win)
            if self.belongs(info, pids):
                self._known.add(win)
                if info.pid is not None:
                    roots.add(info.pid)
                self._on_new(win)
            elif info.pid is not None or info.app_ids or now - first_seen.setdefault(win, now) > self.INCOMPLETE_GRACE:
                self._known.add(win)  # someone else's window (or one that never says whose it is)


def describe_match(how: Match, command: list[str]) -> None:
    """Log how the window was found; warn when it is not a window of its own."""
    program = command[0]
    if how == 'class-new':
        logger.info(f"The window of {program} was recognised by its window class (not by its process).")
    elif how == 'class-existing':
        logger.warning(f"{program} handed its arguments to an instance that was already running; censoring that "
                       f"instance's window. For a separate window, use the program's own option if it has one "
                       f"(e.g. celluloid --new-window).")
