"""
Start a program and find the window it opens, for `python main.py overlay -- <command>`.

A window belongs to the program when its _NET_WM_PID is the started process or one of its children. Programs
that don't set it correctly (Flatpak, Snap) are matched by WM_CLASS against the program's name. Single-instance
programs (Celluloid, Firefox, ...) hand their arguments to an instance that is already running and exit; their
existing window is then used, also matched by WM_CLASS.

The matching (WindowInfo + program names) is meant to be reused by an automatic mode that attaches overlays to
windows of certain programs.
"""
from __future__ import annotations

import os
import select
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Literal

from loguru import logger

from . import x11

Match = Literal['pid', 'class-new', 'class-existing']

# words in front of the actual program, e.g. "flatpak run io.github.celluloid_player.Celluloid"
WRAPPERS = {'flatpak', 'snap', 'run', 'env', 'nice', 'nohup', 'setsid', 'gtk-launch', 'gtk4-launch', 'exec'}

# how long a program that exited (handed its arguments to a running instance) gets to show a window of its own
EXITED_GRACE = 2.0


class LaunchError(RuntimeError):
    pass


def program_names(command: list[str]) -> set[str]:
    """Lowercase names the program's windows may have as WM_CLASS: the program, after any wrapper commands."""
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


def class_matches(names: Iterable[str], wm_class: Iterable[str]) -> bool:
    """Whether WM_CLASS (instance, class) belongs to a program with one of these names."""
    wanted = set().union(*(_name_variants(n) for n in names)) if names else set()
    return any(_name_variants(c) & wanted for c in wm_class)


def parent_map() -> dict[int, int]:
    """pid -> parent pid of every process (Linux /proc)."""
    parents = {}
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit():
            continue
        try:
            with open(f'/proc/{entry.name}/stat', 'rb') as f:
                stat = f.read()
        except OSError:
            continue
        # the command name (2nd field) can contain spaces and parentheses, so split after its closing ')'
        fields = stat[stat.rfind(b')') + 2:].split()
        if len(fields) >= 2:
            parents[int(entry.name)] = int(fields[1])
    return parents


def process_tree(pid: int, parents: dict[int, int]) -> set[int]:
    """pid and all its descendants."""
    children: dict[int, list[int]] = {}
    for child, parent in parents.items():
        children.setdefault(parent, []).append(child)
    tree, todo = set(), [pid]
    while todo:
        p = todo.pop()
        if p not in tree:
            tree.add(p)
            todo.extend(children.get(p, ()))
    return tree


def choose_window(new: list[x11.WindowInfo], pids: set[int], names: set[str]) -> tuple[int, Match] | None:
    """
    The window of the program among windows that appeared after it was started (oldest first): one made by its
    processes, else one whose WM_CLASS matches. Normal windows go before dialogs, splash screens etc.
    """
    for normal_only in (True, False):
        for info in new:
            if info.pid in pids and (info.normal or not normal_only):
                return info.window, 'pid'
    for normal_only in (True, False):
        for info in new:
            if class_matches(names, info.wm_class) and (info.normal or not normal_only):
                return info.window, 'class-new'
    return None


def choose_existing_window(existing: list[x11.WindowInfo], names: set[str]) -> int | None:
    """The topmost (last) normal window of the program among windows that were open before it was started."""
    for info in reversed(existing):
        if info.normal and class_matches(names, info.wm_class):
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


def find_program_window(conn: x11.Connection, proc: subprocess.Popen, command: list[str], before: set[int],
                        timeout: float, tick: Callable[[], None] | None = None) -> tuple[int, Match]:
    """
    Wait for the window of a started program.

    :param conn: an X connection (of the calling thread).
    :param proc: the started program.
    :param before: the windows (_NET_CLIENT_LIST) that existed before the program was started.
    :param timeout: seconds to wait.
    :param tick: called regularly while waiting (e.g. to keep a GUI responsive).
    :return: (window, how it was matched).
    :raises LaunchError: the program exited without a window of its own or of a running instance, or timed out.
    """
    names = program_names(command)
    x = conn.x
    x.XSelectInput(conn.dpy, conn.root, x11.PROPERTY_CHANGE_MASK)  # _NET_CLIENT_LIST changes
    x.XFlush(conn.dpy)
    fd = x.XConnectionNumber(conn.dpy)
    event = x11.XEvent()
    infos: dict[int, x11.WindowInfo] = {}

    def info(win: int) -> x11.WindowInfo:
        cached = infos.get(win)
        if cached is not None:
            return cached
        result = conn.window_info(win)
        if result.pid is not None or result.wm_class:  # not complete yet otherwise: read it again next time
            infos[win] = result
        return result

    deadline = time.monotonic() + timeout
    exited_at = None
    try:
        while True:
            if tick is not None:
                tick()
            clients = conn.client_list()
            new = [info(w) for w in clients if w not in before]
            found = choose_window(new, process_tree(proc.pid, parent_map()), names)
            if found is not None:
                return found

            now = time.monotonic()
            if exited_at is None and proc.poll() is not None:
                exited_at = now
            if exited_at is not None and now - exited_at >= EXITED_GRACE:
                stacking = conn.client_list('_NET_CLIENT_LIST_STACKING') or clients
                existing = choose_existing_window([info(w) for w in stacking if w in before], names)
                if existing is not None:
                    return existing, 'class-existing'
                raise LaunchError(f"{command[0]} exited (code {proc.returncode}) without opening a window.")
            if now >= deadline:
                raise LaunchError(f"{command[0]} did not open a window within {timeout:.0f} s.")

            select.select([fd], [], [], 0.02)
            while x.XPending(conn.dpy):
                x.XNextEvent(conn.dpy, event)
    finally:
        x.XSelectInput(conn.dpy, conn.root, 0)
        x.XFlush(conn.dpy)


def describe_match(how: Match, command: list[str]) -> None:
    """Log how the window was found; warn when it is not a window of its own."""
    program = command[0]
    if how == 'class-new':
        logger.info(f"The window of {program} was recognised by its window class (not by its process).")
    elif how == 'class-existing':
        logger.warning(f"{program} handed its arguments to an instance that was already running; censoring that "
                       f"instance's window. For a separate window, use the program's own option if it has one "
                       f"(e.g. celluloid --new-window).")
