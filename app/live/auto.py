"""
Auto mode of overlay mode: censor every window of certain programs, also windows they open later.

Which windows: rules from overlay_auto_programs in config.yml, plus programs given with --auto or added from the
tray. A rule names a program (matched by WM_CLASS like launch.class_matches), a title (a regular expression,
case-insensitive), or both:

    overlay_auto_programs: [celluloid, mpv, {program: firefox, title: 'Private Browsing'}]

New windows are noticed as early as X11 allows: when the window manager starts managing them (WM_STATE is set),
which is about when they are shown. A frame of a new window can still be visible before its overlay covers it.
Titles are checked again when they change, so a window is censored as soon as its title matches; a censored
window stays censored when its title changes back.
"""
from __future__ import annotations

import ctypes
import os
import re
import select
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from loguru import logger

from . import x11
from .launch import class_matches

# window types that are never censored: panels and the desktop
IGNORED_TYPES = {'DOCK', 'DESKTOP'}


@dataclass(frozen=True)
class AutoRule:
    program: str | None = None  # matched against WM_CLASS
    title: str | None = None  # regular expression, searched in the title, case-insensitive

    def matches(self, wm_class: Iterable[str], title: str) -> bool:
        if self.program is not None and not class_matches({self.program}, wm_class):
            return False
        if self.title is not None and not re.search(self.title, title or '', re.IGNORECASE):
            return False
        return True

    def __str__(self):
        if self.title is None:
            return self.program
        return f"{self.program or 'any program'} with title /{self.title}/"


def parse_rules(entries, name: str = 'overlay_auto_programs') -> list[AutoRule]:
    """
    Rules from a config value: a list of program names and {program: ..., title: ...} mappings.

    :raises ValueError: if the value is not valid.
    """
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list, got {entries!r}.")
    rules = []
    for entry in entries:
        if isinstance(entry, str) and entry.strip():
            rules.append(AutoRule(program=entry.strip()))
            continue
        if isinstance(entry, dict) and entry and set(entry) <= {'program', 'title'}:
            program, title = entry.get('program'), entry.get('title')
            if all(v is None or (isinstance(v, str) and v.strip()) for v in (program, title)):
                if title is not None:
                    try:
                        re.compile(title)
                    except re.error as e:
                        raise ValueError(f"{name}: invalid title pattern {title!r}: {e}.") from None
                rules.append(AutoRule(program=program.strip() if program else None, title=title))
                continue
        raise ValueError(f"{name}: each entry must be a program name or a mapping with 'program' and/or 'title', "
                         f"got {entry!r}.")
    return rules


def program_name_of(wm_class: tuple[str, ...]) -> str | None:
    """The name to save for a window's program: its class without a reverse-domain prefix, in lowercase."""
    if not wm_class:
        return None
    return wm_class[-1].rsplit('.', 1)[-1].lower() or None


class AutoWatcher(threading.Thread):
    """
    Reports windows that match the rules (on_match, from this thread), each once, and application windows that are
    gone (on_gone). Windows of this process (the overlays) are skipped.
    """

    POLL_INTERVAL = 0.5

    def __init__(self, rules: Callable[[], list[AutoRule]], on_match: Callable[[int], None],
                 on_gone: Callable[[int], None] | None = None):
        super().__init__(name='overlay-auto-watcher', daemon=True)
        self._rules = rules
        self._on_match = on_match
        self._on_gone = on_gone
        self._reported: set[int] = set()
        self._watched: set[int] = set()  # windows whose properties are followed
        self._rescan = threading.Event()
        self._stop_event = threading.Event()
        self._pid = os.getpid()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    def rescan(self):
        """Report all matching windows again (e.g. auto mode was switched on again)."""
        self._rescan.set()

    def run(self):
        try:
            conn = x11.Connection()
        except x11.X11Error as e:
            logger.error(f"Auto mode can't watch for windows: {e}")
            return
        try:
            self._watch(conn)
        except Exception as e:
            logger.error(f"Auto mode stopped watching for windows: {e}")
        finally:
            conn.close()

    def _watch(self, conn: x11.Connection):
        x = conn.x
        x.XSelectInput(conn.dpy, conn.root, x11.SUBSTRUCTURE_NOTIFY_MASK | x11.PROPERTY_CHANGE_MASK)
        x.XFlush(conn.dpy)
        fd = x.XConnectionNumber(conn.dpy)
        event = x11.XEvent()
        names = {conn.atom(n) for n in ('WM_STATE', 'WM_CLASS', '_NET_WM_NAME', 'WM_NAME')}
        client_list = conn.atom('_NET_CLIENT_LIST')
        last_rules = None
        last_check = 0.0
        while not self._stop_event.is_set():
            check = set()  # windows to check now
            check_all = False
            while x.XPending(conn.dpy):
                x.XNextEvent(conn.dpy, ctypes.byref(event))
                if event.type == x11.CREATE_NOTIFY and event.create.parent == conn.root \
                        and not event.create.override_redirect:
                    self._follow(conn, event.create.window)  # to see WM_STATE being set: when it is managed
                elif event.type == x11.PROPERTY_NOTIFY:
                    if event.property.window == conn.root:
                        check_all = check_all or event.property.atom == client_list
                    elif event.property.atom in names:
                        check.add(event.property.window)
            if self._rescan.is_set():
                self._rescan.clear()
                self._reported.clear()
                check_all = True
            now = time.monotonic()
            rules = self._rules()
            if rules != last_rules or now - last_check >= self.POLL_INTERVAL:
                check_all = True
            if check_all:
                last_rules, last_check = rules, now
                self._check_all(conn, rules)
            else:
                for win in check:
                    self._check(conn, win, rules)
            select.select([fd], [], [], 0.1)

    def _follow(self, conn: x11.Connection, win: int):
        if win not in self._watched:
            self._watched.add(win)
            conn.x.XSelectInput(conn.dpy, win, x11.PROPERTY_CHANGE_MASK)
            conn.x.XFlush(conn.dpy)

    def _check_all(self, conn: x11.Connection, rules: list[AutoRule]):
        clients = set(conn.client_list())
        for win in clients:
            self._follow(conn, win)  # titles can change
            self._check(conn, win, rules)
        # forget windows that are gone; windows that were created but are not managed yet are kept
        gone = [w for w in self._watched if w not in clients and conn.attributes(w) is None]
        for win in gone:
            self._watched.discard(win)
            self._reported.discard(win)
            if self._on_gone is not None:
                self._on_gone(win)

    def _check(self, conn: x11.Connection, win: int, rules: list[AutoRule]):
        if win in self._reported or not rules:
            return
        if conn.cardinals(win, 'WM_STATE', 1) is None:
            return  # not managed (yet): not shown, or not an application window at all
        info = conn.window_info(win)
        if info.pid == self._pid or conn.window_type(win) in IGNORED_TYPES:
            return
        title = conn.title(win)
        if any(rule.matches(info.wm_class, title) for rule in rules):
            self._reported.add(win)
            self._on_match(win)
