"""
Auto mode of overlay mode: censor every window of certain programs, also windows they open later.

Which windows: rules from overlay_auto_programs in config.yml, plus programs given with --auto or added from the
tray. A rule names a program (matched by window class like launch.class_matches), a title (a regular expression,
case-insensitive), or both:

    overlay_auto_programs: [celluloid, mpv, {program: firefox, title: 'Private Browsing'}]

New windows are noticed as early as the window system allows (X11: when the window manager starts managing them,
which is about when they are shown). A frame of a new window can still be visible before its overlay covers it.
Titles are checked again when they change, so a window is censored as soon as its title matches; a censored
window stays censored when its title changes back.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from loguru import logger

from . import windowing
from .launch import class_matches


@dataclass(frozen=True)
class AutoRule:
    program: str | None = None  # matched against the window class (app_ids)
    title: str | None = None  # regular expression, searched in the title, case-insensitive

    def matches(self, app_ids: Iterable[str], title: str) -> bool:
        if self.program is not None and not class_matches({self.program}, app_ids):
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


def program_name_of(app_ids: tuple[str, ...]) -> str | None:
    """The name to save for a window's program: its (last) class without a reverse-domain prefix, in lowercase."""
    if not app_ids:
        return None
    return app_ids[-1].rsplit('.', 1)[-1].lower() or None


class AutoWatcher(threading.Thread):
    """
    Reports windows that match the rules (on_match, from this thread), each once, and application windows that are
    gone (on_gone). Windows of this process (the overlays) and panels are skipped.
    """

    POLL_INTERVAL = 0.5

    def __init__(self, rules: Callable[[], list[AutoRule]], on_match: Callable[[int], None],
                 on_gone: Callable[[int], None] | None = None):
        super().__init__(name='overlay-auto-watcher', daemon=True)
        self._rules = rules
        self._on_match = on_match
        self._on_gone = on_gone
        self._reported: set[int] = set()
        self._listed: set[int] = set()  # the window list at the last check
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
            monitor = windowing.get().monitor(follow_titles=True)
        except windowing.WindowSystemError as e:
            logger.error(f"Auto mode can't watch for windows: {e}")
            return
        try:
            self._watch(monitor)
        except Exception as e:
            logger.error(f"Auto mode stopped watching for windows: {e}")
        finally:
            monitor.close()

    def _watch(self, monitor: windowing.WindowMonitor):
        display = monitor.display
        last_rules = None
        last_check = 0.0
        while not self._stop_event.is_set():
            changes = monitor.wait(0.1)
            check_all = changes.list_changed
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
                self._check_all(display, rules)
            else:
                for win in changes.windows:
                    self._check(display, win, rules)

    def _check_all(self, display: windowing.Display, rules: list[AutoRule]):
        listed = set(display.window_list())
        for win in listed:
            self._check(display, win, rules)
        for win in self._listed - listed:  # closed (or no longer managed)
            self._reported.discard(win)
            if self._on_gone is not None:
                self._on_gone(win)
        self._listed = listed

    def _check(self, display: windowing.Display, win: int, rules: list[AutoRule]):
        if win in self._reported or not rules:
            return
        if not display.is_app_window(win):
            return  # not managed (yet): not shown, or not an application window at all
        info = display.window_info(win)
        if info.pid == self._pid or info.panel:
            return
        title = display.title(win)
        if any(rule.matches(info.app_ids, title) for rule in rules):
            self._reported.add(win)
            self._on_match(win)
