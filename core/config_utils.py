"""Reading config files, warnings and 'did you mean' suggestions."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml

_collected_warnings: ContextVar[list[str] | None] = ContextVar('_collected_warnings', default=None)


def log_warning(msg: str) -> None:
    """
    Prints a config warning to stderr (same format as the C++ backend), or adds it to the list of an enclosing
    collect_warnings().
    """
    collected = _collected_warnings.get()
    if collected is not None:
        collected.append(msg)
        return
    print(f"[censor_core] WARNING: {msg}", file=sys.stderr, flush=True)


@contextmanager
def collect_warnings() -> Iterator[list[str]]:
    """Collects the config warnings logged inside the block (in this thread/task) instead of printing them."""
    collected: list[str] = []
    token = _collected_warnings.set(collected)
    try:
        yield collected
    finally:
        _collected_warnings.reset(token)


def read_config_file(path: str | Path) -> Any:
    """Reads a .yml, .yaml or .json file without interpreting it."""
    path = Path(path)
    if path.suffix == '.json':
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except OSError as e:
            raise RuntimeError(f"Failed to open JSON file: {path}") from e

    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load YAML file '{path}': {e}") from e


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[len(b)]


def closest_match(key: str, candidates: Iterable[str]) -> str:
    """The candidate closest to key (substring or edit distance <= 2), or '' if none is close."""
    best, best_distance = '', 3
    for c in candidates:
        related = bool(key) and (key in c or c in key)
        d = 0 if related else _edit_distance(key, c)
        if d < best_distance:
            best, best_distance = c, d
    return best


def did_you_mean(key: str, candidates: Iterable[str]) -> str:
    match = closest_match(key, candidates)
    return f" (did you mean '{match}'?)" if match else ""


def warn_unknown_keys(d: Any, known: Iterable[str], where: str, hints: Mapping[str, str] | None = None) -> None:
    """Warns about keys of d not in known, suggesting the closest known key."""
    if not isinstance(d, dict):
        return
    known = list(known)
    hints = hints or {}
    for key in d:
        if key in known:
            continue
        if key in hints:
            log_warning(f"Key '{key}' in {where} is no longer supported: {hints[key]}")
            continue
        suggestion = did_you_mean(key, known)
        log_warning(f"Unknown key '{key}' in {where} is ignored{suggestion or ''}.")
