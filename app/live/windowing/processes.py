"""Which processes belong to a program: its process and their descendants."""
from __future__ import annotations

import os
import sys


def parent_map() -> dict[int, int]:
    """pid -> parent pid of every process."""
    if sys.platform.startswith('linux'):
        return _parent_map_proc()
    try:
        import psutil
    except ImportError:
        return {}
    parents = {}
    for proc in psutil.process_iter(['pid', 'ppid']):
        parents[proc.info['pid']] = proc.info['ppid']
    return parents


def _parent_map_proc() -> dict[int, int]:
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
