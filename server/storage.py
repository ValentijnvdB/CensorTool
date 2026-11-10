"""
The files the HTTP server keeps in data/server/: uploaded and downloaded originals, censored images, videos and
debug output.

A file's modification time is its last use: it is set when the file is written, when a downloaded original is
reused, and when a censored image is fetched from /censored/. Files unused for longer than
server_file_retention_hours (config.yml) are removed by a sweep every SWEEP_INTERVAL seconds and at startup.
The files of video jobs the server still knows are never removed by a sweep; the jobs remove them themselves.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger

from app.config import CONFIG

from . import server_config

SWEEP_INTERVAL = 600
"""Seconds between sweeps."""


def retention_seconds() -> float:
    """How long a file may go unused; 0 keeps files forever."""
    return float(CONFIG.server_file_retention_hours) * 3600


def touch(path: Path) -> None:
    """Mark a file as used now."""
    try:
        os.utime(path)
    except OSError:
        pass


def _files(root: Path) -> Iterable[Path]:
    if root.is_dir():
        for path in root.rglob('*'):
            if path.is_file() and not path.is_symlink():
                yield path


def storage_info(root: Path | None = None) -> dict:
    """Files and bytes per folder of data/server/, and in total."""
    root = server_config.SERVER_DIR if root is None else root
    folders: dict[str, dict] = {}
    if root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            files = list(_files(folder))
            folders[folder.name] = {'files': len(files), 'bytes': sum(_size(p) for p in files)}
    loose = [p for p in root.iterdir() if p.is_file()] if root.is_dir() else []
    if loose:
        folders['.'] = {'files': len(loose), 'bytes': sum(_size(p) for p in loose)}
    return {'folders': folders,
            'files': sum(f['files'] for f in folders.values()),
            'bytes': sum(f['bytes'] for f in folders.values())}


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def remove_files(older_than: float | None, protected: set[Path], root: Path | None = None,
                 now: float | None = None) -> dict:
    """
    Remove the files in data/server/ unused for more than older_than seconds (None: every file), except the
    protected ones.

    :return: {'deleted_files': n, 'deleted_bytes': n}
    """
    root = server_config.SERVER_DIR if root is None else root
    now = time.time() if now is None else now
    protected = {p.resolve() for p in protected}
    count = size = 0
    for path in list(_files(root)):
        try:
            stat = path.stat()
            if older_than is not None and now - stat.st_mtime <= older_than:
                continue
            if path.resolve() in protected:
                continue
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning(f"Could not remove {path}: {e}")
            continue
        count += 1
        size += stat.st_size
    return {'deleted_files': count, 'deleted_bytes': size}


def sweep(protected: set[Path], now: float | None = None) -> dict:
    """Remove the files unused for longer than the retention. Does nothing when the retention is 0."""
    retention = retention_seconds()
    if retention <= 0:
        return {'deleted_files': 0, 'deleted_bytes': 0}
    return remove_files(retention, protected, now=now)


async def sweep_periodically(protected: Callable[[], set[Path]], interval: float = SWEEP_INTERVAL) -> None:
    """Sweep now and then every interval seconds."""
    while True:
        try:
            result = await asyncio.to_thread(sweep, protected())
            if result['deleted_files']:
                logger.info(f"Removed {result['deleted_files']} unused server file(s) "
                            f"({result['deleted_bytes'] / 1e6:.1f} MB).")
        except Exception as e:
            logger.error(f"Error removing unused server files: {e}")
        await asyncio.sleep(interval)
