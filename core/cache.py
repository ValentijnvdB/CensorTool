"""
Per-stage detection caches.

Each stage's detections are stored with the stage's fingerprint and are reused while it matches. The files are
JSON and use different names than the C++ backend's binary caches (<hash>.py.json / <stem>.py.json), so both
backends can share a cache directory without reading each other's files.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .geometry import Detection

CACHE_VERSION = 1
SUFFIX = '.py.json'

# stage -> (fingerprint, detections)
CacheEntries = dict[str, tuple[str, list[Detection]]]
# frame number -> entries
VideoCache = dict[int, CacheEntries]


def _det_to_json(d: Detection) -> list:
    return [d.label, float(d.score), [float(v) for v in d.box],
            [part.reshape(-1).tolist() for part in d.polygons], d.parent_stage, int(d.parent)]


def _det_from_json(v: list) -> Detection:
    label, score, box, polygons, parent_stage, parent = v
    return Detection(label=label, score=float(score), box=tuple(float(b) for b in box),
                     polygons=[np.asarray(p, dtype=np.int32).reshape(-1, 2) for p in polygons],
                     parent_stage=parent_stage, parent=int(parent))


def _entries_to_json(entries: CacheEntries) -> dict:
    return {stage: [fp, [_det_to_json(d) for d in dets]] for stage, (fp, dets) in entries.items()}


def _entries_from_json(v: dict) -> CacheEntries:
    return {stage: (fp, [_det_from_json(d) for d in dets]) for stage, (fp, dets) in v.items()}


def cached_detections(entries: CacheEntries, fingerprints: dict[str, str]) -> dict[str, list[Detection]]:
    """The entries whose fingerprint matches the current pipeline."""
    out = {}
    for stage, fp in fingerprints.items():
        entry = entries.get(stage)
        if entry is not None and entry[0] == fp:
            out[stage] = list(entry[1])
    return out


def update_cache(entries: CacheEntries, detections: dict[str, list[Detection]], fingerprints: dict[str, str]) -> None:
    """Stores the current pipeline's detections in entries (other entries are kept)."""
    for stage, dets in detections.items():
        if stage in fingerprints:
            entries[stage] = (fingerprints[stage], list(dets))


def _read(path: Optional[Path], kind: str) -> Optional[Any]:
    if path is None or not path.is_file():
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if data.get('kind') != kind or data.get('version') != CACHE_VERSION:
            return None
        return data['data']
    except Exception:
        return None


def _write(path: Path, kind: str, data: Any) -> None:
    """Writes atomically, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.tmp_', suffix=SUFFIX)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'kind': kind, 'version': CACHE_VERSION, 'data': data}, f, separators=(',', ':'))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def image_cache_path(cache_dir: Path, image_hash: str) -> Path:
    return Path(cache_dir) / f"{image_hash}{SUFFIX}"


def read_image_cache(path: Optional[Path]) -> CacheEntries:
    data = _read(path, 'image')
    try:
        return _entries_from_json(data) if data is not None else {}
    except Exception:
        return {}


def write_image_cache(path: Path, entries: CacheEntries) -> None:
    _write(path, 'image', _entries_to_json(entries))


def video_cache_path(video_path: Path, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{Path(video_path).stem}{SUFFIX}"


def read_video_cache(path: Path) -> VideoCache:
    data = _read(path, 'video')
    try:
        return {int(frame): _entries_from_json(e) for frame, e in data.items()} if data is not None else {}
    except Exception:
        return {}


def write_video_cache(path: Path, cache: VideoCache) -> None:
    _write(path, 'video', {str(frame): _entries_to_json(e) for frame, e in sorted(cache.items())})
