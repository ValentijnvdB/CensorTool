"""Turning detections into censor boxes."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .censor_config import CensorConfig, CensorStyle, Border, Overlay, OvSticker, OvText, Shape, censor_style_priority
from .geometry import (Detection, Polygon, as_polygon, create_ellipse, intersect_polygons, polygon_area,
                       polygons_intersect, scale_shape, union_polygons)

ELLIPSE_N_STEPS = 100


@dataclass
class Box:
    start: float
    end: float
    censor_style: CensorStyle
    label: str
    overlay: str                       # the picked text or sticker path, '' for none
    overlay_config: Optional[Overlay]
    border: Optional[Border]
    inverse: bool
    polygon: Polygon
    score: float
    censor_priority: int


def create_box(start: float, end: float, censor_style: CensorStyle, label: str, overlay: str,
               overlay_config: Optional[Overlay], border: Optional[Border], inverse: bool, points: Polygon,
               shape: Shape, score: float) -> Box:
    box = Box(start, end, censor_style, label, overlay, overlay_config, border, inverse, points, score,
              censor_style_priority(censor_style))
    if shape in (Shape.DEFAULT, Shape.MASK) or len(points) == 0:
        return box

    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    if shape == Shape.RECTANGLE:
        box.polygon = as_polygon([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
        return box

    cx, cy = int((min_x + max_x) / 2), int((min_y + max_y) / 2)
    rx, ry = int((max_x - min_x) / 2), int((max_y - min_y) / 2)
    if shape == Shape.ELLIPSE:
        box.polygon = create_ellipse(cx, cy, rx, ry, ELLIPSE_N_STEPS)
    else:  # CIRCLE
        r = max(rx, ry)
        box.polygon = create_ellipse(cx, cy, r, r, ELLIPSE_N_STEPS)
    return box


# ---------------------------- stickers ----------------------------
_STICKER_MAX_AGE = 5.0  # seconds
_sticker_lock = threading.Lock()
_sticker_cache: dict[tuple, tuple[float, list[Path]]] = {}


def _sticker_files(base: Path, subdirs: Iterable[str]) -> list[Path]:
    """
    The sticker files of a stickers folder + categories. Kept for a few seconds: scanning the folder for every box
    on every frame is slow on live streams and videos.
    """
    subdirs = tuple(subdirs)
    key = (str(base), subdirs)
    now = time.monotonic()
    with _sticker_lock:
        cached = _sticker_cache.get(key)
        if cached is not None and now - cached[0] < _STICKER_MAX_AGE:
            return cached[1]

    targets = [base / s for s in subdirs] if subdirs else [base]
    files = []
    for target in targets:
        if target.is_dir():
            files.extend(p.absolute() for p in sorted(target.rglob('*')) if p.is_file())

    with _sticker_lock:
        _sticker_cache[key] = (now, files)
    return files


def _generate_overlay(ov: Overlay, config: CensorConfig) -> str:
    if isinstance(ov, OvText):
        return random.choice(ov.values) if ov.values else ''
    if isinstance(ov, OvSticker):
        files = _sticker_files(config.stickers_dir, ov.categories)
        return str(random.choice(files)) if files else ''
    return ''


# ---------------------------- to_box / build_boxes ----------------------------
def to_box(points: Polygon, label: str, score: float, clip_to: list[Polygon], timestamp: float,
           config: CensorConfig) -> Box:
    """Builds one censor box from a shape, applying margins, timing, overlay and clipping."""
    censor_box = config.features_to_censor.get(label)
    if censor_box is None:
        raise RuntimeError(f"to_box: label '{label}' not found in features_to_censor")

    # Safety margin: 0.1 makes the shape 10% wider (5% on each side).
    expanded = scale_shape(points, 1.0 + censor_box.width_area_safety, 1.0 + censor_box.height_area_safety)

    half_time = censor_box.time_safety / 2.0
    start = max(timestamp - half_time, 0.0)
    end = timestamp + half_time

    overlay = ''
    if config.enable_overlays and censor_box.overlay is not None:
        if random.random() < censor_box.overlay.probability:
            overlay = _generate_overlay(censor_box.overlay, config)

    box = create_box(start, end, censor_box.style, label, overlay, censor_box.overlay, censor_box.border,
                     censor_box.inverse, expanded, censor_box.shape, score)

    # clip_to: keep the largest intersection with one of the target outlines; without any the shape is kept as is.
    best, best_area = None, 0.0
    for target in clip_to:
        inter = intersect_polygons(box.polygon, target)
        if len(inter) == 0:
            continue
        area = polygon_area(inter)
        if area > best_area:
            best, best_area = inter, area
    if best is not None:
        box.polygon = best
    return box


def _box_corners(b: tuple[float, float, float, float]) -> Polygon:
    x, y, w, h = b
    return as_polygon([[int(x), int(y)], [int(x + w), int(y)], [int(x + w), int(y + h)], [int(x), int(y + h)]])


def build_boxes(detections: dict[str, list[Detection]], stage_order: list[str], timestamp: float,
                config: CensorConfig) -> list[Box]:
    """
    Keeps detections whose label is in features_to_censor and whose score reaches its min_probability. Shapes start
    from the detection's outline (or its box when it has none). The result is clipped to the clip_to outlines.
    """
    # Outlines of every clip_to target that passes its threshold.
    outlines: dict[str, list[Polygon]] = {}
    for target, min_prob in config.clip_min_probability.items():
        parts = outlines.setdefault(target, [])
        for dets in detections.values():
            for d in dets:
                if d.label == target and d.score >= min_prob:
                    parts.extend(d.polygons)

    boxes = []
    for stage in stage_order:
        for d in detections.get(stage, []):
            censor_box = config.features_to_censor.get(d.label)
            if censor_box is None or d.score < censor_box.min_probability:
                continue

            clip_to = [part for target in censor_box.clip_to for part in outlines.get(target, [])]

            if not d.polygons:
                boxes.append(to_box(_box_corners(d.box), d.label, d.score, clip_to, timestamp, config))
            elif censor_box.shape in (Shape.DEFAULT, Shape.MASK):
                # One censor box per outline part.
                for part in d.polygons:
                    boxes.append(to_box(part, d.label, d.score, clip_to, timestamp, config))
            else:
                # ELLIPSE/CIRCLE/RECTANGLE are fitted around all parts together.
                boxes.append(to_box(np.concatenate(d.polygons), d.label, d.score, clip_to, timestamp, config))
    return boxes


# ---------------------------- merging ----------------------------
def merge_boxes(boxes: list[Box]) -> list[Box]:
    """Boxes whose polygons (transitively) intersect are unioned into one."""
    if not boxes:
        return []
    boxes = list(boxes)
    groups: list[list[Box]] = [[boxes.pop()]]
    for box in boxes:
        hits = [i for i, g in enumerate(groups) if any(polygons_intersect(box.polygon, o.polygon) for o in g)]
        if not hits:
            groups.append([box])
        elif len(hits) == 1:
            groups[hits[0]].append(box)
        else:
            merged = [box] + [b for i in hits for b in groups[i]]
            groups = [g for i, g in enumerate(groups) if i not in hits] + [merged]

    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
            continue
        polygon = group[0].polygon
        for b in group[1:]:
            polygon = union_polygons(polygon, b.polygon)
        result.append(replace(
            random.choice(group),
            start=min(b.start for b in group),
            end=max(b.end for b in group),
            score=max(b.score for b in group),
            polygon=polygon,
        ))
    return result
