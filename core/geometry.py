"""Polygon helpers. A polygon is an (N, 2) int32 numpy array of image points."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

Polygon = np.ndarray


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]               # x, y, w, h in image space
    polygons: list[Polygon] = field(default_factory=list)  # outline parts; empty for detect-only models
    parent_stage: str = ''                                # stage of the detection this one was found in
    parent: int = -1                                      # index into that stage's detections


def as_polygon(points) -> Polygon:
    return np.asarray(points, dtype=np.int32).reshape(-1, 2)


def cround(v):
    """Rounds half away from zero, like std::round."""
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def bounding_rect(points: Polygon) -> tuple[int, int, int, int]:
    """Like cv::boundingRect: x, y, w, h with the max point included."""
    return cv2.boundingRect(as_polygon(points))


def intersect_rects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Like cv::Rect &: the overlap, or (0, 0, 0, 0)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2 - x1, y2 - y1


def rect_empty(r: tuple[int, int, int, int]) -> bool:
    return r[2] <= 0 or r[3] <= 0


# ------------------------------------ shapely conversion ------------------------------------
def _to_shapely(points: Polygon) -> BaseGeometry:
    poly = ShapelyPolygon(as_polygon(points).astype(np.float64))
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    return poly


def _parts(geom: BaseGeometry) -> list[ShapelyPolygon]:
    if isinstance(geom, ShapelyPolygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon) or hasattr(geom, 'geoms'):
        out = []
        for g in geom.geoms:
            out.extend(_parts(g))
        return out
    return []


def _largest(geom: BaseGeometry) -> Polygon:
    parts = [p for p in _parts(geom) if p.area > 0]
    if not parts:
        return as_polygon([])
    best = max(parts, key=lambda p: p.area)
    coords = np.asarray(best.exterior.coords)[:-1]  # drop the closing point
    return cround(coords).astype(np.int32)


# ------------------------------------ public helpers ------------------------------------
def polygon_area(points: Polygon) -> float:
    pts = as_polygon(points)
    if len(pts) < 3:
        return 0.0
    return abs(cv2.contourArea(pts))


def polygons_intersect(a: Polygon, b: Polygon) -> bool:
    """True if the two closed polygons share any area or boundary."""
    if len(a) == 0 or len(b) == 0:
        return False
    if rect_empty(intersect_rects(bounding_rect(a), bounding_rect(b))):
        return False
    return _to_shapely(a).intersects(_to_shapely(b))


def intersect_polygons(a: Polygon, b: Polygon) -> Polygon:
    """The largest piece of the intersection, or an empty polygon if they don't overlap."""
    if len(a) < 3 or len(b) < 3:
        return as_polygon([])
    return _largest(_to_shapely(a).intersection(_to_shapely(b)))


def union_polygons(a: Polygon, b: Polygon) -> Polygon:
    """The largest outline of the union (a if the union is empty)."""
    result = _largest(_to_shapely(a).union(_to_shapely(b)))
    return result if len(result) else as_polygon(a)


def scale_shape(points: Polygon, scale_x: float, scale_y: float) -> Polygon:
    """Scales the polygon around the center of its bounding box."""
    pts = as_polygon(points)
    if len(pts) == 0:
        return pts
    x, y, w, h = bounding_rect(pts)
    cx, cy = x + w / 2.0, y + h / 2.0
    out = np.empty_like(pts)
    out[:, 0] = cround(cx + (pts[:, 0] - cx) * np.float32(scale_x))
    out[:, 1] = cround(cy + (pts[:, 1] - cy) * np.float32(scale_y))
    return out


def chaikin(points: Polygon, iterations: int = 2) -> Polygon:
    """Chaikin corner-cutting; each iteration doubles the number of points."""
    pts = as_polygon(points)
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        p0 = pts.astype(np.int64)
        p1 = np.roll(p0, -1, axis=0)
        a = np.trunc((3 * p0 + p1) / 4)
        b = np.trunc((p0 + 3 * p1) / 4)
        pts = np.stack([a, b], axis=1).reshape(-1, 2).astype(np.int32)
    return pts


def create_ellipse(cx: float, cy: float, rx: float, ry: float, steps: int = 100) -> Polygon:
    angles = 2.0 * math.pi * np.arange(steps) / steps
    xs = np.trunc(cx + rx * np.cos(angles))
    ys = np.trunc(cy + ry * np.sin(angles))
    return np.stack([xs, ys], axis=1).astype(np.int32)
