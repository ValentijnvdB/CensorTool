"""PFLD 68-point facial landmarks, used to find the eyes in a face."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .geometry import Polygon

LEFT_EYE = (36, 41)
RIGHT_EYE = (42, 47)
EXPECTED_VALUES = 68 * 2

EYE_BOX_W_FRACTION = 0.80  # width of the two-eye bar as a fraction of the face width
EYE_BOX_H_FRACTION = 0.18  # height of the two-eye bar as a fraction of the face height


def preprocess_face_crop(face: np.ndarray) -> np.ndarray:
    """Resize to 112x112, BGR -> RGB, CHW, [0, 1]."""
    return cv2.dnn.blobFromImage(face, 1.0 / 255.0, (112, 112), swapRB=True, crop=False)


def _cluster_box(lm: np.ndarray, span: tuple[int, int], face: tuple[int, int, int, int]) -> Optional[tuple]:
    pts = lm[span[0]:span[1] + 1]
    min_x, min_y = np.clip(pts.min(axis=0), 0.0, 1.0)
    max_x, max_y = np.clip(pts.max(axis=0), 0.0, 1.0)
    if max_x <= min_x or max_y <= min_y:
        return None
    fx, fy, fw, fh = face
    return fx + min_x * fw, fy + min_y * fh, fx + max_x * fw, fy + max_y * fh


def postprocess_landmarks(output: list[np.ndarray], face: tuple[int, int, int, int],
                          w_fraction: float = EYE_BOX_W_FRACTION, h_fraction: float = EYE_BOX_H_FRACTION) -> Polygon:
    """
    The eye region from raw PFLD output, in image space:
      - 2 eyes: a parallelogram spanning both eyes, tilted with the head.
      - 1 eye:  the box of the detected cluster.
      - 0 eyes: an empty polygon.
    """
    empty = np.zeros((0, 2), dtype=np.int32)
    if not output:
        return empty
    flat = output[0].reshape(-1)
    if flat.size < EXPECTED_VALUES or face[2] <= 0 or face[3] <= 0:
        return empty
    lm = flat[:EXPECTED_VALUES].reshape(68, 2).astype(np.float32)

    left = _cluster_box(lm, LEFT_EYE, face)
    right = _cluster_box(lm, RIGHT_EYE, face)
    if left is None and right is None:
        return empty

    if left is None or right is None:
        x1, y1, x2, y2 = left or right
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).astype(np.int32)

    fx, fy, fw, fh = face
    left_cy = fy + lm[LEFT_EYE[0]:LEFT_EYE[1] + 1, 1].mean() * fh
    right_cy = fy + lm[RIGHT_EYE[0]:RIGHT_EYE[1] + 1, 1].mean() * fh

    half_w = fw * w_fraction / 2.0
    half_h = fh * h_fraction / 2.0

    face_cx = fx + fw / 2.0
    x1 = max(float(fx), face_cx - half_w)
    x2 = min(float(fx + fw), face_cx + half_w)

    left_y1 = max(float(fy), left_cy - half_h)
    left_y2 = min(float(fy + fh), left_cy + half_h)
    right_y1 = max(float(fy), right_cy - half_h)
    right_y2 = min(float(fy + fh), right_cy + half_h)

    # Clockwise: top-left, top-right, bottom-right, bottom-left
    return np.array([[x1, left_y1], [x2, right_y1], [x2, right_y2], [x1, left_y2]], dtype=np.float64).astype(np.int32)
