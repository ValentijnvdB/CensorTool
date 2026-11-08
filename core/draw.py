"""Applying censor boxes to an image."""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np

from .boxes import Box, merge_boxes
from .censor_config import CensorConfig, CsAIRemove, CsBar, CsBlur, CsDebug, CsPixel, OvSticker, OvText
from .geometry import Polygon, bounding_rect, intersect_rects, rect_empty


# -------------------------------- helpers --------------------------------
def _clipped_rect(image: np.ndarray, points: Polygon) -> tuple[int, int, int, int]:
    return intersect_rects(bounding_rect(points), (0, 0, image.shape[1], image.shape[0]))


def _mask(rect: tuple[int, int, int, int], points: Polygon) -> np.ndarray:
    x, y, w, h = rect
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2) - np.array([x, y], dtype=np.int32)], 255)
    return mask


def _copy_masked(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> None:
    m = mask > 0
    dst[m] = src[m]


# -------------------------------- censor styles --------------------------------
def blur_area(image: np.ndarray, points: Polygon, strength: int) -> None:
    if len(points) < 3:
        return
    if strength % 2 == 0:
        strength += 1
    strength = max(1, strength)
    rect = _clipped_rect(image, points)
    if rect_empty(rect):
        return
    x, y, w, h = rect
    roi = image[y:y + h, x:x + w]
    _copy_masked(cv2.blur(roi, (strength, strength)), roi, _mask(rect, points))


def pixelate_area(image: np.ndarray, points: Polygon, factor: int) -> None:
    if len(points) < 3 or factor <= 1:
        return
    rect = _clipped_rect(image, points)
    if rect_empty(rect):
        return
    x, y, w, h = rect
    roi = image[y:y + h, x:x + w]
    small = cv2.resize(roi, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    _copy_masked(pixelated, roi, _mask(rect, points))


def draw_bar(image: np.ndarray, points: Polygon, color, thickness: int = cv2.FILLED) -> None:
    if len(points) < 2:
        return
    contour = [points.reshape(-1, 1, 2)]
    if thickness == cv2.FILLED:
        cv2.fillPoly(image, contour, color)
    else:
        cv2.polylines(image, contour, True, color, thickness)


def draw_debug(image: np.ndarray, box: Box) -> None:
    x, y, w, h = bounding_rect(box.polygon)
    color = box.censor_style.color
    cv2.rectangle(image, (x, y, w, h), color, 3)
    text = f"{box.label} {box.score:.6f}"[:len(box.label) + 5]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (x, y - th - 5), (x + tw, y), color, cv2.FILLED)
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def draw_censor_box(image: np.ndarray, box: Box) -> np.ndarray:
    """Applies the box. Returns the image: in place, except for ai_remove, which returns a new one."""
    style = box.censor_style
    if isinstance(style, CsBlur):
        blur_area(image, box.polygon, style.strength)
    elif isinstance(style, CsBar):
        draw_bar(image, box.polygon, style.color)
    elif isinstance(style, CsPixel):
        pixelate_area(image, box.polygon, style.factor)
    elif isinstance(style, CsAIRemove):
        return remove_feature(image, box.polygon, style)
    elif isinstance(style, CsDebug):
        draw_debug(image, box)
    return image


# -------------------------------- inverse --------------------------------
def apply_inverse_censor(image: np.ndarray, boxes: list[Box], config: CensorConfig) -> None:
    """Censors everything outside the box polygons (in place)."""
    regions = []
    for box in boxes:
        if len(box.polygon) == 0:
            continue
        rect = _clipped_rect(image, box.polygon)
        if rect_empty(rect):
            continue
        x, y, w, h = rect
        regions.append((rect, image[y:y + h, x:x + w].copy(), _mask(rect, box.polygon)))

    style = config.inverse_censor_style
    if isinstance(style, CsBar):
        image[:] = np.array(style.color[:3], dtype=np.float64).round().astype(np.uint8)
    elif isinstance(style, CsBlur):
        k = style.strength + 1 if style.strength % 2 == 0 else style.strength
        k = max(1, k)
        image[:] = cv2.blur(image, (k, k))
    elif isinstance(style, CsPixel):
        if style.factor > 1:
            h, w = image.shape[:2]
            small = cv2.resize(image, (max(1, w // style.factor), max(1, h // style.factor)),
                               interpolation=cv2.INTER_LINEAR)
            image[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    for (x, y, w, h), original, mask in regions:
        _copy_masked(original, image[y:y + h, x:x + w], mask)


# -------------------------------- overlays --------------------------------
def apply_text_overlay(image: np.ndarray, ov: OvText, text: str, polygon: Polygon) -> None:
    if not ov.values or len(polygon) == 0:
        return
    cx, cy = (int(v) for v in polygon.astype(np.float32).mean(axis=0))
    angle = random.uniform(-15.0, 15.0)

    (tw, th), _ = cv2.getTextSize(text, ov.font, ov.font_scale, ov.thickness)
    mask = np.zeros((th + 20, tw + 20, 3), dtype=np.uint8)
    cv2.putText(mask, text, (10, th + 10), ov.font, ov.font_scale, ov.color, ov.thickness, cv2.LINE_AA)

    rot = cv2.getRotationMatrix2D((mask.shape[1] / 2.0, mask.shape[0] / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(mask, rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

    xo, yo = cx - mask.shape[1] // 2, cy - mask.shape[0] // 2
    x0, y0 = max(0, xo), max(0, yo)
    x1, y1 = min(image.shape[1], xo + rotated.shape[1]), min(image.shape[0], yo + rotated.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    src = rotated[y0 - yo:y1 - yo, x0 - xo:x1 - xo]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _copy_masked(src, image[y0:y1, x0:x1], gray)


_sticker_lock = threading.Lock()
_stickers: dict[str, Optional[tuple[np.ndarray, Optional[np.ndarray]]]] = {}


def _load_sticker(path: str) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
    """Decodes each sticker file once (failures too, so a missing file is not retried every frame)."""
    with _sticker_lock:
        if path in _stickers:
            return _stickers[path]
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    sticker = None
    if image is not None:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=255.0 / max(1, np.iinfo(image.dtype).max))
        sticker = (image[:, :, :3].copy(), image[:, :, 3].copy()) if image.shape[2] == 4 else (image, None)
    with _sticker_lock:
        return _stickers.setdefault(path, sticker)


def apply_sticker_overlay(image: np.ndarray, path: str, polygon: Polygon) -> None:
    if len(polygon) == 0:
        return
    sticker = _load_sticker(path)
    if sticker is None:
        return
    bgr, alpha = sticker

    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    box_w, box_h = int(max_x - min_x), int(max_y - min_y)
    sh, sw = bgr.shape[:2]
    aspect = sw / sh
    if box_w / sw < box_h / sh:
        new_w, new_h = box_w, int(box_w / aspect)
    else:
        new_h, new_w = box_h, int(box_h * aspect)
    if new_w <= 0 or new_h <= 0:
        return

    xo, yo = int(min_x) + (box_w - new_w) // 2, int(min_y) + (box_h - new_h) // 2
    x0, y0 = max(0, xo), max(0, yo)
    x1, y1 = min(image.shape[1], xo + new_w), min(image.shape[0], yo + new_h)
    if x0 >= x1 or y0 >= y1:
        return

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)[y0 - yo:y1 - yo, x0 - xo:x1 - xo]
    roi = image[y0:y1, x0:x1]
    if alpha is None:
        roi[:] = resized
        return
    a = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)[y0 - yo:y1 - yo, x0 - xo:x1 - xo]
    a = a.astype(np.uint16)[:, :, None]
    roi[:] = ((a * resized + (255 - a) * roi + 127) // 255).astype(np.uint8)


def apply_overlay(image: np.ndarray, box: Box) -> None:
    if not box.overlay or box.overlay_config is None:
        return
    if isinstance(box.overlay_config, OvText):
        apply_text_overlay(image, box.overlay_config, box.overlay, box.polygon)
    elif isinstance(box.overlay_config, OvSticker):
        apply_sticker_overlay(image, box.overlay, box.polygon)


# -------------------------------- watermark --------------------------------
_watermark_lock = threading.Lock()
_watermarks: dict[tuple, Optional[tuple[np.ndarray, np.ndarray, tuple[int, int]]]] = {}


def _render_watermark(size: tuple[int, int], text: str, index: int, factor: float):
    width, height = size
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
    mw, mh = tw + 20, th + 20
    max_factor = min(width / mw, (height - 20 * index) / mh)
    if max_factor <= 0:
        return None
    factor = min(factor, max_factor)

    mask = np.zeros((mh, mw, 3), dtype=np.uint8)
    cv2.putText(mask, text, (10, th + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    patch = cv2.resize(mask, None, fx=factor, fy=factor)
    if patch.size == 0:
        return None
    pos_y = height - 20 * index - patch.shape[0]
    if pos_y < 0 or patch.shape[1] > width:
        return None
    _, m = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    return patch, m, (0, pos_y)


def apply_watermark(image: np.ndarray, text: str = "Censored using CensorTool", index: int = 0,
                    factor: float = 0.6) -> None:
    key = (image.shape[1], image.shape[0], text, index, factor)
    with _watermark_lock:
        if key not in _watermarks:
            if len(_watermarks) >= 64:
                _watermarks.clear()
            _watermarks[key] = _render_watermark((image.shape[1], image.shape[0]), text, index, factor)
        wm = _watermarks[key]
    if wm is None:
        return
    patch, mask, (x, y) = wm
    _copy_masked(patch, image[y:y + patch.shape[0], x:x + patch.shape[1]], mask)


# -------------------------------- ai_remove (ComfyUI) --------------------------------
def remove_feature(image: np.ndarray, polygon: Polygon, style: CsAIRemove) -> np.ndarray:
    """Inpaints the polygon with a ComfyUI workflow and returns ComfyUI's output image."""
    import requests

    base = style.comfy_host if '://' in style.comfy_host else f"http://{style.comfy_host}"
    base = f"{base.rstrip('/')}:{style.comfy_port}"
    name = str(uuid.uuid4())

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 255)

    def upload(img: np.ndarray, filename: str) -> str:
        ok, buf = cv2.imencode('.png', img)
        if not ok:
            raise RuntimeError(f"Failed to encode image '{filename}' as PNG.")
        res = requests.post(f"{base}/upload/image", files={'image': (filename, buf.tobytes(), 'image/png')},
                            data={'type': 'input', 'overwrite': 'true'}, timeout=(10, 60))
        if res.status_code != 200:
            raise RuntimeError(f"Failed to upload image '{filename}' to ComfyUI.")
        return res.json()['name']

    image_name = upload(image, f"{name}.png")
    mask_name = upload(mask, f"{name}_mask.png")

    with open(style.comfy_workflow, 'r') as f:
        workflow = json.load(f)

    def node(title: str) -> dict:
        for n in workflow.values():
            if isinstance(n, dict) and n.get('_meta', {}).get('title') == title:
                return n
        raise RuntimeError(f"No workflow node with title '{title}' found.")

    node('IMAGE')['inputs']['image'] = image_name
    node('MASK')['inputs']['image'] = mask_name

    res = requests.post(f"{base}/prompt", json={'prompt': workflow, 'client_id': 'censorcore-client'},
                        timeout=(10, 60))
    if res.status_code != 200:
        raise RuntimeError("Failed to queue ComfyUI prompt.")
    prompt_id = res.json()['prompt_id']

    while True:
        res = requests.get(f"{base}/history/{prompt_id}", timeout=(10, 60))
        if res.status_code != 200:
            raise RuntimeError("Failed to poll ComfyUI history.")
        history = res.json()
        if prompt_id in history:
            entry = history[prompt_id]
            break
        time.sleep(1.0)

    for output in entry.get('outputs', {}).values():
        for info in output.get('images', []):
            params = {'filename': info['filename'], 'type': info.get('type', 'output')}
            if info.get('subfolder'):
                params['subfolder'] = info['subfolder']
            res = requests.get(f"{base}/view", params=params, timeout=(10, 60))
            if res.status_code != 200:
                raise RuntimeError("Failed to fetch output image from ComfyUI.")
            result = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
            if result is None:
                raise RuntimeError("Failed to decode output image from ComfyUI.")
            return result
    raise RuntimeError("No image outputs found in ComfyUI history entry.")


# -------------------------------- apply --------------------------------
def apply_censor_boxes(image: np.ndarray, boxes: list[Box], config: CensorConfig) -> np.ndarray:
    """
    Applies the censor boxes to image. Works in place and returns the image
    (a new array only when an ai_remove box replaced it).
    """
    boxes = sorted(boxes, key=lambda b: b.censor_priority)
    inverse = [b for b in boxes if b.inverse]
    normal = [b for b in boxes if not b.inverse]
    with_border = [b for b in boxes if b.border is not None]

    if inverse or config.force_inverse_censor:
        apply_inverse_censor(image, inverse, config)

    for b in normal:
        image = draw_censor_box(image, b)

    if config.merge_overlapping_borders and len(with_border) > 1:
        with_border = merge_boxes(with_border)
    for b in with_border:
        draw_bar(image, b.polygon, b.border.color, b.border.thickness)

    if config.enable_overlays:
        for b in boxes:
            apply_overlay(image, b)

    if config.enable_watermark:
        apply_watermark(image)
    return image
