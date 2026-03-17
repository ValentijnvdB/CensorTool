from __future__ import annotations

import numpy as np
import cv2
from typing import Literal, List

from shapely import GeometryCollection, Polygon, MultiPolygon, make_valid

import constants

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps (backend, size) → official Ultralytics model name
_MODEL_NAMES: dict[str, str] = {
    # YOLOv8-seg variants  (n=nano … x=extra-large)
    "yolov8n": "yolov8n-seg.pt",
    "yolov8s": "yolov8s-seg.pt",
    "yolov8m": "yolov8m-seg.pt",
    "yolov8l": "yolov8l-seg.pt",
    "yolov8x": "yolov8x-seg.pt",
    # YOLOv11-seg variants
    "yolov11n": "yolo11n-seg.pt",
    "yolov11s": "yolo11s-seg.pt",
    "yolov11m": "yolo11m-seg.pt",
    "yolov11l": "yolo11l-seg.pt",
    "yolov11x": "yolo11x-seg.pt",
}

Backend = Literal[
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_to_polygons(mask: np.ndarray) -> List[Polygon]:
    """
    Convert a binary uint8 mask (H×W) to a list of Shapely Polygons,
    one per connected component / contour.

    Uses RETR_EXTERNAL to get only outer contours (no holes) and
    CHAIN_APPROX_NONE to keep every boundary pixel for maximum accuracy.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[Polygon] = []
    for cnt in contours:
        if cnt.shape[0] < 4:          # need ≥4 points for a valid polygon
            continue
        pts = cnt[:, 0, :]            # shape (N, 2) → (x, y)
        poly = Polygon(pts.tolist())
        if not poly.is_valid:
            poly = make_valid(poly)   # fix self-intersections, etc.
        if poly.is_empty or poly.area < 1.0:
            continue
        polygons.append(poly)
    return polygons

def _extract_polygons(geom) -> List[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        return [max(polys, key=lambda p: p.area)] if polys else []
    return []

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def find_human_polygons(
    image: np.ndarray,
    *,
    backend: Backend = "yolov8s",
    device: str = "cuda",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    verbose: bool = False,
) -> List[Polygon]:
    """
    Detect all humans in *image* and return their outlines as Shapely Polygons.

    Parameters
    ----------
    image : np.ndarray
        Input image.  NumPy arrays may be BGR (OpenCV) or RGB — the model
        handles both because Ultralytics normalises internally.
    backend : str
        Which YOLO segmentation model to use.
        Format: "yolov8{n|s|m|l|x}" or "yolov11{n|s|m|l|x}".
        Larger variants are slower but more accurate.
        Default: "yolov8s"  (second quickest YOLOv8).
    device : str
        PyTorch device string, e.g. "cuda", "cuda:0", "mps", "cpu".
        Default: "cuda".
    conf : float
        Minimum confidence threshold for detections.  Default: 0.25.
    iou : float
        IoU threshold for non-maximum suppression.  Default: 0.45.
    imgsz : int
        Inference image size (pixels, square).  Larger = more accurate but
        slower.  Must be a multiple of 32.  Default: 640.
    verbose : bool
        If True, print Ultralytics model output.  Default: False.

    Returns
    -------
    List[Polygon]
        One Shapely Polygon per detected person.  Coordinates are in the
        original image pixel space (x=col, y=row).
        Returns an empty list when no humans are detected.
    """
    # ------------------------------------------------------------------
    # 1. Lazy-import Ultralytics (avoids paying import cost if unused)
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required:  pip install ultralytics"
        ) from exc

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    model_name = _MODEL_NAMES.get(backend)
    if model_name is None:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Choose from: {sorted(_MODEL_NAMES)}"
        )

    model_path = constants.model_root / 'yolo' / model_name
    model = YOLO(model_path)  # downloads weights on first use, then caches

    # ------------------------------------------------------------------
    # 3. Load / validate image
    # ------------------------------------------------------------------
    h, w = image.shape[:2]

    # ------------------------------------------------------------------
    # 4. Run inference  (classes=[0] → person only, skips NMS on others)
    # ------------------------------------------------------------------
    return model.predict(
        source=image,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=[0],          # COCO class 0 = "person"
        retina_masks=True,    # full-resolution masks (most accurate)
        verbose=verbose,
    )

def to_polygons(model_output, height, width):
    polygons: List[Polygon] = []

    for result in model_output:
        if result.masks is None:
            continue

        # masks.data: Tensor of shape (N, H_mask, W_mask) on GPU/CPU
        masks_np = result.masks.data.cpu().numpy()  # float32, values in [0,1]

        for mask_f in masks_np:
            # Binarise and ensure uint8 for findContours
            mask_u8 = (mask_f > 0.5).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

            # Mask may be at a different resolution than the source image;
            # resize back if needed (retina_masks=True keeps it at src size).
            if mask_u8.shape != (height, width):
                mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)

            polys = _mask_to_polygons(mask_u8)
            final_polygons = []
            for p in polys:
                final_polygons.extend(_extract_polygons(p))
            polygons.extend(final_polygons)

    return polygons