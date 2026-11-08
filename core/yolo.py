"""YOLO ONNX models: YOLOv8/11 or end-to-end YOLOv10/26, detect or segment."""
from __future__ import annotations

import math
import re
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config_utils import did_you_mean
from .geometry import Detection, Polygon, chaikin, cround
from .inference import DetectorOutput, InferenceWorker, OnnxSession
from .pipeline_config import Device, YoloOptions, auto_threads, layout_from_string, resize_mode_from_string

MIN_PART_AREA = 100.0       # px^2
MIN_PART_FRACTION = 0.005   # of the largest part
CLASS_OFFSET = 7680.0       # per-class NMS in one call by shifting each class to its own region


def parse_yolo_names(text: str) -> list[str]:
    """Parses Ultralytics' names metadata, e.g. "{0: 'person', 1: 'bicycle'}"."""
    pairs: list[tuple[int, str]] = []
    for m in re.finditer(r"""(\d+)\s*:\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")""", text):
        pairs.append((int(m.group(1)), m.group(2) if m.group(2) is not None else m.group(3)))
    if not pairs:
        for i, m in enumerate(re.finditer(r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\"""", text)):
            pairs.append((i, m.group(1) if m.group(1) is not None else m.group(2)))
    if not pairs:
        return []
    names = [f"class_{i}" for i in range(max(i for i, _ in pairs) + 1)]
    for i, name in pairs:
        names[i] = name
    return names


def parse_yolo_imgsz(text: str) -> tuple[int, int]:
    """Parses Ultralytics' imgsz metadata, e.g. "[640, 640]" (H, W) or "640". Returns (W, H)."""
    values = [int(v) for v in re.findall(r"\d+", text)]
    if len(values) == 1:
        return values[0], values[0]
    if len(values) >= 2:
        return values[1], values[0]
    return 0, 0


@dataclass
class InputTransform:
    """Maps model-input coordinates back to the original image: orig = (model - pad) / scale."""
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    original: tuple[int, int] = (0, 0)  # W, H
    input: tuple[int, int] = (0, 0)     # W, H


class YoloDecoder:
    """
    Pre- and post-processing for one YOLO ONNX export. Reads class names, input size, task and layout
    from the model (Ultralytics metadata + tensor shapes), with YoloOptions as overrides.
    """

    def __init__(self, session: OnnxSession, options: YoloOptions):
        self.options = replace(options, classes=list(options.classes))
        self.model_name = session.name

        def fail(msg: str):
            raise RuntimeError(f"[{self.model_name}] {msg}")

        meta = session.metadata

        task = meta.get('task', '')
        if task and task not in ('detect', 'segment'):
            fail(f"unsupported task '{task}'; only detect and segment models are supported")

        # ------------------------------------ input ------------------------------------
        inp = session.input_shape
        if len(inp) != 4:
            fail("expected a 4D image input (N, C, H, W)")
        if inp[1] > 0 and inp[1] != 3:
            fail(f"expected 3 input channels, got {inp[1]}")
        self.dynamic = inp[2] <= 0 or inp[3] <= 0

        size = (640, 640)
        if not self.dynamic:
            size = (inp[3], inp[2])
        elif self.options.imgsz > 0:
            size = (self.options.imgsz, self.options.imgsz)
        elif meta.get('imgsz'):
            size = parse_yolo_imgsz(meta['imgsz'])
        if size[0] * size[1] <= 0:
            size = (640, 640)
        self.input_size = size

        if self.options.resize == 'rect' and not self.dynamic:
            fail(f"resize 'rect' needs a model with a dynamic input size; this model expects {size[0]}x{size[1]}")

        # ------------------------------------ names ------------------------------------
        names: list[str] = []
        if self.options.names:
            names = list(self.options.names)
        elif meta.get('names'):
            names = parse_yolo_names(meta['names'])

        # ------------------------------------ outputs ------------------------------------
        outs = session.output_shapes
        if not outs or len(outs[0]) != 3:
            fail("expected a 3D first output")
        self.num_masks = 0
        if len(outs) >= 2 and len(outs[1]) == 4:
            self.num_masks = outs[1][1]
            if self.num_masks <= 0:
                fail("segmentation prototype output has a dynamic channel count")

        d1, d2 = outs[0][1], outs[0][2]
        nc, nm = len(names), self.num_masks

        # ------------------------------------ layout ------------------------------------
        layout = self.options.layout
        if layout == 'auto':
            if meta.get('end2end', '').lower() == 'true':
                layout = 'end2end'
            elif d2 == 6 + nm and not (nc > 0 and d1 == 4 + nc + nm):
                layout = 'end2end'
            else:
                layout = 'v8'
        self.layout = layout

        if layout == 'v8':
            if 0 < d1 <= 4 + nm:
                fail(f"output shape [1, {d1}, N] has no class scores")
            if nc > 0 and d1 > 0 and d1 != 4 + nc + nm:
                fail(f"model has {d1 - 4 - nm} classes but {nc} names")
        elif d2 > 0 and d2 != 6 + nm:
            fail(f"end2end output should have {6 + nm} values per row, got {d2}")

        # ------------------------------------ labels ------------------------------------
        self.num_classes = nc if nc > 0 else (d1 - 4 - nm if layout == 'v8' and d1 > 0 else 0)
        if not names:
            names = [f"class_{i}" for i in range(self.num_classes)]

        model_names = list(names)
        for src in self.options.label_map:
            if src not in model_names:
                fail(f"labels: the model has no class '{src}'{did_you_mean(src, model_names)}")
        self.names = [self.options.label_map.get(n, self.options.label_prefix + n) for n in model_names]

        # Classes may be given by model name or by output label.
        for wanted in self.options.class_names:
            if wanted in model_names:
                cid = model_names.index(wanted)
            elif wanted in self.names:
                cid = self.names.index(wanted)
            else:
                fail(f"classes: the model has no class '{wanted}'{did_you_mean(wanted, model_names + self.names)}")
            self.options.classes.append(cid)

        self._allowed: np.ndarray | None = None
        if self.options.classes:
            n = max(max(self.options.classes) + 1, self.num_classes, 1)
            self._allowed = np.zeros(n, dtype=bool)
            self._allowed[[c for c in self.options.classes if c >= 0]] = True

        print(f"[{self.model_name}] YOLO {self.task}, {self.layout} layout, "
              f"{len(self.names) if self.names else 'unnamed'} classes, input "
              f"{'dynamic' if self.dynamic else f'{size[0]}x{size[1]}'}, resize {self.options.resize}", flush=True)

    # ------------------------------------ properties ------------------------------------
    @property
    def task(self) -> str:
        return 'segment' if self.num_masks > 0 else 'detect'

    def label(self, class_id: int) -> str:
        if 0 <= class_id < len(self.names):
            return self.names[class_id]
        return f"class_{class_id}"

    def class_allowed(self, class_id: int) -> bool:
        if self._allowed is None:
            return True
        return 0 <= class_id < len(self._allowed) and bool(self._allowed[class_id])

    def output_labels(self) -> list[str]:
        return [n for i, n in enumerate(self.names) if self.class_allowed(i)]

    # ------------------------------------ preprocessing ------------------------------------
    def prepare(self, image: np.ndarray, requested_size: int) -> tuple[np.ndarray, InputTransform]:
        """Resizes and normalizes the image. requested_size is used only for dynamic-input models."""
        rows, cols = image.shape[:2]
        t = InputTransform(original=(cols, rows))

        W, H = self.input_size
        if self.dynamic:
            target = requested_size if requested_size > 0 else max(W, H)
            if self.options.resize == 'rect':
                # Like Ultralytics' LetterBox(auto=True): fit in target x target, then pad to a multiple of 32.
                r = min(np.float32(target) / np.float32(rows), np.float32(target) / np.float32(cols))
                new_w, new_h = int(cround(cols * r)), int(cround(rows * r))
                W = new_w + (target - new_w) % 32
                H = new_h + (target - new_h) % 32
            else:
                W = H = target
        t.input = (W, H)

        if self.options.resize in ('letterbox', 'rect'):
            r = min(np.float32(H) / np.float32(rows), np.float32(W) / np.float32(cols))
            new_w, new_h = int(cround(cols * r)), int(cround(rows * r))
            left = int(cround((W - new_w) / 2.0 - 0.1))
            top = int(cround((H - new_h) / 2.0 - 0.1))
            resized = image if (new_w, new_h) == (cols, rows) else \
                cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            sized = cv2.copyMakeBorder(resized, top, H - new_h - top, left, W - new_w - left,
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))
            t.scale_x, t.scale_y = new_w / cols, new_h / rows
            t.pad_x, t.pad_y = float(left), float(top)
        else:
            sized = cv2.resize(image, (W, H))
            t.scale_x, t.scale_y = W / cols, H / rows

        if sized.ndim == 2:
            sized = cv2.cvtColor(sized, cv2.COLOR_GRAY2BGR)
        blob = cv2.dnn.blobFromImage(sized, 1.0 / 255.0, swapRB=True)  # (1, 3, H, W) RGB float32
        return blob, t

    # ------------------------------------ decoding ------------------------------------
    def _decode_v8(self, output: DetectorOutput) -> list[tuple[int, float, float, float, float, float, int]]:
        data = output[0][0]  # (C, A)
        nc = data.shape[0] - 4 - self.num_masks
        if nc <= 0:
            raise RuntimeError(f"[{self.model_name}] output has no class scores")

        scores = data[4:4 + nc]
        cls = np.argmax(scores, axis=0)
        best = scores[cls, np.arange(scores.shape[1])]
        keep = (best > 0) & (best >= self.options.min_confidence)
        if self._allowed is not None:
            allowed = np.zeros_like(keep)
            in_range = cls < len(self._allowed)
            allowed[in_range] = self._allowed[cls[in_range]]
            keep &= allowed
        idx = np.nonzero(keep)[0]
        if len(idx) == 0:
            return []

        cx, cy, w, h = data[0, idx], data[1, idx], data[2, idx], data[3, idx]
        x1, y1, x2, y2 = cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h
        cls_k, best_k = cls[idx], best[idx]

        off = np.zeros(len(idx), np.float64) if self.options.agnostic_nms else cls_k * CLASS_OFFSET
        boxes = np.stack([x1 + off, y1 + off, x2 - x1, y2 - y1], axis=1).astype(np.float64)
        kept = cv2.dnn.NMSBoxes(boxes.tolist(), best_k.astype(np.float32).tolist(), 0.0, self.options.nms_iou,
                                1.0, max(0, self.options.max_det))
        kept = np.asarray(kept, dtype=np.int64).reshape(-1)
        return [(int(cls_k[i]), float(best_k[i]), float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), int(idx[i]))
                for i in kept]

    def _decode_end2end(self, output: DetectorOutput, t: InputTransform):
        data = output[0][0]  # (rows, 6 + nm)
        W, H = float(t.input[0]), float(t.input[1])
        x1 = np.clip(data[:, 0], 0, W)
        y1 = np.clip(data[:, 1], 0, H)
        x2 = np.clip(data[:, 2], 0, W)
        y2 = np.clip(data[:, 3], 0, H)
        conf = data[:, 4]
        cls = data[:, 5].astype(np.int64)
        keep = (x2 > x1) & (y2 > y1) & (conf > 0) & (conf >= self.options.min_confidence)
        rows = [r for r in np.nonzero(keep)[0] if self.class_allowed(int(cls[r]))]
        if self.options.max_det > 0:
            rows = rows[:self.options.max_det]
        return [(int(cls[r]), float(conf[r]), float(x1[r]), float(y1[r]), float(x2[r]), float(y2[r]), int(r))
                for r in rows]

    def _mask_polygons(self, c, output: DetectorOutput, t: InputTransform) -> list[Polygon]:
        cls_id, score, cx1, cy1, cx2, cy2, index = c
        protos = output[1][0]  # (nm, mh, mw)
        nm, mh, mw = protos.shape

        if self.layout == 'v8':
            data = output[0][0]
            nc = data.shape[0] - 4 - nm
            coeffs = data[4 + nc:4 + nc + nm, index]
        else:
            coeffs = output[0][0][index, 6:6 + nm]

        # Mask logits at prototype resolution, inside the box plus a 1-pixel margin for upsampling.
        in_w, in_h = float(t.input[0]), float(t.input[1])
        rx, ry = mw / in_w, mh / in_h
        px1 = int(np.clip(math.floor(cx1 * rx) - 1, 0, mw))
        py1 = int(np.clip(math.floor(cy1 * ry) - 1, 0, mh))
        px2 = int(np.clip(math.ceil(cx2 * rx) + 1, 0, mw))
        py2 = int(np.clip(math.ceil(cy2 * ry) + 1, 0, mh))
        if px2 <= px1 or py2 <= py1:
            return []

        logits = np.full((mh, mw), -50.0, dtype=np.float32)
        region = protos[:, py1:py2, px1:px2].reshape(nm, -1)
        logits[py1:py2, px1:px2] = (coeffs.astype(np.float32) @ region).reshape(py2 - py1, px2 - px1)

        # Box in original-image pixels.
        ow, oh = t.original
        bx1 = int(np.clip(math.floor((cx1 - t.pad_x) / t.scale_x), 0, ow))
        by1 = int(np.clip(math.floor((cy1 - t.pad_y) / t.scale_y), 0, oh))
        bx2 = int(np.clip(math.ceil((cx2 - t.pad_x) / t.scale_x), 0, ow))
        by2 = int(np.clip(math.ceil((cy2 - t.pad_y) / t.scale_y), 0, oh))
        if bx2 <= bx1 or by2 <= by1:
            return []

        # Bilinear upsampling of just the box (the prototype grid covers the padded model input).
        sx, sy = rx * t.scale_x, ry * t.scale_y
        ox = (bx1 + 0.5) * sx + t.pad_x * rx - 0.5
        oy = (by1 + 0.5) * sy + t.pad_y * ry - 0.5
        affine = np.array([[sx, 0, ox], [0, sy, oy]], dtype=np.float64)
        box_logits = cv2.warpAffine(logits, affine, (bx2 - bx1, by2 - by1),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

        th = min(max(self.options.mask_threshold, 1e-4), 1 - 1e-4)
        cut = math.log(th / (1 - th))
        binary = (box_logits > cut).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(bx1, by1))
        if not contours:
            return []

        # Keep every part except specks: a detached hand matters, a few stray pixels do not.
        areas = [cv2.contourArea(cnt) for cnt in contours]
        min_area = max(MIN_PART_AREA, MIN_PART_FRACTION * max(areas))
        parts = []
        for cnt, area in zip(contours, areas):
            if area < min_area:
                continue
            simplified = cv2.approxPolyDP(cnt, 5.0, True).reshape(-1, 2)
            if len(simplified) >= 3:
                parts.append(chaikin(simplified, 2))
        parts.sort(key=lambda p: cv2.contourArea(p), reverse=True)
        return parts

    def decode(self, output: DetectorOutput, t: InputTransform) -> list[Detection]:
        if not output or output[0].ndim != 3:
            raise RuntimeError(f"[{self.model_name}] unexpected output")
        if self.num_masks > 0 and (len(output) < 2 or output[1].ndim != 4):
            raise RuntimeError(f"[{self.model_name}] missing segmentation prototype output")

        candidates = self._decode_v8(output) if self.layout == 'v8' else self._decode_end2end(output, t)
        W, H = float(t.original[0]), float(t.original[1])

        result = []
        for c in candidates:
            cls_id, score, cx1, cy1, cx2, cy2, _ = c
            x1 = min(max((cx1 - t.pad_x) / t.scale_x, 0.0), W)
            y1 = min(max((cy1 - t.pad_y) / t.scale_y, 0.0), H)
            x2 = min(max((cx2 - t.pad_x) / t.scale_x, 0.0), W)
            y2 = min(max((cy2 - t.pad_y) / t.scale_y, 0.0), H)
            if x2 <= x1 or y2 <= y1:
                continue
            det = Detection(label=self.label(cls_id), score=score, box=(x1, y1, x2 - x1, y2 - y1))
            if self.num_masks > 0:
                det.polygons = self._mask_polygons(c, output, t)
            result.append(det)
        return result


@dataclass
class PendingYolo:
    future: Future
    transform: InputTransform


class YoloRunner:
    """A YOLO model together with the worker thread that runs it."""

    def __init__(self, name: str, model_path: str, device: Device, options: YoloOptions):
        self.worker = InferenceWorker(OnnxSession(model_path, device, name))
        self.model = YoloDecoder(self.worker.session, options)
        self.name = name

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    def submit(self, image: np.ndarray, requested_size: int) -> PendingYolo:
        tensor, transform = self.model.prepare(image, requested_size)
        return PendingYolo(self.worker.submit([tensor]), transform)

    def collect(self, pending: PendingYolo) -> list[Detection]:
        outputs = pending.future.result()
        if len(outputs) != 1:
            raise RuntimeError(f"[{self.name}] inference returned no output")
        return self.model.decode(outputs[0], pending.transform)

    def detect(self, image: np.ndarray, requested_size: int) -> list[Detection]:
        return self.collect(self.submit(image, requested_size))


def detection_to_dict(d: Detection) -> dict[str, Any]:
    return {
        'label': d.label,
        'score': float(d.score),
        'box': tuple(float(v) for v in d.box),
        'polygons': [[(int(x), int(y)) for x, y in part] for part in d.polygons],
    }


def check_image(image: Any) -> np.ndarray:
    """A contiguous HxWx3 uint8 copy of image, or RuntimeError."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise RuntimeError("Image array must be HxWxC (3 dimensions).")
    if array.shape[2] != 3:
        raise RuntimeError("Image array must have 3 channels (BGR).")
    return np.array(array, dtype=np.uint8, copy=True, order='C')


class YoloModel:
    """
    A YOLO ONNX model (YOLOv8/11 or end-to-end YOLOv10/26, detect or segment).

    Class names, input size, task and output layout are read from the model; the keyword arguments override them.
    """

    def __init__(self, path: str, *, gpu: bool = False, cuda_device: int = 0, resize: str = 'letterbox',
                 layout: str = 'auto', min_confidence: float = 0.25, nms_iou: float = 0.7, agnostic_nms: bool = False,
                 max_det: int = 300, mask_threshold: float = 0.5, classes: list[int | str] = (),
                 names: list[str] = (), imgsz: int = 0, labels: dict[str, str] | None = None,
                 label_prefix: str = '', threads: int = 0):
        options = YoloOptions(
            resize=resize_mode_from_string(resize),
            layout=layout_from_string(layout),
            min_confidence=float(min_confidence),
            nms_iou=float(nms_iou),
            agnostic_nms=bool(agnostic_nms),
            max_det=int(max_det),
            mask_threshold=float(mask_threshold),
            classes=[int(c) for c in classes if not isinstance(c, str)],
            class_names=[c for c in classes if isinstance(c, str)],
            names=list(names),
            label_map=dict(labels or {}),
            label_prefix=label_prefix,
            imgsz=int(imgsz),
        )
        device = Device(gpu, cuda_device, threads if threads > 0 else (1 if gpu else auto_threads(1)))
        self._runner = YoloRunner(Path(path).name, str(path), device, options)
        self._runner.start()

    def __del__(self):
        runner = getattr(self, '_runner', None)
        if runner is not None:
            runner.stop()

    def detect(self, image: np.ndarray, size: int = 0) -> list[dict[str, Any]]:
        """Runs the model on a BGR uint8 image. size sets the input size of dynamic-input models."""
        return [detection_to_dict(d) for d in self._runner.detect(check_image(image), size)]

    @property
    def names(self) -> list[str]:
        return list(self._runner.model.names)

    @property
    def labels(self) -> list[str]:
        return self._runner.model.output_labels()

    @property
    def task(self) -> str:
        return self._runner.model.task

    @property
    def layout(self) -> str:
        return self._runner.model.layout

    @property
    def dynamic_input(self) -> bool:
        return self._runner.model.dynamic

    @property
    def input_size(self) -> tuple[int, int]:
        return tuple(self._runner.model.input_size)
