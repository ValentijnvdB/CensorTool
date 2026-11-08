"""The pipeline config (pipeline.yml): which models a pipeline loads."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config_utils import read_config_file, warn_unknown_keys

LANDMARKS_STAGE = 'landmarks'

ROOT_KEYS = ["device", "threads", "models", "landmarks"]
MODEL_KEYS = [
    "name", "path", "device", "resize", "layout", "imgsz", "min_confidence", "nms_iou", "agnostic_nms",
    "max_det", "mask_threshold", "classes", "names", "labels", "label_prefix", "run_on", "crop_padding", "threads",
]
LANDMARK_KEYS = ["path", "device", "run_on", "min_face_score", "threads"]

RESIZE_MODES = ('letterbox', 'stretch', 'rect')


@dataclass
class Device:
    gpu: bool = False
    cuda_device_id: int = 0
    threads: int = 1  # ONNX Runtime intra-op CPU threads; 0 = resolve automatically


@dataclass
class YoloOptions:
    resize: str = 'letterbox'          # letterbox | stretch | rect
    layout: str = 'auto'               # auto | v8 | end2end
    min_confidence: float = 0.25
    nms_iou: float = 0.7               # v8 layout only
    agnostic_nms: bool = False         # NMS across classes instead of per class
    max_det: int = 300                 # 0 = no limit
    mask_threshold: float = 0.5
    classes: list[int] = field(default_factory=list)        # keep only these class ids; empty keeps all
    class_names: list[str] = field(default_factory=list)    # keep only these classes, by (model or output) name
    names: list[str] = field(default_factory=list)          # overrides the class names in the model
    label_map: dict[str, str] = field(default_factory=dict)  # model name -> output label
    label_prefix: str = ''             # prepended to names not in label_map
    imgsz: int = 0                     # input size if the model doesn't fix it; 0 = metadata or 640


@dataclass
class ModelSpec:
    name: str
    path: str
    device: Device
    options: YoloOptions
    run_on: list[str] = field(default_factory=list)  # cascade: run on crops of these labels (empty = whole image)
    crop_padding: float = 0.1                        # grow each crop by this fraction of its size


@dataclass
class LandmarkSpec:
    path: str
    device: Device
    run_on: dict[str, str]          # face label -> eye label
    min_face_score: float = 0.05


@dataclass
class PipelineConfig:
    models: list[ModelSpec]
    landmarks: Optional[LandmarkSpec] = None


def auto_threads(cpu_sessions: int) -> int:
    """Intra-op threads for one of cpu_sessions CPU models that share the machine."""
    return max(1, (os.cpu_count() or 1) // max(1, cpu_sessions))


def parse_device(value: Any) -> Device:
    v = str(value).lower()
    if v == 'cpu':
        return Device(False, 0)
    if v in ('gpu', 'cuda'):
        return Device(True, 0)
    if v.startswith('cuda:'):
        try:
            return Device(True, int(v[5:]))
        except ValueError:
            pass
    raise ValueError(f"unknown device '{value}' (expected cpu, cuda or cuda:N)")


def resize_mode_from_string(value: str) -> str:
    v = str(value).lower()
    if v not in RESIZE_MODES:
        raise ValueError(f"unknown resize mode '{value}' (expected letterbox, stretch or rect)")
    return v


def layout_from_string(value: str) -> str:
    v = str(value).lower()
    if v == 'auto':
        return 'auto'
    if v in ('v8', 'v11'):
        return 'v8'
    if v in ('end2end', 'endtoend'):
        return 'end2end'
    raise ValueError(f"unknown layout '{value}' (expected auto, v8 or end2end)")


def _parse_threads(value: Any, where: str) -> int:
    """threads: a positive number or 'auto' (returned as 0, resolved once every model is known)."""
    if value == 'auto':
        return 0
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    raise RuntimeError(f"{where + '.' if where else ''}threads must be a positive number or 'auto'")


def _resolve_path(p: str, base_dir: Path) -> str:
    path = Path(p)
    return os.path.normpath(path if path.is_absolute() else base_dir / path)


def _string_list(value: Any, where: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not all(isinstance(v, str) for v in value):
            raise RuntimeError(f"{where} must contain only labels")
        return list(value)
    raise RuntimeError(f"{where} must be a label or a list of labels")


def _parse_model(d: Any, default_device: Device, base_dir: Path, index: int) -> ModelSpec:
    where = f"models[{index}]"
    if not isinstance(d, dict):
        raise RuntimeError(f"{where} must be a mapping")
    if 'path' not in d:
        raise RuntimeError(f"{where} is missing 'path'")
    warn_unknown_keys(d, MODEL_KEYS, where)

    path = _resolve_path(str(d['path']), base_dir)
    name = str(d.get('name', Path(path).stem))
    device = parse_device(d['device']) if 'device' in d else Device(default_device.gpu, default_device.cuda_device_id)
    here = f"model '{name}'"
    device.threads = _parse_threads(d['threads'], here) if 'threads' in d else default_device.threads

    o = YoloOptions()
    if 'resize' in d:
        o.resize = resize_mode_from_string(d['resize'])
    if 'layout' in d:
        o.layout = layout_from_string(d['layout'])
    if 'imgsz' in d:
        o.imgsz = int(d['imgsz'])
    if 'min_confidence' in d:
        o.min_confidence = float(d['min_confidence'])
    if 'nms_iou' in d:
        o.nms_iou = float(d['nms_iou'])
    if 'agnostic_nms' in d:
        o.agnostic_nms = bool(d['agnostic_nms'])
    if 'max_det' in d:
        o.max_det = int(d['max_det'])
    if 'mask_threshold' in d:
        o.mask_threshold = float(d['mask_threshold'])
    if 'names' in d:
        o.names = _string_list(d['names'], f"{here}.names")
    if 'label_prefix' in d:
        o.label_prefix = str(d['label_prefix'])

    if 'classes' in d:
        classes = d['classes']
        if not isinstance(classes, list):
            raise RuntimeError(f"{here}.classes must be a list")
        for c in classes:
            if isinstance(c, int) and not isinstance(c, bool):
                o.classes.append(c)
            elif isinstance(c, str):
                o.class_names.append(c)
            else:
                raise RuntimeError(f"{here}.classes must contain class names or ids")
    if 'labels' in d:
        if not isinstance(d['labels'], dict):
            raise RuntimeError(f"{here}.labels must map model names to labels")
        o.label_map = {str(k): str(v) for k, v in d['labels'].items()}

    spec = ModelSpec(name=name, path=path, device=device, options=o)
    if 'run_on' in d:
        spec.run_on = _string_list(d['run_on'], f"{here}.run_on")
    if 'crop_padding' in d:
        spec.crop_padding = float(d['crop_padding'])
    if spec.crop_padding < 0:
        raise RuntimeError(f"{here}.crop_padding must be >= 0")
    return spec


def _parse_landmarks(d: Any, default_device: Device, base_dir: Path) -> LandmarkSpec:
    if not isinstance(d, dict):
        raise RuntimeError("landmarks must be a mapping")
    if 'path' not in d:
        raise RuntimeError("landmarks is missing 'path'")
    if not isinstance(d.get('run_on'), dict) or not d['run_on']:
        raise RuntimeError("landmarks.run_on must map face labels to eye labels, e.g. {FACE_MALE: EYES_MALE}")
    warn_unknown_keys(d, LANDMARK_KEYS, "landmarks")

    device = parse_device(d['device']) if 'device' in d else Device(default_device.gpu, default_device.cuda_device_id)
    device.threads = _parse_threads(d['threads'], "landmarks") if 'threads' in d else default_device.threads
    return LandmarkSpec(
        path=_resolve_path(str(d['path']), base_dir),
        device=device,
        run_on={str(k): str(v) for k, v in d['run_on'].items()},
        min_face_score=float(d.get('min_face_score', 0.05)),
    )


def pipeline_config_from_dict(root: Any, base_dir: Path) -> PipelineConfig:
    """
    Parses a pipeline config. Relative model paths are resolved against base_dir
    (the folder of the YAML file, or the working directory for Python dicts).
    """
    if not isinstance(root, dict):
        raise RuntimeError("the pipeline config must be a mapping")
    warn_unknown_keys(root, ROOT_KEYS, "the pipeline config")

    device = parse_device(root.get('device', 'cpu'))
    device.threads = _parse_threads(root['threads'], '') if 'threads' in root else 0

    models = root.get('models')
    if models is not None and not isinstance(models, list):
        raise RuntimeError("models must be a list")
    config = PipelineConfig(models=[_parse_model(m, device, base_dir, i) for i, m in enumerate(models or [])])
    if root.get('landmarks') is not None:
        config.landmarks = _parse_landmarks(root['landmarks'], device, base_dir)

    if not config.models:
        raise RuntimeError("the pipeline config has no models")

    # threads: auto -> the CPU models split the cores; a GPU model needs only one.
    devices = [m.device for m in config.models] + ([config.landmarks.device] if config.landmarks else [])
    cpu_sessions = sum(1 for d in devices if not d.gpu)
    for d in devices:
        if d.threads == 0:
            d.threads = 1 if d.gpu else auto_threads(cpu_sessions)

    names = set()
    for m in config.models:
        if m.name == LANDMARKS_STAGE:
            raise RuntimeError("model name 'landmarks' is reserved")
        if m.name in names:
            raise RuntimeError(f"two models are named '{m.name}'; give them different names")
        names.add(m.name)
    return config


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Loads a pipeline config from a .yml/.yaml/.json file; relative paths are relative to the file."""
    return pipeline_config_from_dict(read_config_file(path), Path(path).absolute().parent)
