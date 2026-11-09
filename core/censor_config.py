"""The censor config: what to censor and how. Same format and validation as the C++ backend."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from .config_utils import did_you_mean, log_warning, read_config_file, warn_unknown_keys


class Shape(Enum):
    DEFAULT = 'DEFAULT'      # the detection's own shape: its outline if it has one, otherwise its box
    ELLIPSE = 'ELLIPSE'
    CIRCLE = 'CIRCLE'
    RECTANGLE = 'RECTANGLE'
    MASK = 'MASK'            # the detection's segmentation outline (segmentation models only)


Color = tuple[int, int, int]  # BGR


# Defaults for settings a censor config leaves out (documented in docs/censor_config.md).
DEFAULT_BLUR_STRENGTH = 31   # also the fallback style for a missing or unknown type, and the inverse censor style
DEFAULT_PIXEL_FACTOR = 20
DEFAULT_AREA_SAFETY = 0.1    # width_area_safety and height_area_safety


# -------------------------------- Censor styles --------------------------------
@dataclass(frozen=True)
class CsBlur:
    strength: int


@dataclass(frozen=True)
class CsPixel:
    factor: int


@dataclass(frozen=True)
class CsBar:
    color: Color


@dataclass(frozen=True)
class CsDebug:
    color: Color


@dataclass(frozen=True)
class CsAIRemove:
    comfy_host: str
    comfy_port: int
    comfy_workflow: str


CensorStyle = Union[CsBlur, CsPixel, CsBar, CsAIRemove, CsDebug]


# -------------------------------- Overlays --------------------------------
@dataclass(frozen=True)
class OvText:
    values: tuple[str, ...]
    probability: float
    color: Color
    thickness: int
    font_scale: float
    font: int


@dataclass(frozen=True)
class OvSticker:
    probability: float
    categories: tuple[str, ...]


Overlay = Union[OvText, OvSticker]


@dataclass(frozen=True)
class Border:
    color: Color
    thickness: int


@dataclass
class CensorBox:
    style: CensorStyle
    overlay: Optional[Overlay]
    shape: Shape
    border: Optional[Border]
    width_area_safety: float
    height_area_safety: float
    time_safety: float
    min_probability: float
    inverse: bool
    clip_to: list[str] = field(default_factory=list)  # clip the shape to these (segmentation) labels


@dataclass
class CensorConfig:
    features_to_censor: dict[str, CensorBox]
    merge_overlapping_borders: bool
    enable_overlays: bool
    force_inverse_censor: bool
    inverse_censor_style: Union[CsBlur, CsPixel, CsBar]
    enable_watermark: bool
    stickers_dir: Path
    clip_min_probability: dict[str, float]  # clip_to target -> score threshold


# -------------------------------------- Known keys --------------------------------------
ROOT_KEYS = [
    "features_to_censor", "default_censor_config", "feature_overrides",
    "merge_overlapping_borders", "enable_overlays",
    "force_inverse_censor", "inverse_censor_style", "enable_watermark", "stickers_dir",
]
REMOVED_ROOT_KEYS = {
    "body_conf_threshold": "set min_probability on the segmentation label (e.g. feature_overrides.BODY).",
    "body_seg_override": "models now run when the censor config needs their labels.",
    "merge_overlapping_censor_boxes": "it merged duplicate detections from multi-pass detection, "
                                      "which censor_core never had.",
}
CENSOR_BOX_KEYS = [
    "censor_style", "overlay", "shape", "border", "width_area_safety", "height_area_safety",
    "time_safety", "min_probability", "inverse", "clip_to",
]
REMOVED_BOX_KEYS = {
    "intersect_human": "use clip_to: [<segmentation label>], e.g. clip_to: [BODY].",
}


# -------------------------------------- Helpers --------------------------------------
def _parse_color(value: Any) -> Color:
    """RGB in the config, BGR in the image."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        log_warning("Invalid or missing color, defaulting to black.")
        return 0, 0, 0
    r, g, b = (float(v) for v in value[:3])
    return b, g, r


def _parse_shape(value: Any) -> Shape:
    shape = str(value).upper()
    try:
        return Shape(shape)
    except ValueError:
        log_warning(f"Unknown shape '{shape}', defaulting to DEFAULT{did_you_mean(shape, [s.value for s in Shape])}.")
        return Shape.DEFAULT


def _parse_censor_style(d: Any, where: str) -> CensorStyle:
    if not isinstance(d, dict) or 'type' not in d:
        log_warning(f"Missing censor_style type, defaulting to blur({DEFAULT_BLUR_STRENGTH}).")
        return CsBlur(DEFAULT_BLUR_STRENGTH)

    kind = str(d['type'])
    here = f"{where} (type '{kind}')"
    known = {
        'blur': ['type', 'strength'],
        'pixel': ['type', 'factor'],
        'bar': ['type', 'color'],
        'debug': ['type', 'color'],
        'ai_remove': ['type', 'comfy_host', 'comfy_port', 'comfy_workflow'],
    }
    if kind in known:
        warn_unknown_keys(d, known[kind], here)

    if kind == 'blur':
        return CsBlur(int(d.get('strength', DEFAULT_BLUR_STRENGTH)))
    if kind == 'pixel':
        return CsPixel(int(d.get('factor', DEFAULT_PIXEL_FACTOR)))
    if kind == 'bar':
        return CsBar(_parse_color(d.get('color', [])))
    if kind == 'ai_remove':
        if not all(k in d for k in ('comfy_host', 'comfy_port', 'comfy_workflow')):
            raise RuntimeError("censor_style 'ai_remove' requires 'comfy_host', 'comfy_port' and 'comfy_workflow'.")
        return CsAIRemove(str(d['comfy_host']), int(d.get('comfy_port', 8188)), str(d['comfy_workflow']))
    if kind == 'debug':
        return CsDebug(_parse_color(d.get('color', [])))

    log_warning(f"Unknown censor_style type '{kind}', defaulting to blur({DEFAULT_BLUR_STRENGTH}).")
    return CsBlur(DEFAULT_BLUR_STRENGTH)


def _parse_overlay(d: Any, where: str) -> Optional[Overlay]:
    if not isinstance(d, dict) or 'type' not in d:
        return None

    kind = str(d['type'])
    if kind == 'text':
        warn_unknown_keys(d, ['type', 'values', 'probability', 'color', 'thickness', 'font_scale', 'font'], where)
        values = d.get('values')
        return OvText(
            values=tuple(str(v) for v in values) if isinstance(values, list) else (),
            probability=float(d.get('probability', 1.0)),
            color=_parse_color(d.get('color', [])),
            thickness=int(d.get('thickness', 1)),
            font_scale=float(d.get('font_scale', 1.0)),
            font=int(d.get('font', 0)),
        )
    if kind == 'sticker':
        warn_unknown_keys(d, ['type', 'probability', 'categories'], where)
        categories = d.get('categories')
        return OvSticker(
            probability=float(d.get('probability', 1.0)),
            categories=tuple(str(c) for c in categories) if isinstance(categories, list) else (),
        )

    log_warning(f"Unknown overlay type '{kind}', skipping overlay.")
    return None


def _parse_border(d: Any, where: str) -> Optional[Border]:
    if d is None:
        return None
    if not isinstance(d, dict):
        d = {}
    warn_unknown_keys(d, ['color', 'thickness'], where)
    return Border(color=_parse_color(d.get('color', [])), thickness=int(d.get('thickness', 1)))


def _parse_label_list(d: dict, key: str, where: str) -> list[str]:
    value = d.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise RuntimeError(f"{where}.{key} must be a label or a list of labels.")


def _parse_censor_box(d: Any, where: str) -> CensorBox:
    if not isinstance(d, dict) or 'censor_style' not in d:
        raise RuntimeError(f"{where} is missing required key 'censor_style'.")
    warn_unknown_keys(d, CENSOR_BOX_KEYS, where, REMOVED_BOX_KEYS)

    return CensorBox(
        style=_parse_censor_style(d['censor_style'], f"{where}.censor_style"),
        overlay=_parse_overlay(d['overlay'], f"{where}.overlay") if 'overlay' in d else None,
        shape=_parse_shape(d.get('shape', 'DEFAULT')),
        border=_parse_border(d['border'], f"{where}.border") if 'border' in d else None,
        width_area_safety=float(d.get('width_area_safety', DEFAULT_AREA_SAFETY)),
        height_area_safety=float(d.get('height_area_safety', DEFAULT_AREA_SAFETY)),
        time_safety=float(d.get('time_safety', 0.0)),
        min_probability=float(d.get('min_probability', 0.5)),
        inverse=bool(d.get('inverse', False)),
        clip_to=_parse_label_list(d, 'clip_to', where),
    )


def _merge_censor_box(base: CensorBox, d: Any, where: str) -> CensorBox:
    if not isinstance(d, dict):
        return replace(base, clip_to=list(base.clip_to))
    warn_unknown_keys(d, CENSOR_BOX_KEYS, where, REMOVED_BOX_KEYS)

    out = replace(base, clip_to=list(base.clip_to))
    if 'censor_style' in d:
        out.style = _parse_censor_style(d['censor_style'], f"{where}.censor_style")
    if 'overlay' in d:
        out.overlay = _parse_overlay(d['overlay'], f"{where}.overlay")
    if 'border' in d:
        out.border = _parse_border(d['border'], f"{where}.border")
    if 'shape' in d:
        out.shape = _parse_shape(d['shape'])
    for key in ('width_area_safety', 'height_area_safety', 'time_safety', 'min_probability'):
        if key in d:
            setattr(out, key, float(d[key]))
    if 'inverse' in d:
        out.inverse = bool(d['inverse'])
    if 'clip_to' in d:
        out.clip_to = _parse_label_list(d, 'clip_to', where)
    return out


# -------------------------------------- Public API --------------------------------------
def censor_config_from_dict(root: Any) -> CensorConfig:
    """Parses and validates a censor config dict. Unknown keys are reported as warnings on stderr."""
    if not isinstance(root, dict):
        raise RuntimeError("the censor config must be a mapping")
    if 'default_censor_config' not in root:
        raise RuntimeError("Missing required key: default_censor_config")
    if 'features_to_censor' not in root:
        raise RuntimeError("Missing required key: features_to_censor")

    warn_unknown_keys(root, ROOT_KEYS, "the censor config", REMOVED_ROOT_KEYS)
    default_box = _parse_censor_box(root['default_censor_config'], "default_censor_config")

    overrides = root.get('feature_overrides') or {}
    if not isinstance(overrides, dict):
        overrides = {}

    features: dict[str, CensorBox] = {}
    for feature in root['features_to_censor'] or []:
        name = str(feature)
        if name in overrides:
            features[name] = _merge_censor_box(default_box, overrides[name], f"feature_overrides.{name}")
        else:
            features[name] = replace(default_box, clip_to=list(default_box.clip_to))

    inverse_style: Union[CsBlur, CsPixel, CsBar] = CsBlur(DEFAULT_BLUR_STRENGTH)
    if 'inverse_censor_style' in root:
        parsed = _parse_censor_style(root['inverse_censor_style'], "inverse_censor_style")
        if isinstance(parsed, (CsBlur, CsPixel, CsBar)):
            inverse_style = parsed
        else:
            log_warning(f"inverse_censor_style must be blur, pixel, or bar. Defaulting to blur({DEFAULT_BLUR_STRENGTH}).")
    else:
        log_warning(f"Missing key 'inverse_censor_style', defaulting to blur({DEFAULT_BLUR_STRENGTH}).")

    # Score threshold for each clip_to target: its own min_probability when it is
    # censored or overridden, otherwise the default.
    clip_min_probability: dict[str, float] = {}
    for box in features.values():
        for target in box.clip_to:
            min_prob = default_box.min_probability
            if target in features:
                min_prob = features[target].min_probability
            elif isinstance(overrides.get(target), dict) and 'min_probability' in overrides[target]:
                min_prob = float(overrides[target]['min_probability'])
            clip_min_probability[target] = min_prob

    return CensorConfig(
        features_to_censor=features,
        merge_overlapping_borders=bool(root.get('merge_overlapping_borders', False)),
        enable_overlays=bool(root.get('enable_overlays', True)),
        force_inverse_censor=bool(root.get('force_inverse_censor', False)),
        inverse_censor_style=inverse_style,
        enable_watermark=bool(root.get('enable_watermark', False)),
        stickers_dir=Path(str(root.get('stickers_dir', './data/stickers'))),
        clip_min_probability=clip_min_probability,
    )


def optional_censor_config(config: Any) -> Optional[CensorConfig]:
    if config is None:
        return None
    if isinstance(config, CensorConfig):
        return config
    return censor_config_from_dict(config)


def load_censor_config(path: str | Path) -> dict[str, Any]:
    """
    Read and validate a censor configuration from path (YAML or JSON).

    Returns the configuration as a plain dict that can be inspected, modified and passed as Job(config=...).
    Raises RuntimeError on an invalid file; unknown keys are reported as warnings on stderr.
    """
    raw = read_config_file(path)
    censor_config_from_dict(raw)  # validate now so errors/warnings point at the file
    return raw


def censor_style_priority(style: CensorStyle) -> int:
    """Lower = drawn first."""
    if isinstance(style, CsAIRemove):
        return 0
    if isinstance(style, CsBlur):
        return 5
    if isinstance(style, CsPixel):
        return 10
    if isinstance(style, CsBar):
        return 20
    return 100
