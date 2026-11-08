"""
Censor configurations used by the golden tests.

Each scenario targets one code path:

  faces_bar        box censoring of a detect model's labels
  faces_intersect  clip_to: faces clipped to the BODY outlines (sequential: the
                   segmentation model only runs when a face needs clipping)
  faces_intersect_parallel
                   the same in parallel mode (segmentation runs up front)
  full_body        a segmentation label censored by its outline
  eyes             the landmark stage (eye bars from the faces)
  margins          width/height safety margins

BODY uses min_probability 0.5, the v1 body_conf_threshold default, so the
goldens carry over from v1.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

_BASE: dict[str, Any] = {
    "default_censor_config": {
        "censor_style": {"type": "bar", "color": [0, 0, 0]},
        "min_probability": 0.3,
        "width_area_safety": 0.0,
        "height_area_safety": 0.0,
    },
    "inverse_censor_style": {"type": "blur", "strength": 31},
    "enable_overlays": False,
    "merge_overlapping_borders": False,
}


def _cfg(features: list[str], overrides: dict[str, Any] | None = None, **top: Any) -> dict[str, Any]:
    cfg = deepcopy(_BASE)
    cfg["features_to_censor"] = features
    if overrides:
        cfg["feature_overrides"] = overrides
    cfg.update(top)
    return cfg


FACES = ["FACE_FEMALE", "FACE_MALE"]
_BODY_THRESHOLD = {"BODY": {"min_probability": 0.5}}
_CLIPPED_FACES = {f: {"clip_to": ["BODY"], "censor_style": {"type": "pixel", "factor": 10}} for f in FACES}
_BODY_BLUR = {"BODY": {"censor_style": {"type": "blur", "strength": 25}, "min_probability": 0.5}}

# name -> (pipeline mode, censor config)
SCENARIOS: dict[str, tuple[str, dict[str, Any]]] = {
    "faces_bar": ("s", _cfg(FACES)),
    "faces_intersect": ("s", _cfg(FACES, {**_CLIPPED_FACES, **_BODY_THRESHOLD})),
    "faces_intersect_parallel": ("p", _cfg(FACES, {**_CLIPPED_FACES, **_BODY_THRESHOLD})),
    "full_body": ("s", _cfg(["BODY"], _BODY_BLUR)),
    "full_body_parallel": ("p", _cfg(["BODY"], _BODY_BLUR)),
    "eyes": ("s", _cfg(["EYES_FEMALE", "EYES_MALE"])),
    "margins": ("s", _cfg(["FEET_COVERED"], {"FEET_COVERED": {"width_area_safety": 0.2, "height_area_safety": 0.2}})),
}

# No censor config: every model runs and the raw detections are returned.
DETECTION_CONFIG = None
