from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2


BODY_DETECTION_MODEL = Literal['fastest', 'faster', 'fast', 'medium-fast',
                               'medium', 'medium-slow', 'slow', 'slower', 'slowest']

@dataclass
class GeneralConfig:
    gpu_enabled: bool = False
    cuda_device_id: int = 0
    debug: bool = False
    body_detection_model: BODY_DETECTION_MODEL = 'fastest'


FEATURE = Literal['exposed_anus','exposed_armpits','covered_belly','exposed_belly','covered_buttocks','exposed_buttocks',
            'face_femme','face_masc','covered_feet','exposed_feet','covered_breast','exposed_breast',
            'covered_vulva','exposed_vulva','exposed_chest','exposed_penis','eyes_femme','eyes_masc']

@dataclass
class CensorBox:
    censor_style: CENSOR_STYLE
    width_area_safety: float
    height_area_safety: float
    time_safety: float
    border: BORDER_TYPE
    overlay: OVERLAY_TYPE
    min_prob: float
    shape: Literal['default', 'ellipse', 'circle', 'rectangle']
    inverse: bool
    intersect_human: bool


@dataclass
class CensorConfig:
    features_to_censor: dict[FEATURE, CensorBox]

    merge_overlapping_censor_boxes: bool
    merge_overlapping_borders: bool
    enable_overlays: bool
    force_inverse_censor: bool
    inverse_censor_style: INVERSE_CENSOR_STYLE

    enable_watermark: bool

######################################### CENSOR_STYLE #########################################

@dataclass
class CSBlur:
    strength: int

@dataclass
class CSPixel:
    factor: int

@dataclass
class CSBar:
    color: tuple[int, int, int]

@dataclass
class CSAIRemove:
    comfy_base_url: str
    comfy_workflow: str|Path

@dataclass
class CSDebug:
    color: tuple[int, int, int]

CENSOR_STYLE         = CSBlur | CSPixel | CSBar | CSAIRemove | CSDebug
INVERSE_CENSOR_STYLE = CSBlur | CSPixel | CSBar


############################################ OVERLAY ############################################

@dataclass
class OVText:
    values: list[str] | str | Path                  # The possible strings that are placed on censored area.
    probability: float                              # The probability an overlay is placed. Must be in [0,1].
    color: tuple[int, int, int] = (255, 255, 255)   # The color of the text (RGB).
    thickness: int = 3                              # The base thickness of the letters.
    font_scale: float = 1.2                         # How much should the text be scaled.
    font: int = cv2.FONT_HERSHEY_SIMPLEX            # The font to use.

@dataclass
class OVSticker:
    probability: float
    categories: list[Path]

OVERLAY_TYPE = OVText | OVSticker | None


############################################ BORDER #############################################

@dataclass
class Border:
    color: tuple[int, int, int]
    thickness: int

BORDER_TYPE = Border | None