from copy import deepcopy, copy
from threading import Lock
from typing import Any

from loguru import logger

from .datatypes import *

_lock = Lock()
_general_config: GeneralConfig = GeneralConfig()
def set_general_config(**kwargs):
    global _general_config
    with _lock:
        _general_config = GeneralConfig(**kwargs)

def get_general_config() -> GeneralConfig:
    with _lock:
        return _general_config

def construct_censor_config(form_config: dict[str, Any]) -> CensorConfig:
    """
    Construct a CensorConfig object from a dictionary.
    Dictionary must have the shape as in the default_censor_config.yml
    ```
    {
        features_to_censor: [ <list_of_features> ],
        default_censor_config: { <censor_box> },
        feature_overrides: { <feature1>: <(partial) censor_box1>, <feature2>: <(partial) censor_box2>, ... },

        merge_overlapping_censor_boxes: <bool>
        merge_overlapping_borders: <bool>
        enable_overlays: <bool>
        force_inverse_censor: <bool>
        inverse_censor_style: <censor_style>

        enable_watermark: <bool>
    }
    ```
    """
    form_config = copy(form_config)

    default_censor_box: CensorBox = _construct_censor_box(form_config['default_censor_config'])

    features_to_censor: dict[FEATURE, CensorBox] = {}
    for feature in form_config['features_to_censor']:
        box_config = form_config['feature_overrides'][feature] if feature in form_config['feature_overrides'] else {}
        features_to_censor[feature] = _overwrite_censor_box(
            box_config=box_config,
            default=default_censor_box
        )

    del form_config['features_to_censor']
    del form_config['feature_overrides']
    del form_config['default_censor_config']

    form_config['features_to_censor'] = features_to_censor

    return CensorConfig(**form_config)


############################################## HELPERS ##############################################

def _construct_censor_box(box_config: dict[str, Any]) -> CensorBox:
    box_config['overlay'] = _construct_overlay(box_config['overlay'])
    box_config['censor_style'] = _construct_censor_style(box_config['censor_style'])
    box_config['border'] = _construct_border(box_config['border'])
    return CensorBox(**box_config)

def _overwrite_censor_box(box_config: dict[str, Any], default: CensorBox) -> CensorBox:
    """Create a copy of 'default' where every item present in 'box_config' will be overwritten."""

    # make a copy of the default, then overwrite all values that are provided in the box_config
    output = deepcopy(default)

    # overwrite
    if 'overlay' in box_config:
        output.overlay = _construct_overlay(box_config['overlay'])
        del box_config['overlay']

    if 'border' in box_config:
        output.border = _construct_border(box_config['border'])
        del box_config['border']

    if 'censor_style' in box_config:
        cs = _construct_censor_style(box_config['censor_style'])
        if cs:
            output.censor_style = cs
        del box_config['censor_style']

    # override remaining values
    for key, value in box_config.items():
        output.__setattr__(key, value)

    return output


def _construct_overlay(ov_config: dict[str, Any] | None) -> OVERLAY_TYPE:
    if ov_config is None:
        return None

    overlay_type = ov_config.pop('type')
    if overlay_type == 'sticker':
        return OVSticker(**ov_config)
    elif overlay_type == 'text':
        return OVText(**ov_config)
    else:
        logger.warning(f"Unknown overlay type: {overlay_type}. No overlay will be used.")
        return None


def _construct_border(border_dict: dict[str, Any] | None) -> BORDER_TYPE:
    if border_dict is None:
        return None
    return Border(**border_dict)


def _construct_censor_style(cs_config: dict[str, Any]) -> CENSOR_STYLE | None:
    if cs_config is None:
        logger.warning(f"cs_config is None. Default censor_style will be used.")
        return None

    censor_style_type = cs_config.pop('type')
    if censor_style_type == 'blur':
        return CSBlur(**cs_config)
    elif censor_style_type == 'pixel':
        return CSPixel(**cs_config)
    elif censor_style_type == 'bar':
        return CSBar(**cs_config)
    elif censor_style_type == 'ai_remove':
        return CSAIRemove(**cs_config)
    elif censor_style_type == 'debug':
        return CSDebug(**cs_config)
    else:
        logger.warning(f"Unknown censor style: {censor_style_type}. Using default.")
        return None