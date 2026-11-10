"""
A JSON Schema (draft 2020-12) of the censor config a client sends, for GET /censor_config/schema.

It describes every setting with its type, default and allowed values, so a client can build a settings form. It is
a description, not the validator: POST /censor_config/validate checks a config the way a censor request does
(shapes are case-insensitive, unknown keys are warnings, labels are checked against the loaded models).
Keep it in sync with core/censor_config.py (tests/server/test_censor_config_schema.py checks the keys).
"""
from __future__ import annotations

import copy

from core.censor_config import DEFAULT_AREA_SAFETY, DEFAULT_BLUR_STRENGTH, DEFAULT_PIXEL_FACTOR, Shape

_COLOR = {
    'type': 'array',
    'description': 'An RGB color: [R, G, B], each 0-255.',
    'items': {'type': 'integer', 'minimum': 0, 'maximum': 255},
    'minItems': 3,
    'maxItems': 3,
    'default': [0, 0, 0],
}


def _style(kind: str, description: str, properties: dict | None = None) -> dict:
    return {
        'title': kind,
        'description': description,
        'type': 'object',
        'properties': {'type': {'const': kind}, **(properties or {})},
        'required': ['type'],
        'additionalProperties': False,
    }


_BLUR = _style('blur', 'Blurs the area.', {
    'strength': {'type': 'integer', 'minimum': 1, 'default': DEFAULT_BLUR_STRENGTH,
                 'description': 'Higher is stronger.'}})
_PIXEL = _style('pixel', 'Pixelates the area.', {
    'factor': {'type': 'integer', 'minimum': 1, 'default': DEFAULT_PIXEL_FACTOR,
               'description': 'Block size: 20 turns a 200x400 region into 10x20 blocks.'}})
_BAR = _style('bar', 'A solid color.', {'color': _COLOR})
_AI_REMOVE = _style('ai_remove', 'Inpaints the area with the ComfyUI workflow set in the server\'s config.yml. '
                                 'Needs a running ComfyUI server. Slow.')
_DEBUG = _style('debug', 'Draws the outline, label and score instead of censoring. For tuning configs.',
                {'color': _COLOR})

CENSOR_STYLES = {'blur': _BLUR, 'pixel': _PIXEL, 'bar': _BAR, 'ai_remove': _AI_REMOVE, 'debug': _DEBUG}
"""Every censor_style type."""

INVERSE_CENSOR_STYLES = ('blur', 'pixel', 'bar')

_TEXT_OVERLAY = {
    'title': 'text',
    'description': 'One of the values, picked at random, on top of the area.',
    'type': 'object',
    'properties': {
        'type': {'const': 'text'},
        'values': {'type': 'array', 'items': {'type': 'string'}, 'default': []},
        'probability': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 1.0,
                        'description': 'Chance the overlay is placed.'},
        'color': {**_COLOR, 'default': [0, 0, 0]},
        'thickness': {'type': 'integer', 'minimum': 1, 'default': 1},
        'font_scale': {'type': 'number', 'exclusiveMinimum': 0, 'default': 1.0},
        'font': {'type': 'integer', 'minimum': 0, 'default': 0, 'description': 'An OpenCV font id.'},
    },
    'required': ['type'],
    'additionalProperties': False,
}

_STICKER_OVERLAY = {
    'title': 'sticker',
    'description': 'A random sticker image on top of the area. See GET /stickers.',
    'type': 'object',
    'properties': {
        'type': {'const': 'sticker'},
        'probability': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 1.0,
                        'description': 'Chance the overlay is placed.'},
        'stickers': {'type': 'array', 'items': {'type': 'string'}, 'default': [],
                     'description': 'Folders and/or images, relative to stickers_dir. Empty: all stickers.'},
        'categories': {'type': 'array', 'items': {'type': 'string'}, 'deprecated': True,
                       'description': 'Deprecated: put the folders in stickers instead.'},
    },
    'required': ['type'],
    'additionalProperties': False,
}

_BORDER = {
    'description': 'A colored outline around the area.',
    'type': ['object', 'null'],
    'properties': {
        'color': _COLOR,
        'thickness': {'type': 'integer', 'minimum': 1, 'default': 1},
    },
    'additionalProperties': False,
    'default': None,
}

_LABELS = {'type': 'array', 'items': {'type': 'string'},
           'description': 'Labels, as listed by GET /profile.'}

_CENSOR_BOX_PROPERTIES = {
    'censor_style': {'description': 'How the area is hidden.', 'oneOf': [{'$ref': f'#/$defs/style_{k}'}
                                                                          for k in CENSOR_STYLES]},
    'min_probability': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.5,
                        'description': 'Minimum score a detection needs to be censored.'},
    'shape': {'type': 'string', 'enum': [s.value.lower() for s in Shape], 'default': 'default',
              'description': 'The shape of the censored area (case-insensitive). mask: only for labels of '
                             'segmentation models.'},
    'width_area_safety': {'type': 'number', 'default': DEFAULT_AREA_SAFETY,
                          'description': 'Extra width, as a fraction: 0.1 makes the area 10% wider.'},
    'height_area_safety': {'type': 'number', 'default': DEFAULT_AREA_SAFETY,
                           'description': 'Extra height, as a fraction.'},
    'time_safety': {'type': 'number', 'minimum': 0, 'default': 0.0,
                    'description': 'Videos only: keep the censor this many seconds after the feature was last '
                                   'detected.'},
    'border': {'$ref': '#/$defs/border'},
    'overlay': {'description': 'Text or a sticker on top of the area.', 'default': None,
                'oneOf': [{'$ref': '#/$defs/overlay_text'}, {'$ref': '#/$defs/overlay_sticker'}, {'type': 'null'}]},
    'inverse': {'type': 'boolean', 'default': False,
                'description': 'Keep this feature visible and censor everything else with inverse_censor_style.'},
    'clip_to': {**_LABELS, 'default': [],
                'description': 'Clip the area to the outline of these segmentation labels.'},
}

SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'title': 'Censor config',
    'description': 'What is censored and how. See docs/censor_config.md.',
    'type': 'object',
    'properties': {
        'features_to_censor': {**_LABELS, 'description': 'The labels to censor (see GET /profile).'},
        'default_censor_config': {'$ref': '#/$defs/censor_box_default',
                                  'description': 'The censor box used for every feature.'},
        'feature_overrides': {
            'type': 'object',
            'description': 'Per label, the censor box settings that differ from default_censor_config. Nested '
                           'settings such as censor_style are replaced as a whole.',
            'additionalProperties': {'$ref': '#/$defs/censor_box'},
            'default': {},
        },
        'inverse_censor_style': {'description': 'The style for inverse censoring.', 'default': {
            'type': 'blur', 'strength': DEFAULT_BLUR_STRENGTH},
            'oneOf': [{'$ref': f'#/$defs/style_{k}'} for k in INVERSE_CENSOR_STYLES]},
        'force_inverse_censor': {'type': 'boolean', 'default': False,
                                 'description': 'Apply the inverse censor even when no inverse feature is found.'},
        'merge_overlapping_borders': {'type': 'boolean', 'default': False,
                                      'description': 'One border around overlapping censored areas.'},
        'enable_overlays': {'type': 'boolean', 'default': True, 'description': 'Master switch for overlays.'},
        'enable_watermark': {'type': 'boolean', 'default': False,
                             'description': 'Add a small "Censored using CensorTool" watermark.'},
        'stickers_dir': {'type': 'string', 'default': 'data/stickers',
                         'description': 'data/stickers or one of its subfolders, relative to the server\'s working '
                                        'directory. See GET /stickers.'},
    },
    'required': ['features_to_censor', 'default_censor_config'],
    'additionalProperties': False,
    '$defs': {
        **{f'style_{k}': v for k, v in CENSOR_STYLES.items()},
        'overlay_text': _TEXT_OVERLAY,
        'overlay_sticker': _STICKER_OVERLAY,
        'border': _BORDER,
        'censor_box': {
            'description': 'How one feature is censored.',
            'type': 'object',
            'properties': _CENSOR_BOX_PROPERTIES,
            'additionalProperties': False,
        },
        'censor_box_default': {
            'description': 'How one feature is censored. censor_style is required here.',
            'type': 'object',
            'properties': _CENSOR_BOX_PROPERTIES,
            'required': ['censor_style'],
            'additionalProperties': False,
        },
    },
}


def censor_config_schema() -> dict:
    return copy.deepcopy(SCHEMA)
