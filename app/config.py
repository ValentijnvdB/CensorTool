from dataclasses import field, dataclass
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger

import core
from utils import hash_utils


@dataclass
class LiveCensorConfig:
    cap_monitor: int = 0
    cap_top: int = 0
    cap_left: int = 0
    cap_height: int = 0
    cap_width: int = 0

    mode: Literal['quick', 'precise'] = 'quick'
    show_fps: bool = False
    interpolate_frames: bool = False
    cursor_color: tuple[int, int, int] = (168, 93, 253)

    delay: float = 0.2 # only used if mode == 'precise'


@dataclass
class Config:
    picture_sizes: list[int] = field(default_factory=lambda: [1280])

    video_censor_fps: int = 15

    live: LiveCensorConfig = field(default_factory=LiveCensorConfig())

    input_delete_probability: float = 0.0

    censor_overlap_strategy: dict = field(default_factory=lambda: {})
    censor_scale_strategy: dict = field(default_factory=lambda: {})

    default_censor_config: core.CensorConfig = field(default_factory=lambda: DEFAULT_CENSOR_CONFIG)

    debug: bool = field(default=False)

    comfy_base_url: str = "http://localhost:8188/"
    comfy_workflow: str|Path =  "./data/assets/workflow.json"


def load_censor_config_from_file(path: Path | str = None) -> core.CensorConfig:
    """
    Read and construct a CensorConfig object from a YAML file.

    :param path: Path to YAML file. If not provided, the default censor_config is used.

    :return: CensorConfig object
    """
    if path is None:
        path = _default_censor_config_path

    with open(path, 'r') as f:
        default_config = yaml.safe_load(f)

    logger.info(f"Loaded default censor censor_config from {path}.")

    # add the comfy url and workflow if they are not present
    default_censor_style = default_config['default_censor_config']['censor_style']
    if default_censor_style['type'] == 'ai_remove':
        if 'comfy_base_url' not in default_censor_style:
            default_censor_style['comfy_base_url'] = CONFIG.comfy_base_url
        if 'comfy_workflow' not in default_censor_style:
            default_censor_style['comfy_workflow'] = CONFIG.comfy_workflow

    if 'feature_overrides' in default_config:
        if default_config['feature_overrides'] is None:
            default_config['feature_overrides'] = {}

        for feature, censor_box in default_config['feature_overrides'].items():
            style = censor_box['censor_style']
            if style['type'] == 'ai_remove':
                if 'comfy_base_url' not in style:
                    style['comfy_base_url'] = CONFIG.comfy_base_url
                if 'comfy_workflow' not in style:
                    style['comfy_workflow'] = CONFIG.comfy_workflow

    return core.config.construct_censor_config(default_config)


def load_censor_config_from_file_w_hash(path: Path | str = None) -> tuple[core.CensorConfig, str]:
    if path is None:
        path = _default_censor_config_path

    hash_path = hash_utils.md5_for_file(path, 16)

    return load_censor_config_from_file(path), hash_path


def load_config(path: Path | str = None, debug: bool = False) -> Config:
    if path is None:
        path = '../config.yml'

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    logger.info(f"Loaded censor_config from {path}.")

    lcc = {}
    for key, value in list(config_dict.items()):
        if key.startswith('live_'):
            nkey = key[5:]
            lcc[nkey] = value
            del config_dict[key]

    core.config.set_general_config(
        gpu_enabled=config_dict.pop('gpu_enabled'),
        cuda_device_id=config_dict.pop('cuda_device_id'),
        body_detection_model=config_dict.pop('body_detection_model'),
        debug=debug
    )

    live_censor_config = LiveCensorConfig(**lcc)
    config_dict['live'] = live_censor_config
    return Config(**config_dict, debug=debug)



def censor_config_has_changed(prev_hash: str, path: Path | str = None) -> Config:
    """Returns whether the censor censor_config file has changed."""
    if path is None:
        path = _default_censor_config_path

    new_hash = hash_utils.md5_for_file(path, 16)
    return new_hash != prev_hash


_default_censor_config_path = Path('./default_censor_config.yml')

# Public variables
DEFAULT_CENSOR_CONFIG: core.CensorConfig|None = None
CONFIG: Config|None = None


def init_configs(config_file: str|Path|None = None,
                 default_censor_config: str|Path|None = None,
                 debug: bool = False):
    global CONFIG, DEFAULT_CENSOR_CONFIG

    DEFAULT_CENSOR_CONFIG = load_censor_config_from_file(default_censor_config)
    CONFIG = load_config(config_file, debug)
    core.set_default_censor_config(DEFAULT_CENSOR_CONFIG)



