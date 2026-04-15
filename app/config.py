from dataclasses import field, dataclass
from pathlib import Path
from typing import Literal, Any

import yaml
from loguru import logger

import censor_core as cc

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


@dataclass
class Config:
    picture_size: int                      = 640

    video_censor_fps: int                   = 15

    live: LiveCensorConfig                  = field(default_factory=LiveCensorConfig)

    input_delete_probability: float         = 0.0

    default_censor_config: dict[str, Any]   = field(default_factory=lambda: {})

    debug: bool                             = field(default=False)

    n_workers: int                  = 4
    max_concurrent_jobs: int        = 128

    comfy_host: str                         = "localhost"
    comfy_port: int                         = 8188
    comfy_workflow: str|Path                =  "./data/assets/workflow.json"


@dataclass
class CoreConfig:
    """
    Contains all settings that will be passed to the censor core system at initialization.
    """
    nudenet_gpu_enabled: bool       = False
    nn_cuda_device_id: int          = 0

    hd_gpu_enabled: bool            = False
    hd_cuda_device_id: int          = 0

    ed_gpu_enabled: bool            = False
    ed_cuda_device_id: int          = 0

    debug: bool                     = False

    body_detection_model: str       = './models/yolo/yolo11n-seg.pt'
    nudenet_min_confidence: float   = 0.0
    hd_min_confidence: float        = 0.0


def read_censor_config_from_file(path: Path | str | None = None) -> dict[str, Any]:
    """
    Read and construct a CensorConfig object from a YAML file.

    :param path: Path to YAML file. If not provided, the default censor_config is used.

    :return: CensorConfig object
    """
    if path is None:
        path: Path = _default_censor_config_path

    with open(path, 'r') as f:
        default_config = yaml.safe_load(f)

    logger.info(f"Loaded default censor censor_config from {path}.")

    # add the comfy url and workflow if they are not present
    default_censor_style = default_config['default_censor_config']['censor_style']
    if default_censor_style['type'] == 'ai_remove':
        if 'comfy_host' not in default_censor_style:
            default_censor_style['comfy_host'] = CONFIG.comfy_host
        if 'comfy_port' not in default_censor_style:
            default_censor_style['comfy_port'] = CONFIG.comfy_port
        if 'comfy_workflow' not in default_censor_style:
            default_censor_style['comfy_workflow'] = CONFIG.comfy_workflow

    if 'feature_overrides' in default_config:
        if default_config['feature_overrides'] is None:
            default_config['feature_overrides'] = {}

        for feature, censor_box in default_config['feature_overrides'].items():
            style = censor_box['censor_style']
            if style['type'] == 'ai_remove':
                if 'comfy_host' not in style:
                    style['comfy_host'] = CONFIG.comfy_host
                if 'comfy_port' not in style:
                    style['comfy_port'] = CONFIG.comfy_port
                if 'comfy_workflow' not in style:
                    style['comfy_workflow'] = CONFIG.comfy_workflow

    return default_config


def read_censor_config_from_file_w_hash(path: Path = None, prev_hash: str = None) -> tuple[bool, dict[str, Any]|None, str]:
    """
    Compares the prev_hash against the hash of path.
    If they differ, reads path.

    :param path: Path to YAML file. If not provided, the default censor_config is used.
    :param prev_hash: Previous hash to compare against

    :return: read_file: whether the file was read | config: the new config if read, None otherwise | hash: the new hash

    """
    if path is None:
        path: Path = _default_censor_config_path

    hash_path = hash_utils.md5_for_file(path, 16)

    if prev_hash is None or prev_hash != hash_path:
        return True, read_censor_config_from_file(path), hash_path
    else:
        return False, None, hash_path


def load_config(path: Path | str = None, debug: bool = False) -> Config:
    if path is None:
        path = './config.yml'

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    logger.info(f"Loaded censor_config from {path}.")

    lcc = {}
    for key, value in list(config_dict.items()):
        if key.startswith('live_'):
            nkey = key[5:]
            lcc[nkey] = value
            del config_dict[key]

    live_censor_config = LiveCensorConfig(**lcc)

    return Config(
        debug=debug,
        picture_size=config_dict['picture_size'],
        video_censor_fps=config_dict['video_censor_fps'],
        live=live_censor_config,
        input_delete_probability=config_dict['input_delete_probability'],
        comfy_host=config_dict['comfy_host'],
        comfy_port=config_dict['comfy_port'],
        comfy_workflow=config_dict['comfy_workflow'],
        n_workers=config_dict['n_workers'],
        max_concurrent_jobs=config_dict['max_concurrent_jobs']
    )


def read_general_config(path_to_config: Path | str) -> dict[str, Any]:
    with open(path_to_config, 'r') as file:
        form_config = yaml.safe_load(file)

    if "gpu_enabled" not in form_config or not form_config['gpu_enabled']:
        # disable gpu acceleration
        form_config["nudenet_gpu_enabled"] = False
        form_config["hd_gpu_enabled"] = False
        form_config["ed_gpu_enabled"] = False

    return form_config


def init_app_config(path_to_config: Path|str|None, debug: bool) -> None:
    global _app_initialized
    if _app_initialized:
        return
    if path_to_config is None:
        path_to_config: str = './config.yml'

    config = read_general_config(path_to_config)

    flags = cc.CoreConfig()
    flags.nn_gpu_enabled = config['nudenet_gpu_enabled']
    flags.hd_gpu_enabled = config['hd_gpu_enabled']
    flags.ed_gpu_enabled = config['ed_gpu_enabled']
    flags.hd_model = config['hd_model']
    flags.debug = debug
    flags.nudenet_min_confidence =config['nudenet_min_confidence']
    flags.hd_min_confidence = config['hd_min_confidence']
    cc.AppConfig.instance().initialize(flags)

    _app_initialized = True


def censor_config_has_changed(prev_hash: str, path: Path = None) -> Config:
    """Returns whether the censor censor_config file has changed."""
    if path is None:
        path = _default_censor_config_path

    new_hash = hash_utils.md5_for_file(path, 16)
    return new_hash != prev_hash


# Private variables
_app_initialized = False
_default_censor_config_path = Path('./default_censor_config.yml')

# Public variables
CONFIG: Config = Config()


def init_configs(config_file: str|Path|None = None,
                 default_censor_config: str|Path|None = None,
                 debug: bool = False):
    global CONFIG

    CONFIG = load_config(config_file, debug)
    init_app_config(config_file, debug)



