from dataclasses import field, dataclass
from pathlib import Path
from typing import Literal, Any

import yaml
from loguru import logger

import censor_core as cc

import constants
from utils import hash_utils


DEFAULT_PROFILE = 'default'
CONFIG_FILE = 'config.yml'
PIPELINE_FILE = 'pipeline.yml'
CENSOR_CONFIG_FILE = 'censor_config.yml'


class ProfileError(Exception):
    """Raised when a profile or one of its config files cannot be found."""


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

    # pipeline.yml listing the models censor_core loads (model paths in it are relative to the working dir)
    pipeline_path: Path                     = constants.profiles_path / DEFAULT_PROFILE / PIPELINE_FILE


def list_profiles() -> list[str]:
    """Names of the profiles in the configs directory: 'default' first, the others alphabetically."""
    if not constants.profiles_path.is_dir():
        return []
    names = [p.name for p in constants.profiles_path.iterdir() if p.is_dir()]
    return sorted(names, key=lambda name: (name != DEFAULT_PROFILE, name.lower(), name))


def resolve_profile_dir(profile: str | Path | None) -> Path:
    """
    Find the directory of a profile.
    An existing directory is used as is, otherwise the profile is looked up by name in the configs directory.
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    path = Path(profile)
    if path.is_dir():
        return path

    path = constants.profiles_path / str(profile)
    if path.is_dir():
        return path

    available = ', '.join(list_profiles()) or 'none'
    raise ProfileError(f"Profile '{profile}' not found. Available profiles: {available}.")


def resolve_profile_files(profile: str | Path | None = None,
                          config_file: str | Path | None = None,
                          censor_config_file: str | Path | None = None,
                          pipeline_file: str | Path | None = None) -> tuple[Path, Path, Path]:
    """
    Determine which config files to use: the profile's files, with any explicitly given file overriding the
    profile's one. A profile without a config.yml falls back to the one of the default profile.

    :return: (config, censor_config, pipeline) paths
    """
    profile_dir = resolve_profile_dir(profile)

    def pick(override: str | Path | None, filename: str, fallback: Path | None = None) -> Path:
        if override is not None:
            path = Path(override)
            if not path.is_file():
                raise ProfileError(f"Config file '{path}' does not exist.")
            return path

        path = profile_dir / filename
        if path.is_file():
            return path
        if fallback is not None and fallback.is_file():
            logger.info(f"Profile '{profile_dir.name}' has no {filename}, using '{fallback}'.")
            return fallback
        raise ProfileError(f"Profile '{profile_dir.name}' ({profile_dir}) has no {filename}.")

    config_path = pick(config_file, CONFIG_FILE, fallback=constants.profiles_path / DEFAULT_PROFILE / CONFIG_FILE)
    censor_config_path = pick(censor_config_file, CENSOR_CONFIG_FILE)
    pipeline_path = pick(pipeline_file, PIPELINE_FILE)
    return config_path, censor_config_path, pipeline_path


def load_pipeline_spec(path: Path | str) -> dict[str, Any]:
    """Read a pipeline.yml and make the relative model paths in it relative to the working directory."""
    with open(path, 'r') as f:
        spec = yaml.safe_load(f) or {}

    def resolve(entry: dict) -> dict:
        entry = dict(entry)
        if 'path' in entry:
            model_path = Path(entry['path'])
            if not model_path.is_absolute():
                model_path = Path.cwd() / model_path
            entry['path'] = str(model_path.resolve())
        return entry

    if spec.get('models'):
        spec['models'] = [resolve(m) for m in spec['models']]
    if spec.get('landmarks'):
        spec['landmarks'] = resolve(spec['landmarks'])
    return spec


def _resolved_pipeline_file(path: Path | str) -> Path:
    """
    censor_core resolves model paths relative to the pipeline.yml. Write a copy with absolute model paths
    (resolved against the working directory) to the temp dir and return its path.
    """
    content = yaml.safe_dump(load_pipeline_spec(path), sort_keys=False)
    out = constants.configs_path / f"pipeline_{hash_utils.hash_bytes(content.encode(), 16)}.yml"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)
    return out


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
            style = (censor_box or {}).get('censor_style')
            if style is not None and style.get('type') == 'ai_remove':
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


def load_config(path: Path | str = None, pipeline_path: Path | str | None = None, debug: bool = False) -> Config:
    if path is None:
        path = constants.profiles_path / DEFAULT_PROFILE / CONFIG_FILE

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
        max_concurrent_jobs=config_dict['max_concurrent_jobs'],
        pipeline_path=Path(pipeline_path) if pipeline_path is not None else Config.pipeline_path,
    )


def make_image_pipeline(max_workers: int | None = None, mode: str = 's') -> cc.ImagePipeline:
    """Create an ImagePipeline from the configured pipeline.yml."""
    if max_workers is None:
        max_workers = CONFIG.n_workers
    return cc.ImagePipeline.from_yaml(str(_resolved_pipeline_file(CONFIG.pipeline_path)),
                                      max_workers=max_workers, mode=mode)


def make_video_pipeline(max_workers: int | None = None) -> cc.VideoPipeline:
    """Create a VideoPipeline (with its own ImagePipeline) from the configured pipeline.yml."""
    if max_workers is None:
        max_workers = CONFIG.n_workers
    return cc.VideoPipeline.from_yaml(str(_resolved_pipeline_file(CONFIG.pipeline_path)), max_workers=max_workers,
                                      max_concurrent_jobs=CONFIG.max_concurrent_jobs)


def censor_config_has_changed(prev_hash: str, path: Path = None) -> Config:
    """Returns whether the censor censor_config file has changed."""
    if path is None:
        path = _default_censor_config_path

    new_hash = hash_utils.md5_for_file(path, 16)
    return new_hash != prev_hash


# Private variables
_default_censor_config_path = constants.profiles_path / DEFAULT_PROFILE / CENSOR_CONFIG_FILE

# Public variables
CONFIG: Config = Config()


def init_configs(profile: str | Path | None = None,
                 config_file: str | Path | None = None,
                 default_censor_config: str | Path | None = None,
                 pipeline_file: str | Path | None = None,
                 debug: bool = False):
    """
    Load the configs of a profile. Explicitly given files override the profile's ones.

    :raises ProfileError: if the profile or one of the config files cannot be found.
    """
    global CONFIG, _default_censor_config_path

    config_path, censor_config_path, pipeline_path = resolve_profile_files(
        profile, config_file, default_censor_config, pipeline_file)

    logger.info(f"Using profile '{profile or DEFAULT_PROFILE}': config={config_path}, "
                f"censor_config={censor_config_path}, pipeline={pipeline_path}")

    CONFIG = load_config(config_path, pipeline_path, debug)
    _default_censor_config_path = censor_config_path
