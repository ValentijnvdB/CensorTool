from __future__ import annotations

from dataclasses import field, dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from core import backend
from core.backend import BackendError, cc

import constants
from utils import hash_utils


CONFIG_FILE = 'config.yml'
PIPELINE_FILE = 'pipeline.yml'
CENSOR_CONFIG_FILE = 'censor_config.yml'

# App settings in pipeline.yml. They are not part of the backend's pipeline format, so they are
# removed before the file is handed to the backend and passed as arguments instead.
DEFAULT_N_WORKERS = 4
DEFAULT_MAX_CONCURRENT_JOBS = 128
PIPELINE_RUNTIME_KEYS = ('backend', 'n_workers', 'max_concurrent_jobs')


class ProfileError(Exception):
    """Raised when a profile or one of its config files cannot be found."""


@dataclass
class LiveCensorConfig:
    cap_top: int = 0
    cap_left: int = 0
    cap_height: int = 0
    cap_width: int = 0

    show_fps: bool = False
    cursor_color: tuple[int, int, int] = (168, 93, 253)

    # optional (may be left out of config.yml)
    input_size: int = 0         # model input size for live frames, 0 = the size in pipeline.yml
    frames_in_flight: int = 2   # frames censored at the same time


@dataclass
class Config:
    # --- from config.yml (reloaded on change during live/webcam censoring) ---
    video_censor_fps: int                   = 15

    live: LiveCensorConfig                  = field(default_factory=LiveCensorConfig)

    comfy_host: str                         = "localhost"
    comfy_port: int                         = 8188
    comfy_workflow: str|Path                = "./data/workflow.json"

    # --- from the command line ---
    debug: bool                             = False

    # --- from pipeline.yml (only read when a pipeline is created) ---
    backend: str                            = backend.DEFAULT_BACKEND  # 'python' or 'cpp'
    n_workers: int                          = DEFAULT_N_WORKERS
    max_concurrent_jobs: int                = DEFAULT_MAX_CONCURRENT_JOBS

    # pipeline.yml listing the models the backend loads (model paths in it are relative to the working dir)
    pipeline_path: Path | None              = None


def list_profiles() -> list[str]:
    """Names of the profiles in the configs directory, alphabetically."""
    if not constants.profiles_path.is_dir():
        return []
    names = [p.name for p in constants.profiles_path.iterdir() if p.is_dir()]
    return sorted(names, key=lambda name: (name.lower(), name))


def resolve_profile_dir(profile: str | Path | None) -> Path:
    """
    Find the directory of a profile.
    An existing directory is used as is, otherwise the profile is looked up by name in the configs directory.
    Without a profile, the first one alphabetically is used (with a warning).
    """
    if profile is None:
        profiles = list_profiles()
        if not profiles:
            raise ProfileError(f"No profiles found in '{constants.profiles_path}'.")
        profile = profiles[0]
        if len(profiles) > 1:
            logger.warning(f"No profile given, using '{profile}' (the first alphabetically). "
                           f"Pick one with --profile <name>. Available profiles: {', '.join(profiles)}.")

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
    profile's one.

    :return: (config, censor_config, pipeline) paths
    """
    profile_dir = resolve_profile_dir(profile)

    def pick(override: str | Path | None, filename: str) -> Path:
        if override is not None:
            path = Path(override)
            if not path.is_file():
                raise ProfileError(f"Config file '{path}' does not exist.")
            return path

        path = profile_dir / filename
        if path.is_file():
            return path
        raise ProfileError(f"Profile '{profile_dir.name}' ({profile_dir}) has no {filename}.")

    config_path = pick(config_file, CONFIG_FILE)
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
    The backend resolves model paths relative to the pipeline.yml. Write a copy with absolute model paths
    (resolved against the working directory) to the temp dir and return its path.
    """
    spec = load_pipeline_spec(path)
    for key in PIPELINE_RUNTIME_KEYS:
        spec.pop(key, None)
    content = yaml.safe_dump(spec, sort_keys=False)
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

    A file that cannot be read (e.g. invalid YAML while it is being edited) is reported and not read;
    it is tried again once the file changes again.
    """
    if path is None:
        path: Path = _default_censor_config_path

    hash_path = hash_utils.md5_for_file(path, 16)

    if prev_hash is not None and prev_hash == hash_path:
        return False, None, hash_path

    try:
        return True, read_censor_config_from_file(path), hash_path
    except Exception as e:
        logger.error(f"Failed to read censor config {path}: {e}")
        return False, None, hash_path


_CONFIG_KEYS = ('video_censor_fps', 'comfy_host', 'comfy_port', 'comfy_workflow')
_LIVE_KEYS = tuple(f.name for f in fields(LiveCensorConfig))
_OPTIONAL_LIVE_KEYS = ('input_size', 'frames_in_flight')


def load_config(path: Path | str) -> Config:
    """
    Read and validate a config.yml. Only the config.yml settings are filled in;
    debug and the pipeline settings keep their defaults.

    :raises ValueError: if the file is not valid.
    """
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError(f"'{path}' is empty or not a mapping.")

    live_dict = {}
    for key, value in config_dict.items():
        if key.startswith('live_') and key[5:] in _LIVE_KEYS:
            live_dict[key[5:]] = value
        elif key in PIPELINE_RUNTIME_KEYS:
            logger.warning(f"'{key}' in {path} is ignored, it is set in pipeline.yml now.")
        elif key == 'picture_size':
            logger.warning(f"'picture_size' in {path} is ignored, set 'imgsz' on the model in pipeline.yml instead.")
        elif key not in _CONFIG_KEYS:
            logger.warning(f"Unknown key '{key}' in {path} is ignored.")

    missing = [k for k in _CONFIG_KEYS if k not in config_dict] + \
              [f'live_{k}' for k in _LIVE_KEYS if k not in live_dict and k not in _OPTIONAL_LIVE_KEYS]
    if missing:
        raise ValueError(f"'{path}' is missing: {', '.join(missing)}.")

    config = Config(live=LiveCensorConfig(**live_dict), **{k: config_dict[k] for k in _CONFIG_KEYS})
    validate_config(config)
    return config


def validate_config(config: Config):
    """
    Check the values of the config.yml settings. Normalizes cursor_color to a tuple.

    :raises ValueError: if a value is not valid.
    """
    def is_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    if not isinstance(config.video_censor_fps, (int, float)) or config.video_censor_fps < 0:
        raise ValueError(f"video_censor_fps must be >= 0, got {config.video_censor_fps!r}.")
    if not is_int(config.comfy_port):
        raise ValueError(f"comfy_port must be an integer, got {config.comfy_port!r}.")

    live = config.live
    for name in ('cap_top', 'cap_left'):
        if not is_int(getattr(live, name)):
            raise ValueError(f"live_{name} must be an integer, got {getattr(live, name)!r}.")
    for name in ('cap_width', 'cap_height'):
        if not is_int(getattr(live, name)) or getattr(live, name) <= 0:
            raise ValueError(f"live_{name} must be a positive integer, got {getattr(live, name)!r}.")
    if not isinstance(live.show_fps, bool):
        raise ValueError(f"live_show_fps must be true or false, got {live.show_fps!r}.")
    if not is_int(live.input_size) or live.input_size < 0 or live.input_size % 32 != 0:
        raise ValueError(f"live_input_size must be 0 or a positive multiple of 32, got {live.input_size!r}.")
    if not is_int(live.frames_in_flight) or live.frames_in_flight < 1:
        raise ValueError(f"live_frames_in_flight must be a positive integer, got {live.frames_in_flight!r}.")

    color = live.cursor_color
    if not isinstance(color, (list, tuple)) or len(color) != 3 or not all(is_int(c) and 0 <= c <= 255 for c in color):
        raise ValueError(f"live_cursor_color must be 3 integers between 0 and 255, got {color!r}.")
    live.cursor_color = tuple(color)


def load_pipeline_runtime(path: Path | str) -> tuple[int, int, str]:
    """
    Read the app settings from a pipeline.yml.

    :return: (n_workers, max_concurrent_jobs, backend)
    :raises ValueError: if a value is not valid.
    """
    with open(path, 'r') as f:
        spec = yaml.safe_load(f) or {}

    try:
        backend_name = backend.normalize(spec.get('backend'))
    except BackendError as e:
        raise ValueError(f"backend in '{path}': {e}") from e

    n_workers = spec.get('n_workers', DEFAULT_N_WORKERS)
    max_concurrent_jobs = spec.get('max_concurrent_jobs', DEFAULT_MAX_CONCURRENT_JOBS)

    if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 1:
        raise ValueError(f"n_workers in '{path}' must be a positive integer, got {n_workers!r}.")
    if not isinstance(max_concurrent_jobs, int) or isinstance(max_concurrent_jobs, bool) or max_concurrent_jobs < 0:
        raise ValueError(f"max_concurrent_jobs in '{path}' must be >= 0, got {max_concurrent_jobs!r}.")
    return n_workers, max_concurrent_jobs, backend_name


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
# Set by init_configs.
_default_censor_config_path: Path | None = None
_config_path: Path | None = None

# Public variables
# Always updated in place (never rebound), so modules that did 'from config import CONFIG' see the changes.
CONFIG: Config = Config()


def _apply_config(new: Config):
    """Copy all values of new into the global CONFIG object."""
    for f in fields(Config):
        setattr(CONFIG, f.name, getattr(new, f.name))


def config_file_hash() -> str:
    """Hash of the config.yml in use."""
    return hash_utils.md5_for_file(_config_path, 16)


def reload_config_if_changed(prev_hash: str) -> tuple[bool, str]:
    """
    Reload the config.yml settings into CONFIG if the file has changed since prev_hash.
    An invalid file is reported and CONFIG is left unchanged; it is tried again once the file changes again.
    debug and the pipeline settings are not touched.

    :return: reloaded: whether CONFIG was updated | hash: the new hash of the file
    """
    new_hash = config_file_hash()
    if new_hash == prev_hash:
        return False, new_hash

    try:
        new = load_config(_config_path)
    except Exception as e:
        logger.error(f"Not reloading {_config_path}, keeping the previous settings: {e}")
        return False, new_hash

    new.debug = CONFIG.debug
    new.backend = CONFIG.backend
    new.n_workers = CONFIG.n_workers
    new.max_concurrent_jobs = CONFIG.max_concurrent_jobs
    new.pipeline_path = CONFIG.pipeline_path
    _apply_config(new)
    logger.info(f"Reloaded {_config_path}.")
    return True, new_hash


def init_configs(profile: str | Path | None = None,
                 config_file: str | Path | None = None,
                 default_censor_config: str | Path | None = None,
                 pipeline_file: str | Path | None = None,
                 debug: bool = False):
    """
    Load the configs of a profile. Explicitly given files override the profile's ones.

    :raises ProfileError: if the profile or one of the config files cannot be found.
    """
    global _default_censor_config_path, _config_path

    profile_dir = resolve_profile_dir(profile)
    config_path, censor_config_path, pipeline_path = resolve_profile_files(
        profile_dir, config_file, default_censor_config, pipeline_file)

    logger.info(f"Using profile '{profile_dir.name}': config={config_path}, "
                f"censor_config={censor_config_path}, pipeline={pipeline_path}")

    try:
        config = load_config(config_path)
        config.n_workers, config.max_concurrent_jobs, config.backend = load_pipeline_runtime(pipeline_path)
    except ValueError as e:
        raise ProfileError(str(e)) from e

    # The backend must be chosen before the first pipeline or job is created.
    try:
        backend.select(config.backend)
    except BackendError as e:
        raise ProfileError(str(e)) from e
    logger.info(f"Using the '{config.backend}' backend.")
    config.debug = debug
    config.pipeline_path = Path(pipeline_path)

    _apply_config(config)
    _config_path = Path(config_path)
    _default_censor_config_path = censor_config_path
    logger.info(f"Loaded config from {config_path}.")
