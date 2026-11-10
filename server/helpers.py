from __future__ import annotations

import asyncio
import copy
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

import constants

from core import backend
from core.backend import cc
from core.censor_config import Shape, censor_config_from_dict
from core.config_utils import collect_warnings, did_you_mean

from app.config import (COMFY_KEYS, CensorConfigError, check_client_censor_config, current_profile,
                        make_image_pipeline, read_censor_config_from_file, resolve_stickers_dir)
from core.boxes import is_sticker_file
from . import utils, server_config
from .errors import ApiError, CONFIG_NOT_ALLOWED, GIF_MESSAGE, INVALID_CONFIG, INVALID_IMAGE, USE_CENSOR_VIDEO

# censoring pipelines, created by start_pipeline()
_image_pipeline: cc.ImagePipeline | None = None
_video_pipeline: cc.VideoPipeline | None = None
_debug = False
_started_at: float | None = None

# Censor config used when a request does not provide one.
# Both backends treat a job without a config as detection-only (nothing is censored).
_default_censor_config: dict | None = None


@dataclass
class _Activity:
    """What the server is busy with, for GET /health."""
    image_jobs: list = field(default_factory=list)  # handles of submitted image jobs; finished ones are pruned
    video_jobs_running: int = 0
    video_jobs_waiting: int = 0
    open_frame_streams: int = 0

    def pending_image_jobs(self) -> int:
        self.image_jobs = [h for h in self.image_jobs if not h.done()]
        return len(self.image_jobs)


activity = _Activity()

# One video at a time, as before; it runs in a thread so the server keeps answering other requests.
_video_lock: asyncio.Lock | None = None


def start_pipeline(debug: bool):
    global _debug, _image_pipeline, _video_pipeline, _default_censor_config, _started_at
    _default_censor_config = read_censor_config_from_file()
    _image_pipeline = make_image_pipeline(mode='s')
    _image_pipeline.start()
    _video_pipeline = cc.VideoPipeline(_image_pipeline)
    _debug = debug
    _started_at = time.time()


def stop_pipeline():
    if _image_pipeline is not None:
        _image_pipeline.stop()


def get_labels() -> dict[str, str]:
    """Every label the loaded pipeline can produce, mapped to the model that produces it."""
    if _image_pipeline is None:
        raise RuntimeError("The pipeline has not been started.")
    return _image_pipeline.labels


def get_profile() -> str | None:
    """Name of the loaded profile."""
    return current_profile()


def get_health() -> dict:
    """Whether the pipeline runs, and what the server is busy with."""
    running = _image_pipeline is not None and bool(_image_pipeline.is_running)
    return {
        'status': 'ok' if running else 'unavailable',
        'pipeline_running': running,
        'profile': get_profile(),
        'backend': backend.name() if running else None,
        'uptime_seconds': round(time.time() - _started_at, 1) if _started_at is not None else 0.0,
        'pending_image_jobs': activity.pending_image_jobs(),
        'running_video_jobs': activity.video_jobs_running,
        'waiting_video_jobs': activity.video_jobs_waiting,
        'open_frame_streams': activity.open_frame_streams,
    }


# -------------------------------------- censor configs --------------------------------------

def get_default_censor_config() -> dict:
    """
    The censor config used when a request does not provide one, as read from the profile's censor_config.yml
    (not parsed), without the ComfyUI settings.
    """
    if _default_censor_config is None:
        raise RuntimeError("The pipeline has not been started.")

    config = copy.deepcopy(_default_censor_config)

    def strip(style):
        if isinstance(style, dict):
            for key in COMFY_KEYS:  # server settings (host, local path), not handed out
                style.pop(key, None)

    # an absolute path after restrict_sticker_paths; show it relative to the working dir, as it is written in a config
    stickers_dir = config.get('stickers_dir')
    if stickers_dir is not None and Path(stickers_dir).is_relative_to(constants.instance_dir):
        config['stickers_dir'] = Path(stickers_dir).relative_to(constants.instance_dir).as_posix()

    strip((config.get('default_censor_config') or {}).get('censor_style'))
    strip(config.get('inverse_censor_style'))
    for box in (config.get('feature_overrides') or {}).values():
        if isinstance(box, dict):
            strip(box.get('censor_style'))
    return config


@dataclass
class ConfigCheck:
    """The outcome of inspect_censor_config."""
    config: dict | None                 # the checked config, ready for a job; None when there are errors
    errors: list[str]
    warnings: list[str]
    code: str | None = None             # error code of the errors (CONFIG_NOT_ALLOWED or INVALID_CONFIG)

    @property
    def valid(self) -> bool:
        return not self.errors


def _pipeline_errors(parsed, warnings: list[str]) -> list[str]:
    """
    Checks a parsed censor config against the loaded models: unknown labels are warnings; clip_to targets that no
    model produces, and (Python backend only) shape MASK or clip_to on a detect model, are errors.
    """
    if _image_pipeline is None:
        return []
    labels = dict(_image_pipeline.labels)
    graph = getattr(_image_pipeline, 'graph', None)  # only the Python backend tells which models segment

    def is_segmentation(label: str) -> bool | None:
        if graph is None:
            return None
        return bool(graph.stage(labels[label]).polygons)

    errors = []
    for label, box in parsed.features_to_censor.items():
        if label not in labels:
            warnings.append(f"No model produces label '{label}'{did_you_mean(label, labels)}; it is never censored.")
            continue
        if box.shape == Shape.MASK and is_segmentation(label) is False:
            errors.append(f"shape MASK for '{label}' needs a segmentation model, but '{labels[label]}' is a "
                          f"detect model.")
        for target in box.clip_to:
            if target not in labels:
                errors.append(f"clip_to of '{label}': no model produces label '{target}'"
                              f"{did_you_mean(target, labels)}.")
            elif is_segmentation(target) is False:
                errors.append(f"clip_to of '{label}': '{target}' comes from detect model '{labels[target]}'; "
                              f"clip_to needs a segmentation model.")
    return errors


def inspect_censor_config(config: Any) -> ConfigCheck:
    """
    Check a censor config sent by a client, the same way a censor request does: allowed paths, the structure, and
    the labels against the loaded models. The config itself is not changed.
    """
    if not isinstance(config, dict):
        return ConfigCheck(None, ["The censor config must be a JSON object."], [], INVALID_CONFIG)
    try:
        checked = check_client_censor_config(copy.deepcopy(config))
    except CensorConfigError as e:
        return ConfigCheck(None, [str(e)], [], CONFIG_NOT_ALLOWED)

    with collect_warnings() as warnings:
        try:
            parsed = censor_config_from_dict(checked)  # same format for both backends
        except Exception as e:
            return ConfigCheck(None, [f"Invalid censor config: {e}"], list(warnings), INVALID_CONFIG)
    errors = _pipeline_errors(parsed, warnings)
    if errors:
        return ConfigCheck(None, errors, warnings, INVALID_CONFIG)
    return ConfigCheck(checked, [], warnings)


def check_censor_config(config: Any) -> dict:
    """
    The censor config sent by a client, checked and ready to use in a job.

    :raises ApiError: (400) if it is not valid or not allowed.
    """
    result = inspect_censor_config(config)
    if not result.valid:
        raise ApiError(' '.join(result.errors), result.code)
    return result.config


# -------------------------------------- stickers --------------------------------------

def get_stickers(stickers_dir: str | None = None) -> dict:
    """
    The sticker categories and images in a stickers folder.

    :param stickers_dir: as in a censor config: data/stickers or one of its subfolders, relative to the working
        directory. None is data/stickers.
    :return: {'stickers_dir': the folder relative to the working dir,
              'categories': every subfolder, relative to the folder (usable as overlay.stickers entries),
              'stickers': [{'path': relative to the folder, 'url': where GET /stickers/<path> serves it}]}
    :raises CensorConfigError: if stickers_dir is outside data/stickers.
    """
    base = resolve_stickers_dir(stickers_dir)
    root = constants.stickers_root_path.resolve()

    categories, stickers = [], []
    if base.is_dir():
        for p in sorted(base.rglob('*')):
            if not p.resolve().is_relative_to(root):  # a symlink out of data/stickers
                continue
            if p.is_dir():
                categories.append(p.relative_to(base).as_posix())
            elif is_sticker_file(p):
                stickers.append({'path': p.relative_to(base).as_posix(),
                                 'url': '/stickers/' + quote(p.relative_to(root).as_posix())})

    return {'stickers_dir': base.relative_to(constants.instance_dir).as_posix()
                            if base.is_relative_to(constants.instance_dir) else str(base),
            'categories': categories,
            'stickers': stickers}


def get_sticker_file(path: str) -> Path:
    """
    The sticker image at path, relative to data/stickers.

    :raises FileNotFoundError: if there is no sticker image there, or it is outside data/stickers.
    """
    root = constants.stickers_root_path.resolve()
    file = (root / path).resolve()
    if not file.is_relative_to(root) or not is_sticker_file(file):
        raise FileNotFoundError(f"No sticker '{path}'.")
    return file


# -------------------------------------- cache --------------------------------------

def get_cache_info() -> dict:
    """Size of the server's detection cache."""
    files = [p for p in server_config.CACHE_DIR.rglob('*') if p.is_file()] if server_config.CACHE_DIR.exists() else []
    return {'cache_dir': _relative(server_config.CACHE_DIR),
            'files': len(files),
            'bytes': sum(p.stat().st_size for p in files)}


def clear_cache() -> dict:
    """Delete everything in the server's detection cache. Returns how much was deleted."""
    info = get_cache_info()
    if server_config.CACHE_DIR.exists():
        for item in server_config.CACHE_DIR.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    return {'deleted_files': info['files'], 'deleted_bytes': info['bytes']}


def _relative(path: Path) -> str:
    path = Path(path)
    return path.relative_to(constants.instance_dir).as_posix() if path.is_relative_to(constants.instance_dir) \
        else str(path)


# -------------------------------------- jobs --------------------------------------

def unique_path(directory: Path, name: str) -> Path:
    """
    Reserve a file name in directory: name itself, or name with '_1', '_2', ... added to the stem when it is taken.
    The (empty) file is created, so two requests never get the same name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 0
    while True:
        path = directory / (f"{stem}_{i}{suffix}" if i else name)
        try:
            with open(path, 'xb'):
                return path
        except FileExistsError:
            i += 1


async def submit_video_job(video_path: Path, output_path: Path, early_exit: bool, censor_config: dict | None,
                           cache_dir: Path | None = None) -> cc.VideoJob:
    """
    Censor a video (or GIF) in a thread, one at a time. Other requests are answered meanwhile.

    :param cache_dir: where its detections are cached; default the server's cache folder.
    """
    global _video_lock
    if _video_lock is None:
        _video_lock = asyncio.Lock()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # never the output itself, which it would be for an .avi output
    avi_path = output_path.parent / f"{output_path.stem}.{uuid.uuid4().hex[:8]}.tmp.avi"

    job = cc.VideoJob(
        job_id=video_path.stem,
        video_path=str(video_path),
        output_path=str(output_path),
        avi_path=str(avi_path),
        early_exit=early_exit,
        override_cache=False,
        use_cache=True,
        cache_base_dir=str(cache_dir if cache_dir is not None else server_config.CACHE_DIR),
        config=censor_config if censor_config is not None else _default_censor_config,
    )

    activity.video_jobs_waiting += 1
    waiting = True
    try:
        async with _video_lock:
            activity.video_jobs_waiting -= 1
            waiting = False
            activity.video_jobs_running += 1
            try:
                return await asyncio.to_thread(_video_pipeline.submit, job)
            finally:
                activity.video_jobs_running -= 1
    finally:
        if waiting:  # cancelled while waiting for its turn
            activity.video_jobs_waiting -= 1


def decode_image(image: bytes) -> np.ndarray:
    """
    :raises ApiError: (400) if the bytes are not an image, or are a GIF (which is censored as a video).
    """
    if image[:4] == b'GIF8':  # also when it was sent with another extension or mime_type
        raise ApiError(GIF_MESSAGE, USE_CENSOR_VIDEO)
    decoded = utils.bytes_to_np(image) if image else None
    if decoded is None:
        raise ApiError("The image could not be read: it is empty or not a supported image.", INVALID_IMAGE)
    return decoded


def _submit_job(image: bytes, output_path: Path|None, early_exit: bool, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    image: np.ndarray = decode_image(image)

    job_id = str(uuid.uuid4())

    if output_path is None and _debug:
        output_path: Path = constants.data_root / 'server' / 'debug' / (job_id + '.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Without a config the backend runs every model and censors nothing: fine for detection,
    # but censoring jobs fall back to the default censor config.
    if censor_config is None and not early_exit:
        censor_config = _default_censor_config

    job = cc.Job(
        job_id=job_id,
        image=image,
        output_path=str(output_path) if output_path else None,
        early_exit=early_exit,
        override_cache=False,
        use_cache=True,
        cache_base_dir=str(server_config.CACHE_DIR),
        config=censor_config,
    )

    handle = _image_pipeline.submit(job)
    activity.image_jobs.append(handle)
    return job, handle


def submit_censoring_job(image: bytes, output_path: Path|None, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    return _submit_job(image, output_path, early_exit=False, censor_config=censor_config)


def submit_detection_job(image: bytes, output_path: Path|None, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    return _submit_job(image, output_path, early_exit=True, censor_config=censor_config)


async def wait_for_job(handle: cc.JobHandle) -> cc.Job:
    """Wait for an image job without blocking the server."""
    if handle.done():
        return handle.wait()
    return await asyncio.to_thread(handle.wait)
