from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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


@dataclass(eq=False)
class VideoTicket:
    """A video job and its place in the video queue."""
    job: Any                    # cc.VideoJob
    state: str = 'new'          # new | queued | running | finished
    started_at: float | None = None     # time.time() when it started running
    finished_at: float | None = None


@dataclass
class _Activity:
    """What the server is busy with, for GET /health."""
    # handles of submitted image jobs; removed once their result is taken (job_result) or when found finished
    image_jobs: set = field(default_factory=set)
    video_queue: list[VideoTicket] = field(default_factory=list)  # waiting for their turn, in order
    video_running: VideoTicket | None = None
    open_frame_streams: int = 0

    def prune_image_jobs(self):
        """Forget finished image jobs, so their images are not kept in memory."""
        self.image_jobs = {h for h in self.image_jobs if not _is_finished(h)}

    def pending_image_jobs(self) -> int:
        self.prune_image_jobs()
        return len(self.image_jobs)


def _is_finished(handle) -> bool:
    try:
        return handle.done()
    except RuntimeError:
        # censor_core < 2.1.1: done() raises "std::future_error: No associated state" after wait()
        return True


def job_result(handle: cc.JobHandle) -> cc.Job:
    """Wait for an image job and return it; it no longer counts as pending."""
    try:
        return handle.wait()
    finally:
        activity.image_jobs.discard(handle)


activity = _Activity()

# One video at a time, as before; it runs in a thread so the server keeps answering other requests.
_video_lock: asyncio.Lock | None = None


def start_pipeline(debug: bool):
    global _debug, _image_pipeline, _video_pipeline, _default_censor_config, _started_at
    _default_censor_config = read_censor_config_from_file()
    # 'p': independent models run in parallel, as in the app's live mode (faster frames for /censor_frame)
    _image_pipeline = make_image_pipeline(mode='p')
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
        'running_video_jobs': int(activity.video_running is not None),
        'waiting_video_jobs': len(activity.video_queue),
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
    model produces, and shape MASK or clip_to on a detect model, are errors.
    """
    if _image_pipeline is None:
        return []
    labels = dict(_image_pipeline.labels)
    graph = _image_pipeline.graph

    def is_segmentation(label: str) -> bool:
        return bool(graph.stage(labels[label]).polygons)

    errors = []
    for label, box in parsed.features_to_censor.items():
        if label not in labels:
            warnings.append(f"No model produces label '{label}'{did_you_mean(label, labels)}; it is never censored.")
            continue
        if box.shape == Shape.MASK and not is_segmentation(label):
            errors.append(f"shape MASK for '{label}' needs a segmentation model, but '{labels[label]}' is a "
                          f"detect model.")
        for target in box.clip_to:
            if target not in labels:
                errors.append(f"clip_to of '{label}': no model produces label '{target}'"
                              f"{did_you_mean(target, labels)}.")
            elif not is_segmentation(target):
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


def effective_censor_config(censor_config: dict | None) -> dict | None:
    """The censor config a video is censored with: the request's, else the profile's."""
    return censor_config if censor_config is not None else _default_censor_config


def censor_config_key(censor_config: dict | None) -> str:
    """
    Identifies the censor config a video is censored with, so jobs with the same config can share their work. A
    request without a config gets the profile's current one, so a changed censor_config.yml gives another key.
    """
    canonical = json.dumps(effective_censor_config(censor_config), sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_video_job(video_path: Path, output_path: Path, censor_config: dict | None, cache_dir: Path | None = None,
                   early_exit: bool = False) -> cc.VideoJob:
    """
    A video job for the video pipeline.

    :param censor_config: None for the profile's.
    :param cache_dir: where its detections are cached; default the server's cache folder.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # never the output itself, which it would be for an .avi output
    avi_path = output_path.parent / f"{output_path.stem}.{uuid.uuid4().hex[:8]}.tmp.avi"
    return cc.VideoJob(
        job_id=video_path.stem,
        video_path=str(video_path),
        output_path=str(output_path),
        avi_path=str(avi_path),
        early_exit=early_exit,
        override_cache=False,
        use_cache=True,
        cache_base_dir=str(cache_dir if cache_dir is not None else server_config.CACHE_DIR),
        config=effective_censor_config(censor_config),
    )


def queue_position(ticket: VideoTicket) -> int | None:
    """1 for the next video to run, 2 after that, ...; None when it is not waiting."""
    try:
        return activity.video_queue.index(ticket) + 1
    except ValueError:
        return None


async def run_video_job(ticket: VideoTicket) -> cc.VideoJob:
    """
    Censor a video in a thread, one video at a time, in the order they are submitted. Other requests are answered
    meanwhile. Cancelling the awaiting task takes a waiting video out of the queue; a running one is stopped with
    ticket.job.cancel().
    """
    global _video_lock
    if _video_lock is None:
        _video_lock = asyncio.Lock()  # FIFO: waiters get the lock in the order they asked for it

    ticket.state = 'queued'
    activity.video_queue.append(ticket)
    try:
        async with _video_lock:
            activity.video_queue.remove(ticket)
            ticket.state = 'running'
            ticket.started_at = time.time()
            activity.video_running = ticket
            done = asyncio.ensure_future(asyncio.to_thread(_video_pipeline.submit, ticket.job))
            try:
                return await asyncio.shield(done)
            except asyncio.CancelledError:
                # a thread cannot be interrupted: ask the job to stop, and keep the lock until it has
                if hasattr(ticket.job, 'cancel'):
                    ticket.job.cancel()
                await done
                raise
            finally:
                activity.video_running = None
    finally:
        if ticket in activity.video_queue:  # cancelled while waiting for its turn
            activity.video_queue.remove(ticket)
        ticket.state = 'finished'
        ticket.finished_at = time.time()


async def submit_video_job(video_path: Path, output_path: Path, early_exit: bool, censor_config: dict | None,
                           cache_dir: Path | None = None) -> cc.VideoJob:
    """Censor a video (see run_video_job) and wait for it."""
    job = make_video_job(video_path, output_path, censor_config, cache_dir, early_exit)
    return await run_video_job(VideoTicket(job))


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


def _submit_job(image: bytes, output_path: Path|None, early_exit: bool, censor_config: dict|None,
                use_cache: bool = True) -> tuple[cc.Job, cc.JobHandle]:
    image: np.ndarray = decode_image(image)

    job_id = str(uuid.uuid4())

    if output_path is None and _debug:
        output_path: Path = server_config.DEBUG_DIR / (job_id + '.png')
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
        use_cache=use_cache,
        cache_base_dir=str(server_config.CACHE_DIR),
        config=censor_config,
    )

    handle = _image_pipeline.submit(job)
    activity.prune_image_jobs()  # also when GET /health is never called
    activity.image_jobs.add(handle)
    return job, handle


def submit_censoring_job(image: bytes, output_path: Path|None, censor_config: dict|None,
                         use_cache: bool = True) -> tuple[cc.Job, cc.JobHandle]:
    """
    :param use_cache: read and write the detections cache (a file per image). Off for live video frames: they would
        fill the cache folder with a file per frame.
    """
    return _submit_job(image, output_path, early_exit=False, censor_config=censor_config, use_cache=use_cache)


def submit_detection_job(image: bytes, output_path: Path|None, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    return _submit_job(image, output_path, early_exit=True, censor_config=censor_config)


async def wait_for_job(handle: cc.JobHandle) -> cc.Job:
    """Wait for an image job without blocking the server."""
    if handle.done():
        return job_result(handle)
    return await asyncio.to_thread(job_result, handle)
