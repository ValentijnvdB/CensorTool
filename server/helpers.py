from __future__ import annotations

import copy
import uuid
from pathlib import Path
from urllib.parse import quote

import numpy as np

import constants

from core.backend import cc

from app.config import (COMFY_KEYS, current_profile, make_image_pipeline,
                        read_censor_config_from_file, resolve_stickers_dir)
from core.boxes import is_sticker_file
from . import utils, server_config

# censoring pipelines, created by start_pipeline()
_image_pipeline: cc.ImagePipeline | None = None
_video_pipeline: cc.VideoPipeline | None = None
_debug = False

# Censor config used when a request does not provide one.
# Both backends treat a job without a config as detection-only (nothing is censored).
_default_censor_config: dict | None = None


def start_pipeline(debug: bool):
    global _debug, _image_pipeline, _video_pipeline, _default_censor_config
    _default_censor_config = read_censor_config_from_file()
    _image_pipeline = make_image_pipeline(mode='s')
    _image_pipeline.start()
    _video_pipeline = cc.VideoPipeline(_image_pipeline)
    _debug = debug


def get_labels() -> dict[str, str]:
    """Every label the loaded pipeline can produce, mapped to the model that produces it."""
    if _image_pipeline is None:
        raise RuntimeError("The pipeline has not been started.")
    return _image_pipeline.labels


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


def get_profile() -> str | None:
    """Name of the loaded profile."""
    return current_profile()


def stop_pipeline():
    if _image_pipeline is not None:
        _image_pipeline.stop()


async def submit_video_job(video_path: Path, output_path: Path, early_exit: bool, censor_config: dict | None) -> cc.VideoJob:
    avi_path = output_path.parent / (output_path.stem + '.avi')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job = cc.VideoJob(
        job_id=video_path.stem,
        video_path=str(video_path),
        output_path=str(output_path),
        avi_path=str(avi_path),
        early_exit=early_exit,
        override_cache=False,
        use_cache=True,
        cache_base_dir=str(server_config.CACHE_DIR),
        config=censor_config if censor_config is not None else _default_censor_config,
    )

    return _video_pipeline.submit(job)


async def submit_gif_job(image: bytes, output_path: Path|None, early_exit: bool, censor_config: dict|None) -> cc.VideoJob:
    job_id = str(uuid.uuid4())
    input_path: Path = constants.data_root / 'server' / 'input_gif' / (job_id + '.gif')
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'wb') as f:
        f.write(image)

    if output_path is None:
        output_path: Path = constants.data_root / 'server' / 'gifs' / (job_id + '.gif')

    return await submit_video_job(input_path, output_path, early_exit, censor_config=censor_config)



def _submit_job(image: bytes, output_path: Path|None, early_exit: bool, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    image: np.ndarray = utils.bytes_to_np(image)

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

    return job, _image_pipeline.submit(job)


def submit_censoring_job(image: bytes, output_path: Path|None, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    return _submit_job(image, output_path, early_exit=False, censor_config=censor_config)

def submit_detection_job(image: bytes, output_path: Path|None, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    return _submit_job(image, output_path, early_exit=True, censor_config=censor_config)

