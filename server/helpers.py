import uuid
from pathlib import Path

import numpy as np

import constants

import censor_core as cc

from app.config import make_image_pipeline, read_censor_config_from_file
from . import utils, server_config

# censoring pipelines, created by start_pipeline()
_image_pipeline: cc.ImagePipeline | None = None
_video_pipeline: cc.VideoPipeline | None = None
_debug = False

# Censor config used when a request does not provide one.
# censor_core 2.0 treats a job without a config as detection-only (nothing is censored).
_default_censor_config: dict | None = None


def start_pipeline(debug: bool):
    global _debug, _image_pipeline, _video_pipeline, _default_censor_config
    _default_censor_config = read_censor_config_from_file()
    _image_pipeline = make_image_pipeline(mode='s')
    _image_pipeline.start()
    _video_pipeline = cc.VideoPipeline(_image_pipeline)
    _debug = debug


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

    # Without a config censor_core runs every model and censors nothing: fine for detection,
    # but censoring jobs fall back to the default censor config.
    if censor_config is None and not early_exit:
        censor_config = _default_censor_config

    job = cc.Job(
        job_id=job_id,
        image=image,
        size=1280,
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

