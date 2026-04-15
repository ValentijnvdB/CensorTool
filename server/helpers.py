import uuid
from pathlib import Path

import numpy as np

import constants

import py_censor_core as cc

from . import utils, server_config

# TODO: use config for vars

# censoring pipeline
_image_pipeline = cc.ImagePipeline(max_workers=8)
_video_pipeline: cc.VideoPipeline = None
_debug = False


def start_pipeline(debug: bool):
    global _debug, _video_pipeline
    _image_pipeline.start()
    _video_pipeline = cc.VideoPipeline(_image_pipeline)
    _debug = debug


def stop_pipeline():
    _image_pipeline.stop()
    _video_pipeline.stop()


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
        cache_base_dir=server_config.CACHE_DIR
    )
    if censor_config is not None:
        job.config = censor_config

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

    job = cc.Job(
        job_id=job_id,
        image=image,
        size=1280,
        output_path=str(output_path) if output_path else None,
        early_exit=early_exit,
        override_cache=False,
        use_cache=True,
        cache_base_dir=server_config.CACHE_DIR
    )
    if censor_config is not None:
        job.config = censor_config

    return job, _image_pipeline.submit(job)


def submit_censoring_job(image: bytes, output_path: Path|None, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    return _submit_job(image, output_path, early_exit=False, censor_config=censor_config)

def submit_detection_job(image: bytes, output_path: Path|None, censor_config: dict|None) -> tuple[cc.Job, cc.JobHandle]:
    return _submit_job(image, output_path, early_exit=True, censor_config=censor_config)

