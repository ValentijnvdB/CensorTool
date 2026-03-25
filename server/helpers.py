import uuid
from concurrent.futures import Future
from pathlib import Path

import numpy as np

import constants
from app.video.types import VideoJob
from app.video.video import censor_video
from core import Job, ImageInput, CensorConfig, ImagePipeline

from . import utils, server_config


# censoring pipeline
_pipeline = ImagePipeline(max_workers=4)
_debug = False


def start_pipeline(debug: bool):
    global _debug
    _pipeline.start()
    _debug = debug


def stop_pipeline():
    _pipeline.stop()


async def submit_video_job(video: Path, output_path: Path, early_exit: bool, censor_config: CensorConfig|None) -> Job:
    avi_path = output_path.parent / (output_path.stem + '.avi')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job = VideoJob(
        job_id=video.stem,
        video=video,
        output_path=output_path,
        avi_path=avi_path,
        early_exit=early_exit,
        override_cache=False,
        skip_cache_write=False,
        cache_base_dir=server_config.CACHE_DIR
    )
    if censor_config is not None:
        job.config = censor_config

    return censor_video(job, _pipeline)


async def submit_gif_job(image: bytes, output_path: Path|None, early_exit: bool, censor_config: CensorConfig|None) -> Job:
    job_id = str(uuid.uuid4())
    input_path: Path = constants.data_root / 'server' / 'input_gif' / (job_id + '.gif')
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'wb') as f:
        f.write(image)

    if output_path is None:
        output_path = constants.data_root / 'server' / 'gifs' / (job_id + '.gif')

    return await submit_video_job(input_path, output_path, early_exit, censor_config=censor_config)



def _submit_job(image: bytes, exp_extension: str, output_path: Path|None, early_exit: bool, censor_config: CensorConfig|None) -> tuple[Job, Future[Job]]:
    image: np.ndarray = utils.bytes_to_np(image)

    job_id = str(uuid.uuid4())

    if output_path is None and _debug:
        output_path = constants.data_root / 'server' / 'debug' / (job_id + '.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)

    job = Job(
        job_id=job_id,
        image=image,
        output_path=output_path,
        early_exit=early_exit,
        override_cache=False,
        skip_cache_write=False,
        cache_base_dir=server_config.CACHE_DIR,
        output_extension=exp_extension,
        return_bytes=True
    )
    if censor_config is not None:
        job.config = censor_config

    return job, _pipeline.submit(job)


def submit_censoring_job(image: bytes, exp_extension: str, output_path: Path|None, censor_config: CensorConfig|None) -> tuple[Job, Future[Job]]:
    return _submit_job(image, exp_extension, output_path, early_exit=False, censor_config=censor_config)

def submit_detection_job(image: bytes, exp_extension: str, output_path: Path|None, censor_config: CensorConfig|None) -> tuple[Job, Future[Job]]:
    return _submit_job(image, exp_extension, output_path, early_exit=True, censor_config=censor_config)

