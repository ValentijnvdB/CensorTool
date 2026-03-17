from pathlib import Path

import numpy as np

from core import Job, ImageInput

from app.image.image import ImagePipeline


from . import utils, server_config

# censoring pipeline
_pipeline = ImagePipeline(max_workers=4)


def start_pipeline():
    _pipeline.start()

def stop_pipeline():
    _pipeline.stop()


def _submit_job(image: bytes|ImageInput, output_path: Path|None, early_exit: bool) -> Job:
    if isinstance(image, bytes):
        image: np.ndarray = utils.bytes_to_np(image)

    job = Job(
        image=image,
        output_path=output_path,
        early_exit=early_exit,
        override_cache=False,
        skip_cache_write=False,
        cache_base_dir=server_config.CACHE_DIR,
    )
    return _pipeline.submit(job).result()


def submit_censoring_job(image: bytes|ImageInput, output_path: Path|None) -> Job:
    return _submit_job(image, output_path, early_exit=False)

def submit_detection_job(image: bytes|ImageInput, output_path: Path|None) -> Job:
    return _submit_job(image, output_path, early_exit=True)


