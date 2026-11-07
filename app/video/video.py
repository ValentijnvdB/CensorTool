from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm
from loguru import logger

import constants
from ..config import *

import censor_core as cc


def censor_videos(
        video_paths: list[Path],
        output_dir: Path|None = None,
        override_cache: bool = False,
        skip_existing: bool = False,
        only_analyze: bool = False,
        censor_config: Path | str = None) -> None:
    """
    Censor all images

    :param video_paths: the videos to censor.
    :param output_dir: the output directory, if None images are not written to file.
    :param override_cache: whether to overwrite cached versions.
    :param skip_existing: whether to skip already existing censored images (in the output directory).
    :param only_analyze: whether to only analyze the images for features and bodies.
    :param censor_config: the censoring configuration.
    """
    if len(video_paths) == 0:
        return

    # load config
    censor_config: dict = read_censor_config_from_file(censor_config)

    with make_video_pipeline() as pl:
        for video_path in tqdm(video_paths):
            if output_dir is None:
                output_dir = video_path.parent
            output_path = str(output_dir / video_path.name)
            avi_path = str(output_dir / (video_path.stem + '.avi'))

            vid_job = cc.VideoJob(
                job_id=video_path.stem,
                video_path=str(video_path),
                avi_path=avi_path,
                output_path=output_path,
                early_exit=only_analyze,
                override_cache=override_cache,
                skip_existing=skip_existing,
                use_cache=True,
                cache_base_dir=str(constants.video_cache_path),
                config=censor_config,
                size=CONFIG.picture_size,
                censor_fps=CONFIG.video_censor_fps
            )
            completed = pl.submit(vid_job)

            if completed.success:
                logger.info(f"{video_path.name} completed successfully!")
            else:
                logger.error(f"Error: {completed.error}")



