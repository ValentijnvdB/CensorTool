import py_censor_core as cc

import yaml
from tqdm import tqdm
from loguru import logger

import constants
from ..config import *


def censor_images(
        image_paths: list[Path],
        output_dir: Path|None = None,
        override_cache: bool = False,
        skip_existing: bool = False,
        only_analyze: bool = False,
        censor_config: Path | str = None) -> None:
    """
    Censor all images

    :param image_paths: the images to censor.
    :param output_dir: the output directory, if None images are not written to file.
    :param override_cache: whether to overwrite cached versions.
    :param skip_existing: whether to skip already existing censored images (in the output directory).
    :param only_analyze: whether to only analyze the images for features and bodies.
    :param censor_config: the censoring configuration.
    """
    if len(image_paths) == 0:
        return

    futures: list = []
    success_count = 0
    error_count = 0
    skipped_count = 0

    # load config
    censor_config: dict = read_censor_config_from_file(censor_config)

    with cc.ImagePipeline(max_workers=CONFIG.n_workers, mode='s') as pl:
        def add_image(image_or_path: Path, save_path: Path|None) -> bool:
            """Drop-in callback for your external API."""
            job = cc.Job(
                image=str(image_or_path),
                output_path=str(save_path),
                early_exit=only_analyze,
                override_cache=override_cache,
                skip_existing=skip_existing,
                use_cache=True,
                cache_base_dir=str(constants.image_cache_path),
                config=censor_config,
                size=CONFIG.picture_size
            )
            futures.append( pl.submit(job) )
            return True

        current_image: int = 0
        def create_jobs():
            """Create jobs until max_concurrent_jobs is reached."""
            nonlocal current_image
            while len(futures) < CONFIG.max_concurrent_jobs and current_image < len(image_paths):
                image = image_paths[current_image]
                path = output_dir / image.name
                if skip_existing and path.exists():
                    continue
                success = add_image(image, path)
                current_image += 1
                if not success:
                    break

        # Create the initial jobs.
        create_jobs()

        pbar = tqdm(total=len(image_paths))
        while futures:
            # Block until at least one analyzes job is done
            done = cc.wait_any(futures)
            completed = done.wait()
            futures.remove(done)

            # process result
            if completed.success:
                success_count += 1
                # logger.info(f"features: {len(completed.result['features'])}, bodies: {len(completed.result['bodies'])}")
            else:
                error_count += 1
                log_str = f"[{completed.job_id}] error: {completed.error}."
                logger.error(log_str)

            if completed.skipped:
                skipped_count += 1

            pbar.update(1)

            # create new jobs
            create_jobs()

        pbar.close()

    logger.info(f"Censored {success_count} successfully. {error_count} errors. (total {error_count + success_count})")
    logger.info(f"Skipped: {skipped_count}")


