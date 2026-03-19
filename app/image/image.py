from concurrent.futures import Future, wait, FIRST_COMPLETED

from tqdm import tqdm

from ..config import CONFIG, load_censor_config_from_file

from core import *


def censor_images(
        image_paths: list[Path],
        output_dir: Path|None = None,
        override_cache: bool = False,
        skip_existing: bool = False,
        only_analyze: bool = False,
        censor_config: dict[str, Any] | Path | str = None) -> None:
    """
    Censor all images

    :param image_paths: the images to censor.
    :param output_dir: the output directory, if None images are not written to file.
    :param override_cache: whether to overwrite cached versions.
    :param skip_existing: whether to skip already existing censored images (in the output directory).
    :param only_analyze: whether to only analyze the images for features and bodies.
    :param censor_config: the censoring configuration.
    """
    futures: dict[str, Future] = {}
    success_count = 0
    error_count = 0

    if isinstance(censor_config, dict):
        censor_config = CensorConfig(**censor_config)
    else:
        censor_config = load_censor_config_from_file(censor_config)

    with ImagePipeline(max_workers=CONFIG.n_workers) as pl:
        def add_image(image_or_path: ImageInput, save_path: Path|None) -> bool:
            """Drop-in callback for your external API."""
            job = Job(
                image=image_or_path,
                output_path=save_path,
                early_exit=only_analyze,
                override_cache=override_cache,
                sizes=CONFIG.picture_sizes,
                config=censor_config
            )
            futures[save_path.name] = pl.submit(job)
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
            done, _ = wait(futures.values(), return_when=FIRST_COMPLETED)

            # look for the finished job with the lowest frame_number
            for job_id, future in futures.items():
                if future in done:
                    completed: Job = future.result()
                    del futures[job_id]
                    break

            # process result
            if completed.success:
                success_count += 1
            else:
                error_count += 1
                log_str = f"[{completed.job_id}] error: {completed.error}."
                if CONFIG.debug:
                    log_str += f" stacktrace: {completed.stacktrace}"
                logger.error(log_str)

            pbar.update(1)

            # create new jobs
            create_jobs()

        pbar.close()

    logger.info(f"Censored {success_count} successfully. {error_count} errors. (total {error_count + success_count})")


