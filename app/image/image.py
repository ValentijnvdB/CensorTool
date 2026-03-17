from concurrent.futures import Future

from tqdm import tqdm

from app.config import CONFIG, load_censor_config_from_file

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
    futures: list[Future] = []
    success_count = 0
    error_count = 0

    if isinstance(censor_config, dict):
        censor_config = CensorConfig(**censor_config)
    else:
        censor_config = load_censor_config_from_file(censor_config)

    with ImagePipeline(max_workers=4) as pl:
        def add_image(image_or_path: ImageInput, save_path: Path|None) -> None:
            """Drop-in callback for your external API."""
            job = Job(
                image=image_or_path,
                output_path=save_path,
                early_exit=only_analyze,
                override_cache=override_cache,
                sizes=CONFIG.picture_sizes,
                config=censor_config
            )
            futures.append(pl.submit(job))

        for image_path in image_paths:
            output_path = output_dir / image_path.name
            if skip_existing and output_path.exists():
                continue
            add_image(image_path, output_path)

        for future in tqdm(futures, total=len(futures)):
            result = future.result()
            if result.success:
                success_count += 1
            else:
                error_count += 1
                log_str = f"[{result.job_id}] error: {result.error}."
                if CONFIG.debug:
                    log_str += f" stacktrace: {result.stacktrace}"
                logger.error(log_str)

    logger.info(f"Censored {success_count} successfully. {error_count} errors. (total {error_count + success_count})")


