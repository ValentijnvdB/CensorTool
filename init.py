from loguru import logger
from constants import *


def init_dirs(input_path: Path = None, output_path: Path = None):
    logger.info("Setting up directories")

    if input_path is None:
        input_path = uncensored_path
    if output_path is None:
        output_path = uncensored_path

    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    stickers_root_path.mkdir(parents=True, exist_ok=True)
    debug_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    model_root.mkdir(parents=True, exist_ok=True)

    certs_path.mkdir(parents=True, exist_ok=True)

    image_cache_path.mkdir(parents=True, exist_ok=True)
    video_cache_path.mkdir(parents=True, exist_ok=True)
    trash_bin_cache_path.mkdir(parents=True, exist_ok=True)

    profiles_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    init_dirs()

