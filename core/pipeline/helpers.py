import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from .. import get_general_config
from ..draw import draw_debug_bodies
from ..models import nudenet as nn, human_detection as hd, detect_eyes
from ..censor import process_multiple_passes, censor_image_from_boxes
from ..datatypes import *

from . import Elements

from utils import hash_utils
import constants



def load_image(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def check_cache(image: np.ndarray, sizes: list[int], cache_base_dir: Path, override_cache: bool) -> list[Elements]:
    """Check the cache for each of the sizes."""
    image_hash = hashlib.md5(image.tobytes()).hexdigest()[16:]

    out: list[Elements] = []

    for size in sizes:
        cache_path = (cache_base_dir / f'{image_hash}_{constants.picture_saved_box_version}_{size}.gz')

        if (not override_cache) and cache_path.exists() and cache_path.is_file():
            logger.debug(f"Found image in cache: {cache_path}")
            result = hash_utils.read_object(cache_path)
            out.append(Elements(
                features=result['features'],
                bodies=result['bodies'],
                cache_path=cache_path
            ))

        else:
            logger.debug(f"Did not find image in cache: {cache_path}")
            out.append(Elements(
                features=None,
                bodies=None,
                cache_path=cache_path
            ))

    return out

def write_cache(item: Elements) -> None:
    """Persist all_raw_boxes (and optionally bodies) to cache."""
    hash_utils.write_object({
        'features': item.features,
        'bodies': item.bodies
    }, item.cache_path)


def prepare_image_variants(image: np.ndarray, sizes: list[int]) -> tuple[list[np.ndarray], list[float]]:
    """
    Return a list of preprocessed images and there scales (compared to the original)
    """
    adj_images = []
    scales = []
    for size in sizes:
        img, scale = nn.prep_img_for_nudenet(image, size)
        adj_images.append(img)
        scales.append(scale)

    return adj_images, scales


def process_raw_nudenet_output(
        model_output: tuple[np.ndarray, np.ndarray, np.ndarray],
        scales: list[float],
        timestamp: float,
        needs_to_detect_features: list[bool],
        cached_items: list[Elements]) -> list[list[RawBox]]:
    """
    Process the raw output of nudenet and zips it up with the raw boxes taken from the cache
    """
    model_output = nn.clean_nudenet_output(model_output)

    all_raw_boxes = nn.raw_boxes_from_model_output(model_output, scales=scales, timestamp=timestamp)

    if any([not b for b in needs_to_detect_features]):
        # also used cached items
        # zip up cached and non-cached items

        final_raw_boxes: list[list[RawBox]] = []
        for i, newly_computed in enumerate(needs_to_detect_features):
            final_raw_boxes.append(all_raw_boxes.pop(0) if newly_computed else cached_items[i].features)
    else:
        # there were no cached items
        final_raw_boxes = all_raw_boxes

    return final_raw_boxes


def postprocess(
        image: np.ndarray,
        adj_images: list[np.ndarray],
        scales: list[float],
        timestamp: float,
        needs_to_detect_features: list[bool],
        needs_to_detect_bodies: list[bool],
        raw_nn_output: tuple[np.ndarray, np.ndarray, np.ndarray],
        raw_hd_output: list[Any] | None,
        cached_items: list[Elements],
        skip_cache_write: bool
) -> list[Elements]:
    """
    Turn raw model outputs into usable data and writes new data to the cache.
    Returns the processed data as a list of Elements.
    """
    cleaned_nn_output = nn.clean_nudenet_output(raw_nn_output)

    # process the raw model outputs into the correct datatypes
    all_raw_boxes = nn.raw_boxes_from_model_output(cleaned_nn_output, scales=scales, timestamp=timestamp)
    all_bodies = hd.process_raw_output(raw_hd_output,
                                       [img for (i, img) in enumerate(adj_images) if needs_to_detect_bodies[i]],
                                       model=get_general_config().body_detection_model)
    all_bodies = hd.scale_polygons(all_bodies, scales)

    # process into Elements
    for i, (new_features, new_bodies, scale) in enumerate(zip(needs_to_detect_features, needs_to_detect_bodies, scales)):
        rbs: list[RawBox] | None = None
        if new_features:
            rbs = all_raw_boxes[i]
            # detect eyes
            rbs.extend(detect_eyes(image, rbs, timestamp))

        bodies = None
        if new_bodies:
            bodies = all_bodies.pop(0)

        if cached_items[i].features is None:
            assert rbs is not None
            cached_items[i].features = rbs

        if cached_items[i].bodies is None:
            cached_items[i].bodies = bodies

        # write to cache if something has changed
        if (not skip_cache_write) and (new_features or new_bodies):
            write_cache(cached_items[i])

    return cached_items


def apply_censor(image: np.ndarray, items: list[Elements], write_path: Path|None, censor_config: CensorConfig) -> np.ndarray:
    """
    Apply the processed data to censor the image.
    If write_path is not None, write the censored image to write_path.
    Returns the censored image.
    """
    boxes = process_multiple_passes(
        all_passes=[item.features for item in items],
        all_passes_bodies=[item.bodies for item in items],
        censor_config=censor_config
    )

    if get_general_config().debug:
        image = draw_debug_bodies(image, [item.bodies for item in items])
    censored_image = censor_image_from_boxes(image, boxes, censor_config)

    if write_path is not None:
        if write_path.exists():
            count = 1
            stem = write_path.stem
            ext = write_path.suffix
            directory = write_path.parent
            while write_path.exists():
                write_path = directory / f"{stem}_{count}{ext}"
                count += 1

        cv2.imwrite(str(write_path), censored_image)

    return censored_image