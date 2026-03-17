import json
import math
import subprocess
from collections import deque
from typing import Generator

from loguru import logger
from shapely import Polygon

import constants
from utils import hash_utils

from core import *
from core import process_multiple_passes

from ..video.ffmpeg import get_ffprobe



##########################################################################
## Cache form. A cache file for a video and size combination.
##  tuple (raw_boxes, bodies)
##      raw_boxes: list[RawBox]
##      bodies: list[Polygon]
##########################################################################

def check_cache(video_path: Path, sizes: list[int], override_cache: bool) \
        -> tuple[list[dict[int, list[RawBox]]], list[dict[int, list[Polygon]]], list[Path]]:
    """Check the cache for each of the sizes."""
    file_hash = hash_utils.md5_for_file(video_path, 16)

    raw_boxes = []
    bodies = []
    cache_paths = []

    for size in sizes:
        cache_path = (constants.video_cache_path / f'{file_hash}_{constants.picture_saved_box_version}_{size}.gz')
        cache_paths.append(cache_path)

        if (not override_cache) and cache_path.exists() and cache_path.is_file():
            logger.debug(f"Found video in cache: {cache_path.relative_to(constants.video_cache_path)}")
            result: tuple[dict[int, list], list[dict[int, list]]] = hash_utils.read_object(cache_path)
            raw_boxes.append(result[0])
            bodies.append(result[1])

        else:
            logger.debug(f"Did not find video in cache: {cache_path.relative_to(constants.video_cache_path)}")
            raw_boxes.append({})
            bodies.append({})

    return raw_boxes, bodies, cache_paths


def write_cache(path: Path, features: dict[int, list[RawBox]], bodies: dict[int, list[Polygon]]) -> None:
    """Persist all_raw_boxes (and optionally bodies) to cache."""
    hash_utils.write_object((features, bodies), path)


def censor_frames(censor_fps: float, video_fps: float, max_frame: int) -> Generator[int, None, None]:
    factor = video_fps / censor_fps
    for i in range(max_frame):
        frame = math.floor(i * factor)
        if frame >= max_frame:
            break

        yield frame


def process_raw_data(bodies_per_pass: list[dict[int, list[Polygon]]],
                     raw_boxes_per_pass: list[dict[int, list[RawBox]]],
                     censor_config: CensorConfig) -> deque[Box]:
    """
    Process cached features and bodies into a sorted list of boxes.
    """
    # restructure the data structure
    # from  sizes     ->    frame_id   ->   RawBox/Polygon
    # to    frame_id  ->    size       ->   RawBox/Polygon
    all_raw_boxes: dict[int, list[list[RawBox]]] = {}
    all_bodies: dict[int, list[list[Polygon]]] = {}
    for raw_boxes_dict, bodies_dict in zip(raw_boxes_per_pass, bodies_per_pass):
        for frame_id, rb in raw_boxes_dict.items():
            if frame_id not in all_raw_boxes:
                all_raw_boxes[frame_id] = [rb]
            else:
                all_raw_boxes[frame_id].append(rb)
        for frame_id, body in bodies_dict.items():
            if frame_id not in all_bodies:
                all_bodies[frame_id] = [body]
            else:
                all_bodies[frame_id].append(body)

    # convert to Boxes
    boxes: list[Box] = []
    for frame_id in set(all_raw_boxes.keys()):
        raw_boxes = all_raw_boxes.pop(frame_id)
        bodies = all_bodies.pop(frame_id)

        boxes.extend(process_multiple_passes(raw_boxes, bodies, censor_config))

    boxes.sort()
    return deque(boxes)


def get_frame_count(path: Path | str) -> int:
    result = subprocess.run(
        [get_ffprobe(), "-v", "quiet", "-print_format", "json",
         "-show_streams", str(path)],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    try:
        for stream in info["streams"]:
            if stream["codec_type"] == "video":
                return int(stream["nb_frames"])
    except:
        pass

    return -1

