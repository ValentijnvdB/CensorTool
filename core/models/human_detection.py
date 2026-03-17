from typing import Literal

import numpy as np
from shapely import Polygon
from shapely.affinity import scale as scale_poly

from . import pphumanseg
from . import yolo

_MODELS = Literal['fastest', 'faster', 'fast', 'medium-fast', 'medium', 'medium-slow', 'slow', 'slower', 'slowest']

_YOLO_MODEL_MAPPING = {
    'faster': 'yolov8n',
    'fast': 'yolov8s',
    'medium-fast': 'yolov8m',
    'medium': 'yolov11s',
    'medium-slow': 'yolov11m',
    'slow': 'yolov8l',
    'slower': 'yolov8x',
    'slowest': 'yolov11x',
}

def find_human_polygons(images: list[np.ndarray], model: _MODELS = 'fastest') -> list[Polygon]:
    """
    Find all humans detected in the image.

    :param images: The images to find humans in.
    :param model: the model to use.

    :return: for each image: the list of humans detected in the image in the non-scaled image space.
    """
    all_polygons = []
    for image in images:
        if model == 'fastest':
            all_polygons.append(pphumanseg.find_human_polygons(image))
        else:
            all_polygons.append(yolo.find_human_polygons(image, backend=_YOLO_MODEL_MAPPING[model]))

    return all_polygons


def process_raw_output(model_output, images: list[np.ndarray] = None, model: _MODELS = 'fastest') -> list[list[Polygon]]:
    """
    Convert the raw output of the model to shapely polygons.
    """
    if model == 'fastest':
        to_polygons = pphumanseg.to_polygons
    elif model in _YOLO_MODEL_MAPPING:
        to_polygons = yolo.to_polygons
    else:
        raise ValueError(f'Unknown model: {model}')

    all_polygons = []
    for result, image in zip(model_output, images):
        height, width, _ = image.shape

        all_polygons.append( to_polygons(result, height=height, width=width) )

    return all_polygons


def scale_polygons(all_polygons: list[list[Polygon]], scales: list[float]) -> list[list[Polygon]]:
    """
    Transform polygons from a downscaled image's coordinate space back to
    the original image's coordinate space.

    :param all_polygons: a list of lists of the polygons to scale the image's coordinate space.
    :param scales: a list of scales to scale the image's coordinate space.

    :return: a list of lists of the scaled polygons.
    """
    result = []
    for polygons, scale in zip(all_polygons, scales):
        factor = 1.0 / scale
        result.append( [
            scale_poly(poly, xfact=factor, yfact=factor, origin=(0, 0))
            for poly in polygons
        ])

    return result





