import numpy as np

from shapely import Polygon

from .censor import draw_bar


def draw_border(image: np.ndarray,
                shape: Polygon,
                thickness: int,
                color: tuple[int, int, int] | list[int]) -> np.ndarray:
    """
    Draw a border around the area set by shape in image.

    :param image: the image
    :param shape: the shape of the border
    :param thickness: the thickness of the border
    :param color: the color of the border

    :return: the modified image
    """
    return draw_bar(image=image, polygon=shape, color=color, thickness=thickness)