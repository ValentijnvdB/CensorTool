import math
from typing import Any

import cv2

import numpy as np

from ..datatypes import CensorConfig


def watermark_image(image: np.ndarray, config: CensorConfig) -> np.ndarray:
    """
    Add a watermark to the image.

    :param image: the image
    :param config: the censoring configuration

    :return: the modified image
    """
    if config.enable_watermark:
        image = np.ascontiguousarray(image)
        (h, w, _) = image.shape
        scale = max(min(w / 750, h / 750), 1)
        return (cv2.putText(image, 'Censored with CensoringApp', (20, math.ceil(20 * scale)), cv2.FONT_HERSHEY_PLAIN,
                            scale, (0, 0, 255), math.floor(scale)))
    else:
        return image


def annotate_image_shape(image: np.ndarray) -> np.ndarray:
    """
    Draw the shape of the image in the top left of the image

    :param image: the image

    :return: the modified image
    """
    return annotate_image(image, str(image.shape))


def annotate_image(image: np.ndarray, text: Any, index: int = 0) -> np.ndarray:
    """
    Annotate image with text in the top left.

    :param image: the image
    :param text: the text to annotate
    :param index: the row of the text will be annotated

    :return: the annotated image
    """
    return cv2.putText(image, str(text), (20, 20*(index+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
