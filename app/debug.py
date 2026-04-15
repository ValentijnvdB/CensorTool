from typing import Any

import cv2
import numpy as np


def annotate_image(image: np.ndarray, text: Any, index: int = 0) -> np.ndarray:
    """
    Annotate image with text in the top left.

    :param image: the image
    :param text: the text to annotate
    :param index: the row of the text will be annotated

    :return: the annotated image
    """
    return cv2.putText(image, str(text), (20, 20*(index+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)