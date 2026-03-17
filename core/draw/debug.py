import cv2
import numpy as np
from shapely import Polygon

from .censor import draw_bar



def draw_debug_bodies(image: np.ndarray, all_bodies: list[list[Polygon]], alpha: float = 0.5) -> np.ndarray:

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    original_image = image.copy()

    i = 0
    for bodies in all_bodies:
        for body in bodies:
            color = colors[i % len(colors)]

            image = draw_bar(image, body, color)
            i += 1

    image = cv2.addWeighted(original_image, 1-alpha, image, alpha, 0)

    return image