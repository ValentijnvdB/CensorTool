import numpy as np

from shapely import Polygon

from ..datatypes import *


def inverse_censor_image(image: np.ndarray, boxes: list[Box], config: CensorConfig) -> np.ndarray:
    """
    Apply inverse censoring to the image.

    :param image: the image
    :param boxes: the boxes that should not be censored
    :param config: the censoring configuration

    :return: the censored image
    """
    inverse_config = config.inverse_censor_style
    if isinstance(inverse_config, CSBar):
        censored = inverse_draw_bar(image, color=inverse_config['color'])
    elif isinstance(inverse_config, CSBlur):
        censored = inverse_blur_image(image, blur_strength=inverse_config['blur_strength'])
    elif isinstance(inverse_config, CSPixel):
        censored = inverse_pixelate_image(image, factor=inverse_config['factor'])
    else:
        raise ValueError(f"Unknown censor_style: {inverse_config.__class__.__name__} at 'inverse_censor_box'. "
                         f"Please check 'inverse_censor_box' in the censor_box_config file.")

    return combine_images_from_shape(image, censored, [box.polygon for box in boxes])


def combine_images_from_shape(first_image: np.ndarray, second_image: np.ndarray, shapes: list[Polygon]) -> np.ndarray:
    """
    Combine two images into a single image.
    Take the pixel from first_image if pixel lies in one of the shapes otherwise take pixel from second_image.

    :param first_image: the first image
    :param second_image: the second image
    :param shapes: list of shapes

    :return: the combined image
    """
    mask = np.zeros(first_image.shape[:2], dtype=np.uint8)
    for shape in shapes:
        polygon_pts = np.array(shape.exterior.coords, np.int32)
        cv2.fillPoly(mask, [polygon_pts], 255)

    return np.where(mask[..., None] == 255, first_image, second_image)


def inverse_blur_image(image: np.ndarray, blur_strength: int) -> np.ndarray:
    """
    Apply a blur to everything except the area in shape

    :param image: the image
    :param blur_strength: the blur strength

    :return: the blurred image
    """
    if blur_strength % 2 == 0:
        blur_strength += 1

    # Apply Gaussian blur to the whole image
    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

    return blurred


def inverse_pixelate_image(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Pixelate everything except the area in shape

    :param image: the image
    :param factor: the pixelation factor

    :return: the pixelated image
    """
    small = cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return pixelated


def inverse_draw_bar(image: np.ndarray, color: tuple[int, int, int] | list[int])  -> np.ndarray:
    """
    Fills the whole area with color except for the area in shape.

    :param image: the image
    :param color: the color of the pixels

    :return: the modified image
    """
    return np.full(image.shape, color, dtype=np.uint8)

