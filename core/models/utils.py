import numpy as np


def get_resize_scale(img_w: int, img_h: int, max_length: int) -> float:
    """
    Get how much the image has to be resized before sending it to the datatypes

    :param img_h: original image height
    :param img_w: original image width
    :param max_length: max length of the resized image

    :return: scale factor
    """
    if max_length == 0:
        return 1
    else:
        return max_length / max(img_h, img_w)


def get_image_resize_scale(raw_img: np.ndarray, max_length: int) -> float:
    """
    Get how much the image has to be resized before sending it to the datatypes

    :param raw_img: the original (raw) image
    :param max_length: max length of the resized image

    :return: scale factor
    """
    (height, width, _) = raw_img.shape
    return get_resize_scale(img_w=width, img_h=height, max_length=max_length)