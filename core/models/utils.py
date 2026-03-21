from pathlib import Path

import numpy as np
import requests
from loguru import logger

import constants


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


def download_model(url: str, write_dir: Path = None) -> Path:
    """
    Download the model from url and save it to write_dir.

    :param url: model url
    :param write_dir: directory to save the model

    :return: path to saved model
    """

    file_name = url.split('/')[-1]
    logger.info(f"Downloading '{file_name}'.")

    if write_dir is None:
        write_dir = constants.model_root

    file_path = write_dir / file_name

    # Download the model file
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Save the file to the specified path
    with open(file_path, 'wb') as f:
        f.write(response.content)

    logger.info(f"Successfully downloaded '{file_name}'.")

    return file_path
