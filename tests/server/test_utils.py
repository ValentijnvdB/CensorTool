"""
Unit tests for server/utils.py

Run with:
    pytest tests/server/test_utils.py -v
"""

import cv2
import numpy as np
import pytest

from server.utils import bytes_to_np


def encode(image: np.ndarray, ext: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return buffer.tobytes()


@pytest.mark.parametrize("image", [
    pytest.param(np.full((6, 8, 3), 200, np.uint8), id="bgr"),
    pytest.param(np.full((6, 8), 200, np.uint8), id="grayscale"),
    pytest.param(np.full((6, 8, 4), 200, np.uint8), id="bgra"),
    pytest.param(np.full((6, 8, 3), 50000, np.uint16), id="16-bit"),
])
def test_bytes_to_np_gives_8bit_bgr(image):
    # the censor backends only take 3-channel 8-bit images
    decoded = bytes_to_np(encode(image))
    assert decoded.shape == (6, 8, 3)
    assert decoded.dtype == np.uint8


def test_bytes_to_np_drops_alpha_keeping_colour():
    image = np.zeros((2, 2, 4), np.uint8)
    image[..., :3] = (10, 20, 30)
    image[..., 3] = 0  # fully transparent
    assert bytes_to_np(encode(image))[0, 0].tolist() == [10, 20, 30]


def test_bytes_to_np_not_an_image():
    assert bytes_to_np(b"<html></html>") is None
