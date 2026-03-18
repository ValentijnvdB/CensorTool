import cv2
import numpy as np

from app.config import CONFIG


def shm_name_for_screenshot(size):
    return f'raw_grab_{size}'


def interpolate_images(img1, ts1, img2, ts2, timestamp):
    assert (ts1 < ts2)
    if timestamp < ts1:
        return img1
    if ts2 < timestamp:
        return img2

    pct2 = (timestamp - ts1) / (ts2 - ts1)
    pct1 = 1 - pct2

    return cv2.addWeighted(img1, pct1, img2, pct2, 0)


def vision_adj_img_size(max_length):
    if max_length != 0:
        return max_length, max_length
    else:
        return CONFIG.live.cap_height, CONFIG.live.cap_width



