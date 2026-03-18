import time

import cv2
import numpy as np
import pyautogui

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


def get_next_frame(vid_cap: cv2.VideoCapture = None) -> tuple[float, np.ndarray]:
    if vid_cap is None:
        return get_screenshot()

    if not vid_cap.isOpened():
        raise RuntimeError("Webcam stream is closed!")

    ret, frame = vid_cap.read()
    if not ret:
        raise RuntimeError('Failed to get next image from video capture')

    return time.monotonic(), frame


def push_frame(device, frame: np.ndarray):

    if isinstance(device, str):
        cv2.imshow(device, frame)
    else:
        device.send(frame)
        device.sleep_until_next_frame()


def get_screenshot():
    screenshot = pyautogui.screenshot(region=(CONFIG.live.cap_left,
                                              CONFIG.live.cap_top,
                                              CONFIG.live.cap_width,
                                              CONFIG.live.cap_height))

    # Convert to numpy array
    screenshot_np = np.array(screenshot)

    # If the image has an alpha channel (RGBA), remove it (keep RGB/BGR)
    if screenshot_np.shape[2] == 4:  # RGBA format (with alpha channel)
        screenshot_np = screenshot_np[:, :, :3]  # Remove alpha, keep RGB

    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    # Ensure the image is uint8 (8-bit)
    screenshot_cv = screenshot_cv.astype(np.uint8)

    # return time, screenshot
    return time.monotonic(), screenshot_cv
