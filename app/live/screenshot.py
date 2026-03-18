import time

import cv2
import numpy as np
import pyautogui

from core import models, draw

from app.config import CONFIG





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


def get_adjusted_screenshot() -> tuple[list[float], list[np.ndarray], list[float]]:
    screenshot_time, screenshot_image = get_screenshot()

    adjusted_images = []

    scales = []
    for size in CONFIG.picture_sizes:
        adj_image, scale = models.prep_img_for_nudenet(screenshot_image, size)
        adjusted_images.append( adj_image )
        scales.append( scale )

    if CONFIG.debug:
        cv2.imwrite('debug-vision-raw-screenshot.png', screenshot_image)
        for i, size in enumerate(CONFIG.picture_sizes):
            cv2.imwrite(f'debug-vision-adj-screenshot-{i}-{size}.png',
                        draw.annotate_image_shape(adjusted_images[i]))

    return [screenshot_time], adjusted_images, scales


