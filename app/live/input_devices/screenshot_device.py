import time

import mss
import numpy as np

from .generic import InputDevice
from ...config import CONFIG


class ScreenshotDevice(InputDevice):

    def __init__(self):
        self.device = mss.mss()
        self.area = {
            "top": CONFIG.live.cap_top,
            "left": CONFIG.live.cap_left,
            "width": CONFIG.live.cap_width,
            "height": CONFIG.live.cap_height
        }

    def close(self):
        self.device.close()

    def get_frame(self) -> tuple[float, np.ndarray]:
        screenshot = np.array(self.device.grab(self.area), dtype=np.uint8)  # BGRA
        return time.monotonic(), screenshot[:, :, :3].copy()
