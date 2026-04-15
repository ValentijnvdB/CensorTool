import cv2
import numpy as np

import constants
from .generic import OutputDevice
from ...config import CONFIG


class WindowDevice(OutputDevice):

    def __init__(self, window_name: str):
        cv2.startWindowThread()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CONFIG.live.cap_width, CONFIG.live.cap_height)

        init_image = cv2.imread(constants.init_screen_image, cv2.IMREAD_GRAYSCALE)
        init_image = cv2.resize(init_image, (CONFIG.live.cap_width, CONFIG.live.cap_height))
        cv2.imshow(window_name, init_image)
        self.device = window_name

    def close(self):
        cv2.destroyAllWindows()

    def push_frame(self, frame: np.ndarray):
        cv2.imshow(self.device, frame)
