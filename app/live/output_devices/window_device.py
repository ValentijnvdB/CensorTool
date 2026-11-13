import cv2
import numpy as np

import constants
from .generic import Command, OutputDevice
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
        self._frame_size = (CONFIG.live.cap_width, CONFIG.live.cap_height)

    def close(self):
        cv2.destroyAllWindows()

    def poll_command(self) -> Command | None:
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return 'quit'
        if key == ord('r'):
            return 'reload'
        return None

    def push_frame(self, frame: np.ndarray, timestamp: float | None = None):
        # follow the frame size (capture region changed, or a webcam with another resolution)
        size = (frame.shape[1], frame.shape[0])
        if size != self._frame_size:
            cv2.resizeWindow(self.device, *size)
            self._frame_size = size
        cv2.imshow(self.device, frame)
