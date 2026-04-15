import time

import cv2
import numpy as np

from .generic import InputDevice


class CameraDevice(InputDevice):

    def __init__(self, device_id):
        self.device = cv2.VideoCapture(device_id)

    def close(self):
        self.device.release()

    def get_frame(self) -> tuple[float, np.ndarray]:
        if not self.device.isOpened():
            raise RuntimeError("Webcam stream is closed!")

        ret, frame = self.device.read()
        if not ret:
            raise RuntimeError('Failed to get next image from video capture')

        return time.monotonic(), frame.copy()