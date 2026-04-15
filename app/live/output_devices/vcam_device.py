import numpy as np
from loguru import logger
from pyvirtualcam import Camera, PixelFormat

from .generic import OutputDevice


class VcamDevice(OutputDevice):

    def __init__(self, w, h, fps):
        self.device = Camera(width=w, height=h, fps=fps, fmt=PixelFormat.BGR)
        logger.info(f"Outputting to {self.device.device}")

    def close(self):
        self.device.close()

    def push_frame(self, frame: np.ndarray):
        self.device.send(frame)
        self.device.sleep_until_next_frame()