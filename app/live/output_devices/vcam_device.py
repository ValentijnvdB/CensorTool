import cv2
import numpy as np
from loguru import logger
from pyvirtualcam import Camera, PixelFormat

from .generic import OutputDevice


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to fit in width x height, keeping its aspect ratio, and pad the rest with black."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame

    scale = min(width / w, height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


class VcamDevice(OutputDevice):

    def __init__(self, w, h, fps):
        self.device = Camera(width=w, height=h, fps=fps, fmt=PixelFormat.BGR)
        logger.info(f"Outputting to {self.device.device}")

    def close(self):
        self.device.close()

    def push_frame(self, frame: np.ndarray, timestamp: float | None = None, source=None):
        # the virtual camera only accepts frames of exactly its own size
        self.device.send(letterbox(frame, self.device.width, self.device.height))
        self.device.sleep_until_next_frame()
