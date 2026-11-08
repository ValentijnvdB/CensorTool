import threading
import time

import cv2
import mss
import numpy as np
from loguru import logger

from .generic import InputDevice
from ...config import CONFIG


class ScreenshotDevice(InputDevice):

    # maximum number of grabs per frame when looking for two that agree (see get_frame)
    MAX_GRABS = 3

    def __init__(self):
        # mss instances must be used in the thread that created them (Windows), and frames are captured in a
        # background thread in live mode, so each thread gets its own instance (see _sct).
        self._local = threading.local()
        self._instances: list = []
        self._instances_lock = threading.Lock()
        self._prev_grab: np.ndarray | None = None
        self.area: dict[str, int] | None = None
        self._rejected_area: dict[str, int] | None = None
        self._update_area()
        if self.area is None:
            screen = self._sct().monitors[0]
            self.close()
            raise ValueError(f"The live capture region {self._rejected_area} is not within the screen {screen}.")

    def close(self):
        with self._instances_lock:
            instances, self._instances = self._instances, []
        for sct in instances:
            try:
                sct.close()
            except Exception as e:
                logger.debug(f"Closing a screen capture instance failed: {e}")

    def close_thread(self):
        sct = getattr(self._local, 'sct', None)
        if sct is None:
            return
        self._local.sct = None
        with self._instances_lock:
            if sct in self._instances:
                self._instances.remove(sct)
        sct.close()

    def _sct(self):
        """The mss instance of the calling thread."""
        sct = getattr(self._local, 'sct', None)
        if sct is None:
            sct = mss.mss()
            self._local.sct = sct
            with self._instances_lock:
                self._instances.append(sct)
        return sct

    def _update_area(self):
        """
        Use the capture region of the config (it can change while running).
        A region that is (partly) off the screen is rejected and the previous region is kept.
        """
        live = CONFIG.live
        area = {"top": live.cap_top, "left": live.cap_left, "width": live.cap_width, "height": live.cap_height}
        if area == self.area or area == self._rejected_area:
            return

        screen = self._sct().monitors[0]  # bounding box of all monitors
        on_screen = (area["left"] >= screen["left"] and area["top"] >= screen["top"]
                     and area["left"] + area["width"] <= screen["left"] + screen["width"]
                     and area["top"] + area["height"] <= screen["top"] + screen["height"])
        if not on_screen:
            self._rejected_area = area
            if self.area is not None:
                logger.error(f"Live capture region {area} is not within the screen {screen}, "
                             f"keeping the previous region {self.area}.")
            return

        if self.area is not None:
            logger.info(f"Live capture region changed to {area}.")
        self.area = area
        self._rejected_area = None

    def _grab(self) -> np.ndarray:
        grab = self._sct().grab(self.area)
        # a view on the grabbed buffer, much faster than np.array(grab)
        return np.frombuffer(grab.raw, np.uint8).reshape(grab.height, grab.width, 4)  # BGRA

    def get_frame(self) -> tuple[float, np.ndarray]:
        """
        Capture the screen region.

        Sometimes a grab can land mid-repaint (happened on X11 w a compositor), so the frame contains a band of
        half-drawn screen (e.g. desktop background over a window). Such a torn grab is never
        repeated by the next grab, so a frame is only accepted when it equals the previous
        frame (static screen, no extra cost) or when two consecutive grabs agree.
        When the screen keeps changing (e.g. a video) the last grab is used after MAX_GRABS.

        This approach majorly reduces the number of bands, with almost no performance penalty.
        """
        self._update_area()
        frame = self._grab()
        for _ in range(self.MAX_GRABS - 1):
            if np.array_equal(frame, self._prev_grab):
                break
            confirm = self._grab()
            if np.array_equal(frame, confirm):
                break
            frame = confirm
        self._prev_grab = frame
        return time.monotonic(), cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def screen_region(self) -> tuple[int, int, int, int]:
        return self.area["left"], self.area["top"], self.area["width"], self.area["height"]
