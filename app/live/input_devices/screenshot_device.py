import time

import mss
import numpy as np
from loguru import logger

from .generic import InputDevice
from ...config import CONFIG


class ScreenshotDevice(InputDevice):

    # maximum number of grabs per frame when looking for two that agree (see get_frame)
    MAX_GRABS = 3

    def __init__(self):
        self.device = mss.mss()
        self._prev_grab: np.ndarray | None = None
        self.area: dict[str, int] | None = None
        self._rejected_area: dict[str, int] | None = None
        self._update_area()
        if self.area is None:
            screen = self.device.monitors[0]
            self.device.close()
            raise ValueError(f"The live capture region {self._rejected_area} is not within the screen {screen}.")

    def close(self):
        self.device.close()

    def _update_area(self):
        """
        Use the capture region of the config (it can change while running).
        A region that is (partly) off the screen is rejected and the previous region is kept.
        """
        live = CONFIG.live
        area = {"top": live.cap_top, "left": live.cap_left, "width": live.cap_width, "height": live.cap_height}
        if area == self.area or area == self._rejected_area:
            return

        screen = self.device.monitors[0]  # bounding box of all monitors
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
        return np.array(self.device.grab(self.area), dtype=np.uint8)  # BGRA

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
        return time.monotonic(), frame[:, :, :3].copy()

    def screen_region(self) -> tuple[int, int, int, int]:
        return self.area["left"], self.area["top"], self.area["width"], self.area["height"]
