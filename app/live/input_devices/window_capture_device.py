import threading
import time

import cv2
import numpy as np

from .generic import InputDevice
from .. import windowing


class WindowCaptureDevice(InputDevice):
    """
    Captures one window, also while other windows (the overlay) cover it. Frames are captured in the thread
    that calls get_frame; each thread gets its own grabber (X11: its own connection).

    While the window is hidden (minimised, other workspace) or gone, get_frame keeps returning the last frame,
    so the live loop sees an unchanged screen.
    """

    HIDDEN_POLL_INTERVAL = 0.05
    REFRESH_INTERVAL = 0.1  # how often the window's position and size are checked
    MAX_FAILED_GRABS = 50

    def __init__(self, client: int, window_system: windowing.WindowSystem | None = None):
        self.target = client
        self._window_system = window_system or windowing.get()
        self._local = threading.local()
        self._grabbers: list[windowing.Grabber] = []
        self._grabbers_lock = threading.Lock()
        self._last: np.ndarray | None = None

    def _grabber(self) -> windowing.Grabber:
        """The grabber of the calling thread."""
        grabber = getattr(self._local, 'grabber', None)
        if grabber is None:
            grabber = self._window_system.grabber(self.target)
            self._local.grabber = grabber
            self._local.last_refresh = 0.0
            with self._grabbers_lock:
                self._grabbers.append(grabber)
        return grabber

    def get_frame(self) -> tuple[float, np.ndarray]:
        failed = 0
        while True:
            timestamp = time.monotonic()
            grabber = self._grabber()
            if timestamp - self._local.last_refresh >= self.REFRESH_INTERVAL or grabber.failures:
                grabber.refresh()
                self._local.last_refresh = timestamp
            frame = grabber.grab()
            if frame is not None:
                self._last = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return timestamp, self._last
            if not grabber.visible:
                time.sleep(self.HIDDEN_POLL_INTERVAL)
                if self._last is None:  # hidden from the start: nothing to show yet
                    width, height = grabber.area[2:] if grabber.area[2] > 0 else (64, 64)
                    self._last = np.zeros((height, width, 3), np.uint8)
                return time.monotonic(), self._last
            failed += 1
            if failed >= self.MAX_FAILED_GRABS:
                raise windowing.WindowSystemError(f"Capturing window {grabber.window:#x} keeps failing.")
            time.sleep(0.01)

    def close_thread(self):
        grabber = getattr(self._local, 'grabber', None)
        if grabber is not None:
            with self._grabbers_lock:
                if grabber in self._grabbers:
                    self._grabbers.remove(grabber)
            self._local.grabber = None
            grabber.close()

    def close(self):
        with self._grabbers_lock:
            grabbers, self._grabbers = self._grabbers, []
        for grabber in grabbers:
            try:
                grabber.close()
            except Exception:
                pass
