import threading
import time

import cv2
import numpy as np

from .generic import InputDevice
from .. import x11


class WindowCaptureDevice(InputDevice):
    """
    Captures one X11 window, also while other windows (the overlay) cover it. The target can be changed while
    capturing (set_target). Frames are captured in the thread that calls get_frame; each thread gets its own
    X connection.

    While the window is hidden (minimised, other workspace) or gone, get_frame keeps returning the last frame,
    so the live loop sees an unchanged screen.
    """

    HIDDEN_POLL_INTERVAL = 0.05
    REFRESH_INTERVAL = 0.1  # how often the window's position and size are checked

    def __init__(self, client: int):
        self._target = client
        self._generation = 0
        self._target_lock = threading.Lock()
        self._local = threading.local()
        self._grabbers: list[x11.WindowGrabber] = []
        self._grabbers_lock = threading.Lock()
        self._last: np.ndarray | None = None

    @property
    def target(self) -> int:
        return self._target

    def set_target(self, client: int):
        """
        Capture another window from now on. Frames whose timestamp is at or after the moment this returns are
        captures of the new window.
        """
        with self._target_lock:
            self._target = client
            self._generation += 1

    def _grabber(self) -> tuple[x11.WindowGrabber, int]:
        """The grabber of the calling thread for the current target, and the generation it belongs to."""
        with self._target_lock:
            target, generation = self._target, self._generation
        grabber = getattr(self._local, 'grabber', None)
        if grabber is not None and getattr(self._local, 'generation', None) != generation:
            self._close_grabber(grabber)
            grabber = None
        if grabber is None:
            grabber = x11.WindowGrabber(target)
            self._local.grabber, self._local.generation = grabber, generation
            self._local.last_refresh = 0.0
            with self._grabbers_lock:
                self._grabbers.append(grabber)
        return grabber, generation

    def _close_grabber(self, grabber: x11.WindowGrabber):
        with self._grabbers_lock:
            if grabber in self._grabbers:
                self._grabbers.remove(grabber)
        self._local.grabber = None
        grabber.close()

    MAX_FAILED_GRABS = 50

    def get_frame(self) -> tuple[float, np.ndarray]:
        failed = 0
        while True:
            # the timestamp is taken before the target is read, see set_target
            timestamp = time.monotonic()
            grabber, generation = self._grabber()
            if timestamp - self._local.last_refresh >= self.REFRESH_INTERVAL or grabber.failures:
                grabber.refresh()
                self._local.last_refresh = timestamp
            frame = grabber.grab()
            with self._target_lock:
                if generation != self._generation:
                    continue  # the target changed while capturing
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
                raise x11.X11Error(f"Capturing window {grabber.client:#x} keeps failing.")
            time.sleep(0.01)

    def close_thread(self):
        grabber = getattr(self._local, 'grabber', None)
        if grabber is not None:
            self._close_grabber(grabber)

    def close(self):
        with self._grabbers_lock:
            grabbers, self._grabbers = self._grabbers, []
        for grabber in grabbers:
            try:
                grabber.close()
            except Exception:
                pass
