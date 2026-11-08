from __future__ import annotations

import time
import traceback
from collections import deque
from pathlib import Path
from threading import Condition, Event, Thread

import cv2
import numpy as np
import pyautogui
from loguru import logger

from core.backend import cc

from .input_devices.generic import InputDevice
from .output_devices.generic import OutputDevice
from ..config import (CONFIG, read_censor_config_from_file_w_hash, make_image_pipeline, config_file_hash,
                      reload_config_if_changed)

# How often (seconds) the config files are checked for changes. Pressing 'r' reloads them immediately.
CONFIG_CHECK_INTERVAL = 0.5

# How long (seconds) the loop waits for a new frame before checking the window, keys and jobs again.
FRAME_POLL_INTERVAL = 0.002

Region = tuple[int, int, int, int] | None


class CaptureThread(Thread):
    """
    Captures frames from the input device in the background, so capturing overlaps with censoring
    and displaying. Only the newest frame is kept; frames that are not taken in time are dropped.
    """

    def __init__(self, input_device: InputDevice):
        super().__init__(name='live-capture', daemon=True)
        self._device = input_device
        self._cond = Condition()
        self._frame: tuple[float, np.ndarray, Region] | None = None
        self._error: Exception | None = None
        self._stop_event = Event()

    def run(self):
        try:
            self._capture()
        finally:
            self._device.close_thread()

    def _capture(self):
        while not self._stop_event.is_set():
            try:
                timestamp, frame = self._device.get_frame()
                item = (timestamp, frame, self._device.screen_region())
            except Exception as e:
                with self._cond:
                    self._error = e
                    self._cond.notify_all()
                self._stop_event.wait(0.1)  # don't spin on a failing device
                continue

            with self._cond:
                self._frame = item
                self._cond.notify_all()

    def take(self, timeout: float) -> tuple[float, np.ndarray, Region] | None:
        """
        Take the newest frame, waiting at most timeout seconds for one.

        :return: (timestamp, frame, screen region), or None if no new frame arrived.
        :raises: the error of the input device, if capturing failed.
        """
        with self._cond:
            if self._frame is None and self._error is None:
                self._cond.wait(timeout)
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            item, self._frame = self._frame, None
            return item

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)


def draw_cursor(frame: np.ndarray, region: Region) -> np.ndarray:
    """Draw the mouse cursor on (a copy of) frame, when the frames are a capture of the screen."""
    frame = frame.copy()
    if region is None:
        return frame

    left, top, width, height = region
    cx, cy = pyautogui.position()
    cx -= left
    cy -= top
    if 5 < cx < width and 5 < cy < height:
        color = tuple(reversed(CONFIG.live.cursor_color))
        frame[cy - 5:cy + 5, cx - 5:cx + 5] = color
        if CONFIG.debug:
            frame = cv2.putText(frame, f'({cx}, {cy})', (max(cx - 10, 0), max(cy - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def live_censor_loop(stop_event: Event, input_device: InputDevice, output_device: OutputDevice, censor_config_path: Path = None):
    """
    The main live censor function. Used for both 'live' and 'webcam' mode.

    Frames are captured in a background thread and up to CONFIG.live.frames_in_flight frames are censored at
    the same time, so the fps is set by the slowest step instead of the sum of all steps.
    Finished frames are shown in capture order: a frame that finishes after a newer one has been shown is dropped.

    :param stop_event: Stops when stop_event is set.
    :param input_device: The input device.
    :param output_device: The output device.
    :param censor_config_path: The path to the censor config.
    """
    was_read, censor_config, censor_config_hash = read_censor_config_from_file_w_hash(censor_config_path)
    if not was_read:
        raise RuntimeError("Failed to read censor config from file.")
    config_hash = config_file_hash()
    last_config_check = time.monotonic()

    pending: dict[float, tuple[cc.JobHandle, Region]] = {}  # capture timestamp -> (job, screen region)
    last_submitted: np.ndarray | None = None  # the last frame that was submitted, to skip unchanged frames
    force_submit = True  # submit the next frame even if it did not change (after an error or config reload)

    shown: np.ndarray | None = None  # the last censored frame, without cursor
    shown_region: Region = None
    shown_ts = float('-inf')

    censored_times: deque[float] = deque(maxlen=11)  # for the fps counter

    capture = CaptureThread(input_device)
    capture.start()
    try:
        with make_image_pipeline(mode='p') as pipeline:
            while not stop_event.is_set():
                try:
                    # check user input
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        stop_event.set()
                        break
                    force_reload = key == ord("r")

                    # reload config.yml and the censor config if they changed. The censor config is also reread
                    # when config.yml changed, because it takes the comfy settings from it.
                    now = time.monotonic()
                    if force_reload or now - last_config_check >= CONFIG_CHECK_INTERVAL:
                        last_config_check = now
                        config_changed, config_hash = reload_config_if_changed(None if force_reload else config_hash)
                        was_read, new_config, censor_config_hash = read_censor_config_from_file_w_hash(
                            path=censor_config_path,
                            prev_hash=None if force_reload or config_changed else censor_config_hash)
                        if was_read:
                            censor_config = new_config
                        if was_read or config_changed:
                            force_submit = True

                    # submit the newest frame if there is room. Only wait for one when no jobs are running.
                    push_unchanged = False
                    if len(pending) < CONFIG.live.frames_in_flight:
                        item = capture.take(timeout=FRAME_POLL_INTERVAL if pending else 0.1)
                        if item is not None:
                            timestamp, frame, region = item
                            if force_submit or not np.array_equal(frame, last_submitted):
                                job = cc.Job(
                                    image=frame,
                                    timestamp=timestamp,
                                    size=CONFIG.live.input_size,
                                    output_path=None,
                                    early_exit=False,
                                    override_cache=False,
                                    use_cache=False,
                                    config=censor_config,
                                )
                                pending[timestamp] = (pipeline.submit(job), region)
                                last_submitted = frame
                                force_submit = False
                            elif not pending and shown is not None:
                                # nothing changed on screen, still update the cursor
                                push_unchanged = True
                                shown_region = region

                    # collect the finished jobs. When no new frame can be submitted, wait for one to finish.
                    if pending and len(pending) >= CONFIG.live.frames_in_flight:
                        cc.wait_any([handle for handle, _ in pending.values()])
                    finished = sorted(ts for ts, (handle, _) in pending.items() if handle.done())

                    new_result = False
                    for ts in finished:
                        handle, region = pending.pop(ts)
                        job = handle.wait()
                        if job is None or not job.success or job.result is None:
                            logger.error(f"Censoring a frame failed: {getattr(job, 'error', None)}")
                            force_submit = True
                            continue
                        if ts < shown_ts:
                            continue  # a newer frame is already shown

                        shown = job.result["image"]
                        shown_region = region
                        shown_ts = ts
                        new_result = True

                    if new_result:
                        if CONFIG.live.show_fps:
                            now = time.monotonic()
                            censored_times.append(now)
                            fps = (len(censored_times) - 1) / (now - censored_times[0]) if len(censored_times) > 1 else 0
                            print(f"fps: {fps:.1f}, latency: {(now - shown_ts) * 1000:.0f}ms, "
                                  f"in flight: {len(pending)}")
                    elif not push_unchanged:
                        continue

                    output_device.push_frame(draw_cursor(shown, shown_region))
                except Exception as e:
                    log_str = f"Error: {e}"
                    if CONFIG.debug:
                        log_str += f" {traceback.format_exc()}"
                    logger.error(log_str)
                    force_submit = True
    finally:
        capture.stop()
