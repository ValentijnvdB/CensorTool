from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Hashable

import cv2
import numpy as np
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

    def __init__(self, input_device: InputDevice, cond: Condition | None = None):
        """
        :param cond: notified when a frame (or error) arrives. Capture threads that share one can be waited for
            together, see wait_for_frames.
        """
        super().__init__(name='live-capture', daemon=True)
        self._device = input_device
        self._cond = cond if cond is not None else Condition()
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

    def ready(self) -> bool:
        """Whether take would return something now. Call with the condition held."""
        return self._frame is not None or self._error is not None

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)


def wait_for_frames(cond: Condition, captures: list[CaptureThread], timeout: float):
    """Wait until one of the capture threads (which share cond) has a frame or an error, at most timeout seconds."""
    with cond:
        if not any(c.ready() for c in captures):
            cond.wait(timeout)


SourceKey = Hashable


class Sources(ABC):
    """
    The input devices the live loop censors, by key. The set may change while the loop runs (overlay mode adds and
    removes windows); the loop starts and stops a capture thread for each device.
    """

    @abstractmethod
    def devices(self) -> dict[SourceKey, InputDevice]:
        """The current devices. Called from the loop's thread."""

    def priority(self) -> SourceKey | None:
        """The source that gets the most turns (e.g. the focused window), or None."""
        return None


class SingleSource(Sources):
    KEY = 0

    def __init__(self, device: InputDevice):
        self._devices = {self.KEY: device}

    def devices(self) -> dict[SourceKey, InputDevice]:
        return self._devices


class TurnScheduler:
    """
    Decides which source may submit the next frame, when several have one (stride scheduling). Every source gets
    turns in a fair rotation; the priority source gets PRIORITY_WEIGHT times as many turns. A source that had
    nothing to censor (an unchanged window) does not save up turns.
    """
    PRIORITY_WEIGHT = 3.0

    def __init__(self):
        self._pass: dict[SourceKey, float] = {}  # a source's next turn, in virtual time
        self._clock = 0.0  # virtual time of the last turn

    def order(self, keys: list[SourceKey], priority: SourceKey | None) -> list[SourceKey]:
        """The sources in the order they should try to submit a frame."""
        return sorted(keys, key=lambda k: (max(self._pass.get(k, self._clock), self._clock), k != priority))

    def took_turn(self, key: SourceKey, priority: SourceKey | None):
        start = max(self._pass.get(key, self._clock), self._clock)
        self._clock = start
        self._pass[key] = start + (1 / self.PRIORITY_WEIGHT if key == priority else 1.0)

    def forget(self, key: SourceKey):
        self._pass.pop(key, None)


def draw_cursor(frame: np.ndarray, region: Region) -> np.ndarray:
    """Draw the mouse cursor on (a copy of) frame, when the frames are a capture of the screen."""
    frame = frame.copy()
    if region is None:
        return frame

    import pyautogui  # only needed for screen captures, and it needs a display when imported

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


@dataclass
class _SourceState:
    """What the loop keeps per source."""
    capture: CaptureThread
    last_submitted: np.ndarray | None = None  # the last frame that was submitted, to skip unchanged frames
    force_submit: bool = True  # submit the next frame even if it did not change (after an error or config reload)
    shown: np.ndarray | None = None  # the last censored frame, without cursor
    shown_region: Region = None
    shown_ts: float = float('-inf')


def live_censor_loop(stop_event: Event, input_device: InputDevice | Sources, output_device: OutputDevice,
                     censor_config_path: Path = None):
    """
    The main live censor function. Used for 'live', 'webcam' and 'overlay' mode.

    Frames are captured in a background thread per input device and up to CONFIG.live.frames_in_flight frames are
    censored at the same time, so the fps is set by the slowest step instead of the sum of all steps.
    Finished frames are shown in capture order: a frame that finishes after a newer one (of the same source) has
    been shown is dropped. With several sources (overlay mode), they take turns (TurnScheduler); a source whose
    frame did not change lets the next one go.

    :param stop_event: Stops when stop_event is set.
    :param input_device: The input device, or the input devices (Sources).
    :param output_device: The output device. Frames are pushed with the key of their source.
    :param censor_config_path: The path to the censor config.
    """
    sources = input_device if isinstance(input_device, Sources) else SingleSource(input_device)

    was_read, censor_config, censor_config_hash = read_censor_config_from_file_w_hash(censor_config_path)
    if not was_read:
        raise RuntimeError("Failed to read censor config from file.")
    config_hash = config_file_hash()
    last_config_check = time.monotonic()

    cond = Condition()  # shared by the capture threads
    states: dict[SourceKey, _SourceState] = {}
    scheduler = TurnScheduler()
    pending: dict[tuple[SourceKey, float], tuple[cc.JobHandle, Region]] = {}  # (source, capture time) -> job
    censored_times: deque[float] = deque(maxlen=11)  # for the fps counter

    def sync_sources():
        devices = sources.devices()
        for key in [k for k in states if k not in devices]:
            states.pop(key).capture.stop()
            scheduler.forget(key)
        for key, device in devices.items():
            if key not in states:
                capture = CaptureThread(device, cond)
                capture.start()
                states[key] = _SourceState(capture)

    try:
        with make_image_pipeline(mode='p') as pipeline:
            while not stop_event.is_set():
                try:
                    # check user input
                    command = output_device.poll_command()
                    if command == 'quit':
                        stop_event.set()
                        break
                    force_reload = command == 'reload'

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
                            for state in states.values():
                                state.force_submit = True

                    sync_sources()
                    if not states:
                        stop_event.wait(0.05)  # nothing to censor (overlay mode without windows)
                        continue

                    # submit the newest frame of the source whose turn it is, if there is room. A source without
                    # a changed frame lets the next one go. Only wait for frames when no jobs are running.
                    to_push: set[SourceKey] = set()
                    if len(pending) < CONFIG.live.frames_in_flight:
                        priority = sources.priority()
                        submitted = False
                        for key in scheduler.order(list(states), priority):
                            state = states[key]
                            try:
                                item = state.capture.take(timeout=0)
                            except Exception as e:
                                logger.error(f"Capturing failed: {e}")
                                state.force_submit = True
                                continue
                            if item is None:
                                continue
                            timestamp, frame, region = item
                            if state.force_submit or not np.array_equal(frame, state.last_submitted):
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
                                pending[(key, timestamp)] = (pipeline.submit(job), region)
                                state.last_submitted = frame
                                state.force_submit = False
                                scheduler.took_turn(key, priority)
                                submitted = True
                                break
                            if region is not None and state.shown is not None \
                                    and not any(k == key for k, _ in pending):
                                # nothing changed on screen, still update the cursor
                                state.shown_region = region
                                to_push.add(key)
                        if not submitted and not to_push:
                            wait_for_frames(cond, [s.capture for s in states.values()],
                                            FRAME_POLL_INTERVAL if pending else 0.1)

                    # collect the finished jobs. When no new frame can be submitted, wait for one to finish.
                    if pending and len(pending) >= CONFIG.live.frames_in_flight:
                        cc.wait_any([handle for handle, _ in pending.values()])
                    finished = sorted((k for k, (handle, _) in pending.items() if handle.done()), key=lambda k: k[1])

                    new_results = 0
                    for key, ts in finished:
                        handle, region = pending.pop((key, ts))
                        job = handle.wait()
                        state = states.get(key)  # None: the source was removed meanwhile
                        if job is None or not job.success or job.result is None:
                            logger.error(f"Censoring a frame failed: {getattr(job, 'error', None)}")
                            if state is not None:
                                state.force_submit = True
                            continue
                        if state is None or ts < state.shown_ts:
                            continue  # a newer frame is already shown

                        state.shown = job.result["image"]
                        state.shown_region = region
                        state.shown_ts = ts
                        to_push.add(key)
                        new_results += 1

                    if new_results and CONFIG.live.show_fps:
                        now = time.monotonic()
                        censored_times.extend([now] * new_results)
                        fps = (len(censored_times) - 1) / (now - censored_times[0]) \
                            if len(censored_times) > 1 and now > censored_times[0] else 0
                        latency = now - max(states[k].shown_ts for k in to_push)
                        sources_text = f", sources: {len(states)}" if len(states) > 1 else ""
                        print(f"fps: {fps:.1f}, latency: {latency * 1000:.0f}ms, "
                              f"in flight: {len(pending)}{sources_text}")

                    for key in to_push:
                        state = states[key]
                        output_device.push_frame(draw_cursor(state.shown, state.shown_region),
                                                 timestamp=state.shown_ts, source=key)
                except Exception as e:
                    log_str = f"Error: {e}"
                    if CONFIG.debug:
                        log_str += f" {traceback.format_exc()}"
                    logger.error(log_str)
                    for state in states.values():
                        state.force_submit = True
    finally:
        for state in states.values():
            state.capture.stop()
