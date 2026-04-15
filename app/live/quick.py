import time
import traceback
from collections import deque
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pyautogui
from loguru import logger

import py_censor_core as cc

from input_devices.generic import InputDevice
from output_devices.generic import OutputDevice
from ..config import CONFIG, read_censor_config_from_file_w_hash


def live_censor_loop(stop_event: Event, input_device: InputDevice, output_device: OutputDevice, censor_config_path: Path = None):
    """
    The main live censor function. Used for both 'live' and 'webcam' mode.

    :param stop_event: Stops when stop_event is set.
    :param input_device: The input device.
    :param output_device: The output device.
    :param censor_config_path: The path to the censor config.
    """
    prev_image_sum = 0
    previous_censored_screenshots: np.ndarray = None

    was_read, censor_config, prev_hash = read_censor_config_from_file_w_hash(censor_config_path)
    if not was_read:
        raise RuntimeError("Failed to read censor config from file.")

    with cc.ImagePipeline(max_workers=CONFIG.n_workers) as pipeline:
        futures: list[cc.JobHandle] = []
        cancel_events: dict[float, Event] = {}
        def add_image(image: np.ndarray, ts: float) -> None:
            """Drop-in callback for your external API."""
            c_event = Event()
            job = cc.Job(
                image=image,
                timestamp=ts,
                output_path=None,
                early_exit=False,
                override_cache=False,
                use_cache=False,
                config=censor_config,
                size=1280
            )
            futures.append(pipeline.submit(job))
            cancel_events[ts] = c_event

        start_times: deque[float] = deque()
        checkpoints: deque[float] = deque()

        frames_put_out = 0
        force_reload = False
        reloaded_config = False
        errored = False
        while not stop_event.is_set():

            checkpoints.clear()
            checkpoints.append(time.time()) # S0

            try:
                # open window and check user input
                key = cv2.waitKey(1)
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    stop_event.set()
                    break
                elif key == ord("r"):
                    force_reload = True

                start_times.append(time.time())
                checkpoints.append(time.time())  # S1

                # reload cache
                was_read, new_config, prev_hash = read_censor_config_from_file_w_hash(
                    path=censor_config_path,
                    prev_hash=None if force_reload else prev_hash)
                if was_read:
                    assert new_config is not None
                    censor_config = new_config
                    force_reload = False
                    reloaded_config = True

                timestamp, screenshot = input_device.get_frame()

                # compute sum to compare against previous
                new_sum = np.sum(screenshot)

                force_update = errored or reloaded_config
                if force_update or new_sum != prev_image_sum:
                    checkpoints.append(time.time())  # S2
                    # Submit the screenshot for censoring
                    add_image(screenshot, timestamp)

                    # Block until at least one analyzes job is done
                    done = cc.wait_any(futures)
                    completed = done.wait()
                    futures.remove(done)

                    if completed is None or not completed.success:
                        continue

                    result = completed.result
                    if result is None:
                        continue
                    frame = result["image"]
                    previous_censored_screenshots = frame.copy()
                    prev_image_sum = new_sum

                    checkpoints.append(time.time())  # S3

                else:
                    # if nothing changed on screen, we still want to update in case of mouse movements.
                    frame = previous_censored_screenshots.copy()

                checkpoints.append(time.time())  # S4

                # cursor
                cx, cy = pyautogui.position()
                cx = cx - CONFIG.live.cap_left
                cy = cy - CONFIG.live.cap_top

                if 5 < cx < CONFIG.live.cap_width and 5 < cy < CONFIG.live.cap_height:
                    color = tuple(reversed(CONFIG.live.cursor_color))
                    frame[cy - 5:cy + 5, cx - 5:cx + 5] = color
                    if CONFIG.debug:
                        frame = cv2.putText(frame, f'({cx}, {cy})', (max(cx - 10, 0), max(cy - 10, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                checkpoints.append(time.time())  # S5

                # fps
                if CONFIG.live.show_fps:
                    while len(start_times) > 10:
                        start_times.popleft()
                    frames_put_out += 1
                    try:
                        took = time.time() - start_times[0]
                        fps = 10 / took
                    except ZeroDivisionError:
                        fps = 0

                    print(f"took: {(checkpoints[-1]-checkpoints[0])*1000}ms, fps: {fps}")

                output_device.push_frame(frame)
                errored = False
            except Exception as e:
                log_str = f"Error: {e}"
                if CONFIG.debug:
                    log_str += f" {traceback.format_exc()}"
                logger.error(log_str)
                errored = True