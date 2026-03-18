import hashlib
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait, Future
from threading import Event
from typing import Any

import cv2
import numpy as np
import pyautogui
from loguru import logger

from core import CensorConfig, ImagePipeline, Job, ImageInput, ProcessedResult, draw

from app.config import CONFIG

from .utils import get_next_frame, push_frame


def quick_live_censor(stop_event: Event, reload_config, output_device: Any, vid_cap: cv2.VideoCapture|None):
    """
    The main live censor function. Used for both 'live' and 'webcam' mode.

    :param stop_event: Stops when stop_event is set.
    :param reload_config: the function that reloads the censor config
    :param output_device: the output device. if str, will output to cv2 window, otherwise use pyvirtualcam.Camera.
    :param vid_cap: the video capture device. if None, will take screenshots.
    """
    prev_image_sum = 0
    previous_censored_screenshots: np.ndarray = None

    censor_config, file_hash, _ = reload_config(None, '', force=True)

    with ImagePipeline(max_workers=4) as pipeline:
        futures: dict[float, Future] = {}
        cancel_events: dict[float, Event] = {}
        def add_image(image_or_path: ImageInput, ts: float, image_sum: int, cc: CensorConfig) -> None:
            """Drop-in callback for your external API."""
            c_event = Event()
            job = Job(
                image=image_or_path,
                timestamp=ts,
                output_path=None,
                early_exit=False,
                override_cache=False,
                skip_cache_write=True,
                sizes=CONFIG.picture_sizes,
                cancelled=c_event,
                data={
                    'sum': image_sum
                },
                config=cc
            )
            futures[ts] = pipeline.submit(job)
            cancel_events[ts] = c_event

        start_times: deque[float] = deque()
        frames_put_out = 0
        force_reload = False
        errored = False
        while not stop_event.is_set():

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

                # reload box_config if it has changed
                censor_config, file_hash, reloaded_config = reload_config(censor_config, file_hash, force=force_reload)
                force_reload = False

                timestamp, screenshot = get_next_frame(vid_cap)

                # compute sum to compare against previous
                new_sum = np.sum(screenshot)

                force_update = errored or reloaded_config
                if force_update or new_sum != prev_image_sum:
                    # Submit the screenshot for censoring
                    add_image(screenshot, timestamp, new_sum, censor_config)

                    # Block until at least one analyzes job is done
                    done, _ = wait(futures.values(), return_when=FIRST_COMPLETED)

                    # look for the finished job with the lowest frame_number
                    for job_id, future in futures.items():
                        if future in done:
                            completed: Job = future.result()
                            del futures[job_id]
                            del cancel_events[job_id]
                            break

                    if completed is None or not completed.success:
                        continue

                    result = completed.result
                    assert isinstance(result, ProcessedResult)
                    result_timestamp = completed.timestamp
                    prev_image_hash = completed.data.get('hash')
                    prev_image_sum = completed.data.get('sum')
                    frame = result.image
                    previous_censored_screenshots = frame.copy()

                    # cancel all jobs for screenshots taken before the completed
                    for t in cancel_events:
                        if result_timestamp < t:
                            break
                        logger.debug(f'Cancelling job {t}')
                        cancel_events[t].set()
                        del cancel_events[t]
                        del futures[t]
                else:
                    # if nothing changed on screen, we still want to update in case of mouse movements.
                    frame = previous_censored_screenshots.copy()


                cx, cy = pyautogui.position()
                cx = cx - CONFIG.live.cap_left
                cy = cy - CONFIG.live.cap_top

                if 5 < cx < CONFIG.live.cap_width and 5 < cy < CONFIG.live.cap_height:
                    color = tuple(reversed(CONFIG.live.cursor_color))
                    frame[cy - 5:cy + 5, cx - 5:cx + 5] = color
                    if CONFIG.debug:
                        frame = cv2.putText(frame, f'({cx}, {cy})', (max(cx - 10, 0), max(cy - 10, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                while len(start_times) > 10:
                    start_times.popleft()
                frames_put_out += 1

                if CONFIG.live.show_fps:
                    try:
                        fps = 10 / (time.time() - start_times[0])
                    except ZeroDivisionError:
                        fps = 0
                    draw.annotate_image(frame, f"frames: {frames_put_out}", 0)
                    draw.annotate_image(frame, f"FPS: {fps}", 2)


                push_frame(output_device, frame)
                errored = False
            except Exception as e:
                log_str = f"Error: {e}"
                if CONFIG.debug:
                    log_str += f" {traceback.format_exc()}"
                logger.error(log_str)
                errored = True