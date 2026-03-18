import hashlib
import traceback
from concurrent.futures import FIRST_COMPLETED, wait, Future
from threading import Event

import cv2
import numpy as np
import pyautogui
from loguru import logger

from core import CensorConfig, ImagePipeline, Job, ImageInput, ProcessedResult

from app.config import CONFIG

from .screenshot import get_next_image


def quick_live_censor(stop_event: Event, reload_config, window_name: str, device_id: int):
    prev_image_sum = 0
    prev_image_hash = 0
    previous_censored_screenshots: np.ndarray = None

    censor_config, file_hash, _ = reload_config(None, '', force=True)

    vid_cap = None
    if device_id >= 0:
        vid_cap = cv2.VideoCapture(device_id)

    with ImagePipeline(max_workers=4) as pipeline:
        futures: dict[float, Future] = {}
        cancel_events: dict[float, Event] = {}
        def add_image(image_or_path: ImageInput, ts: float, image_sum: int, image_hash: bytes|int, cc: CensorConfig) -> None:
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
                    'sum': image_sum,
                    'hash': image_hash
                },
                config=cc
            )
            futures[ts] = pipeline.submit(job)
            cancel_events[ts] = c_event


        i = 0
        force_reload = False
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

                # reload box_config if it has changed
                censor_config, file_hash, reloaded_config = reload_config(censor_config, file_hash, force=force_reload)
                force_reload = False

                timestamp, screenshot = get_next_image(vid_cap)

                ### we don't want to censor again if image is unchanged
                ### hashing at size 1280 takes 30ms, which is not nothing
                ### summing takes 10ms, which is a lot less overhead.
                ### so start with a very fast check (just sum the image)
                ### if the sum is unchanged, proceed to hash
                ### this means we will Detect the same image twice in a
                ### row, but not more than twice
                new_sum = np.sum(screenshot)

                if new_sum == prev_image_sum:
                    new_hash = hashlib.md5(screenshot.tobytes()).digest()
                else:
                    new_hash = 0

                force_update = errored or reloaded_config
                if force_update or new_sum != prev_image_sum or new_hash != prev_image_hash:
                    # Submit the screenshot for censoring
                    add_image(screenshot, timestamp, new_sum, new_hash, censor_config)

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


                cv2.imshow(window_name, frame)
                i += 1
                errored = False
            except Exception as e:
                log_str = f"Error: {e}"
                if CONFIG.debug:
                    log_str += f" {traceback.format_exc()}"
                logger.error(log_str)
                errored = True