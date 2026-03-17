import hashlib
from concurrent.futures import Future, wait, FIRST_COMPLETED
from queue import Queue
from threading import Event

import numpy as np
from loguru import logger

from core import ProcessedResult, ImageInput, Job, ImagePipeline

from app.config import CONFIG

from .screenshot import get_screenshot


def detect_loop(stop_event: Event, message_queue: Queue[ProcessedResult]):

    prev_image_sum = 0
    prev_image_hash = 0

    with ImagePipeline(max_workers=4) as pipeline:
        futures: dict[float, Future] = {}
        cancel_events: dict[float, Event] = {}
        def add_image(image_or_path: ImageInput, ts: float, image_sum: int, image_hash: bytes|int) -> None:
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
                }
            )
            futures[ts] = pipeline.submit(job)
            cancel_events[ts] = c_event


        while not stop_event.is_set():

            timestamp, screenshot = get_screenshot()

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

            # Submit the screenshot for censoring
            if new_sum != prev_image_sum or new_hash != prev_image_hash:
                add_image(screenshot, timestamp, new_sum, new_hash)

                # Block until at least one analyzes job is done
                done, _ = wait(futures.values(), return_when=FIRST_COMPLETED)

                # look for the finished job with the lowest frame_number
                for job_id, future in futures.items():
                    if future in done:
                        completed: Job = future.result()
                        del futures[job_id]
                        del cancel_events[job_id]
                        break

                if completed is None:
                    continue

                result_timestamp = completed.timestamp
                prev_image_hash = completed.data.get('hash')
                prev_image_sum = completed.data.get('sum')
                message_queue.put(completed.result)

                # cancel all jobs for screenshots taken before the completed
                for t in cancel_events:
                    if result_timestamp < t:
                        break
                    logger.info(f'Cancelling job {t}')
                    cancel_events[t].set()
                    del cancel_events[t]
                    del futures[t]


