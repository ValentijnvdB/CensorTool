from concurrent.futures import Future, wait, FIRST_COMPLETED
from queue import Queue
from threading import Event

import numpy as np
from loguru import logger

from core import ProcessedResult, ImageInput, Job, ImagePipeline

from ..config import CONFIG

from .utils import get_next_frame


def detect_loop(stop_event: Event, message_queue: Queue[ProcessedResult], input_device):

    prev_image_sum = 0

    with ImagePipeline(max_workers=CONFIG.n_workers) as pipeline:
        futures: dict[float, Future] = {}
        cancel_events: dict[float, Event] = {}
        def add_image(image_or_path: ImageInput, ts: float, image_sum: int) -> None:
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
                }
            )
            futures[ts] = pipeline.submit(job)
            cancel_events[ts] = c_event


        while not stop_event.is_set():

            timestamp, frame = get_next_frame(input_device)

            new_sum = np.sum(frame)

            # Submit the screenshot for censoring
            if new_sum != prev_image_sum:
                add_image(frame, timestamp, new_sum)

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


