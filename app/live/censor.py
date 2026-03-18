import time
from collections import deque
from queue import Queue, Empty
from threading import Event

import cv2
import numpy as np
import pyautogui
from loguru import logger

from core import models, draw, process_multiple_passes, censor_image_from_boxes, Box

from app.config import CONFIG

from core import ProcessedResult

from . import utils
from .utils import get_next_frame, push_frame


def censor_loop(stop_event: Event, message_queue: Queue[ProcessedResult], reload_config, input_device: cv2.VideoCapture|None, output_device):
    scale_array = []
    for size in CONFIG.picture_sizes:
        scale_array.append(
            models.get_resize_scale(img_w=CONFIG.live.cap_width,
                                    img_h=CONFIG.live.cap_height,
                                    max_length=size)
        )

    ### set up for censoring
    live_boxes: deque[Box] = deque()
    boxes_buffer: deque[Box] = deque()
    image_buffer: deque[tuple[float, np.ndarray]] = deque()

    censor_config, file_hash, _ = reload_config(None, '', force=True)
    force = False

    frames_put_out = 0
    start_times: deque[float] = deque()
    while not stop_event.is_set():
        try:
            # open window and check user input
            pressed_key = cv2.waitKey(1)
            if pressed_key == ord("q"):
                cv2.destroyAllWindows()
                stop_event.set()
                break
            elif pressed_key == ord("r"):
                force = True

            # reload box_config if it has changed
            censor_config, file_hash, _ = reload_config(censor_config, file_hash, force=force)
            force = False

            start_times.append(time.time())

            # check queue for new detection out
            try:
                result = message_queue.get(block=True, timeout=1)
                logger.debug("CENSOR received new message.")
            except Empty:
                logger.debug("CENSOR: raw_model_output queue is empty")
                continue

            # take screenshot
            image_buffer.append(get_next_frame(input_device))

            boxes: list[Box] = process_multiple_passes(result.features, result.bodies, censor_config)
            boxes.sort(reverse=True)
            assert not boxes_buffer or boxes[0] >= boxes_buffer[-1], "Precondition violated"
            boxes_buffer.extend(boxes)  # O(k)

            # move all active boxes from boxes_buffer to live_boxes
            while boxes_buffer and boxes_buffer[0].start < time.monotonic() - CONFIG.live.delay:
                live_boxes.append(boxes_buffer.popleft())

            # remove all boxes that have become inactive
            while live_boxes and live_boxes[0].end < time.monotonic() - CONFIG.live.delay:
                live_boxes.popleft()

            # remove images that are not needed
            while len(image_buffer) > 1 and time.monotonic() - image_buffer[1][0] > CONFIG.live.delay:
                image_buffer.popleft()

            frame_timestamp = time.monotonic() - CONFIG.live.delay

            # nothing in the buffer is old enough
            if image_buffer[0][0] > frame_timestamp:
                continue

            if CONFIG.live.interpolate_frames:
                frame = utils.interpolate_images(
                    image_buffer[0][1], image_buffer[0][0], image_buffer[1][1], image_buffer[1][0],
                    frame_timestamp
                )
            else:
                frame = image_buffer[0][1]

            frame = censor_image_from_boxes(frame.copy(), live_boxes, censor_config)

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
            message_queue.task_done()

        except ValueError as e:
            logger.error(f"Error: {e}")
