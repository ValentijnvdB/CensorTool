from queue import Queue
from threading import Event, Thread
from typing import Any

import cv2

from core import CensorConfig
from app.config import CONFIG, censor_config_has_changed, load_censor_config_from_file_w_hash

import constants
from core import ProcessedResult

from .censor import censor_loop
from .detect import detect_loop
from .quick import quick_live_censor


####################################################################################################################
# MODES: 'quick', 'precise'
# 'Quick' runs the censor pipeline on each frame separately, making it much quicker,
# but if it could happen that some stuff is visible if the models does not detect a feature for a few frames
# 'Precise' solves this keeping track of the boxes and
# applies the time_safety property to keep the boxes for some time when the feature is not detected anymore
# at the cost of increased latency
####################################################################################################################

def start_live_censor(mode: str, device_id: int, to_v_cam: bool, width: int, height: int, fps: int):
    """
    Start the live censoring mode
    """
    window_name = 'Live Censoring'

    stop_event = Event()

    # setup output device
    if to_v_cam:
        from pyvirtualcam import Camera, PixelFormat
        output_device = Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR)
    else:
        cv2.startWindowThread()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CONFIG.live.cap_width, CONFIG.live.cap_height)

        init_image = cv2.imread(constants.init_screen_image, cv2.IMREAD_GRAYSCALE)
        init_image = cv2.resize(init_image, (CONFIG.live.cap_width, CONFIG.live.cap_height))
        cv2.imshow(window_name, init_image)
        output_device = window_name

    # setup input device
    input_device = None
    if device_id >= 0:
        input_device = cv2.VideoCapture(device_id)

    try:
        if mode == "quick":
            _start_quick(stop_event, output_device, input_device)
        elif mode == "precise":
            _start_precise(stop_event, window_name)
        else:
            raise Exception(f"Invalid mode: {mode}")
    except Exception as e:
        raise e
    finally:
        # close everything gracefully
        if to_v_cam:
            output_device.close()
        else:
            cv2.destroyAllWindows()

        if input_device is not None:
            input_device.release()



def reload_censor_config(censor_config: CensorConfig|None, file_hash: str = '', force: bool = False) -> tuple[CensorConfig, str, bool]:
    changed = False
    if force or censor_config is None or censor_config_has_changed(file_hash):
        censor_config, file_hash = load_censor_config_from_file_w_hash()
        censor_config.enable_overlays = False
        censor_config.merge_overlapping_borders = False
        censor_config.merge_overlapping_censor_box = False
        changed = True

    return censor_config, file_hash, changed


def _start_precise(stop_event: Event, window_name: str):
    message_queue: Queue[ProcessedResult] = Queue()

    detect_thread = Thread(target=detect_loop,
                           args=(stop_event, message_queue),
                           daemon=True)

    detect_thread.start()

    try:
        censor_loop(stop_event, message_queue, reload_censor_config, window_name)

    except KeyboardInterrupt:
        print("Interrupted...")
        stop_event.set()
        detect_thread.join()


def _start_quick(stop_event: Event, output_device: Any, input_device: cv2.VideoCapture|None):
    try:
        quick_live_censor(stop_event, reload_censor_config, output_device, input_device)

    except KeyboardInterrupt:
        print("Interrupted...")
        stop_event.set()
