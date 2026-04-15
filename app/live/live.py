from pathlib import Path
from threading import Event
from typing import Any

import cv2
import mss
from loguru import logger

import constants
from input_devices.generic import InputDevice
from .output_devices.generic import OutputDevice

from ..config import CONFIG

from .quick import live_censor_loop


####################################################################################################################
# MODES: 'quick', 'precise'
# 'Quick' runs the censor pipeline on each frame separately, making it much quicker,
# but if it could happen that some stuff is visible if the models does not detect a feature for a few frames
# 'Precise' solves this keeping track of the boxes and
# applies the time_safety property to keep the boxes for some time after the feature is not detected anymore
# at the cost of increased latency
####################################################################################################################

def start_live_censor(input_device_id: int, to_v_cam: bool, vcam_w: int = 1920, vcam_h: int = 1080, vcam_fps: int = 10, censor_config: Path|str = None):
    """
    Start the live/webcam censoring mode.

    :param input_device_id: the id of the input device. <0 for screenshot mode
    :param to_v_cam: If true, output to a virtual cam, otherwise use a cv2 window
    :param vcam_w: the width of the virtual camera.
    :param vcam_h: the height of the virtual camera.
    :param vcam_fps: target fps of the virtual camera.
    :param censor_config: config file path
    """
    stop_event = Event()

    # setup output device
    if to_v_cam:
        from .output_devices.vcam_device import VcamDevice
        output_device = VcamDevice(vcam_w, vcam_h, vcam_fps)
    else:
        from .output_devices.window_device import WindowDevice
        output_device = WindowDevice(window_name='CensorTool: Live Censoring')

    # setup input device
    if input_device_id >= 0:
        from .input_devices.camera_device import CameraDevice
        input_device = CameraDevice(input_device_id)
    else:
        from .input_devices.screenshot_device import ScreenshotDevice
        input_device = ScreenshotDevice()

    try:
        if isinstance(censor_config, str):
            censor_config: Path = Path(censor_config)
        live_censor_loop(stop_event, input_device, output_device, censor_config)
    except KeyboardInterrupt:
        print("Interrupted...")
        stop_event.set()
    except Exception as e:
        raise e
    finally:
        # close everything gracefully
        input_device.close()
        output_device.close()
