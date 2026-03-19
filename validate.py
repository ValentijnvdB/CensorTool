import numpy as np
from loguru import logger

import constants


def validate_install():

    to_test = ['config', 'onnxruntime', 'onnxruntime_gpu', 'vcam', 'screenshot']

    validated: dict[str, bool] = {}
    for item in to_test:
        validated[item] = False

    ########################## CONFIG ##########################
    from app.config import CONFIG
    if CONFIG is None:
        logger.error("Config not loaded.")
    else:
        logger.info(f"Config loaded.")
        validated['config'] = True

    ########################## ONNXRUNTIME ##########################
    success, msg = validate_onnxruntime(False)
    if success:
        logger.info(f"Onnxruntime CPU validated successfully.")
    else:
        logger.error(f"Onnxruntime CPU failed: {msg}")
    validated['onnxruntime'] = success

    success, msg = validate_onnxruntime(True)
    if success:
        logger.info(f"Onnxruntime GPU validated successfully.")
    else:
        logger.error(f"Onnxruntime GPU failed: {msg}")
    validated['onnxruntime_gpu'] = success

    ########################## VCAM ##########################
    success, msg = validate_vcam()
    if success:
        logger.info(f"Vcam validated successfully.")
    else:
        logger.error(f"Vcam failed: {msg}")
    validated['vcam'] = success

    ########################## SCREENSHOT ##########################
    success, msg = validate_screenshot(CONFIG)
    if success:
        logger.info(f"Screenshot validated successfully.")
    else:
        logger.error(f"Screenshot failed: {msg}")
    validated['screenshot'] = success


    ########################## END REPORT ##########################
    works_str = 'should work'
    not_working_str = 'NOT WORKING (check logs above)'

    onnx = validated['onnxruntime'] or validated['onnxruntime_gpu']
    onnx_str = f"(CPU {'and GPU' if validated['onnxruntime_gpu'] else 'only'})"

    image_censor = onnx
    image_str = f"{works_str} {onnx_str}." if image_censor else not_working_str

    video_censor = onnx
    video_str = f"{works_str} {onnx_str}." if video_censor else not_working_str

    live_censor = onnx and validated['vcam'] and validated['screenshot']
    live_str = f"{works_str}" if live_censor else not_working_str

    logger.info("Validation completed.")
    logger.info("NOTE: this script is not completely done yet, so not everything is tested. "
                "Things still may not work even if it report says it should.")
    logger.info(
        "Note with live censor: if, for example, the virtual camera or taking screenshots does not work, then it will say 'not working' above. "
        "However, if you are not interested in this feature, live censoring without them could still work.")
    logger.info("===========================================================")
    logger.info("Mode overview:")
    logger.info(f"  Image censor: {image_str}")
    logger.info(f"  Video censor: {video_str}")
    logger.info(f"  Live/webcam censor: {live_str}")
    logger.info("===========================================================")


def validate_onnxruntime(gpu: bool) -> tuple[bool, str]:
    """Start an onnxruntime session with CPU or GPU and run on a small image."""
    try:
        import onnxruntime

        if gpu:
            providers = [('CUDAExecutionProvider', {'device_id': 0})]
        else:
            providers = [('CPUExecutionProvider', {})]

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3

        model_path = constants.model_root / 'detector_v2_default_checkpoint.onnx'
        if not model_path.exists():
            return False, f'Nudenet model not found at {model_path}. Make sure you have downloaded it first.'

        session = onnxruntime.InferenceSession(
            path_or_bytes=model_path,
            sess_options=sess_options,
            providers=providers)

        image = np.zeros((120, 120, 3), dtype=np.uint8)

        if session is None:
            return False, 'session is None'

        session.run(constants.model_outputs, {constants.model_input: [image]})

        return True, 'success'

    except Exception as e:
        return False, str(e)

def validate_vcam() -> tuple[bool, str]:
    """Open a virtual camera, send a frame and close it."""
    try:
        import pyvirtualcam

        w, h, fps = 120, 120, 15

        cam = pyvirtualcam.Camera(height=h, width=w, fps=fps)

        frame = np.zeros((h, w, 3), dtype=np.uint8)

        cam.send(frame)

        cam.close()

        return True, 'success'

    except Exception as e:
        return False, str(e)

def validate_screenshot(config) -> tuple[bool, str]:
    """Take a screenshot of the configured region and make sure it is not all black."""
    try:
        import pyautogui

        screenshot = pyautogui.screenshot(
            region=(config.live.cap_left, config.live.cap_top, config.live.cap_width, config.live.cap_height)
        )

        array = np.array(screenshot, dtype=np.uint8)

        # make sure it is not all black
        if array.sum() > 0:
            return True, 'success'
        else:
            return False, 'Screenshot was all black'

    except Exception as e:
        return False, str(e)