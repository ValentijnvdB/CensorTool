from pathlib import Path

import numpy as np
from loguru import logger

import constants


def validate_install():

    to_test = ['config', 'pipeline', 'pipeline_gpu', 'vcam', 'screenshot']

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

    ########################## PIPELINE ##########################
    success, msg = validate_pipeline(CONFIG.pipeline_path, 'cpu')
    if success:
        logger.info(f"Pipeline ({CONFIG.pipeline_path}) on CPU validated successfully: {msg}")
    else:
        logger.error(f"Pipeline ({CONFIG.pipeline_path}) on CPU failed: {msg}")
    validated['pipeline'] = success

    success, msg = validate_pipeline(CONFIG.pipeline_path, 'cuda')
    if success:
        logger.info(f"Pipeline ({CONFIG.pipeline_path}) on GPU validated successfully: {msg}")
    else:
        logger.error(f"Pipeline ({CONFIG.pipeline_path}) on GPU failed: {msg}")
    validated['pipeline_gpu'] = success

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

    onnx = validated['pipeline'] or validated['pipeline_gpu']
    onnx_str = f"(CPU {'and GPU' if validated['pipeline_gpu'] else 'only'})"

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


def validate_pipeline(pipeline_path: Path, device: str) -> tuple[bool, str]:
    """
    Load every model in the pipeline config on the given device (ignoring the devices set in the file)
    and run a detection-only job on a small image.
    """
    try:
        import yaml
        import censor_core as cc

        pipeline_path = Path(pipeline_path)
        if not pipeline_path.exists():
            return False, f"Pipeline config not found at '{pipeline_path}'."

        with open(pipeline_path, 'r') as f:
            spec = yaml.safe_load(f)

        def resolve(entry: dict) -> dict:
            entry = dict(entry)
            entry.pop('device', None)
            model_path = Path(entry['path'])
            if not model_path.is_absolute():
                model_path = pipeline_path.parent / model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model '{entry.get('name', model_path.name)}' not found at '{model_path}'.")
            entry['path'] = str(model_path)
            return entry

        models = [resolve(m) for m in spec.get('models', [])]
        landmarks = resolve(spec['landmarks']) if spec.get('landmarks') else None

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        with cc.ImagePipeline(models=models, landmarks=landmarks, device=device, max_workers=1) as pipeline:
            job = pipeline.submit(cc.Job(image=image, use_cache=False)).wait()
            stages = pipeline.models

        if not job.success:
            return False, str(job.error)

        return True, f"models: {stages}"

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