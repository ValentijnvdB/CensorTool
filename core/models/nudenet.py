import math

import cv2
import numpy as np
import onnxruntime

import constants

from .. import get_general_config
from ..datatypes import RawBox
from . import utils


# The NudeNet session. Should not be interacted with directly. Use get_nudenet_session() instead.
_session = None


def get_nudenet_session():
    """
    Get the NudeNet session.
    If one does not exist, create a new one.

    return the onnxruntime session.

    """
    global _session
    if _session is not None:
        return _session

    gen_config = get_general_config()
    if gen_config.gpu_enabled:
        providers = [('CUDAExecutionProvider', {'device_id': gen_config.cuda_device_id})]
    else:
        providers = [('CPUExecutionProvider', {})]

    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = 3

    model_path = constants.model_root / 'detector_v2_default_checkpoint.onnx'
    _session = onnxruntime.InferenceSession(
        path_or_bytes=model_path,
        sess_options=sess_options,
        providers=providers)

    return _session


def prep_img_for_nudenet(raw_img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """
    Prep the image for the nudenet datatypes.

    :param raw_img: the original image
    :param size: the size of the resized image

    :return: the prepped image, how much the image was resized
    """
    scale = utils.get_image_resize_scale(raw_img, size)

    adj_img = cv2.resize(raw_img, None, fx=scale, fy=scale)

    if size > 0:
        (h, w, _) = adj_img.shape
        adj_img = cv2.copyMakeBorder(adj_img, 0, size - h, 0, size - w, cv2.BORDER_CONSTANT, value=0)

    adj_img = adj_img.astype(np.float32)
    adj_img -= [103.939, 116.779, 123.68]
    return adj_img, scale


def get_raw_nudenet_output(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the nudenet model on the images.

    :param images: a list of images

    :return: the raw datatypes output. Format: Boxes, Scores, Classes
    """

    boxes = np.zeros([len(images), 300, 4], dtype=np.float32)
    scores = np.zeros([len(images), 300], dtype=np.float32)
    classes = np.zeros([len(images), 300], dtype=np.int32)

    session = get_nudenet_session()

    for i in range(len(images)):
        boxes[i], scores[i], classes[i] = session.run(constants.model_outputs,
                                                      {constants.model_input: [images[i]]})

    return boxes, scores, classes


def clean_nudenet_output(model_output: tuple[np.ndarray, np.ndarray, np.ndarray]) \
        -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Remove empty rows and rows for which the confidence is lower than the threshold from the model output.
    """
    boxes, scores, classes = model_output

    cleaned_boxes = []
    cleaned_scores = []
    cleaned_classes = []
    for i in range(len(boxes)):
        to_remove = [(confidence <= constants.global_min_prob or class_id == -1)
                     for confidence, class_id in zip(scores[i], classes[i])]

        cleaned_boxes.append(np.delete(boxes[i], to_remove, axis=0))
        cleaned_scores.append(np.delete(scores[i], to_remove, axis=0))
        cleaned_classes.append(np.delete(classes[i], to_remove, axis=0))

    return cleaned_boxes, cleaned_scores, cleaned_classes


def raw_boxes_from_model_output(model_output: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
                                scales: list[float | int],
                                timestamp: int | float) -> list[list[RawBox]]:
    """
    Process the datatypes output into RawBoxes

    :param model_output: the datatypes output.
        model_output[0] = the boxes,
        model_output[1] = the scores,
        model_output[2] = the classes
    :param scales: how much the images were scaled up/down before sending it to the datatypes
    :param timestamp: the timestamp of the image

    :return: A list of lists of RawBoxes. Each list of raw boxes corresponds to one image.
    """
    all_raw_boxes: list[list[RawBox]] = []
    all_boxes = model_output[0]
    all_scores = model_output[1]
    all_classes = model_output[2]
    for boxes, scores, classes, scale in zip(all_boxes, all_scores, all_classes, scales):
        raw_boxes: list[RawBox] = []
        for box, score, class_id in zip(boxes, scores, classes):
            if score > constants.global_min_prob:
                x1 = float(math.floor(box[0] / scale))
                x2 = float(math.ceil(box[2] / scale))

                y1 = float(math.floor(box[1] / scale))
                y2 = float(math.ceil(box[3] / scale))

                points = [
                    (x1, y1),
                    (x1, y2),
                    (x2, y2),
                    (x2, y1)
                ]

                points = [(int(x), int(y)) for x, y in points]

                raw_boxes.append(RawBox.from_points(class_id=class_id,
                                                    points=points,
                                                    score=score,
                                                    timestamp=timestamp)
                                 )
        all_raw_boxes.append(raw_boxes)
    return all_raw_boxes


