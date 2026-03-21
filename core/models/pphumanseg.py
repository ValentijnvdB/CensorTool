import math
from pathlib import Path
from threading import Lock

import cv2

import numpy as np
from shapely import Polygon

import constants
from core.models.utils import download_model

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU]
]

_lock: Lock = Lock()


class PPHumanSeg:
    def __init__(self, model_path, backend_id=0, target_id=0):
        self._modelPath = model_path
        self._backendId = backend_id
        self._targetId = target_id

        self._model = cv2.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._inputNames = ''
        self._outputNames = ['save_infer_model/scale_0.tmp_1']
        self._currentInputSize = None
        self._inputSize = [192, 192]
        self._mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]

    @property
    def name(self):
        return self.__class__.__name__

    def set_backend_and_target(self, backend_id, target_id):
        self._backendId = backend_id
        self._targetId = target_id
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self._currentInputSize = image.shape
        image = cv2.resize(image, (192, 192))

        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        return cv2.dnn.blobFromImage(image)

    def infer(self, image):
        # Preprocess
        input_blob = self._preprocess(image)

        # Forward
        self._model.setInput(input_blob, self._inputNames)
        output_blob = self._model.forward()

        # Postprocess
        results = self._postprocess(output_blob)

        return results

    def _postprocess(self, output_blob):
        output_blob = output_blob[0]
        output_blob = cv2.resize(output_blob.transpose(1, 2, 0), (self._currentInputSize[1], self._currentInputSize[0]),
                                 interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[np.newaxis, ...]

        result = np.argmax(output_blob, axis=1).astype(np.uint8)
        return result


def to_polygons(mask, height, width) -> list[Polygon]:
    # Find contours (edges of each person)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    s_tolerance = math.ceil(max(height, width) / 1000) * 7
    a_tolerance = height * width // 5

    human_polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > a_tolerance:  # filter out noise
            # Convert contour to Shapely Polygon
            coords = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]
            poly = Polygon(coords)
            if poly.is_valid:
                human_polygons.append(poly.simplify(tolerance=s_tolerance))

    return human_polygons


def find_human_polygons(image,
                        backend_target: int = 0):
    backend_id = backend_target_pairs[backend_target][0]
    target_id = backend_target_pairs[backend_target][1]

    # Instantiate PPHumanSeg
    model = constants.model_root / 'human_segmentation_pphumanseg_2023mar.onnx'
    with _lock:
        if not model.exists():
            download_model(url="https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx")

    model = PPHumanSeg(model_path=str(model), backend_id=backend_id, target_id=target_id)

    height, width, _ = image.shape

    # Resize to 192x192
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _image = cv2.resize(image, dsize=(192, 192))

    # Inference
    result = model.infer(_image)
    return cv2.resize(result[0, :, :], dsize=(width, height), interpolation=cv2.INTER_NEAREST)


