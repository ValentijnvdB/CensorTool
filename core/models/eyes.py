import math

import cv2
import numpy as np

import constants
from ..datatypes import RawBox


def _find_eyes(image: np.ndarray,
               x_offset: int = 0,
               y_offset: int = 0,
               merge_eyes: bool = True) -> None | list[list[tuple[int, int]]]:
    """
    Run the detect eyes datatypes on the image.

    :param image: the original image. Recommended: only send in the part of the image that contains the face.
    :param x_offset: the x offset to be added to the coordinates
    :param y_offset: the y offset to be added to the coordinates
    :param merge_eyes: merge the eyes boxes into a single RawBox

    :return: None if no eyes were detected, list of corners of the eye boxes
    """
    eye_cascade = cv2.CascadeClassifier(str(constants.model_root / 'haarcascade_eye.xml'))

    if image.dtype != 'uint8':
        image = cv2.convertScaleAbs(image)
    eyes = eye_cascade.detectMultiScale(image, minSize=(20, 20))

    if len(eyes) == 0:
        return None

    if merge_eyes:
        min_x = 100000000
        max_x = -1
        left_coords = []
        right_coords = []
        for ex, ey, ew, eh in eyes:
            x1 = int(ex + x_offset)
            x2 = int(ex + x_offset + ew)
            y1 = int(ey + y_offset)
            y2 = int(ey + y_offset + eh)
            if x1 < min_x:
                min_x = x1
                left_coords = [
                    (x1, y1),
                    (x1, y2)
                ]

            if x2 > max_x:
                max_x = x2
                right_coords = [
                    (x2, y2),
                    (x2, y1)
                ]

        return [left_coords + right_coords]

    else:
        eyes_coords = []
        for (ex, ey, ew, eh) in eyes:
            x1 = int(ex + x_offset)
            x2 = int(ex + x_offset + ew)
            y1 = int(ey + y_offset)
            y2 = int(ey + y_offset + eh)
            points = [
                (x1, y1),
                (x1, y2),
                (x2, y2),
                (x2, y1)
            ]

            eyes_coords.append(points)

        return eyes_coords


def detect_eyes(image,
                raw_boxes: list[RawBox],
                timestamp: int | float) -> list[RawBox]:
    """
    Detect eyes in all faces in features

    :param image: the original image
    :param raw_boxes: the raw boxes (processed output of the nudenet datatypes)
    :param timestamp: the timestamp of the image

    :return: the list of RawBoxes, one for each face
    """
    out = []
    for i, raw_box in enumerate(raw_boxes):
        class_id = raw_box.class_id
        x1, y1, x2, y2 = raw_box.polygon.bounds
        confidence = raw_box.score

        if class_id in [6, 7] and confidence > constants.global_min_prob:
            # a face has been detected
            face_img = image[int(y1):math.ceil(y1 + (y2 - y1) / 2), int(x1):int(x2)]
            eyes = _find_eyes(image=face_img, x_offset=int(x1), y_offset=int(y1))
            if eyes:
                for eye in eyes:
                    out.append(RawBox.from_points(points=eye,
                                                  score=confidence,
                                                  class_id=class_id + 10,
                                                  timestamp=timestamp))

    return out

