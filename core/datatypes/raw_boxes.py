from typing import get_args

import numpy as np
from shapely import Polygon

from .config_types import FEATURE


_FEATURES = list(get_args(FEATURE))


class RawBox:

    def __init__(self,
                 class_id: int,
                 shape: Polygon,
                 score: float,
                 timestamp: int | float):
        self.class_id = class_id
        self.score = score
        self.timestamp = timestamp

        self.polygon = shape

        if not self.polygon.is_valid:
            raise ValueError("Invalid polygon")

    @classmethod
    def from_points(cls,
                    class_id: int,
                    points: list[list[int | float]] | np.ndarray,
                    score: float,
                    timestamp: int | float,
                    from_bounds: bool = False):
        """
        Returns a new RawBox instance where the shape is a polygon where points indicate the borders.
        """

        points = np.intp(points)
        if from_bounds:
            polygon = Polygon.from_bounds(*points)
        else:
            polygon = Polygon(points)

        if not polygon.is_valid:
            raise ValueError("Invalid polygon")

        return RawBox(class_id=class_id,
                      shape=polygon,
                      score=score,
                      timestamp=timestamp)

    @property
    def label(self) -> FEATURE:
        return _FEATURES[self.class_id]



