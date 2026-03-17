import numpy as np
from shapely import Polygon, MultiPolygon, Point, affinity

from .config_types import *


class Box:

    def __init__(self,
                 start: int | float,
                 end: int | float,
                 polygon: Polygon,
                 censor_style: CENSOR_STYLE,
                 label: str,
                 score: float,
                 overlay: dict[str, str|Path] | None,
                 overlay_config: OVERLAY_TYPE,
                 border: BORDER_TYPE,
                 inverse: bool):

        self.start = start
        self.end = end
        self.censor_style = censor_style
        self.label = label
        self.score = score
        self.overlay = overlay
        self.overlay_config = overlay_config
        self.border = border
        self.inverse = inverse

        self.polygon = polygon

    @classmethod
    def from_points(cls,
                    shape: str,
                    start: int | float,
                    end: int | float,
                    points: list[list[int | float]] | np.ndarray,
                    censor_style: CENSOR_STYLE,
                    label: str,
                    score: float,
                    overlay: dict[str, str|Path] | None,
                    overlay_config: OVERLAY_TYPE,
                    border: BORDER_TYPE,
                    inverse: bool,
                    intersect: bool,
                    other_shapes: list[Polygon]):
        """
        Returns a new Box instance where the polygon is created based on 'shape' and 'points'.
        If other_shape is provided,
        then the resulting polygon is the intersection with the highest area between the censor poly and other_shape.
        """
        # compute some significant coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
        width, height = x_max - x_min, y_max - y_min
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        # compute polygon
        if shape == 'ellipse':
            polygon = Box._create_ellipse_polygon((center_x, center_y), int(width/2), int(height/2))
        elif shape == 'circle':
            r = int(max(width, height) / 2)
            polygon = Box._create_ellipse_polygon((center_x, center_y), r, r)
        elif shape == 'rectangle':
            polygon = Polygon.from_bounds(x_min, y_min, x_max, y_max)
        else:
            polygon = Polygon(points)

        if intersect:
            best_poly = None
            highest_area = 0
            for shape in other_shapes:
                if isinstance(shape, MultiPolygon):
                    for other in shape.geoms:
                        better, poly = _test_intersection(polygon, other, highest_area)
                        if better and not isinstance(poly, MultiPolygon):
                            best_poly = poly
                            highest_area = best_poly.area
                else:
                    better, poly = _test_intersection(polygon, shape, highest_area)
                    if better and not isinstance(poly, MultiPolygon):
                        best_poly = poly
                        highest_area = best_poly.area


            if best_poly is not None:
                polygon = best_poly

        return Box(
            start=start,
            end=end,
            polygon=polygon,
            censor_style=censor_style,
            label=label,
            score=score,
            overlay=overlay,
            overlay_config=overlay_config,
            border=border,
            inverse=inverse
        )

    def censor_style_priority(self) -> int:
        """
        Get the priority of each censor_style.
        Lower means censoring earlier.

        :return: the priority of the censor_style of self.
        """
        if isinstance(self.censor_style, CSAIRemove):
            return 0
        if isinstance(self.censor_style, CSBlur):
            return 5
        if isinstance(self.censor_style, CSPixel):
            return 10
        if isinstance(self.censor_style, CSBar):
            return 20
        if isinstance(self.censor_style, CSDebug):
            return 100

        raise ValueError(f"Unknown censor_style {self.censor_style.__class__.__name__}")

    def override_timestamp(self, new_timestamp, time_safety: float):
        self.start = max(new_timestamp - time_safety / 2, 0)
        self.end = new_timestamp + time_safety / 2

    @classmethod
    def _create_ellipse_polygon(cls, center: tuple[int, int], rx: int, ry: int):
        """
        Create an ellipse polygon and set self.polygon

        :param center: center point (x,y) coordinates
        :param rx: the radius in x-axis (before rotation)
        :param ry: the radius in y-axis (before rotation)
        """
        # Function from : https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely
        # Let create a circle of radius 1 around center point:
        circ = Point(center).buffer(1)

        # Let create the ellipse along x and y:
        return affinity.scale(circ, rx, ry)

    def __lt__(self, other):
        if self.start == other.start:
            return self.end < other.end
        return self.start < other.start

    def __le__(self, other):
        if self.start == other.start:
            return self.end <= other.end
        return self.start <= other.start


def _test_intersection(polygon: Polygon, other_shape: Polygon, highest_area: float):
    if polygon.intersects(other_shape):
        temp_poly = polygon.intersection(other_shape)
        if temp_poly.area > highest_area:
            return True, temp_poly
    return False, polygon


