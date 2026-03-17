import cv2
import numpy as np
from shapely import Polygon, GeometryCollection

from ..datatypes import Box, CSBar, CSBlur, CSPixel, CSDebug, CSAIRemove
from .remove_feature import remove_feature


def generic_cv_manipulation(apply_operation_func,
                            image: np.ndarray,
                            shape: Polygon) -> np.ndarray:
    """
    Does preparations and finishes everything before and after applying an opencv operation.
    Not all opencv operation need this helper function.

    :param apply_operation_func: the specific function that will be applied to the image
    :param image: the input image
    :param shape: the area where the operation will be applied

    :return: the modified image
    """
    # Convert Shapely Polygon to OpenCV mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    polygon_pts = np.array(shape.exterior.coords, np.int32)
    cv2.fillPoly(mask, [polygon_pts], 255)

    # Get bounding box of the polygon to optimize processing
    min_x, min_y, max_x, max_y = shape.bounds
    min_x, min_y, = max(int(min_x), 0), max(int(min_y), 0),
    max_x, max_y = int(max_x), int(max_y)

    # Extract region of interest (ROI)
    roi = image[min_y:max_y, min_x:max_x]

    # call the operation specific function
    modified = apply_operation_func(roi)

    # Apply pixelation only to the masked area
    modified_roi = np.where(mask[min_y:max_y, min_x:max_x, None] == 255, modified, roi)

    # Replace the region in the original image
    image[min_y:max_y, min_x:max_x] = modified_roi

    return image


def pixelate_image(image: np.ndarray,
                   polygon: Polygon,
                   factor: int | float) -> np.ndarray:
    """
    Pixelate the area set by shape in image.

    :param image: the image
    :param polygon: the shape to pixelate
    :param factor: the pixelation factor

    :return: the pixelated image
    """
    def operation(roi):
        small = cv2.resize(roi, (roi.shape[1] // factor, roi.shape[0] // factor), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    return generic_cv_manipulation(operation, image, polygon)


def blur_image(image: np.ndarray, polygon: Polygon, blur_strength: int) -> np.ndarray:
    """
    Blur the area set by shape in image.

    :param image: the image
    :param polygon: the shape to blur
    :param blur_strength: the blur strength

    :return: the blurred image
    """
    # blur strength must be odd
    if blur_strength % 2 == 0:
        blur_strength += 1

    def operation(roi: np.ndarray):
        # Apply Gaussian blur to the ROI
        return cv2.GaussianBlur(roi, [blur_strength, blur_strength], 0)

    return generic_cv_manipulation(operation, image, polygon)


def draw_bar(image: np.ndarray,
             polygon: Polygon,
             color: tuple[int, int, int] | list[int],
             thickness=cv2.FILLED) -> np.ndarray:
    """
    Draw a filled bar in the area set by shape in image.

    :param image: the image
    :param polygon: the shape of the bar
    :param color: the color of the bar
    :param thickness: the thickness of the bar. Default: cv2.FILLED

    :return: the modified image
    """
    if isinstance(polygon, GeometryCollection):
        polygon = polygon.convex_hull

    points = np.array(polygon.exterior.coords, np.int32)
    return cv2.drawContours(image=image, contours=[points], contourIdx=0, color=color, thickness=thickness)


def draw_debug_info(image: np.ndarray, box: Box) -> np.ndarray:
    """
    Draw debug info from box on image.

    :param image: the image
    :param box: the box

    :return: the debug image
    """
    color = tuple(reversed(box.censor_style.color))
    # points = np.array(box.polygon.exterior.coords, np.int32)
    min_x, min_y, max_x, max_y = box.polygon.bounds

    # cv2.drawContours(image=image, contours=[points], contourIdx=0, color=color, thickness=2)
    image = cv2.putText(image, f"({min_x},{min_y})", (int(min_x + 10), int(min_y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)
    image = cv2.putText(image, f"({max_x},{max_y})", (int(min_x + 10), int(min_y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)
    image = cv2.putText(image, box.label, (int(min_x + 10), int(min_y + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    image = cv2.putText(image, f"{box.score}, {box.start}, {box.end}", (int(min_x + 10), int(min_y + 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def censor_image(image: np.ndarray, box: Box) -> np.ndarray:
    """
    Censor the image from box

    :param image: the image
    :param box: the box

    :return: the censored image
    """
    censor_style = box.censor_style

    if isinstance(censor_style, CSBar):
        return draw_bar(image, box.polygon, color=box.censor_style.color)
    if isinstance(censor_style, CSBlur):
        return blur_image(image, box.polygon, blur_strength=box.censor_style.strength)
    if isinstance(censor_style, CSPixel):
        return pixelate_image(image, box.polygon, factor=box.censor_style.factor)
    if isinstance(censor_style, CSDebug):
        return draw_debug_info(image, box)
    if isinstance(censor_style, CSAIRemove):
        return remove_feature(image, box.polygon,
                              base_url=censor_style.comfy_base_url, workflow_path=censor_style.comfy_workflow)

    raise ValueError(f"Unknown censor_style: {censor_style.__cls__.__name__}, inverse: {box.inverse}.")







