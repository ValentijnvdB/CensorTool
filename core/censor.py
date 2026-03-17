import random
from pathlib import Path
from typing import Collection

import numpy as np
from shapely import Polygon, unary_union, MultiPolygon

from .config import CensorConfig, CensorBox

from .draw import (censor_image, inverse_censor_image, generate_overlay,
                              apply_overlay, draw_border, watermark_image)
from .datatypes import Box, RawBox


def process_multiple_passes(all_passes: list[list[RawBox]],
                            all_passes_bodies: list[list[Polygon]],
                            censor_config: CensorConfig) -> list[Box]:
    """
    Merge multiple passes into a single pass by merging intersecting polygons of the same class
    and process the RawBoxes into Boxes.

    :param all_passes: the passes to process
    :param all_passes_bodies: the bodies to process
    :param censor_config: the censoring box_config

    :return: The merged pass containing Boxes
    """
    def process_to_box(rbs: list[RawBox], bds: list[Polygon]) -> list[Box]:
        output_boxes = []
        for rb in rbs:
            box = process_raw_box(rb, bds, censor_config)
            if box:
                output_boxes.append(box)
        return output_boxes

    # TODO: merge all the bodies from all the passes into one (Multi)Polygon
    bodies = all_passes_bodies[0]

    if len(all_passes) == 1:
        return process_to_box(all_passes[0], bodies)

    # unwrap passes
    raw_boxes_per_class = [[] for _ in range(18)]
    for single_pass in all_passes:
        for raw_box in single_pass:
            raw_boxes_per_class[raw_box.class_id].append(raw_box)

    final_raw_boxes = []
    # merge boxes as much as possible
    for single_class in raw_boxes_per_class:
        if len(single_class) == 0:
            continue

        if not censor_config.merge_overlapping_censor_boxes:
            final_raw_boxes.extend(single_class)
            continue

        # merge the overlapping features
        final_raw_boxes.extend(merge_boxes(single_class))

    ######### Process RawBoxes into Boxes #########
    return process_to_box(final_raw_boxes, bodies)


def merge_boxes(boxes: list[Box | RawBox]) -> list[Box | RawBox]:
    """
    Merge overlapping boxes as much as possible

    :param boxes: the boxes to merge

    :return: the merged boxes
    """
    ######### Merge Boxes or RawBoxes of the same feature if they overlap ########
    # Each list in 'intersections' is such that:
    #   1. for each polygon in the list there is at least one other polygon it intersects with
    #     (unless it is in a list on its own)
    #   2. the union of all polygons in the list is exactly one polygon
    if len(boxes) == 0:
        return []

    expected_type = type(boxes[0])
    if expected_type not in [Box, RawBox]:
        raise ValueError(f"Expected type Box or RawBox. Got '{expected_type}'")
    for box in boxes:
        if not isinstance(box, expected_type):
            raise ValueError("An element in input was not of the expected type. "
                             f"Expected '{expected_type}', got '{type(box)}'")

    intersections = [[boxes.pop()]]
    for raw_box in boxes:

        found_intersections = []

        # check all list in 'intersections' for intersections
        for i, intersection_list in enumerate(intersections):
            for other_box in intersection_list:
                if raw_box.polygon.intersects(other_box.polygon):
                    found_intersections.append(i)
                    break

        if len(found_intersections) == 0:
            # add raw_box as its own list
            intersections.append([raw_box])
        elif len(found_intersections) == 1:
            # at raw_box to the list with which it intersected
            index = found_intersections[0]
            intersections[index].append(raw_box)
        else:
            # merge all lists that intersect_human with raw_box into one
            # keep all others the same
            new_total_list = []
            new_intersections_of_raw_box = [raw_box]
            for i, intersection_list in enumerate(intersections):
                if i in found_intersections:
                    new_intersections_of_raw_box.extend(intersection_list)
                else:
                    new_total_list.append(intersection_list)

            new_total_list.append(new_intersections_of_raw_box)
            intersections = new_total_list

    # process the different sets: merge polygon and create a new Box/RawBox
    final_boxes = []
    for intersection in intersections:
        if len(intersection) == 0:
            continue

        if len(intersection) == 1:
            final_boxes.append(intersection[0])
            continue

        polygon = unary_union([raw_box.polygon for raw_box in intersection])
        if not isinstance(polygon, Polygon):
            raise RuntimeError("union was not a polygon!")
        if isinstance(polygon, MultiPolygon):
            raise RuntimeError("union was a MultiPolygon!")

        if expected_type == RawBox:
            score = max([raw_box.score for raw_box in intersection])
            class_id = intersection[0].class_id
            timestamp = intersection[0].timestamp
            final_boxes.append(RawBox(shape=polygon,
                                      score=score,
                                      class_id=class_id,
                                      timestamp=timestamp))
        else:
            # expected type is Box
            # pick extremes for start, end, and scores
            # for others we pick a random value from the available values
            start = min([box.start for box in intersection])
            end = max([box.end for box in intersection])
            score = max([box.score for box in intersection])
            censor_style = random.choice([box.censor_style for box in intersection])
            label = random.choice([box.label for box in intersection])
            overlay = random.choice([box.overlay for box in intersection])
            overlay_config = random.choice([box.overlay_config for box in intersection])
            border = random.choice([box.border for box in intersection])
            inverse = random.choice([box.inverse for box in intersection])

            final_boxes.append(Box(start=start,
                                   end=end,
                                   score=score,
                                   censor_style=censor_style,
                                   label=label,
                                   overlay=overlay,
                                   overlay_config=overlay_config,
                                   border=border,
                                   inverse=inverse,
                                   polygon=polygon))

    return final_boxes


def process_raw_box(raw: RawBox, bodies: list[Polygon], config: CensorConfig) -> Box | None:
    """
    Processes a RawBox into a Box object by applying the relevant box_config information.

    returns None if the RawBox should not be censored, because it is not in the 'features_to_censor' list
    or because the confidence of the datatypes was too low.

    :param raw: The RawBox object to be processed
    :param bodies: the list of Polygons representing the outline of the all_polygons in the image
    :param config: the censoring configuration

    :return: None if the RawBox should not be censored, otherwise the Box object
    """
    features_to_censor = config.features_to_censor
    if raw.label not in features_to_censor:
        # we should not censor this feature
        return None

    censor_box: CensorBox = features_to_censor[raw.label]

    if raw.score <= censor_box.min_prob:
        # we are not confident enough
        return None

    x_area_safety = censor_box.width_area_safety
    y_area_safety = censor_box.height_area_safety
    time_safety   = censor_box.time_safety

    min_x, min_y, max_x, max_y = raw.polygon.bounds
    width = max_x - min_x
    height = max_y - min_y

    # expand the area
    new_points = expand_shape(points=raw.polygon.exterior.coords,
                              offset=(width * x_area_safety / 2, height * y_area_safety / 2),
                              round_to_int=True)

    overlay: dict[str, str|Path] | None = None
    overlay_config = config.features_to_censor[raw.label].overlay
    if censor_box.overlay is not None:
        roll = random.random()
        if roll < censor_box.overlay.probability:
            overlay = generate_overlay(censor_box.overlay, config)

    return Box.from_points(
        shape=censor_box.shape,
        start=max(raw.timestamp - time_safety / 2, 0),
        end=raw.timestamp + time_safety / 2,
        points=new_points,
        censor_style=censor_box.censor_style,
        label=raw.label,
        score=raw.score,
        overlay=overlay,
        overlay_config=overlay_config,
        border=censor_box.border,
        inverse=censor_box.inverse,
        other_shapes=bodies,
        intersect=censor_box.intersect_human,
    )


def expand_shape(points: Collection[list[int | float]],
                 offset: int | float | tuple[int | float, int | float],
                 round_to_int: bool = True) -> list:
    """
    Expand the shape created by points by offset in all directions.

    :param points: a list of the (x, y) points
    :param offset: how much to expand in each direction
    :param round_to_int: whether to round to the x and y coords to integers

    :return: a list of the expanded (x, y) points
    """
    if isinstance(offset, int | float):
        offset = (offset, offset)

    # Convert points to a NumPy array
    points = np.array(points, dtype=float)

    # Compute the centroid of the shape
    centroid = np.mean(points, axis=0)

    # Compute vectors from centroid to each point
    vectors = points - centroid

    # Normalize vectors to keep direction but set magnitude to 1
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Scale unit vectors by the offset
    expanded_points = points + unit_vectors * offset

    if round_to_int:
        return [(int(x), int(y)) for x, y in expanded_points]
    else:
        return list(expanded_points)


def censor_image_from_boxes(image, boxes: Collection[Box], censor_config: CensorConfig):
    """
    Applies the boxes to image.

    :param image: the image to censor
    :param boxes: the boxes that will be applied
    :param censor_config: the censoring box_config

    :return: the censored image
    """
    if not isinstance(boxes, list):
        boxes = list(boxes)

    # sort on censor style
    boxes.sort(key=lambda box: box.censor_style_priority())

    # split boxes on inverse and normal censors
    # and save boxes that need borders
    inverse_boxes = []
    normal_boxes = []
    boxes_with_borders = []
    for box in boxes:
        if box.inverse:
            inverse_boxes.append(box)
        else:
            normal_boxes.append(box)

        if box.border:
            boxes_with_borders.append(box)

    # first, perform the inverse censors
    if len(inverse_boxes) > 0 or censor_config.force_inverse_censor:
        image = inverse_censor_image(image, inverse_boxes, censor_config)

    # second, perform normal censor
    for box in normal_boxes:
        image = censor_image(image, box)

    # if required, merge overlapping borders
    if censor_config.merge_overlapping_borders:
        boxes_with_borders = merge_boxes(boxes_with_borders)

    # draw the borders
    for box in boxes_with_borders:
        image = draw_border(image=image,
                            shape=box.polygon,
                            thickness=box.border['thickness'],
                            color=box.border['color'])

    # apply overlays
    if censor_config.enable_overlays:
        for box in boxes:
            image = apply_overlay(image, box)

    image = watermark_image(image, censor_config)

    return image
