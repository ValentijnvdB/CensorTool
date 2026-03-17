import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

import constants

from ..datatypes import Box, CensorConfig, OVText, OVSticker


def apply_text_overlay(image: np.ndarray, box: Box) -> np.ndarray:
    """
    Draw text on the image at a slight angle

    :param image: the image
    :param box: the box

    :return: the image with text overlay
    """
    ov_config = box.overlay_config
    text = box.overlay['text']

    center_x, center_y = int(box.polygon.centroid.x), int(box.polygon.centroid.y)
    angle = random.uniform(-15, 15)  # Random angle in range [-15, 15]

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text,
                                                          fontFace=ov_config.font,
                                                          fontScale=ov_config.font_scale,
                                                          thickness=ov_config.thickness)

    # Create a blank mask to draw the text
    text_mask = np.zeros((text_height + 20, text_width + 20, 3), dtype=np.uint8)

    # Put text on the mask
    text_x = 10  # Small padding
    text_y = text_height + 10  # Ensure proper positioning inside the mask
    cv2.putText(imag=text_mask,
                text=text,
                org=(text_x, text_y),
                fontFace=ov_config.font,
                fontScale=ov_config.font_scale,
                color=ov_config.color,
                thickness=ov_config.thickness,
                lineType=cv2.LINE_AA)

    # Compute the center of the text mask
    center = (text_mask.shape[1] // 2, text_mask.shape[0] // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the text mask
    rotated_text = cv2.warpAffine(text_mask, rotation_matrix, (text_mask.shape[1], text_mask.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Find the region to place the text
    x_offset = center_x - text_mask.shape[1] // 2
    y_offset = center_y - text_mask.shape[0] // 2

    # Overlay the rotated text onto the image
    for i in range(3):  # Copy only non-black pixels from the rotated text
        image[y_offset:y_offset + text_mask.shape[0], x_offset:x_offset + text_mask.shape[1], i] = np.where(
            rotated_text[:, :, i] > 0, rotated_text[:, :, i],
            image[y_offset:y_offset + text_mask.shape[0], x_offset:x_offset + text_mask.shape[1], i]
        )

    return image


def apply_sticker_overlay(image: np.ndarray, box: Box) -> np.ndarray:
    """
    Apply a sticker overlay on the image

    :param image: the image
    :param box: the box

    :return: the image with the sticker applied
    """
    sticker_path = box.overlay['sticker']
    sticker_image = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    sticker_h, sticker_w, sticker_c = sticker_image.shape

    min_x, min_y, max_x, max_y = box.polygon.bounds
    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)

    # Compute available width and height
    box_width = max_x - min_x
    box_height = max_y - min_y

    # Get original sticker dimensions
    sticker_h, sticker_w = sticker_image.shape[:2]
    sticker_aspect = sticker_w / sticker_h  # Aspect ratio (width/height)

    # Resize sticker to fit within the bounding box while maintaining aspect ratio
    if (box_width / sticker_w) < (box_height / sticker_h):
        # Fit based on width
        new_w = box_width
        new_h = int(new_w / sticker_aspect)
    else:
        # Fit based on height
        new_h = box_height
        new_w = int(new_h * sticker_aspect)

    x_offset = min_x + (box_width - new_w) // 2
    y_offset = min_y + (box_height - new_h) // 2

    # resize RGB channels
    sticker_bgr = sticker_image[:, :, :3]
    sticker_resized = cv2.resize(sticker_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if sticker_c == 4:
        # Resize Alpha channel
        alpha_mask = sticker_image[:, :, 3]
        alpha_resized = cv2.resize(alpha_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return apply_sticker_overlay_alpha(image, sticker_resized, alpha_resized, x_offset, y_offset, new_w, new_h)

    else:
        image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = sticker_resized[:, :, :]
        return image


def apply_sticker_overlay_alpha(image: np.ndarray,
                                sticker_resized: np.ndarray,
                                alpha_resized: np.ndarray,
                                x_offset: int, y_offset: int,
                                new_w: int, new_h: int) -> np.ndarray:
    """
    Helper function for applying stickers with an alpha channel.

    :param image: the image
    :param sticker_resized: the rgb channels of the sticker image
    :param alpha_resized: the alpha channel of the sticker image
    :param x_offset: the x offset of the sticker image
    :param y_offset: the y offset of the sticker image
    :param new_w: the new width of the sticker image
    :param new_h: the new height of the sticker image

    :return: the image with the sticker applied
    """

    # Extract the region of interest (ROI) from the main image
    roi = image[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

    # Normalize alpha mask to range [0, 1]
    alpha = alpha_resized / 255.0

    # Blend images using alpha transparency
    for c in range(3):  # Apply for each channel (B, G, R)
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * sticker_resized[:, :, c]

    # Replace the ROI back into the main image
    image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = roi

    return image


def apply_overlay(image: np.ndarray, box: Box) -> np.ndarray:
    """
    Apply an overlay on the image inside the box

    :param image: the image
    :param box: the box

    :return: the image with overlay
    """
    if box.overlay is None:
        return image

    overlay_type = box.overlay['type']

    if overlay_type == 'text':
        return apply_text_overlay(image, box)

    if overlay_type == 'sticker':
        return apply_sticker_overlay(image, box)

    logger.warning(f"Unknown overlay while applying overlay: {overlay_type}.")
    return image


## GENERATION
def generate_overlay(overlay_config: dict[str, Any], config: CensorConfig) -> dict[str, str|Path] | None:
    """
    Generate a specific overlay for the overlay_config

    :param overlay_config: the overlay box_config
    :param config: the censoring configuration

    :return: the overlay to place (str if text, Path if sticker, or None if no overlay should be placed)
    """
    if (not config.enable_overlays) or overlay_config is None:
        return None

    ov_config = overlay_config

    ############ TEXT OVERLAY ############
    if isinstance(ov_config, OVText):
        pos_values = ov_config.values

        # if not list of values, assume it is a file
        if not isinstance(pos_values, list):
            values_file = Path(__file__).parent.parent.parent.resolve() / pos_values
            with open(values_file, 'r') as file:
                pos_values = file.readlines()

        return {
            'type': 'text',
            'text': random.choice(pos_values)
        }

    ############ STICKER OVERLAY ############
    if isinstance(ov_config, OVSticker):
        categories = ov_config.categories

        if not isinstance(categories, list):
            categories = [categories]

        image_paths = []
        for cat in categories:
            category_dir = constants.stickers_root_path / cat
            if not category_dir.exists():
                logger.warning(f"Sticker category {cat} does not exist! Skipping.")
                continue

            image_paths.extend([p for p in category_dir.rglob("*") if p.suffix.lower() in constants.image_extensions])

        if len(image_paths) == 0:
            logger.warning(f"No images found in subdirs {categories}! Returning without sticker.")
            return None

        return {
            'type': 'sticker',
            'path': random.choice(image_paths)
        }

    return None


def get_sticker_categories():
    from os import walk
    root = constants.stickers_root_path
    return [x[0] for x in walk(root)]

