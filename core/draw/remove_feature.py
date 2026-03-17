import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests
from shapely import Polygon


def _create_mask(width: int, height: int, polygon: Polygon) -> np.ndarray:
    """
    Create a mask covering the area defined in the polygon.
    """
    mask = np.zeros([height, width], dtype=np.uint8)
    polygon_pts = np.array(polygon.exterior.coords, np.int32)
    cv2.fillPoly(mask, [polygon_pts], 255)

    return mask


def _upload_image(array: np.ndarray, filename: str, base_url: str, image_type: str = "input") -> str:
    """
    Encode a numpy image (BGR or single-channel) as PNG and upload it to ComfyUI.
    Returns the filename ComfyUI assigned to the upload.
    """
    # Encode to PNG in memory
    success, buffer = cv2.imencode(".png", array)
    if not success:
        raise RuntimeError(f"Failed to encode image '{filename}' as PNG.")

    response = requests.post(
        f"{base_url}/upload/image",
        files={"image": (filename, buffer.tobytes(), "image/png")},
        data={"type": image_type, "overwrite": "true"},
    )
    response.raise_for_status()
    return response.json()["name"]


def _find_node_by_title(workflow: dict, title: str) -> dict:
    """
    Return the first workflow node whose _meta.title matches `title`.
    """
    for node in workflow.values():
        if isinstance(node, dict) and node.get("_meta", {}).get("title") == title:
            return node
    raise KeyError(f"No workflow node with title '{title}' found.")


def _queue_prompt(workflow: dict, base_url: str) -> str:
    """
    Submit the workflow to ComfyUI and return the prompt_id.
    """
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    response = requests.post(f"{base_url}/prompt", json=payload)
    response.raise_for_status()
    return response.json()["prompt_id"]


def _poll_until_done(prompt_id: str, base_url: str, poll_interval: float = 1.0) -> dict:
    """
    Block until ComfyUI finishes executing prompt_id, then return the history entry.
    """
    url = f"{base_url}/history/{prompt_id}"
    while True:
        response = requests.get(url)
        response.raise_for_status()
        history = response.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(poll_interval)


def _fetch_output_image(history_entry: dict, base_url: str) -> np.ndarray:
    """
    Pull the first image output from a history entry and return it as a BGR np.ndarray.
    """
    outputs = history_entry.get("outputs", {})
    for node_output in outputs.values():
        for img_info in node_output.get("images", []):
            params = {
                "filename": img_info["filename"],
                "type": img_info.get("type", "output"),
            }
            if "subfolder" in img_info and img_info["subfolder"]:
                params["subfolder"] = img_info["subfolder"]

            response = requests.get(f"{base_url}/view", params=params)
            response.raise_for_status()

            buffer = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Failed to decode output image from ComfyUI.")
            return image

    raise RuntimeError("No image outputs found in ComfyUI history entry.")


def remove_feature(image: np.ndarray, polygon: Polygon, base_url: str, workflow_path: Path) -> np.ndarray:
    """
    Run a ComfyUI inpainting workflow to remove a feature from an image.

    :param image: the input image
    :param polygon: the area the feature can be found
    :param base_url: the base url of the ComfyUI
    :param workflow_path: path to the ComfyUI inpainting workflow

    :return: the modified image
    """
    img_height, img_width = image.shape[:2]

    # 1. Create mask
    mask = _create_mask(img_width, img_height, polygon=polygon)

    # 2. Upload image and mask to ComfyUI
    image_name = _upload_image(image, "input_image.png", image_type="input", base_url=base_url)
    mask_name = _upload_image(mask, "input_mask.png", image_type="input", base_url=base_url)

    # 3. Load and patch the workflow
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    _find_node_by_title(workflow, "IMAGE")["inputs"]["image"] = image_name
    _find_node_by_title(workflow, "MASK")["inputs"]["image"] = mask_name

    # 4. Queue the prompt and wait for completion
    prompt_id = _queue_prompt(workflow, base_url)
    history_entry = _poll_until_done(prompt_id, base_url=base_url)

    # 5. Fetch and return the output image
    return _fetch_output_image(history_entry, base_url=base_url)