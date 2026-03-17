import base64
from pathlib import Path

import numpy as np
from aiohttp import web
from aiohttp.web_response import Response
from loguru import logger

from . import utils
from .server_config import CENSORED_PATH


def _construct_bytes_response(image: bytes, name: str) -> Response:
    return web.Response(
        body=image,
        content_type='image/png',
        headers={f'Content-Disposition': f'attachment; filename="{name}"'}
    )

def _construct_url_response(image: bytes, image_path: Path) -> Response:
    if image_path.is_absolute():
        logger.warning(f"Image path was absolute. Trying to make relative to CENSORED_PATH.")
        image_path = image_path.relative_to(CENSORED_PATH)

    body = {
        'image_name': str(image_path)
    }
    return web.json_response(body)

def _construct_base64_response(image: bytes, extension: str) -> Response:
    base64_str = base64.b64encode(image).decode()

    body = {
        'image_data': base64_str,
        'mime_type': f'image/{extension}'
    }

    # Return the image bytes as part of the response
    return web.json_response(body)


def construct_response(expected_response: str, image: bytes|np.ndarray, extension: str, image_path: Path=None, name: str=None) -> Response:
    if isinstance(image, np.ndarray):
        image = utils.np_to_bytes(image, extension)

    if expected_response == 'bytes':
        return _construct_bytes_response(image, name=name)
    elif expected_response == 'url':
        return _construct_url_response(image, image_path=image_path)
    else:
        return _construct_base64_response(image, extension=extension)