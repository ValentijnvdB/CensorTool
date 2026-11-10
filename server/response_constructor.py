import base64
from pathlib import Path
from urllib.parse import quote

import numpy as np
from aiohttp import web
from aiohttp.web_response import Response
from loguru import logger

from . import utils
from .server_config import CENSORED_PATH

MIME_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.bmp': 'image/bmp',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
    '.webp': 'image/webp',
    '.avif': 'image/avif',
    '.gif': 'image/gif',
}

# Fields of the info dict that the 'bytes' response sends as headers
INFO_HEADERS = {
    'width': 'X-Image-Width',
    'height': 'X-Image-Height',
    'processing_ms': 'X-Processing-Time-Ms',
}


def mime_type(extension: str) -> str:
    """'.jpg' -> 'image/jpeg'."""
    extension = extension.lower()
    if not extension.startswith('.'):
        extension = '.' + extension
    return MIME_TYPES.get(extension, 'image/' + extension[1:])


def _construct_bytes_response(image: bytes, name: str, extension: str = '.png', info: dict = None) -> Response:
    headers = {'Content-Disposition': f'attachment; filename="{name}"'}
    for key, header in INFO_HEADERS.items():
        if info and info.get(key) is not None:
            headers[header] = str(info[key])
    return web.Response(body=image, content_type=mime_type(extension), headers=headers)


def _url_body(image_path: Path) -> dict:
    if image_path.is_absolute():
        logger.debug(f"Image path was absolute. Making it relative to CENSORED_PATH.")
        image_path = image_path.relative_to(CENSORED_PATH)
    name = image_path.as_posix()
    return {'image_name': name, 'url': '/censored/' + quote(name)}


def _base64_body(image: bytes, extension: str) -> dict:
    return {'image_data': base64.b64encode(image).decode(), 'mime_type': mime_type(extension)}


def construct_body(expected_response: str, image: bytes, extension: str, image_path: Path = None,
                   info: dict = None) -> dict:
    """The JSON answer for 'base64' or 'url': status, the image (or where to get it), then the info fields."""
    if expected_response == 'url':
        body = _url_body(image_path)
    elif expected_response == 'base64':
        body = _base64_body(image, extension=extension)
    else:
        raise ValueError(f"Unexpected response type: {expected_response}")
    return {'status': 'success', **body, **(info or {})}


def construct_response(expected_response: str, image: bytes|np.ndarray|Path, extension: str, image_path: Path=None,
                       name: str=None, info: dict=None) -> Response:
    """
    Returns the response in the expected format. Choices: 'bytes', 'url', or 'base64'.

    :param info: extra fields for the JSON answers (e.g. width, height, processing_ms, detections). The 'bytes'
        answer sends width, height and processing_ms as X-Image-Width, X-Image-Height and X-Processing-Time-Ms.
    """
    if isinstance(image, np.ndarray):
        image = utils.np_to_bytes(image, extension)
    if isinstance(image, Path):
        with open(image, 'rb') as f:
            image = f.read()

    if expected_response == 'bytes':
        return _construct_bytes_response(image, name=name, extension=extension, info=info)
    return web.json_response(construct_body(expected_response, image, extension, image_path, info))
