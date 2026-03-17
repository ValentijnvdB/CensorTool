import base64
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger
from typing_extensions import deprecated

import constants
from app.config import load_config
from core import construct_censor_config, CensorConfig
from utils.hash_utils import hash_bytes

from server.server_config import UPLOAD_DIR


async def read_uploaded_image(reader) -> tuple[bytes, Path]:
    image_data = b''

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}.png"
    file_path = UPLOAD_DIR / unique_filename

    async for part in reader:
        if part.name == 'file':
            # Save file
            while True:
                chunk = await part.read_chunk()
                if not chunk:
                    break
                image_data += chunk

    return image_data, file_path


def get_image_path_from_url(url: str, extension: str = None) -> Path:
    if extension is None:
        parsed_url = urlparse(url)
        extension = parsed_url.path.rsplit('.', 1)[-1]
    if not extension.startswith('.'):
        extension = '.' + extension
    if extension not in constants.IMAGE_EXT:
        raise Exception(f"Invalid image extension: '{extension}'")
    filename = str(hash_bytes(url.encode('utf-8'), 32)) + extension
    return UPLOAD_DIR / filename


def get_image_from_base64(data: dict) -> tuple[bytes, Path]:
    """
    Convert the image from a base64 string and store at path
    """
    extension = data['mime_type'].split('/')[-1]
    path = get_image_path_from_url(data['image_url'], extension)

    base64_data = data['image_data']

    # Convert base64 string to bytes
    return base64.b64decode(base64_data), path


def get_image_from_source(data: dict) -> tuple[bytes, Path]:
    """
    Get the image from a url source.
    """
    # get the url
    if 'image_url' not in data:
        raise Exception("No image url provided")
    url = data['image_url']

    parsed_url = urlparse(url)

    # ignore request from this machine
    if 'localhost' in parsed_url.netloc or '127.0.0.1' in parsed_url.netloc:
        raise Exception("Invalid image url: localhost")

    # construct the write path
    path = get_image_path_from_url(url)

    # check if we already retrieved the image
    if path.exists() and path.is_file():
        with open(path, 'rb') as f:
            image = f.read()
        return image, path

    return requests.get(url, timeout=5).content, path


def get_image_from_json(as_json: dict) -> tuple[bytes, Path]:
    if not 'type' in as_json:
        raise Exception("No message type provided!")

    msg_type = as_json['type']

    if msg_type == 'base64':
        image, path = get_image_from_base64(as_json)

    elif msg_type == 'source':
        image, path = get_image_from_source(as_json)

    else:
        raise Exception("Unknown image message type!")

    with open(path, 'wb') as f:
        f.write(image)

    return image, path


async def read_request(request) -> tuple[bytes, Path, CensorConfig|None, str]:
    if request.content_type == 'application/json':
        as_json: dict = await request.json()
        image_bytes, image_path = get_image_from_json(as_json)

        expected_response = as_json.get('expected_response', 'base64')

        censor_config = None
        if 'config' in as_json:
            censor_config = as_json['config']
            logger.debug(censor_config)
            censor_config = construct_censor_config(censor_config)

        return image_bytes, image_path, censor_config, expected_response

    else:
        raise ValueError("Only json type is supported.")