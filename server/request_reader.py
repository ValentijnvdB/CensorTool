from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import socket
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import unquote, urljoin, urlparse

import requests

import constants
from utils.hash_utils import hash_bytes

from server.server_config import UPLOAD_DIR
from server import storage
from .errors import (ApiError, DOWNLOAD_FAILED, INVALID_IMAGE, INVALID_REQUEST, URL_NOT_ALLOWED, USE_CENSOR_VIDEO,
                     GIF_MESSAGE)
from .helpers import check_censor_config

EXPECTED_RESPONSES = ('base64', 'bytes', 'url')
"""The formats /censor_image can answer in."""

DOWNLOAD_TIMEOUT = 5
MAX_REDIRECTS = 5
MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024
# some sites (e.g. Wikimedia) refuse requests' default client name
DOWNLOAD_HEADERS = {'User-Agent': 'CensorTool-server/1.0'}


@dataclass
class ImageRequest:
    """An image request for /censor_image, /detect_features or one item of /censor_images."""
    image_bytes: bytes
    image_path: Path                    # where the uploaded image is stored; its name and suffix are used in replies
    config: dict | None = None          # checked censor config, None for the profile's
    expected_response: str = 'base64'
    include_detections: bool = False


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


def image_extension(extension: str) -> str:
    """
    '.jpg', 'JPG', ... -> '.jpg'.

    :raises ApiError: (400) if it is not a supported image type.
    """
    extension = extension.lower()
    if not extension.startswith('.'):
        extension = '.' + extension
    if extension == '.gif':
        raise ApiError(GIF_MESSAGE, USE_CENSOR_VIDEO)
    if extension not in constants.IMAGE_EXT:
        raise ApiError(f"Invalid image extension: '{extension}'. Supported: {', '.join(constants.IMAGE_EXT)}.",
                       INVALID_IMAGE)
    return extension


def get_image_path_from_url(url: str, extension: str = None) -> Path:
    if extension is None:
        parsed_url = urlparse(url)
        extension = parsed_url.path.rsplit('.', 1)[-1]
    extension = image_extension(extension)
    filename = str(hash_bytes(url.encode('utf-8'), 32)) + extension
    return UPLOAD_DIR / filename


def get_image_from_base64(data: dict) -> tuple[bytes, Path]:
    """Convert the image from a base64 string and store at path"""
    if 'image_data' not in data:
        raise ApiError("No 'image_data' provided.")
    try:
        image = base64.b64decode(data['image_data'], validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise ApiError("'image_data' is not valid base64.", INVALID_IMAGE) from None

    mime_type = data.get('mime_type') or data.get('mimetype')
    extension = None
    if mime_type is not None:
        extension = str(mime_type).split('/')[-1]

    url = data.get('image_url')
    if url is None:
        # no URL to name the file after: name it after its contents
        if extension is None:
            raise ApiError("Provide 'mime_type' or an 'image_url' with an extension.")
        url = 'data:' + hash_bytes(image, 32)
    return image, get_image_path_from_url(str(url), extension)


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split('%', 1)[0])  # drop an IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global


def check_url_allowed(url: str, resolve: bool = True) -> None:
    """
    Only download from http(s) URLs on the internet: not from this machine (localhost, 127.0.0.1, [::1], 0.0.0.0,
    ...) or the local network.

    :param resolve: also look up the host name and check every address it resolves to.
    :raises ApiError: (400) if the URL is not allowed, (502) if the host cannot be resolved.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ApiError(f"Invalid URL: only http and https are supported, not '{parsed.scheme}'.",
                       URL_NOT_ALLOWED)
    host = parsed.hostname
    if not host:
        raise ApiError("Invalid URL: no host.", URL_NOT_ALLOWED)
    try:
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    except ValueError:
        raise ApiError("Invalid URL: invalid port.", URL_NOT_ALLOWED) from None

    not_allowed = ApiError(f"Invalid URL: '{host}' is this machine or on the local network.", URL_NOT_ALLOWED)
    if host.lower() == 'localhost' or host.lower().endswith('.localhost'):
        raise not_allowed
    try:
        if not _is_public(host):
            raise not_allowed
        return  # an IP address, nothing to resolve
    except ValueError:
        pass  # a host name

    if not resolve:
        return
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise ApiError(f"Could not download the image: unknown host '{host}'.", DOWNLOAD_FAILED, 502) from None
    if not infos or not all(_is_public(info[4][0]) for info in infos):
        raise not_allowed


def iter_download(url: str, what: str = 'image', max_size: int = MAX_DOWNLOAD_SIZE) -> Iterator[bytes]:
    """
    Download a file in chunks, following redirects only to allowed URLs.

    :param what: what is downloaded, for the error messages.
    :raises ApiError: if a URL is not allowed, the download fails or the file is larger than max_size.
    """
    for _ in range(MAX_REDIRECTS + 1):
        check_url_allowed(url)
        try:
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, allow_redirects=False, stream=True,
                                    headers=DOWNLOAD_HEADERS)
        except requests.RequestException as e:
            raise ApiError(f"Could not download the {what}: {e}", DOWNLOAD_FAILED, 502) from None

        with response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get('Location', ''))
                continue
            if response.status_code >= 400:
                raise ApiError(f"Could not download the {what}: the server answered {response.status_code}.",
                               DOWNLOAD_FAILED, 502)
            size = 0
            try:
                for chunk in response.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > max_size:
                        raise ApiError(f"Could not download the {what}: it is larger than "
                                       f"{max_size // (1024 * 1024)} MB.", DOWNLOAD_FAILED, 502)
                    yield chunk
            except requests.RequestException as e:
                raise ApiError(f"Could not download the {what}: {e}", DOWNLOAD_FAILED, 502) from None
            return

    raise ApiError(f"Could not download the {what}: too many redirects.", DOWNLOAD_FAILED, 502)


def download_image(url: str) -> bytes:
    """
    Download an image, following redirects only to allowed URLs.

    :raises ApiError: if a URL is not allowed or the download fails.
    """
    return b''.join(iter_download(url, 'image'))


def download_video(url: str, path: Path, digest=None, stop: threading.Event | None = None) -> None:
    """
    Download a video to path, following redirects only to allowed URLs.

    :param digest: a hashlib object that is updated with the contents.
    :param stop: when set, the download stops with an ApiError.
    :raises ApiError: if a URL is not allowed, the download fails or is stopped.
    """
    with open(path, 'wb') as f:
        for chunk in iter_download(url, 'video'):
            if stop is not None and stop.is_set():
                raise ApiError("The download was stopped.", DOWNLOAD_FAILED, 502)
            if digest is not None:
                digest.update(chunk)
            f.write(chunk)


def filename_from_url(url: str, default: str = 'video.mp4') -> str:
    """The file name at the end of a URL's path ('https://x.com/a/my%20clip.gif' -> 'my clip.gif'), or default."""
    name = unquote(PurePosixPath(urlparse(url).path).name)
    plain = name and name not in ('.', '..') and not any(c in name for c in '/\\\0')
    return name if plain and Path(name).suffix else default


def get_image_from_source(data: dict) -> tuple[bytes, Path]:
    """Get the image from a url source."""
    # get the url
    if 'image_url' not in data:
        raise ApiError("No 'image_url' provided.")
    url = str(data['image_url'])

    # refuse obvious local URLs straight away, also when the image is stored already
    check_url_allowed(url, resolve=False)

    # construct the write path
    path = get_image_path_from_url(url)

    # reuse the image if we already retrieved it (and it was not removed for being unused)
    try:
        image = path.read_bytes()
    except (FileNotFoundError, IsADirectoryError):
        return download_image(url), path
    storage.touch(path)
    return image, path


def get_image_from_json(as_json: dict) -> tuple[bytes, Path]:
    if not 'type' in as_json:
        raise ApiError("No message 'type' provided: use 'base64' or 'source'.")

    msg_type = as_json['type']

    if msg_type == 'base64':
        image, path = get_image_from_base64(as_json)

    elif msg_type == 'source':
        image, path = get_image_from_source(as_json)

    else:
        raise ApiError(f"Unknown image message type '{msg_type}': use 'base64' or 'source'.")

    with open(path, 'wb') as f:
        f.write(image)

    return image, path


def parse_image_request(data: Any, defaults: dict | None = None) -> ImageRequest:
    """
    Read an image request from its JSON. Downloads the image for a 'source' request.

    :param defaults: fields used when data leaves them out (the shared fields of /censor_images).
    :raises ApiError: if the request is not valid.
    """
    if not isinstance(data, dict):
        raise ApiError("The request must be a JSON object.")
    if defaults:
        data = {**defaults, **data}

    expected_response = data.get('expected_response', 'base64')
    if expected_response not in EXPECTED_RESPONSES:
        raise ApiError(f"Unknown expected_response '{expected_response}': use {', '.join(EXPECTED_RESPONSES)}.")

    include_detections = data.get('include_detections', False)
    if not isinstance(include_detections, bool):
        raise ApiError("'include_detections' must be true or false.")

    # check the config before downloading anything
    config = check_censor_config(data['config']) if data.get('config') is not None else None

    image_bytes, image_path = get_image_from_json(data)
    return ImageRequest(image_bytes, image_path, config, expected_response, include_detections)


async def read_json(request) -> Any:
    """
    The JSON body of a request.

    :raises ApiError: (400) if it is not a JSON request.
    """
    if request.content_type != 'application/json':
        raise ApiError("Only JSON requests (Content-Type: application/json) are supported.", INVALID_REQUEST)
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        raise ApiError("The request body is not valid JSON.", INVALID_REQUEST) from None


async def read_request(request) -> ImageRequest:
    """Read an image request; the image is downloaded in a thread, so the server keeps answering meanwhile."""
    return await asyncio.to_thread(parse_image_request, await read_json(request))
