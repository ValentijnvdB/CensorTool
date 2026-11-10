from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import ssl
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp_cors
import cv2
from aiohttp import web

from loguru import logger

import constants
from core.backend import cc

from app.config import CensorConfigError

from . import request_reader as rr
from .censor_config_schema import censor_config_schema
from .errors import (ApiError, CONFIG_NOT_ALLOWED, INTERNAL_ERROR, INVALID_CONFIG, INVALID_REQUEST, INVALID_VIDEO,
                     NOT_FOUND, PROCESSING_FAILED, error_body, error_response)
from .helpers import (activity, check_censor_config, clear_cache, get_cache_info, get_default_censor_config,
                      get_health, get_labels, get_profile, get_sticker_file, get_stickers, inspect_censor_config,
                      start_pipeline, stop_pipeline, submit_censoring_job, submit_detection_job, submit_video_job,
                      unique_path, wait_for_job)
from .response_constructor import construct_body, construct_response
from .server_config import CACHE_DIR, CENSORED_PATH, UPLOAD_DIR
from .utils import np_to_bytes


def api_handler(handler):
    """Answers ApiErrors with their status and code, and anything unexpected with a 500 internal_error."""
    @functools.wraps(handler)
    async def wrapper(request):
        try:
            return await handler(request)
        except ApiError as e:
            if e.status >= 500:
                logger.warning(f"{handler.__name__}: {e}")
            return e.to_response()
        except CensorConfigError as e:
            return error_response(str(e), CONFIG_NOT_ALLOWED, 400)
        except Exception as e:
            logger.error(f"Error in {handler.__name__}: {e}. Stacktrace: {traceback.format_exc()}")
            return error_response(str(e) or type(e).__name__, INTERNAL_ERROR, 500)
    return wrapper


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


# -------------------------------------- detections --------------------------------------

def serialize_detections(detections) -> list[dict]:
    """The detections of a job result as plain JSON."""
    return [{
        'id': int(d['id']),
        'label': d['label'],
        'score': float(d['score']),
        'box': [float(v) for v in d['box']],  # x, y, w, h
        'polygons': [[[float(x), float(y)] for x, y in poly] for poly in d['polygons']],
        'model': d['model'],
        'parent': d['parent'],
    } for d in detections]


def image_size(image) -> tuple[int, int]:
    """(width, height) of a decoded image."""
    height, width = image.shape[:2]
    return int(width), int(height)


@dataclass
class VideoInfo:
    frames: int
    fps: float
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0


def read_video_info(path: Path) -> VideoInfo:
    """Frame count, frame rate and size of a video (or GIF); zeros for what cannot be read."""
    cap = cv2.VideoCapture(str(path))
    try:
        return VideoInfo(frames=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                         fps=max(0.0, float(cap.get(cv2.CAP_PROP_FPS))),
                         width=max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                         height=max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    finally:
        cap.release()


# -------------------------------------- images --------------------------------------

@dataclass
class CensoredImage:
    image: bytes                        # encoded like the uploaded image
    width: int
    height: int
    detections: list[dict] | None       # only when asked for
    cached_models: list[str] | None


async def censor_request(req: rr.ImageRequest) -> CensoredImage:
    """
    Censor the image of a request. For expected_response 'url' the image is saved in CENSORED_PATH before this
    returns, so it can be fetched straight away.

    :raises ApiError: if censoring fails.
    """
    extension = req.image_path.suffix
    url_path = CENSORED_PATH / req.image_path.name if req.expected_response == 'url' else None

    _, handle = submit_censoring_job(req.image_bytes, None, req.config)
    job: cc.Job = await wait_for_job(handle)
    if not job.success:
        raise ApiError(f"Censoring failed: {job.error}", PROCESSING_FAILED, 500)

    censored = job.result['image']
    image = await asyncio.to_thread(np_to_bytes, censored, extension)
    if url_path is not None:
        await asyncio.to_thread(url_path.write_bytes, image)

    width, height = image_size(censored)
    detections = cached = None
    if req.include_detections:
        detections = serialize_detections(job.result['detections'])
        cached = list(job.result.get('cached_models', []))
    return CensoredImage(image, width, height, detections, cached)


def image_info(censored: CensoredImage, req: rr.ImageRequest, started: float) -> dict:
    """The fields of a censor answer besides the image."""
    info = {'width': censored.width, 'height': censored.height, 'processing_ms': elapsed_ms(started)}
    if req.include_detections:
        info['detections'] = censored.detections
        info['cached_models'] = censored.cached_models
    return info


@api_handler
async def censor_image(request):
    """Censor image and return the image."""
    started = time.perf_counter()
    req = await rr.read_request(request)
    if req.expected_response == 'bytes' and req.include_detections:
        raise ApiError("include_detections needs expected_response 'base64' or 'url'.")

    censored = await censor_request(req)
    return construct_response(expected_response=req.expected_response,
                              image=censored.image,
                              extension=req.image_path.suffix,
                              image_path=CENSORED_PATH / req.image_path.name,
                              name=req.image_path.name,
                              info=image_info(censored, req, started))


MAX_BATCH_IMAGES = 50
BATCH_SHARED_FIELDS = ('config', 'expected_response', 'include_detections')


async def _censor_batch_item(item: Any, shared: dict) -> dict:
    """One answer of /censor_images: the censor answer, or an error with its code."""
    started = time.perf_counter()
    try:
        req = await asyncio.to_thread(rr.parse_image_request, item, shared)
        if req.expected_response == 'bytes':
            raise ApiError("expected_response 'bytes' is not supported by /censor_images: use 'base64' or 'url'.")
        censored = await censor_request(req)
        return construct_body(req.expected_response, censored.image, req.image_path.suffix,
                              CENSORED_PATH / req.image_path.name, image_info(censored, req, started))
    except ApiError as e:
        return e.to_dict()
    except CensorConfigError as e:
        return error_body(str(e), CONFIG_NOT_ALLOWED)
    except Exception as e:
        logger.error(f"Error censoring a batch image: {e}. Stacktrace: {traceback.format_exc()}")
        return error_body(str(e) or type(e).__name__, INTERNAL_ERROR)


@api_handler
async def censor_images(request):
    """Censor several images at once. Every image gets its own answer, in order; one failing does not fail the rest."""
    started = time.perf_counter()
    data = await rr.read_json(request)
    if not isinstance(data, dict) or not isinstance(data.get('images'), list):
        raise ApiError("The request must be a JSON object with an 'images' list.")
    images = data['images']
    if not images:
        raise ApiError("'images' is empty.")
    if len(images) > MAX_BATCH_IMAGES:
        raise ApiError(f"Too many images: at most {MAX_BATCH_IMAGES} per request.")

    shared = {key: data[key] for key in BATCH_SHARED_FIELDS if key in data}
    if shared.get('expected_response') == 'bytes':
        raise ApiError("expected_response 'bytes' is not supported by /censor_images: use 'base64' or 'url'.")
    if shared.get('config') is not None:
        shared['config'] = check_censor_config(shared['config'])  # a bad shared config fails the whole request

    results = await asyncio.gather(*(_censor_batch_item(item, shared) for item in images))
    return web.json_response({'status': 'success', 'processing_ms': elapsed_ms(started), 'results': results})


@api_handler
async def detect_features(request):
    """Run detection on the image and return the result."""
    req = await rr.read_request(request)
    _, handle = submit_detection_job(req.image_bytes, None, req.config)
    job: cc.Job = await wait_for_job(handle)

    if not job.success:
        raise ApiError(f"Detection failed: {job.error}", PROCESSING_FAILED, 500)

    width, height = image_size(job.result['image'])
    return web.json_response({
        'status': 'success',
        'file_name': req.image_path.name,
        'width': width,
        'height': height,
        'detections': serialize_detections(job.result['detections']),
        'cached_models': list(job.result.get('cached_models', [])),
    })


# -------------------------------------- video --------------------------------------

def is_plain_filename(name: str) -> bool:
    """Whether name is just a file name: no folders (so no '..' or absolute paths), and not '.' or '..'."""
    return bool(name) and name not in ('.', '..') and '/' not in name and '\\' not in name and '\0' not in name


VIDEO_FORMATS = {
    'mp4': 'video/mp4',
    'm4v': 'video/mp4',
    'mov': 'video/quicktime',
    'mkv': 'video/x-matroska',
    'webm': 'video/webm',
    'avi': 'video/x-msvideo',
    'gif': 'image/gif',
}
"""Formats /censor_video can return, with their Content-Type."""
DEFAULT_VIDEO_FORMAT = 'mp4'
DEFAULT_VIDEO_NAME = 'video.mp4'


def parse_video_format(text: str) -> str:
    """
    The format asked for in the 'format' part of /censor_video, e.g. 'webm' or '.WebM' -> 'webm'.

    :raises ValueError: if the format is not supported.
    """
    fmt = text.strip().lower().removeprefix('.')
    if fmt not in VIDEO_FORMATS:
        raise ValueError(f"Unsupported format '{text}'. Supported: {', '.join(VIDEO_FORMATS)}.")
    return fmt


def output_video_format(filename: str, requested: str | None) -> str:
    """The requested format, else the format of the uploaded file, else mp4 when that format is not supported."""
    if requested is not None:
        return requested
    fmt = Path(filename).suffix.lower().removeprefix('.')
    return fmt if fmt in VIDEO_FORMATS else DEFAULT_VIDEO_FORMAT


@api_handler
async def censor_video(request):
    """Censor a video in one go."""
    started = time.perf_counter()
    reader = await request.multipart()
    config = None
    requested_format = None
    filename: str | None = None
    video_url: str | None = None
    path: Path | None = None
    digest = hashlib.sha256()

    try:
        # Receive the video, or the URL to download it from
        while True:
            part = await reader.next()
            if part is None: break
            if part.name == 'config':
                try:
                    config = json.loads(await part.text())
                except json.JSONDecodeError as e:
                    raise ApiError(f"'config' is not valid JSON: {e}", INVALID_CONFIG) from None
                config = check_censor_config(config)
            elif part.name == 'filename':
                filename = await part.text()
                if not is_plain_filename(filename):
                    raise ApiError(f"Invalid filename '{filename}'.")
            elif part.name == 'format':
                try:
                    requested_format = parse_video_format(await part.text())
                except ValueError as e:
                    raise ApiError(str(e)) from None
            elif part.name == 'video_url':
                video_url = (await part.text()).strip()
                rr.check_url_allowed(video_url, resolve=False)  # fail early; checked fully when downloading
            elif part.name == 'video':
                if path is not None:
                    raise ApiError("Send one 'video' part.")
                path = unique_path(UPLOAD_DIR, filename or DEFAULT_VIDEO_NAME)
                async with aiofiles.open(path, mode='wb') as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk: break
                        digest.update(chunk)
                        await f.write(chunk)

        if path is not None and video_url is not None:
            raise ApiError("Send either a 'video' or a 'video_url' part, not both.")
        if video_url is not None:
            # parts may come in any order, so the URL is downloaded once all of them are read
            filename = filename or rr.filename_from_url(video_url, DEFAULT_VIDEO_NAME)
            path = unique_path(UPLOAD_DIR, filename)
            await asyncio.to_thread(rr.download_video, video_url, path, digest)
        if path is None:
            raise ApiError("No video: send the video file as the 'video' part, or its URL as 'video_url'.")
        if path.stat().st_size == 0:
            raise ApiError("The video is empty.")
        info = await asyncio.to_thread(read_video_info, path)
        if info.width == 0 or info.height == 0:
            raise ApiError("The video could not be read: it is not a supported video.", INVALID_VIDEO)
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise

    # Censor the video
    video_format = output_video_format(filename or DEFAULT_VIDEO_NAME, requested_format)
    output_path = unique_path(CENSORED_PATH, path.with_suffix('.' + video_format).name)
    # cached per content, so a different video uploaded under a used name never gets the old detections
    cache_dir = CACHE_DIR / 'videos' / digest.hexdigest()[:32]
    result = await submit_video_job(video_path=path, output_path=output_path, censor_config=config, early_exit=False,
                                    cache_dir=cache_dir)
    if not result.success:
        output_path.unlink(missing_ok=True)
        raise ApiError(f"Censoring failed: {result.error}", PROCESSING_FAILED, 500)

    output_path = Path(result.result)

    # Stream the modified video back
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': VIDEO_FORMATS[video_format],
            'Content-Disposition': f'attachment; filename="{output_path.name}"',
            'X-Processing-Time-Ms': str(elapsed_ms(started)),
            'X-Frame-Count': str(info.frames),
            'X-Video-Fps': f"{info.fps:.3f}",
            'X-Video-Duration': f"{info.duration:.3f}",
            'X-Video-Width': str(info.width),
            'X-Video-Height': str(info.height),
        }
    )
    await response.prepare(request)

    async with aiofiles.open(output_path, mode='rb') as f:
        while True:
            chunk = await f.read(1024 * 64)  # Read 64KB chunks
            if not chunk:
                break
            await response.write(chunk)

    return response


@api_handler
async def video_formats(request):
    """The formats /censor_video can return."""
    return web.json_response({
        'status': 'success',
        'default': DEFAULT_VIDEO_FORMAT,
        'formats': [{'format': fmt, 'content_type': content_type} for fmt, content_type in VIDEO_FORMATS.items()],
    })


# -------------------------------------- frame stream --------------------------------------

def _config_from_message(message: dict) -> dict | None:
    """The checked config of a config message, or None to go back to the profile's censor config."""
    if 'config' not in message:
        raise ValueError("Message has no 'config'.")
    config = message['config']
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("'config' must be an object or null.")
    try:
        return check_censor_config(config)
    except ApiError as e:
        if e.code == CONFIG_NOT_ALLOWED:
            raise CensorConfigError(str(e)) from None
        raise ValueError(str(e)) from None


def parse_message(text: str) -> dict:
    """
    A text message sent over the /censor_frame WebSocket: a JSON object with a 'type'.

    :raises ValueError: if it is not.
    """
    try:
        message = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Message is not valid JSON: {e}") from None
    if not isinstance(message, dict):
        raise ValueError("Message must be a JSON object.")
    return message


def read_config_message(text: str) -> dict | None:
    """
    Read a {"type": "config", "config": {...} | null} message sent over the /censor_frame WebSocket.

    :return: the checked censor config, or None to go back to the profile's censor config.
    :raises ValueError: if the message is not valid or the config is not valid.
    :raises CensorConfigError: if the config is not allowed.
    """
    message = parse_message(text)
    if message.get('type') != 'config':
        raise ValueError(f"Unknown message type '{message.get('type')}'.")
    return _config_from_message(message)


@dataclass(eq=False)
class _PendingFrame:
    job: Any
    handle: Any
    header: bytes
    image_hash: str


class FrameStream:
    """One /censor_frame connection."""

    HEADER_SIZE = 13
    EXTENSION = '.webp'
    FPS_WINDOW = 10  # seconds

    def __init__(self, ws: web.WebSocketResponse):
        self.ws = ws
        self.config: dict | None = None             # None is the profile's censor config
        self.include_detections = False
        self.stopped = False
        self.pending: list[_PendingFrame] = []
        # censored frames by the hash of the received frame: (encoded frame, detections, width, height)
        self.remembered: dict[str, tuple[bytes, list, int, int]] = {}
        self.sent_times: deque[float] = deque()
        self.counts = {'frames_received': 0, 'frames_sent': 0, 'frames_from_memory': 0, 'frames_failed': 0,
                       'frames_cancelled': 0}

    # ---------------- sending ----------------
    async def send_error(self, message: str, code: str = INVALID_REQUEST, header: bytes | None = None):
        body = {'type': 'error', 'code': code, 'error': message}
        if header is not None:
            body['header'] = header.hex()
        await self.ws.send_json(body)

    async def send_frame(self, header: bytes, frame: bytes, detections: list, width: int, height: int):
        if self.include_detections:
            # before the frame, so a client has both when the frame arrives
            await self.ws.send_json({'type': 'detections', 'header': header.hex(), 'width': width, 'height': height,
                                     'detections': serialize_detections(detections)})
        await self.ws.send_bytes(header + frame)
        self.counts['frames_sent'] += 1
        self.sent_times.append(time.time())

    def fps(self) -> float:
        threshold = time.time() - self.FPS_WINDOW
        while self.sent_times and self.sent_times[0] < threshold:
            self.sent_times.popleft()
        return len(self.sent_times) / self.FPS_WINDOW

    def stats(self) -> dict:
        return {'type': 'stats', 'fps': round(self.fps(), 2), 'active_jobs': len(self.pending),
                'remembered_frames': len(self.remembered), **self.counts}

    # ---------------- receiving ----------------
    async def on_text(self, text: str):
        try:
            message = parse_message(text)
        except ValueError as e:
            return await self.send_error(str(e))

        kind = message.get('type')
        if kind == 'config':
            try:
                self.config = _config_from_message(message)
            except CensorConfigError as e:
                return await self.send_error(str(e), CONFIG_NOT_ALLOWED)
            except ValueError as e:
                return await self.send_error(str(e), INVALID_CONFIG)
            # remembered frames were censored with the previous config
            self.remembered.clear()
            logger.info("Censor config of the frame stream " +
                        ("reset to the profile's." if self.config is None else "changed."))
            await self.ws.send_json({'type': 'config', 'status': 'success'})

        elif kind == 'options':
            include = message.get('include_detections', self.include_detections)
            if not isinstance(include, bool):
                return await self.send_error("'include_detections' must be true or false.")
            self.include_detections = include
            await self.ws.send_json({'type': 'options', 'status': 'success', 'include_detections': include})

        elif kind == 'stats':
            await self.ws.send_json(self.stats())

        else:
            await self.send_error(f"Unknown message type '{kind}'.")

    def cancel_all(self):
        """Cancel every frame in progress; they are not sent back."""
        for frame in self.pending:
            frame.job.cancel()
        self.counts['frames_cancelled'] += len(self.pending)
        logger.info(f"Cancelled {len(self.pending)} frame(s)")
        self.pending.clear()

    async def on_frame(self, data: bytes):
        if len(data) <= self.HEADER_SIZE:
            if data[:1] == b'\x01':
                return self.cancel_all()
            return await self.send_error(f"A frame needs a {self.HEADER_SIZE}-byte header and an image.")

        header = data[:self.HEADER_SIZE]
        if header[0] == 1:
            return self.cancel_all()

        self.counts['frames_received'] += 1
        image_bytes = data[self.HEADER_SIZE:]
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        if image_hash in self.remembered:
            self.counts['frames_from_memory'] += 1
            return await self.send_frame(header, *self.remembered[image_hash])

        try:
            job, handle = submit_censoring_job(image_bytes, None, self.config)
        except ApiError as e:
            self.counts['frames_failed'] += 1
            return await self.send_error(str(e), e.code, header)
        self.pending.append(_PendingFrame(job, handle, header, image_hash))

    async def receiver(self):
        async for msg in self.ws:
            if msg.type == web.WSMsgType.TEXT:
                await self.on_text(msg.data)
            elif msg.type == web.WSMsgType.BINARY:
                await self.on_frame(msg.data)
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break

    async def finish(self, frame: _PendingFrame):
        job = frame.handle.wait()  # done already
        if not job.success:
            self.counts['frames_failed'] += 1
            logger.warning(f"Error censoring frame: {job.error}")
            return await self.send_error(f"Censoring failed: {job.error}", PROCESSING_FAILED, frame.header)

        censored = job.result['image']
        buffer = await asyncio.to_thread(np_to_bytes, censored, self.EXTENSION)
        width, height = image_size(censored) if hasattr(censored, 'shape') else (0, 0)
        remembered = (buffer, job.result.get('detections', []), width, height)
        self.remembered[frame.image_hash] = remembered
        await self.send_frame(frame.header, *remembered)

    async def sender(self):
        while not self.ws.closed and not self.stopped:
            done = [frame for frame in self.pending if frame.handle.done()]
            if not done:
                await asyncio.sleep(0.005)
                continue
            for frame in done:
                if frame not in self.pending:  # cancelled meanwhile
                    continue
                self.pending.remove(frame)
                try:
                    await self.finish(frame)
                except Exception as e:
                    logger.error(f"Error sending frame: {e}")
            logger.debug(f"Current fps: {self.fps()}. Active jobs: {len(self.pending)}")

    async def run(self):
        receiver = asyncio.create_task(self.receiver())
        sender = asyncio.create_task(self.sender())
        try:
            await receiver
        finally:
            self.stopped = True
            self.cancel_all()  # nobody to send them to
            await sender


async def censor_frame(request):
    """Censor a video frame-by-frame. Used for real-time censoring."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    activity.open_frame_streams += 1
    try:
        await FrameStream(ws).run()
    finally:
        activity.open_frame_streams -= 1
    return ws


# -------------------------------------- info --------------------------------------

@api_handler
async def health(request):
    """Whether the server can censor, and what it is busy with."""
    body = get_health()
    return web.json_response(body, status=200 if body['pipeline_running'] else 503)


@api_handler
async def cache_info(request):
    """Size of the detection cache."""
    return web.json_response({'status': 'success', **await asyncio.to_thread(get_cache_info)})


@api_handler
async def delete_cache(request):
    """Delete all items in the detection cache."""
    return web.json_response({'status': 'success', **await asyncio.to_thread(clear_cache)})


@api_handler
async def profile(request):
    """The loaded profile and every label (class name) its models can produce."""
    label_models = get_labels()
    return web.json_response({
        'status': 'success',
        'profile': get_profile(),
        'labels': sorted(label_models),
        'models': dict(label_models),
    })


@api_handler
async def censor_config(request):
    """The censor config used when a request does not provide one, as in the profile's censor_config.yml."""
    return web.json_response({
        'status': 'success',
        'profile': get_profile(),
        'config': get_default_censor_config(),
    })


@api_handler
async def censor_config_schema_handler(request):
    """A JSON Schema of the censor config."""
    return web.json_response(censor_config_schema())


@api_handler
async def validate_censor_config(request):
    """Check a censor config without censoring anything."""
    data = await rr.read_json(request)
    if not isinstance(data, dict) or 'config' not in data:
        raise ApiError("The request must be a JSON object with a 'config'.")
    result = inspect_censor_config(data['config'])
    body = {'status': 'success', 'valid': result.valid, 'errors': result.errors, 'warnings': result.warnings}
    if not result.valid:
        body['code'] = result.code
    return web.json_response(body)


@api_handler
async def stickers(request):
    """The sticker categories and images in data/stickers, or in the folder given with ?dir=."""
    return web.json_response({'status': 'success', **get_stickers(request.query.get('dir'))})


async def sticker_file(request):
    """One sticker image, by its path relative to data/stickers."""
    try:
        return web.FileResponse(get_sticker_file(request.match_info['path']))
    except FileNotFoundError as e:
        return error_response(str(e), NOT_FOUND, 404)


async def init_app():
    """Initialize aiohttp application"""
    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB max

    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            # allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
    })

    # Add routes
    app.router.add_post('/censor_image', censor_image)
    app.router.add_post('/censor_images', censor_images)
    app.router.add_post('/detect_features', detect_features)
    app.router.add_post('/censor_video', censor_video)
    app.router.add_get('/video_formats', video_formats)

    app.router.add_get('/censor_frame', censor_frame)

    app.router.add_get('/health', health)
    app.router.add_get('/cache', cache_info)
    app.router.add_delete('/cache', delete_cache)
    app.router.add_get('/profile', profile)
    app.router.add_get('/censor_config', censor_config)
    app.router.add_get('/censor_config/schema', censor_config_schema_handler)
    app.router.add_post('/censor_config/validate', validate_censor_config)
    app.router.add_get('/stickers', stickers)
    app.router.add_get('/stickers/{path:.+}', sticker_file)

    app.router.add_static('/censored', path=CENSORED_PATH)
    app.router.add_static('/assets', path=str(constants.assets_path))

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    return app


def start_server(host: str = 'localhost', port: int = 8443, use_https: bool = False, cert_file: str = 'cert.pem', key_file: str = 'key.pem', debug: bool = False):

    start_pipeline(debug=debug)

    try:
        asyncio.run(run(host, port, use_https, cert_file, key_file))
    except KeyboardInterrupt:
        print("Closing server...")

    stop_pipeline()

async def run(host, port, use_https, cert_file, key_file):

    # Start HTTP(S) server
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()

    ssl_context = None
    if use_https:
        # Create SSL context
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(cert_file, key_file)

    site = web.TCPSite(runner, host=host, port=port, ssl_context=ssl_context)

    await site.start()

    print(f"HTTPS Server started on http{'s' if use_https else ''}://{host}:{port}")
    print(f"Using certificate: {cert_file}")
    print(f"Using key: {key_file}")
    print("Press Ctrl+C to stop the server")

    await asyncio.Event().wait()
