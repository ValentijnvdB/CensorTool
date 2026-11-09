from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import ssl
import time
import traceback
from collections import deque
from pathlib import Path

import aiofiles
import aiohttp_cors
from aiohttp import web

from loguru import logger

import constants
from core.backend import cc

from app.config import CensorConfigError, check_client_censor_config
from core.censor_config import censor_config_from_dict

from . import request_reader as rr
from .helpers import get_default_censor_config, get_labels, get_profile, submit_censoring_job, start_pipeline, stop_pipeline, submit_detection_job, submit_gif_job, \
    submit_video_job
from .response_constructor import construct_response
from .server_config import CACHE_DIR, CENSORED_PATH, UPLOAD_DIR
from .utils import np_to_bytes


async def censor_image(request):
    """Censor image and return the image."""
    try:
        image_bytes, image_path, censor_config, exp_response = await rr.read_request(request)

        output_path = None
        if exp_response == 'url':
            output_path = CENSORED_PATH / image_path.name

        is_gif = image_path.suffix == '.gif'

        if is_gif:
            result = await submit_gif_job(image_bytes, output_path, False, censor_config)
        else:
            _, future = submit_censoring_job(image_bytes, output_path, censor_config)
            result: cc.Job = future.wait()

        if not result.success:
            logger.warning(f"Error censoring image: {str(result.error)}")
            return web.json_response({'error': str(result.error)}, status=500)

        if is_gif:
            image = result.result
        else:
            image = result.result["image"]
            image = np_to_bytes(image, image_path.suffix)

        return construct_response(expected_response=exp_response,
                                  image=image,
                                  extension=image_path.suffix,
                                  image_path=output_path,
                                  name=image_path.name)

    except CensorConfigError as e:
        return web.json_response({'error': str(e)}, status=400)

    except Exception as e:
        logger.error(f"Error censoring image: {e}. Stacktrace: {traceback.format_exc()}")
        return web.json_response({'error': str(e)}, status=500)


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


async def censor_video(request):
    """Censor a video in one go."""
    reader = await request.multipart()
    config = None
    requested_format = None
    filename = 'video.mp4'
    path = UPLOAD_DIR / filename

    # Receive the video
    while True:
        part = await reader.next()
        if part is None: break
        if part.name == 'config':
            try:
                config = check_client_censor_config(json.loads(await part.text()))
            except CensorConfigError as e:
                return web.json_response({'error': str(e)}, status=400)
        elif part.name == 'filename':
            filename = await part.text()
            if not is_plain_filename(filename):
                return web.json_response({'error': f"Invalid filename '{filename}'."}, status=400)
        elif part.name == 'format':
            try:
                requested_format = parse_video_format(await part.text())
            except ValueError as e:
                return web.json_response({'error': str(e)}, status=400)
        elif part.name == 'video':
            path = UPLOAD_DIR / filename
            async with aiofiles.open(path, mode='wb') as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk: break
                    await f.write(chunk)

    # Censor the video
    video_format = output_video_format(filename, requested_format)
    output_path = CENSORED_PATH / Path(filename).with_suffix('.' + video_format).name
    result = await submit_video_job(video_path=path, output_path=output_path, censor_config=config, early_exit=False)
    if not result.success:
        logger.warning(f"Error censoring video: {str(result.error)}")
        return web.json_response({'error': str(result.error)}, status=500)

    output_path: Path = result.result

    # Stream the modified video back
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': VIDEO_FORMATS[video_format]}
    )
    await response.prepare(request)

    async with aiofiles.open(output_path, mode='rb') as f:
        while True:
            chunk = await f.read(1024 * 64)  # Read 64KB chunks
            if not chunk:
                break
            await response.write(chunk)

    return response


def read_config_message(text: str) -> dict | None:
    """
    Read a {"type": "config", "config": {...} | null} message sent over the /censor_frame WebSocket.

    :return: the checked censor config, or None to go back to the profile's censor config.
    :raises ValueError: if the message is not valid or the config is not allowed or not valid.
    """
    try:
        message = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Message is not valid JSON: {e}") from None
    if not isinstance(message, dict):
        raise ValueError("Message must be a JSON object.")
    if message.get('type') != 'config':
        raise ValueError(f"Unknown message type '{message.get('type')}'.")
    if 'config' not in message:
        raise ValueError("Message has no 'config'.")

    config = message['config']
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("'config' must be an object or null.")

    config = check_client_censor_config(config)
    try:
        censor_config_from_dict(config)  # same format for both backends; fails here instead of on every frame
    except Exception as e:
        raise ValueError(f"Invalid censor config: {e}") from None
    return config


async def censor_frame(request):
    """Censor a video frame-by-frame. Used for real-time censoring."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    HEADER_SIZE = 13
    EXTENSION = '.webp'

    img_hashes: dict[str, str] = {}
    frames: dict[str, bytes] = {}

    headers: dict[str, bytes] = {}
    jobs: dict[str, cc.Job] = {}
    futures: list[cc.JobHandle] = []

    frames_send = deque()

    # Censor config for this connection, set with a config text message. None is the profile's censor config.
    config: dict | None = None

    def cancel_all():
        pass
        # for k, f in futures.items():
        #     f.cancel()
        #     jobs[k].cancelled.set()
        # futures.clear()
        # jobs.clear()
        # headers.clear()

    async def send_frame_back(job: cc.Job):
        frames_send.append(time.time())
        job_id = job.job_id
        img = job.result["image"]
        buffer = np_to_bytes(img, EXTENSION)
        header = headers[job_id]
        img_hash = img_hashes[job_id]
        frames[img_hash] = buffer
        await ws.send_bytes(header + buffer)
        del jobs[job_id]
        del headers[job_id]
        del img_hashes[job_id]

    def update_fps():
        if len(frames_send) > 0:
            threshold = 10
            time_threshold = time.time() - threshold
            while frames_send and frames_send[0] < time_threshold:
                frames_send.popleft()
            fps = len(frames_send) / threshold if frames_send else 0
            logger.info(f"Current fps: {fps}. Active jobs: {len(futures)}")

    async def set_config(text: str):
        nonlocal config
        try:
            config = read_config_message(text)
        except (ValueError, CensorConfigError) as e:
            await ws.send_json({'type': 'error', 'error': str(e)})
            return
        # remembered frames were censored with the previous config
        frames.clear()
        logger.info("Censor config of the frame stream " + ("reset to the profile's." if config is None else "changed."))
        await ws.send_json({'type': 'config', 'status': 'success'})

    async def receiver():
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await set_config(msg.data)

            elif msg.type == web.WSMsgType.BINARY:
                header_bytes = msg.data[:HEADER_SIZE]

                if header_bytes[0] == 1:
                    logger.info("Cancelled all jobs")
                    cancel_all()
                    continue

                image_bytes = msg.data[HEADER_SIZE:]
                image_hash = hashlib.sha256(image_bytes).hexdigest()

                if image_hash in frames:
                    logger.info("Already seen this frame, returning cached")
                    frames_send.append(time.time())
                    await ws.send_bytes(header_bytes + frames[image_hash])
                    update_fps()
                    continue

                job, future = submit_censoring_job(image_bytes, None, config)
                futures.append(future)
                jobs[job.job_id] = job
                headers[job.job_id] = header_bytes
                img_hashes[job.job_id] = image_hash

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break

    async def sender():
        while not ws.closed:
            if not futures:
                await asyncio.sleep(0.005)
                continue

            done = cc.wait_any(futures)
            completed = done.wait()
            futures.remove(done)

            try:
                await send_frame_back(completed)
            except Exception as e:
                logger.error(f"Error sending frame for job {completed.job_id}: {e}")
            update_fps()
            await asyncio.sleep(0.005)

    await asyncio.gather(receiver(), sender())

    return ws


async def detect_features(request):
    """Run detection on the image and return the result."""
    try:
        image_bytes, image_path, censor_config, exp_response = await rr.read_request(request)

        output_path = None
        if exp_response == 'url':
            output_path = CENSORED_PATH / image_path.name
        _, future = submit_detection_job(image_bytes, output_path, censor_config)
        result: cc.Job = future.wait()

        if not result.success:
            logger.warning(f"Error detecting features: {str(result.error)}")
            return web.json_response({'error': str(result.error)}, status=500)

        detections = []
        for d in result.result["detections"]:
            detections.append({
                'id': int(d['id']),
                'label': d['label'],
                'score': float(d['score']),
                'box': [float(v) for v in d['box']],  # x, y, w, h
                'polygons': [[[float(x), float(y)] for x, y in poly] for poly in d['polygons']],
                'model': d['model'],
                'parent': d['parent'],
            })

        return web.json_response({
            'status': 'success',
            'file_name': image_path.name,
            'detections': detections,
            'cached_models': list(result.result.get('cached_models', [])),
        })

    except CensorConfigError as e:
        return web.json_response({'error': str(e)}, status=400)

    except Exception as e:
        logger.error(f"Error detecting features: {e}")
        return web.json_response({'error': str(e)}, status=500)


async def reset_cache(request):
    """Delete all items in the detection cache."""
    try:
        if CACHE_DIR.exists():
            for item in CACHE_DIR.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        return web.json_response({'status': 'success'})

    except Exception as e:
        logger.error(f"Error resetting cache: {e}")
        return web.json_response({'error': str(e)}, status=500)


async def profile(request):
    """The loaded profile and every label (class name) its models can produce."""
    try:
        label_models = get_labels()
        return web.json_response({
            'status': 'success',
            'profile': get_profile(),
            'labels': sorted(label_models),
            'models': dict(label_models),
        })

    except Exception as e:
        logger.error(f"Error reading profile: {e}")
        return web.json_response({'error': str(e)}, status=500)


async def censor_config(request):
    """The censor config used when a request does not provide one, as in the profile's censor_config.yml."""
    try:
        return web.json_response({
            'status': 'success',
            'profile': get_profile(),
            'config': get_default_censor_config(),
        })

    except Exception as e:
        logger.error(f"Error reading censor config: {e}")
        return web.json_response({'error': str(e)}, status=500)


async def init_app():
    """Initialize aiohttp application"""
    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB max

    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            # allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Add routes
    app.router.add_post('/censor_image', censor_image)
    app.router.add_post('/detect_features', detect_features)
    app.router.add_post('/censor_video', censor_video)

    app.router.add_get('/censor_frame', censor_frame)
    app.router.add_get('/reset_cache', reset_cache)
    app.router.add_get('/profile', profile)
    app.router.add_get('/censor_config', censor_config)

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
