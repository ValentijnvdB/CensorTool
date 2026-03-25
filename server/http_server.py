import asyncio
import hashlib
import json
import ssl
import time
import traceback
from collections import deque
from concurrent.futures import Future, wait, FIRST_COMPLETED
from pathlib import Path

import aiofiles
import aiohttp_cors
from aiohttp import web

from loguru import logger

import constants
from core import Job, construct_censor_config

from . import request_reader as rr
from .helpers import submit_censoring_job, start_pipeline, stop_pipeline, submit_detection_job, submit_gif_job, \
    submit_video_job
from .response_constructor import construct_response
from .server_config import CACHE_DIR, CENSORED_PATH, UPLOAD_DIR


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
            _, future = submit_censoring_job(image_bytes, image_path.suffix, output_path, censor_config)
            result: Job = future.result()

        if not result.success:
            logger.warning(f"Error censoring image: {str(result.error)}")
            return web.json_response({'error': str(result.error)}, status=500)

        if is_gif:
            image = result.result
        else:
            image = result.result.image

        return construct_response(expected_response=exp_response,
                                  image=image,
                                  extension=image_path.suffix,
                                  image_path=output_path,
                                  name=image_path.name)

    except Exception as e:
        logger.error(f"Error censoring image: {e}. Stacktrace: {traceback.format_exc()}")
        return web.json_response({'error': str(e)}, status=500)


async def censor_video(request):
    """Censor a video in one go."""
    reader = await request.multipart()
    config = None
    filename = 'video.mp4'
    path = UPLOAD_DIR / filename

    # Receive the video
    while True:
        part = await reader.next()
        if part is None: break
        if part.name == 'config':
            config = json.loads(await part.text())
        elif part.name == 'filename':
            filename = await part.text()
        elif part.name == 'video':
            path = UPLOAD_DIR / filename
            async with aiofiles.open(path, mode='wb') as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk: break
                    await f.write(chunk)

    # Censor the video
    output_path = CENSORED_PATH / filename
    if config is not None:
        config = construct_censor_config(config)
    result = await submit_video_job(video=path, output_path=output_path, censor_config=config, early_exit=False)
    if not result.success:
        logger.warning(f"Error censoring video: {str(result.error)}")
        return web.json_response({'error': str(result.error)}, status=500)

    output_path: Path = result.result

    # Stream the modified video back
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'video/mp4'}
    )
    await response.prepare(request)

    async with aiofiles.open(output_path, mode='rb') as f:
        while True:
            chunk = await f.read(1024 * 64)  # Read 64KB chunks
            if not chunk:
                break
            await response.write(chunk)

    return response


async def censor_frame(request):
    """Censor a video frame-by-frame. Used for real-time censoring."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    HEADER_SIZE = 13
    img_hashes: dict[str, str] = {}
    frames: dict[str, bytes] = {}

    headers: dict[str, bytes] = {}
    jobs: dict[str, Job] = {}
    futures: dict[str, Future] = {}

    frames_send = deque()

    def cancel_all():
        for k, f in futures.items():
            f.cancel()
            jobs[k].cancelled.set()
        futures.clear()
        jobs.clear()
        headers.clear()

    async def send_frame_back(job_id: str):
        frames_send.append(time.time())
        buffer = jobs[job_id].result.image
        header = headers[job_id]
        img_hash = img_hashes[job_id]
        frames[img_hash] = buffer
        await ws.send_bytes(header + buffer)
        del futures[job_id]
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

    async def receiver():
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
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

                job, future = submit_censoring_job(image_bytes, '.webp', None, None)
                futures[job.job_id] = future
                jobs[job.job_id] = job
                headers[job.job_id] = header_bytes
                img_hashes[job.job_id] = image_hash

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break

    async def sender():
        while not ws.closed:
            for job_id in list(futures.keys()):
                if futures[job_id].done():
                    try:
                        await send_frame_back(job_id)
                    except Exception as e:
                        logger.error(f"Error sending frame for job {job_id}: {e}")
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
        _, future = submit_detection_job(image_bytes, image_path.suffix, output_path, censor_config)
        result: Job = future.result()

        # Create JSON response with detected features
        response_data = {
            'status': 'success',
            'file_name': image_path.name,
            'features': []
        }

        raw_boxes = result.result.features[0]

        for box in raw_boxes:
            response_data['features'].append({
                'class': int(box.class_id),
                'label': box.label,
                'confidence': float(box.score),
                'bbox': [float(b) for b in box.polygon.bounds]
            })

        return web.json_response(response_data)

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
                    item.rmdir()

        return web.json_response({'status': 'success'})

    except Exception as e:
        logger.error(f"Error resetting cache: {e}")
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
