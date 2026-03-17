import asyncio
import base64
import ssl
import traceback

import aiohttp_cors
from aiohttp import web

from loguru import logger

import constants

from . import request_reader as rr, utils
from .helpers import submit_censoring_job, submit_detection_job, start_pipeline, stop_pipeline



CENSORED_PATH = constants.data_root / 'server' / 'censored'


async def censor_image_bytes(request):
    """Censor image and return a stream"""
    try:
        image_bytes, image_path = await rr.read_request(request)

        output_dir = CENSORED_PATH / image_path.name
        result = submit_censoring_job(image_bytes, output_dir)

        if not result.success:
            logger.warning(f"Error handling base64 upload: {str(result.error)}")
            return web.json_response({'error': str(result.error)}, status=500)

        # if is_image:
        mod_image_np = result.image
        mod_image_bytes = utils.np_to_bytes(mod_image_np, image_path.suffix)

        # Return the image bytes as part of the response
        return web.Response(
            body=mod_image_bytes,
            content_type='image/png',
            headers={f'Content-Disposition': f'attachment; filename="{image_path.name}"'}
        )

    except Exception as e:
        print(f"Error handling file upload: {e}. Stacktrace: {traceback.format_exc()}")
        return web.json_response({'error': str(e)}, status=500)


async def censor_image_url(request):
    """Censor image and return a path"""
    logger.debug("Got url message")

    try:
        image_bytes, image_path = await rr.read_request(request)

        output_dir = CENSORED_PATH / image_path.name
        result = submit_censoring_job(image_bytes, output_dir)

        if not result.success:
            logger.warning(f"Error handling base64 upload: {str(result.error)}")
            return web.json_response({'error': str(result.error)}, status=500)

        body = {
            'image_name': str(image_path.relative_to(output_dir))
        }

        # Return the image bytes as part of the response
        return web.json_response(body)

    except Exception as e:
        print(f"Error handling file upload: {e}. Stacktrace: {traceback.format_exc()}")
        return web.json_response({'error': str(e)}, status=500)


async def censor_image_base64(request):
    """Censor image and return as base64"""
    logger.debug("Got base64 message")

    # todo: use provided config

    try:
        image_bytes, image_path = await rr.read_request(request)

        result = submit_censoring_job(image_bytes, None)

        if not result.success:
            logger.warning(f"Error handling base64 upload: {str(result.error)}")
            return web.json_response({'error': str(result.error)}, status=500)

        # if is_image:
        mod_image_np = result.image
        mod_image_bytes = utils.np_to_bytes(mod_image_np, image_path.suffix)
        base64_str = base64.b64encode(mod_image_bytes).decode()

        body = {
            'image_data': base64_str,
            'mime_type': f'image/{image_path.suffix}'
        }

        logger.debug("Responding with json")

        # Return the image bytes as part of the response
        return web.json_response(body)


    except Exception as e:
        print(f"Error handling file upload: {e}. Stacktrace: {traceback.format_exc()}")
        return web.json_response({'error': str(e)}, status=500)


async def detect_features(request):
    """Handle file upload via HTTP POST"""
    try:
        image_bytes, image_path = await rr.read_request(request)

        output_dir = CENSORED_PATH / image_path.name
        result = submit_detection_job(image_bytes, output_dir)

        # Create JSON response with detected features
        response_data = {
            'status': 'success',
            'file_name': image_path.name,
            'features': []
        }

        raw_boxes = result.features[0]

        for box in raw_boxes:
            response_data['features'].append({
                'class': int(box.class_id),
                'label': box.label,
                'confidence': float(box.score),
                'bbox': [float(b) for b in box.polygon.bounds]
            })

        return web.json_response(response_data)

    except Exception as e:
        print(f"Error handling file upload: {e}")
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
    app.router.add_post('/censor_image', censor_image_bytes)
    app.router.add_post('/censor_image_url', censor_image_url)
    app.router.add_post('/censor_image_base64', censor_image_base64)
    app.router.add_post('/detect_features', detect_features)

    app.router.add_static('/censored', path=CENSORED_PATH)
    app.router.add_static('/assets', path=str(constants.assets_path))

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    return app


def start_server(host: str = 'localhost', port: int = 8443, use_https: bool = False, cert_file: str = 'cert.pem', key_file: str = 'key.pem'):

    start_pipeline()

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
