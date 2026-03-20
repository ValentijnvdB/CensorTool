import asyncio
import ssl

import aiohttp_cors
from aiohttp import web

from loguru import logger

import constants

from . import request_reader as rr
from .helpers import submit_censoring_job, start_pipeline, stop_pipeline, submit_detection_job
from .response_constructor import construct_response
from .server_config import CACHE_DIR, CENSORED_PATH


async def censor_image(request):
    """Censor image and return the image."""
    try:
        image_bytes, image_path, censor_config, exp_response = await rr.read_request(request)

        output_path = None
        if exp_response == 'url':
            output_path = CENSORED_PATH / image_path.name
        result = submit_censoring_job(image_bytes, output_path, censor_config)

        if not result.success:
            logger.warning(f"Error censoring image: {str(result.error)}")
            return web.json_response({'error': str(result.error)}, status=500)

        return construct_response(expected_response=exp_response,
                                  image=result.image,
                                  extension=image_path.suffix,
                                  image_path=output_path,
                                  name=image_path.name)

    except Exception as e:
        logger.error(f"Error censoring image: {e}")
        return web.json_response({'error': str(e)}, status=500)



async def detect_features(request):
    """Run detection on the image and return the result."""
    try:
        image_bytes, image_path, censor_config, exp_response = await rr.read_request(request)

        output_path = None
        if exp_response == 'url':
            output_path = CENSORED_PATH / image_path.name
        result = submit_detection_job(image_bytes, output_path, censor_config)

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

    app.router.add_get('/reset_cache', reset_cache)

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
