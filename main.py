import argparse
import sys
import time
from os import makedirs
from pathlib import Path

from loguru import logger, _defaults

import constants
from app import init_app


def format_time(total_ms: float) -> str:
    # Convert total_ms to hours, minutes, seconds, and milliseconds
    hours = int(total_ms // 3600000)
    remaining_ms = total_ms % 3600000
    minutes = int(remaining_ms // 60000)
    remaining_ms = remaining_ms % 60000
    seconds = int(remaining_ms // 1000)
    milliseconds = int(remaining_ms % 1000)

    time_str = ""
    if hours > 0:
        time_str += f"{hours} hour{'s' if hours != 1 else ''}, "
    if minutes > 0:
        time_str += f"{minutes} min, "
    if seconds > 0 or not time_str:
        time_str += f"{seconds} sec, "
    if milliseconds > 0 or not time_str:
        time_str += f"{milliseconds} ms"

    # Remove trailing comma and space if any
    if time_str.endswith(', '):
        time_str = time_str[:-2]
    return time_str


def parse_input_path(input_path: Path, extension_filter: list[str]) -> list[Path]:
    """
    Checks the input_path, if it is a file return a list with that file.
    Otherwise, return all files with the correct extension.
    """
    logger.info(f"Using input path: '{input_path}'")

    if input_path is None:
        raise ValueError("Input path is required")
    if not input_path.exists():
        raise ValueError("Input path does not exist")

    if input_path.is_dir():
        input_files = [f for f in input_path.glob('**/*') if f.suffix.lower() in extension_filter]
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path '{input_path}' does not exist")
        if input_path.suffix not in extension_filter:
            raise ValueError(f"Input path has an incorrect extension: '{input_path.suffix}'. Expected: {extension_filter}")
        return [input_path]

    if len(input_files) == 0:
        raise ValueError("No input files")

    logger.info(f"Found {len(input_files)} input files.")
    return input_files


def start_image_censor(input_path: Path, output_path: Path, **kwargs):
    from app.image import censor_images
    logger.info("Censoring images")

    try:
        input_files = parse_input_path(input_path, constants.IMAGE_EXT)
    except Exception as e:
        logger.error(e)
        return

    start_time = time.perf_counter_ns()
    censor_images(input_files, output_path, **kwargs)
    stop_time = time.perf_counter_ns()

    total_ms = (stop_time - start_time) / 1000000
    time_str = format_time(total_ms)

    logger.info("Censored all images")
    logger.info(f"Took {time_str} to censor images.")


def start_video_censor(input_path: Path, output_path: Path, **kwargs):
    from app.video import censor_videos
    logger.info("Censoring videos")
    try:
        input_files = parse_input_path(input_path, constants.VID_EXT)
    except Exception as e:
        logger.error(e)
        return

    start_time = time.perf_counter_ns()
    censor_videos(input_files, output_path, **kwargs)
    stop_time = time.perf_counter_ns()

    total_ms = (stop_time - start_time) / 1000000
    time_str = format_time(total_ms)

    logger.info("Censored all videos")
    logger.info(f"Took {time_str} to censor videos.")


def censor_all(*args, **kwargs):
    start_image_censor(*args, **kwargs)
    start_video_censor(*args, **kwargs)


def create_dirs():
    makedirs(constants.video_cache_path, exist_ok=True)
    makedirs(constants.image_cache_path, exist_ok=True)
    makedirs(constants.trash_bin_cache_path, exist_ok=True)

    makedirs(constants.censored_path, exist_ok=True)
    makedirs(constants.debug_path, exist_ok=True)
    makedirs(constants.stickers_root_path, exist_ok=True)
    makedirs(constants.uncensored_path, exist_ok=True)
    makedirs(constants.configs_path, exist_ok=True)

    makedirs(constants.model_root, exist_ok=True)


def start_live_censor(mode: str, device_id: int, use_vcam: bool, vcam_width: int, vcam_height, vcam_fps:int):
    from app.live import start_live_censor
    start_live_censor(mode, device_id, use_vcam, vcam_width, vcam_height, vcam_fps)


def serve_http(host, port, ssl_file, ssl_key):
    from server import http_server
    http_server.start_server(host=host, port=port, use_https=True, cert_file=ssl_file, key_file=ssl_key)


def get_mode_overview():
    return (
        "'image': censor images."
        "'video': censor videos. "
        "'all': censor images and videos. "
        "'live': real time censoring. "
        "'webcam': real time webcam censoring. "
        "'http: start the http server."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default='http', help=f"The mode to use. {get_mode_overview()}")
    parser.add_argument('-i', '--input', default=constants.uncensored_path, help='Path to input file or directory.')
    parser.add_argument('-o', '--output', default=constants.censored_path, help='Path to output directory.')

    parser.add_argument('--config', default='./config.yml', help='Path to the main config file.')
    parser.add_argument('-cc', '--censor-config',
                        default='./default_censor_config.yml', help='Path to the censor config file.')

    # live & webcam mode
    lw_parser = parser.add_argument_group(title='Live & Webcam arguments')
    lw_parser.add_argument('--live-mode', default='quick', type=str, help='Live mode (quick or precise)')
    lw_parser.add_argument('--device', default=-1, type=int, help='Device to use (only used for live webcam censor).')
    lw_parser.add_argument('--vcam', action='store_true', help='Whether to output to a virtual camera or not.')
    lw_parser.add_argument('--vcam-w', default=1920, type=int, help='Virtual camera width.')
    lw_parser.add_argument('--vcam-h', default=1080, type=int, help='Virtual camera height.')
    lw_parser.add_argument('--vcam-fps', default=10, type=int, help='Virtual camera target fps.')

    # flags
    flags_parser = parser.add_argument_group(title='General flags')
    flags_parser.add_argument('--override-cache', action='store_true', help='Recompute features by overriding the cache.')
    flags_parser.add_argument('--skip-existing', action='store_true',
                        help='Whether to skip files that already exist in the output directory.')
    flags_parser.add_argument('--only-analyze', action='store_true',
                        help='Whether to only analyze the images and videos (does not apply censoring).')
    flags_parser.add_argument('--debug', action='store_true', help='Whether to enable debug mode.')

    # server arguments
    server_parser = parser.add_argument_group(title='Server arguments')
    server_parser.add_argument('--port', default=8443, type=int, help='Port to run the server on.')
    server_parser.add_argument('--host', default='localhost', type=str, help='Host to run the server on.')
    server_parser.add_argument('--ssl-cert', default='cert.pem', type=str, help='Path to certificate file.')
    server_parser.add_argument('--ssl-key', default='key.pem', type=str, help='Path to key file.')

    args = parser.parse_args()
    create_dirs()

    # Initialize the censor app
    init_app(config_file=args.config, default_censor_config=args.censor_config, debug=args.debug)

    # Remove the default stderr handler
    logger.remove()
    log_format = "{time} | {level} | {message}"

    # Console handler - level depends on debug flag
    console_level = "DEBUG" if args.debug else "INFO"
    logger.add(sys.stderr, level=console_level, format=_defaults.LOGURU_FORMAT)

    # File handler - always logs DEBUG
    logger.add(constants.log_path / 'log.txt', format=log_format, level='DEBUG', rotation='10 MB')

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    ############ Start the correct mode ############
    if args.mode in ['init']:
        logger.info("Setting up directories.")

    elif args.mode in ['live']:
        if args.device != -1:
            logger.warning(f"Device id was given, but is not used with live mode. Using 'webcam' mode if you want to censor a camera..")
        start_live_censor(args.live_mode, device_id=-1, use_vcam=args.vcam, vcam_width=args.vcam_w, vcam_height=args.vcam_h, vcam_fps=args.vcam_fps)

    elif args.mode in ['webcam']:
        if args.device == -1:
            logger.warning(f"No device id provided, using default (0).")
        start_live_censor('quick', device_id=args.device, use_vcam=args.vcam, vcam_width=args.vcam_w, vcam_height=args.vcam_h, vcam_fps=args.vcam_fps)

    elif args.mode in ['image', 'images']:
        start_image_censor(input_path, output_path,
                           override_cache=args.override_cache,
                           skip_existing=args.skip_existing,
                           only_analyze=args.only_analyze,
                           censor_config=args.censor_config)

    elif args.mode in ['video', 'videos']:
        start_video_censor(input_path, output_path,
                           override_cache=args.override_cache,
                           skip_existing=args.skip_existing,
                           only_analyze=args.only_analyze,
                           censor_config=args.censor_config)

    elif args.mode in ['all']:
        censor_all(input_path, output_path,
                    override_cache=args.override_cache,
                    skip_existing=args.skip_existing,
                    only_analyze=args.only_analyze,
                    censor_config=args.censor_config)

    elif args.mode in ['http', 'https']:
        serve_http(args.host, args.port, args.ssl_cert, args.ssl_key)

    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    logger.info("Done")


if __name__ == '__main__':
    main()



