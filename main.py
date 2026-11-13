import argparse
import sys
import time
from os import makedirs
from pathlib import Path
from typing import Literal

from loguru import logger, _defaults

import constants
from app import init_app
from app.config import ProfileError


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


def is_image(file: Path) -> bool:
    return file.suffix in constants.IMAGE_EXT

def is_video(file: Path) -> bool:
    return file.suffix in constants.VID_EXT


def parse_input_path(input_path: Path, recursive: bool) -> dict[Literal['imgs', 'videos'], list[Path]]:
    """
    Checks the input_path, if it is a file return a list with that file.
    Otherwise, return all files with the correct extension.
    """
    logger.info(f"Using input path: '{input_path}'")

    if input_path is None:
        raise ValueError("Input path is required!")
    if not input_path.exists():
        raise ValueError(f"Input path '{input_path}' does not exist!")

    files: dict[Literal['imgs', 'videos'], list[Path]] = {
        'imgs': [],
        'videos': []
    }
    if input_path.is_dir():
        glob_arg = "**/*" if recursive else "*"
        for f in input_path.glob(glob_arg):
            if not f.is_file(): continue
            if is_image(f):
                files['imgs'].append(f)
            elif is_video(f):
                files['videos'].append(f)

    else:
        if is_image(input_path):
            files['imgs'].append(input_path)
        elif is_video(input_path):
            files['videos'].append(input_path)

    total_files = len(files['imgs']) + len(files['videos'])
    if total_files == 0:
        raise ValueError(f"No input files in '{input_path}'.")

    logger.info(f"Found {total_files} input files ({len(files['imgs'])} images and {len(files['videos'])} videos).")
    return files


def censor_files(input_path: Path, output_path: Path, recursive: bool, **kwargs):
    from app.image import censor_images
    from app.video import censor_videos

    try:
        input_files = parse_input_path(input_path, recursive)
    except Exception as e:
        logger.error(e)
        return

    start_time = time.perf_counter_ns()
    images = input_files['imgs']
    videos = input_files['videos']

    if len(images) > 0:
        logger.info("Censoring images...")
        censor_images(input_files['imgs'], output_path, **kwargs)
        logger.info(f"Done with images.")
    if len(videos) > 0:
        logger.info("Censoring videos...")
        censor_videos(input_files['videos'], output_path, **kwargs)
        logger.info("Done with videos.")

    stop_time = time.perf_counter_ns()
    total_ms = (stop_time - start_time) / 1000000
    time_str = format_time(total_ms)
    logger.info(f"Took {time_str} to censor {len(images) + len(videos)} files.")


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


def start_live_censor(device_id: int, use_vcam: bool, vcam_width: int, vcam_height, vcam_fps:int, censor_config: Path|str):
    from app.live import start_live_censor
    start_live_censor(device_id, use_vcam, vcam_width, vcam_height, vcam_fps, censor_config)


def start_overlay_censor(window_ids: list[int] | None, censor_config: Path | str, command: list[str] | None,
                         curtain: bool):
    from app.live.overlay import start_overlay_censor
    try:
        start_overlay_censor(window_ids, censor_config, command=command, curtain=curtain)
    except (RuntimeError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)


def window_id_arg(value: str) -> int:
    """An X11 window id, in hex (0x...) or decimal."""
    try:
        return int(value, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a window id: {value!r} (use e.g. 0x3a00007)") from None


def split_command(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Split the arguments at the first '--': (CensorTool's arguments, the command after it or None)."""
    if '--' not in argv:
        return argv, None
    index = argv.index('--')
    return argv[:index], argv[index + 1:]


def serve_http(host, port, ssl_file, ssl_key, use_ssl, debug):
    from server import http_server
    http_server.start_server(host=host, port=port, use_https=use_ssl, cert_file=ssl_file, key_file=ssl_key, debug=debug)


def select_profile() -> str:
    """
    Let the user pick a profile in the terminal. A single profile is used without asking.
    Returns None when there is nothing to pick from or no terminal to ask in: the first profile alphabetically is
    then used (with a warning).
    """
    from app.config import list_profiles
    from utils import cli_select

    profiles = list_profiles()
    if len(profiles) == 1:
        return profiles[0]
    if not profiles or not cli_select.is_interactive():
        return None

    try:
        return cli_select.select("Select a profile:", profiles)
    except KeyboardInterrupt:
        sys.exit(130)


def get_mode_overview():
    return (
        "'file': censor images and videos from file."
        "'live': real time censoring. "
        "'webcam': real time webcam censoring. "
        "'overlay': censor a window in place (Linux/X11). "
        "'http: start the http server."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default='http', help=f"The mode to use. {get_mode_overview()}")
    parser.add_argument('-i', '--input', default=constants.uncensored_path, help='Path to input file or directory.')
    parser.add_argument('-o', '--output', default=constants.censored_path, help='Path to output directory.')

    parser.add_argument('-p', '--profile', default=None,
                        help="Config profile: a name of a directory in './configs/' or a path to a profile directory. "
                             "If not given, you are asked to select one (or the first one alphabetically is used when not run in a terminal).")
    parser.add_argument('--config', default=None, help="Path to the main config file. Overrides the profile's one.")
    parser.add_argument('-cc', '--censor-config', default=None,
                        help="Path to the censor config file. Overrides the profile's one.")
    parser.add_argument('-pl', '--pipeline', default=None,
                        help="Path to the pipeline config file (the backend and the models it loads). "
                             "Overrides the profile's one.")

    # live & webcam mode
    lw_parser = parser.add_argument_group(title='Live & Webcam arguments')
    lw_parser.add_argument('--device', default=-1, type=int, help='Device to use (only used for live webcam censor).')
    lw_parser.add_argument('--vcam', action='store_true', help='Whether to output to a virtual camera or not.')
    lw_parser.add_argument('--vcam-w', default=1920, type=int, help='Virtual camera width.')
    lw_parser.add_argument('--vcam-h', default=1080, type=int, help='Virtual camera height.')
    lw_parser.add_argument('--vcam-fps', default=10, type=int, help='Virtual camera target fps.')

    # overlay mode
    overlay_parser = parser.add_argument_group(title='Overlay arguments')
    overlay_parser.add_argument('--window', default=None, type=window_id_arg, action='append',
                                help='X11 id of a window to censor (e.g. from xwininfo or wmctrl -l); repeat it '
                                     'for more windows. If not given, you click the window.')
    overlay_parser.add_argument('--curtain', action='store_true',
                                help='With a command after --: cover the screens while the program starts, so its '
                                     'window is never visible uncensored.')

    # flags
    flags_parser = parser.add_argument_group(title='General flags')
    flags_parser.add_argument('--override-cache', action='store_true', help='Recompute features by overriding the cache.')
    flags_parser.add_argument('--skip-existing', action='store_true',
                        help='Whether to skip files that already exist in the output directory.')
    flags_parser.add_argument('--only-analyze', action='store_true',
                        help='Whether to only analyze the images and videos (does not apply censoring).')
    flags_parser.add_argument("-r", "--recursive", action='store_true', help="Recursively censor all files subdirectories of the input directory.")
    flags_parser.add_argument('--debug', action='store_true', help='Whether to enable debug mode.')

    # server arguments
    server_parser = parser.add_argument_group(title='Server arguments')
    server_parser.add_argument('--ssl', action='store_true', help='Whether to enable https.')
    server_parser.add_argument('--port', default=8443, type=int, help='Port to run the server on.')
    server_parser.add_argument('--host', default='localhost', type=str, help='Host to run the server on.')
    server_parser.add_argument('--ssl-cert', default=str(Path('certs') / 'cert.pem'), type=str,
                               help='Path to certificate file.')
    server_parser.add_argument('--ssl-key', default=str(Path('certs') / 'key.pem'), type=str,
                               help='Path to key file.')

    parser.usage = '%(prog)s mode [options] [-- command ...]'
    parser.epilog = ("overlay mode: everything after -- is a program to start and censor, e.g. "
                     "'python main.py overlay -- celluloid --new-window video.mp4'.")
    argv, command = split_command(sys.argv[1:])
    args = parser.parse_args(argv)
    if command is not None:
        if args.mode != 'overlay':
            parser.error("a command after -- can only be used with overlay mode")
        if not command:
            parser.error("no command after --")
        if args.window is not None:
            parser.error("use either --window or a command after --, not both")
    elif args.curtain:
        parser.error("--curtain needs a command after --")
    create_dirs()

    # Initialize the censor app
    profile = args.profile if args.profile is not None else select_profile()

    try:
        init_app(profile=profile, config_file=args.config, default_censor_config=args.censor_config,
                 pipeline_file=args.pipeline, debug=args.debug)
    except ProfileError as e:
        parser.error(str(e))

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

    if args.mode in ['validate']:
        from validate import validate_install
        validate_install()

    elif args.mode in ['live']:
        if args.device != -1:
            logger.warning(f"Device id was given, but is not used with live mode. Using 'webcam' mode if you want to censor a camera.")
        start_live_censor(
            device_id=-1,
            use_vcam=args.vcam,
            vcam_width=args.vcam_w,
            vcam_height=args.vcam_h,
            vcam_fps=args.vcam_fps,
            censor_config=args.censor_config
        )

    elif args.mode in ['webcam']:
        if args.device == -1:
            logger.warning(f"No device id provided, using default (0).")
        start_live_censor(
            device_id=max(args.device, 0),
            use_vcam=args.vcam,
            vcam_width=args.vcam_w,
            vcam_height=args.vcam_h,
            vcam_fps=args.vcam_fps,
            censor_config=args.censor_config
        )

    elif args.mode in ['overlay']:
        start_overlay_censor(window_ids=args.window, censor_config=args.censor_config, command=command,
                             curtain=args.curtain)

    elif args.mode in ['f', 'file', 'files']:
        censor_files(
            input_path, output_path,
            recursive=args.recursive,
            override_cache=args.override_cache,
            skip_existing=args.skip_existing,
            only_analyze=args.only_analyze,
            censor_config=args.censor_config
        )

    elif args.mode in ['http', 'https']:
        serve_http(args.host, args.port, args.ssl_cert, args.ssl_key, args.ssl, args.debug)

    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    logger.info("Done")


if __name__ == '__main__':
    main()



