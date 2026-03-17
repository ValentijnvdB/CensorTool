import platform

import constants


def get_ffmpeg() -> str:

    if platform.system() == 'Windows':
        ffmpeg_path = constants.project_root_dir / 'ffmpeg' / 'bin' / 'ffmpeg.exe'
        return str(ffmpeg_path)
    else:
        return 'ffmpeg'


def get_ffprobe() -> str:

    if platform.system() == 'Windows':
        ffmpeg_path = constants.project_root_dir / 'ffmpeg' / 'bin' / 'ffprobe.exe'
        return str(ffmpeg_path)
    else:
        return 'ffprobe'