import json
import subprocess
from pathlib import Path

from ..video import ffmpeg


def video_file_has_audio(video_path: Path) -> bool:
    """
    Test whether a video file has an audio channel

    :param video_path: Path to the video file

    :return: whether the video file has an audio channel
    """
    try:
        if not video_path.is_absolute():
            video_path = video_path.absolute().resolve()

        # Run ffprobe to get stream information in JSON format
        command = [
            ffmpeg.get_ffprobe(),
            "-v", "error",
            "-select_streams", "a",  # Select only audio streams
            "-show_entries", "stream=index",
            "-of", "json",
            video_path
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False  # Error occurred

        # Parse JSON output
        streams = json.loads(result.stdout).get("streams", [])

        return len(streams) > 0  # True if at least one audio stream exists

    except Exception as e:
        print(f"Error processing file: {e}")
        return False
