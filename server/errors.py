"""Errors returned by the HTTP API: a status, a machine-readable code and a message."""
from __future__ import annotations

from aiohttp import web

# Error codes. Clients can branch on these; the message is for people.
INVALID_REQUEST = 'invalid_request'          # 400: malformed body, missing or wrong fields
INVALID_IMAGE = 'invalid_image'              # 400: the image could not be read or has an unsupported type
INVALID_VIDEO = 'invalid_video'              # 400: the video could not be read
INVALID_CONFIG = 'invalid_config'            # 400: the censor config is not valid
CONFIG_NOT_ALLOWED = 'config_not_allowed'    # 400: the censor config points outside data/stickers
URL_NOT_ALLOWED = 'url_not_allowed'          # 400: the URL points to this machine or the local network
USE_CENSOR_VIDEO = 'use_censor_video'        # 400: a GIF sent to an image endpoint; send it to /censor_video
NOT_FOUND = 'not_found'                      # 404
NOT_READY = 'not_ready'                      # 409: the video job has not finished yet
JOB_FAILED = 'job_failed'                    # 409: the video job failed or was cancelled, so it has no result
QUEUE_FULL = 'queue_full'                    # 429: too many video jobs are waiting or running
DOWNLOAD_FAILED = 'download_failed'          # 502: the image or video URL could not be downloaded
PROCESSING_FAILED = 'processing_failed'      # 500: censoring or detection failed
INTERNAL_ERROR = 'internal_error'            # 500: anything else
UNAVAILABLE = 'unavailable'                  # 503: the pipeline is not running


GIF_MESSAGE = "GIFs are censored as videos: send them to /censor_video (as 'video' or 'video_url')."


class ApiError(Exception):
    """An error with the HTTP status and code to answer it with."""

    def __init__(self, message: str, code: str = INVALID_REQUEST, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status

    def to_dict(self) -> dict:
        return error_body(str(self), self.code)

    def to_response(self) -> web.Response:
        return error_response(str(self), self.code, self.status)


def error_body(message: str, code: str) -> dict:
    return {'status': 'error', 'code': code, 'error': message}


def error_response(message: str, code: str, status: int) -> web.Response:
    return web.json_response(error_body(message, code), status=status)
