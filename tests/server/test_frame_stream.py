"""The /censor_frame WebSocket: stats, options (detections), cancelling and failed frames."""
import asyncio
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

HEADER = bytes([0]) + bytes(range(1, 13))
CANCEL = bytes([1]) + bytes(12)


class FakePipeline:
    """Jobs finish when release() is called (or at once with auto_finish); frames named b'fail...' fail."""

    def __init__(self):
        self.auto_finish = True
        self.jobs = []

    def submit(self, image_bytes, output_path, config):
        job = MagicMock()
        job.cancelled = False
        job.cancel.side_effect = lambda: setattr(job, "cancelled", True)
        job.success = not image_bytes.startswith(b"fail")
        job.error = None if job.success else "model crashed"
        job.result = {"image": np.zeros((6, 8, 3), dtype=np.uint8),
                      "detections": [{"id": 0, "label": "FACE", "score": 0.9, "box": (1, 2, 3, 4), "polygons": [],
                                      "model": "m", "parent": None}],
                      "cached_models": []}
        job.finished = self.auto_finish
        handle = MagicMock()
        handle.done.side_effect = lambda: job.finished
        handle.wait.return_value = job
        self.jobs.append(job)
        return job, handle

    def release(self):
        for job in self.jobs:
            job.finished = True


@pytest_asyncio.fixture
async def stream():
    from aiohttp import web as aio_web
    from aiohttp.test_utils import TestClient, TestServer
    import server.http_server as hs

    pipeline = FakePipeline()
    app = aio_web.Application()
    app.router.add_get('/censor_frame', hs.censor_frame)
    with patch("server.http_server.submit_censoring_job", side_effect=pipeline.submit), \
            patch("server.http_server.np_to_bytes", side_effect=lambda img, ext: b"censored"):
        client = TestClient(TestServer(app))
        await client.start_server()
        ws = await client.ws_connect('/censor_frame')
        ws.pipeline = pipeline
        yield ws
        await ws.close()
        await client.close()


async def stats(ws) -> dict:
    await ws.send_str('{"type": "stats"}')
    return await ws.receive_json(timeout=2)


@pytest.mark.asyncio
async def test_stats_on_request(stream):
    await stream.send_bytes(HEADER + b"a")
    await stream.receive_bytes(timeout=2)
    await stream.send_bytes(HEADER + b"a")
    await stream.receive_bytes(timeout=2)

    reply = await stats(stream)
    assert reply["type"] == "stats"
    assert reply["frames_received"] == 2
    assert reply["frames_sent"] == 2
    assert reply["frames_from_memory"] == 1
    assert reply["active_jobs"] == 0
    assert reply["fps"] > 0


@pytest.mark.asyncio
async def test_detections_are_opt_in_and_come_before_the_frame(stream):
    await stream.send_str('{"type": "options", "include_detections": true}')
    assert await stream.receive_json(timeout=2) == {"type": "options", "status": "success",
                                                   "include_detections": True}

    for _ in range(2):  # censored, then from memory
        await stream.send_bytes(HEADER + b"a")
        detections = await stream.receive_json(timeout=2)
        assert detections["type"] == "detections"
        assert detections["header"] == HEADER.hex()
        assert (detections["width"], detections["height"]) == (8, 6)
        assert detections["detections"][0]["box"] == [1.0, 2.0, 3.0, 4.0]
        assert await stream.receive_bytes(timeout=2) == HEADER + b"censored"

    await stream.send_str('{"type": "options", "include_detections": false}')
    await stream.receive_json(timeout=2)
    await stream.send_bytes(HEADER + b"b")
    assert await stream.receive_bytes(timeout=2) == HEADER + b"censored"


@pytest.mark.asyncio
async def test_bad_options_are_an_error(stream):
    await stream.send_str('{"type": "options", "include_detections": "yes"}')
    reply = await stream.receive_json(timeout=2)
    assert reply["type"] == "error" and "true or false" in reply["error"]


@pytest.mark.asyncio
async def test_cancel_drops_the_frames_in_progress(stream):
    stream.pipeline.auto_finish = False
    await stream.send_bytes(HEADER + b"a")
    await stream.send_bytes(HEADER + b"b")
    assert (await stats(stream))["active_jobs"] == 2

    await stream.send_bytes(CANCEL)
    reply = await stats(stream)
    assert reply["active_jobs"] == 0 and reply["frames_cancelled"] == 2
    assert all(job.cancelled for job in stream.pipeline.jobs)

    # finished cancelled jobs are not sent; the next frame is
    stream.pipeline.release()
    stream.pipeline.auto_finish = True
    await stream.send_bytes(HEADER + b"c")
    assert await stream.receive_bytes(timeout=2) == HEADER + b"censored"
    assert (await stats(stream))["frames_sent"] == 1


def frame_header(video_id: int, seek_id: int, frame_num: int) -> bytes:
    """A header as the extension builds it: type, video id, seek id, frame number (big-endian)."""
    return bytes([0]) + video_id.to_bytes(4, "big") + seek_id.to_bytes(4, "big") + frame_num.to_bytes(4, "big")


@pytest.mark.asyncio
async def test_cancel_with_a_key_drops_only_the_frames_of_that_video_and_seek(stream):
    stream.pipeline.auto_finish = False
    await stream.send_bytes(frame_header(1, 0, 0) + b"a")   # video 1, seek 0: cancelled
    await stream.send_bytes(frame_header(1, 0, 1) + b"b")   # video 1, seek 0: cancelled
    await stream.send_bytes(frame_header(1, 1, 2) + b"c")   # video 1, a later seek: kept
    await stream.send_bytes(frame_header(2, 0, 0) + b"d")   # another video: kept
    assert (await stats(stream))["active_jobs"] == 4

    await stream.send_bytes(bytes([1]) + (1).to_bytes(4, "big") + (0).to_bytes(4, "big") + bytes(4))
    reply = await stats(stream)
    assert reply["active_jobs"] == 2 and reply["frames_cancelled"] == 2
    assert [job.cancelled for job in stream.pipeline.jobs] == [True, True, False, False]

    stream.pipeline.release()
    received = {await stream.receive_bytes(timeout=2), await stream.receive_bytes(timeout=2)}
    assert received == {frame_header(1, 1, 2) + b"censored", frame_header(2, 0, 0) + b"censored"}


@pytest.mark.asyncio
async def test_failed_frames_get_an_error_with_their_header(stream):
    await stream.send_bytes(HEADER + b"fail")
    reply = await stream.receive_json(timeout=2)
    assert reply == {"type": "error", "code": "processing_failed", "error": "Censoring failed: model crashed",
                     "header": HEADER.hex()}
    assert (await stats(stream))["frames_failed"] == 1


@pytest.mark.asyncio
async def test_undecodable_frames_get_an_error(stream):
    from server.errors import ApiError

    with patch("server.http_server.submit_censoring_job", side_effect=ApiError("not an image", "invalid_image")):
        await stream.send_bytes(HEADER + b"garbage")
        reply = await stream.receive_json(timeout=2)
    assert reply["code"] == "invalid_image" and reply["header"] == HEADER.hex()


@pytest.mark.asyncio
async def test_too_short_frames_get_an_error(stream):
    await stream.send_bytes(bytes(5))
    reply = await stream.receive_json(timeout=2)
    assert reply["type"] == "error" and "header" in reply["error"]


@pytest.mark.asyncio
async def test_unknown_message_type(stream):
    await stream.send_str('{"type": "dance"}')
    reply = await stream.receive_json(timeout=2)
    assert reply["type"] == "error" and "Unknown message type" in reply["error"]


@pytest.mark.asyncio
async def test_open_streams_are_counted(stream):
    from server.helpers import activity

    assert activity.open_frame_streams == 1
    await stream.close()
    for _ in range(100):
        if activity.open_frame_streams == 0:
            break
        await asyncio.sleep(0.01)
    assert activity.open_frame_streams == 0
