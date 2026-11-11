"""
Unit tests for server/http_server.py

Run with:
    pytest tests/server/test_http_server.py -v
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_request():
    """Minimal aiohttp request mock."""
    return AsyncMock()


def image_request(name="photo.png", config=None, expected_response="base64", include_detections=False):
    from server.request_reader import ImageRequest
    return ImageRequest(b"img", Path(name), config, expected_response, include_detections)


def read_request_returning(req):
    return patch("server.http_server.rr.read_request", AsyncMock(return_value=req))


def make_detection(id=0, label="FACE_FEMALE", score=0.92, box=(10, 20, 100, 100),
                   polygons=(), model="nudenet", parent=None) -> dict:
    """A detection dict as censor_core 2.0 returns it in result["detections"]."""
    return {"id": id, "label": label, "score": score, "box": box,
            "polygons": list(polygons), "model": model, "parent": parent}


def make_detection_submit(detections: list[dict], cached_models=()):
    """Return value of submit_detection_job: (job, handle), where handle.wait() returns the finished job."""
    job = MagicMock()
    job.success = True
    job.result = {"image": np.zeros((30, 40, 3), dtype=np.uint8), "output_path": None,
                  "detections": detections, "cached_models": list(cached_models)}
    handle = MagicMock()
    handle.wait.return_value = job
    handle.done.return_value = True
    return job, handle


@pytest.fixture()
def detection_job():
    """A detect_features submission with one detection."""
    return make_detection_submit([make_detection()])


# ---------------------------------------------------------------------------
# censor_image
# ---------------------------------------------------------------------------

class TestCensorImage:

    @staticmethod
    def submit(detections=()):
        return patch("server.http_server.submit_censoring_job",
                     return_value=make_detection_submit(list(detections), cached_models=["nudenet"]))

    @pytest.mark.asyncio
    async def test_base64_answer_has_status_size_and_time(self, fake_request):
        from server.http_server import censor_image

        with read_request_returning(image_request("photo.jpg")), self.submit():
            result = await censor_image(fake_request)

        body = json.loads(result.body)
        assert body["status"] == "success"
        assert body["mime_type"] == "image/jpeg"
        assert (body["width"], body["height"]) == (40, 30)
        assert isinstance(body["processing_ms"], int)
        assert "detections" not in body

    @pytest.mark.asyncio
    async def test_detections_are_opt_in(self, fake_request):
        from server.http_server import censor_image

        with read_request_returning(image_request(include_detections=True)), self.submit([make_detection()]):
            result = await censor_image(fake_request)

        body = json.loads(result.body)
        assert body["detections"][0]["label"] == "FACE_FEMALE"
        assert body["detections"][0]["box"] == [10.0, 20.0, 100.0, 100.0]
        assert body["cached_models"] == ["nudenet"]

    @pytest.mark.asyncio
    async def test_bytes_answer(self, fake_request):
        from server.http_server import censor_image

        with read_request_returning(image_request("photo.webp", expected_response="bytes")), self.submit():
            result = await censor_image(fake_request)

        assert result.content_type == "image/webp"
        assert result.headers["X-Image-Width"] == "40"
        assert "X-Processing-Time-Ms" in result.headers

    @pytest.mark.asyncio
    async def test_bytes_answer_with_detections_is_400(self, fake_request):
        from server.http_server import censor_image

        req = image_request(expected_response="bytes", include_detections=True)
        with read_request_returning(req), self.submit() as submit:
            result = await censor_image(fake_request)

        assert result.status == 400
        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_url_answer_is_saved_before_answering(self, fake_request, tmp_path):
        from server.http_server import censor_image

        with read_request_returning(image_request("photo.png", expected_response="url")), self.submit(), \
                patch("server.http_server.CENSORED_PATH", tmp_path), \
                patch("server.response_constructor.CENSORED_PATH", tmp_path):
            result = await censor_image(fake_request)

        body = json.loads(result.body)
        assert body["image_name"] == "photo.png"
        assert body["url"] == "/censored/photo.png"
        assert (tmp_path / "photo.png").read_bytes()[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_forwards_censor_config_to_job(self, fake_request):
        from server.http_server import censor_image

        censor_config = {"features_to_censor": []}
        with read_request_returning(image_request(config=censor_config)), self.submit() as submit:
            await censor_image(fake_request)

        assert submit.call_args.args == (b"img", None, censor_config)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("handler", ["censor_image", "detect_features"])
    @pytest.mark.parametrize("payload", [
        {"type": "source", "image_url": "https://example.com/anim.GIF"},
        {"type": "base64", "mime_type": "image/gif", "image_data": "R0lGODlh"},
        {"type": "base64", "mime_type": "image/png", "image_data": "R0lGODlhAQABAAAAACw="},  # a GIF named .png
    ])
    async def test_gifs_are_sent_to_censor_video(self, tmp_path, handler, payload):
        import server.http_server as hs

        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=payload)
        with patch("server.request_reader.UPLOAD_DIR", tmp_path), \
                patch("server.request_reader.requests.get") as download, \
                patch("server.helpers._image_pipeline", MagicMock()) as pipeline:
            result = await getattr(hs, handler)(request)

        body = json.loads(result.body)
        assert result.status == 400
        assert body["code"] == "use_censor_video" and "/censor_video" in body["error"]
        download.assert_not_called()
        pipeline.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_500_when_job_fails(self, fake_request):
        from server.http_server import censor_image

        job, handle = make_detection_submit([])
        job.success = False
        job.error = "pipeline exploded"
        with read_request_returning(image_request()), \
                patch("server.http_server.submit_censoring_job", return_value=(job, handle)):
            result = await censor_image(fake_request)

        assert result.status == 500
        body = json.loads(result.body)
        assert body["code"] == "processing_failed"
        assert "pipeline exploded" in body["error"]

    @pytest.mark.asyncio
    async def test_api_errors_keep_their_status_and_code(self, fake_request):
        from server.errors import ApiError
        from server.http_server import censor_image

        with patch("server.http_server.rr.read_request", AsyncMock(side_effect=ApiError("bad", "invalid_image"))):
            result = await censor_image(fake_request)

        assert result.status == 400
        assert json.loads(result.body) == {"status": "error", "code": "invalid_image", "error": "bad"}

    @pytest.mark.asyncio
    async def test_returns_500_on_unexpected_exception(self, fake_request):
        from server.http_server import censor_image

        with read_request_returning(image_request()), \
                patch("server.http_server.submit_censoring_job", side_effect=Exception("unexpected")):
            result = await censor_image(fake_request)

        assert result.status == 500
        assert json.loads(result.body)["code"] == "internal_error"


# ---------------------------------------------------------------------------
# censor_images
# ---------------------------------------------------------------------------

class TestCensorImages:

    @staticmethod
    def request(body):
        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=body)
        return request

    @pytest.mark.asyncio
    async def test_each_image_gets_its_own_answer_in_order(self):
        from server.errors import ApiError
        from server.http_server import censor_images

        def parse(item, shared):
            if item == "bad":
                raise ApiError("No 'image_url' provided.")
            return image_request(item, **{k: v for k, v in shared.items() if k != "config"})

        with patch("server.http_server.rr.parse_image_request", side_effect=parse), \
                patch("server.http_server.submit_censoring_job",
                      side_effect=lambda *a: make_detection_submit([make_detection()])):
            result = await censor_images(self.request({"images": ["a.png", "bad", "c.jpg"],
                                                       "include_detections": True}))

        body = json.loads(result.body)
        assert body["status"] == "success"
        a, bad, c = body["results"]
        assert a["status"] == "success" and a["mime_type"] == "image/png" and len(a["detections"]) == 1
        assert bad == {"status": "error", "code": "invalid_request", "error": "No 'image_url' provided."}
        assert c["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body, error", [
        ({}, "'images' list"),
        ({"images": []}, "empty"),
        ({"images": [{}] * 51}, "at most 50"),
        ({"images": [{}], "expected_response": "bytes"}, "not supported"),
        ({"images": [{}], "config": {"features_to_censor": []}}, "Invalid censor config"),
    ])
    async def test_bad_requests_are_400(self, body, error):
        from server.http_server import censor_images

        result = await censor_images(self.request(body))
        assert result.status == 400
        assert error in json.loads(result.body)["error"]


# ---------------------------------------------------------------------------
# detect_features
# ---------------------------------------------------------------------------

class TestDetectFeatures:

    @pytest.mark.asyncio
    async def test_returns_success_status_name_and_size(self, fake_request, detection_job):
        from server.http_server import detect_features

        with read_request_returning(image_request("face.jpg")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert body["status"] == "success"
        assert body["file_name"] == "face.jpg"
        assert (body["width"], body["height"]) == (40, 30)

    @pytest.mark.asyncio
    async def test_returns_correct_detection_fields(self, fake_request, detection_job):
        from server.http_server import detect_features

        with read_request_returning(image_request()):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert len(body["detections"]) == 1
        d = body["detections"][0]
        assert d["id"] == 0
        assert d["label"] == "FACE_FEMALE"
        assert d["score"] == pytest.approx(0.92)
        assert d["box"] == [10.0, 20.0, 100.0, 100.0]
        assert all(isinstance(v, float) for v in d["box"])
        assert d["polygons"] == []
        assert d["model"] == "nudenet"
        assert d["parent"] is None

    @pytest.mark.asyncio
    async def test_polygons_and_parent_are_serialized(self, fake_request):
        from server.http_server import detect_features

        eye = make_detection(id=1, label="EYES_FEMALE", model="landmarks", parent=0,
                             polygons=[np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)])
        submit = make_detection_submit([make_detection(), eye], cached_models=["nudenet"])

        with read_request_returning(image_request()):
            with patch("server.http_server.submit_detection_job", return_value=submit):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert body["detections"][1]["polygons"] == [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        assert body["detections"][1]["parent"] == 0
        assert body["cached_models"] == ["nudenet"]

    @pytest.mark.asyncio
    async def test_returns_multiple_detections_in_order(self, fake_request):
        from server.http_server import detect_features

        submit = make_detection_submit([
            make_detection(id=0, label="BODY", model="people"),
            make_detection(id=1, label="FACE_FEMALE"),
        ])

        with read_request_returning(image_request()):
            with patch("server.http_server.submit_detection_job", return_value=submit):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert [d["label"] for d in body["detections"]] == ["BODY", "FACE_FEMALE"]

    @pytest.mark.asyncio
    async def test_returns_500_when_job_failed(self, fake_request):
        from server.http_server import detect_features

        job, handle = make_detection_submit([])
        job.success = False
        job.error = "pipeline exploded"

        with read_request_returning(image_request()):
            with patch("server.http_server.submit_detection_job", return_value=(job, handle)):
                result = await detect_features(fake_request)

        assert result.status == 500
        assert json.loads(result.body)["code"] == "processing_failed"

    @pytest.mark.asyncio
    async def test_undecodable_image_is_400(self, fake_request):
        from server.http_server import detect_features

        with read_request_returning(image_request()), patch("server.helpers._image_pipeline", MagicMock()):
            result = await detect_features(fake_request)

        assert result.status == 400
        assert json.loads(result.body)["code"] == "invalid_image"


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

class TestCache:

    @pytest.mark.asyncio
    async def test_info_counts_files_and_bytes(self, fake_request, tmp_path):
        from server.http_server import cache_info

        (tmp_path / "a.json").write_bytes(b"12345")
        (tmp_path / "videos" / "abc").mkdir(parents=True)
        (tmp_path / "videos" / "abc" / "clip.json").write_bytes(b"123")

        with patch("server.server_config.CACHE_DIR", tmp_path):
            body = json.loads((await cache_info(fake_request)).body)

        assert (body["status"], body["files"], body["bytes"]) == ("success", 2, 8)

    @pytest.mark.asyncio
    async def test_delete_removes_everything_and_reports_it(self, fake_request, tmp_path):
        from server.http_server import delete_cache

        (tmp_path / "cache1.bin").write_bytes(b"a")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "cache2.bin").write_bytes(b"bc")

        with patch("server.server_config.CACHE_DIR", tmp_path):
            body = json.loads((await delete_cache(fake_request)).body)

        assert body == {"status": "success", "deleted_files": 2, "deleted_bytes": 3}
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_delete_succeeds_when_cache_dir_does_not_exist(self, fake_request, tmp_path):
        from server.http_server import delete_cache

        with patch("server.server_config.CACHE_DIR", tmp_path / "nonexistent"):
            body = json.loads((await delete_cache(fake_request)).body)

        assert body["status"] == "success" and body["deleted_files"] == 0

    @pytest.mark.asyncio
    async def test_returns_500_on_os_error(self, fake_request):
        from server.http_server import delete_cache

        with patch("server.server_config.CACHE_DIR") as mock_dir:
            mock_dir.exists.side_effect = OSError("permission denied")
            result = await delete_cache(fake_request)

        assert result.status == 500
        assert "permission denied" in json.loads(result.body)["error"]

    def test_reset_cache_get_route_is_gone(self):
        import asyncio
        import server.http_server as hs

        app = asyncio.run(hs.init_app())
        routes = {(r.method, r.resource.canonical) for r in app.router.routes()}
        assert ("DELETE", "/cache") in routes
        assert not any(path == "/reset_cache" for _, path in routes)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

class TestProfile:

    @pytest.mark.asyncio
    async def test_returns_profile_sorted_labels_and_models(self, fake_request):
        from server.http_server import profile

        label_models = {"FACE_MALE": "nudenet", "EYES_FEMALE": "landmarks", "FACE_FEMALE": "nudenet"}
        with patch("server.http_server.get_labels", return_value=label_models), \
                patch("server.http_server.get_profile", return_value="nudity"):
            response = await profile(fake_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "success"
        assert body["profile"] == "nudity"
        assert body["labels"] == ["EYES_FEMALE", "FACE_FEMALE", "FACE_MALE"]
        assert body["models"] == label_models

    @pytest.mark.asyncio
    async def test_returns_500_when_pipeline_not_started(self, fake_request):
        from server.http_server import profile

        with patch("server.http_server.get_labels", side_effect=RuntimeError("The pipeline has not been started.")):
            response = await profile(fake_request)

        assert response.status == 500
        assert "not been started" in json.loads(response.body)["error"]


# ---------------------------------------------------------------------------
# censor_config
# ---------------------------------------------------------------------------

class TestCensorConfig:

    @pytest.mark.asyncio
    async def test_returns_profile_and_config(self, fake_request):
        from server.http_server import censor_config

        config = {"features_to_censor": ["FACE_FEMALE"],
                  "default_censor_config": {"censor_style": {"type": "bar", "color": [0, 0, 0]}}}
        with patch("server.http_server.get_default_censor_config", return_value=config), \
                patch("server.http_server.get_profile", return_value="nudity"):
            response = await censor_config(fake_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body == {"status": "success", "profile": "nudity", "config": config}

    @pytest.mark.asyncio
    async def test_returns_500_when_pipeline_not_started(self, fake_request):
        from server.http_server import censor_config

        with patch("server.http_server.get_default_censor_config",
                   side_effect=RuntimeError("The pipeline has not been started.")):
            response = await censor_config(fake_request)

        assert response.status == 500
        assert "not been started" in json.loads(response.body)["error"]


class TestGetDefaultCensorConfig:

    def test_strips_comfy_settings_and_leaves_stored_config_untouched(self):
        from server import helpers

        comfy = {"comfy_host": "localhost", "comfy_port": 8188, "comfy_workflow": "/home/me/workflow.json"}
        stored = {
            "features_to_censor": ["FACE_FEMALE", "FACE_MALE"],
            "default_censor_config": {"censor_style": {"type": "ai_remove", **comfy}},
            "feature_overrides": {"FACE_FEMALE": {"censor_style": {"type": "ai_remove", **comfy}},
                                  "FACE_MALE": None},
            "inverse_censor_style": {"type": "blur", "strength": 31},
        }
        with patch.object(helpers, "_default_censor_config", stored):
            config = helpers.get_default_censor_config()

        assert config["default_censor_config"]["censor_style"] == {"type": "ai_remove"}
        assert config["feature_overrides"]["FACE_FEMALE"]["censor_style"] == {"type": "ai_remove"}
        assert config["feature_overrides"]["FACE_MALE"] is None
        assert config["inverse_censor_style"] == {"type": "blur", "strength": 31}
        assert stored["default_censor_config"]["censor_style"]["comfy_host"] == "localhost"

    def test_raises_when_pipeline_not_started(self):
        from server import helpers

        with patch.object(helpers, "_default_censor_config", None):
            with pytest.raises(RuntimeError, match="not been started"):
                helpers.get_default_censor_config()


# ---------------------------------------------------------------------------
# rejected censor configs
# ---------------------------------------------------------------------------

class TestRejectedCensorConfig:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("handler", ["censor_image", "detect_features"])
    async def test_returns_400(self, fake_request, handler):
        import server.http_server as hs
        from app.config import CensorConfigError

        with patch("server.http_server.rr.read_request",
                   AsyncMock(side_effect=CensorConfigError("stickers_dir '/etc' is not allowed"))):
            response = await getattr(hs, handler)(fake_request)

        assert response.status == 400
        assert "not allowed" in json.loads(response.body)["error"]

    @pytest.mark.asyncio
    async def test_censor_video_returns_400(self):
        from server.http_server import censor_video

        part = MagicMock()
        part.name = "config"
        part.text = AsyncMock(return_value=json.dumps({"stickers_dir": "/etc"}))
        reader = MagicMock()
        reader.next = AsyncMock(side_effect=[part, None])
        request = MagicMock()
        request.multipart = AsyncMock(return_value=reader)

        with patch("server.http_server.submit_video_job") as submit:
            response = await censor_video(request)

        assert response.status == 400
        submit.assert_not_called()


class TestDefaultCensorConfigStickersDir:

    def test_stickers_dir_is_shown_relative_to_the_working_dir(self):
        import constants
        from server import helpers

        stored = {"stickers_dir": str(constants.instance_dir / "data" / "stickers" / "plants")}
        with patch.object(helpers, "_default_censor_config", stored):
            assert helpers.get_default_censor_config()["stickers_dir"] == "data/stickers/plants"


class TestCensorVideoFilename:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("filename", ["../evil.mp4", "..", ".", "", "/etc/evil.mp4", "a/b.mp4", "..\\evil.mp4"])
    async def test_rejects_filenames_with_folders(self, filename):
        from server.http_server import censor_video

        part = MagicMock()
        part.name = "filename"
        part.text = AsyncMock(return_value=filename)
        reader = MagicMock()
        reader.next = AsyncMock(side_effect=[part, None])
        request = MagicMock()
        request.multipart = AsyncMock(return_value=reader)

        with patch("server.http_server.submit_video_job") as submit:
            response = await censor_video(request)

        assert response.status == 400
        submit.assert_not_called()

    @pytest.mark.parametrize("filename", ["video.mp4", "my video..mp4", ".hidden.mp4"])
    def test_plain_filenames_are_allowed(self, filename):
        from server.http_server import is_plain_filename
        assert is_plain_filename(filename)


class TestCensorVideoFormat:

    @staticmethod
    def _request(parts):
        reader = MagicMock()
        reader.next = AsyncMock(side_effect=[*parts, None])
        request = MagicMock()
        request.multipart = AsyncMock(return_value=reader)
        return request

    @staticmethod
    def _text_part(name, text):
        part = MagicMock()
        part.name = name
        part.text = AsyncMock(return_value=text)
        return part

    @pytest.mark.parametrize("text, expected", [("webm", "webm"), (".WebM", "webm"), (" mov ", "mov"), ("gif", "gif")])
    def test_parse_format(self, text, expected):
        from server.http_server import parse_video_format
        assert parse_video_format(text) == expected

    @pytest.mark.parametrize("text", ["", "exe", "mp4/../x", "flv"])
    def test_parse_rejects_unsupported(self, text):
        from server.http_server import parse_video_format
        with pytest.raises(ValueError, match="Unsupported format"):
            parse_video_format(text)

    @pytest.mark.parametrize("filename, requested, expected", [
        ("clip.mp4", None, "mp4"),
        ("clip.WEBM", None, "webm"),
        ("clip.mkv", "mp4", "mp4"),
        ("clip.flv", None, "mp4"),
        ("clip", None, "mp4"),
    ])
    def test_output_format(self, filename, requested, expected):
        from server.http_server import output_video_format
        assert output_video_format(filename, requested) == expected

    @pytest.mark.asyncio
    async def test_unsupported_format_returns_400(self):
        from server.http_server import censor_video

        request = self._request([self._text_part("format", "exe")])
        with patch("server.http_server.submit_video_job") as submit:
            response = await censor_video(request)

        assert response.status == 400
        submit.assert_not_called()

    @staticmethod
    def _video_part(data=b"video data"):
        part = MagicMock()
        part.name = "video"
        part.filename = None
        part.read_chunk = AsyncMock(side_effect=[data, b""])
        return part

    @pytest.fixture()
    def dirs(self, tmp_path):
        """Upload, censored and cache folders in tmp_path."""
        import server.http_server as hs
        upload, censored, cache = tmp_path / "up", tmp_path / "out", tmp_path / "cache"
        upload.mkdir(), censored.mkdir()
        with patch.object(hs, "UPLOAD_DIR", upload), patch.object(hs, "CENSORED_PATH", censored), \
                patch.object(hs, "CACHE_DIR", cache):
            yield upload, censored, cache

    async def _run(self, parts, video_info=None):
        """Run censor_video with a fake video job that 'censors' by copying; returns (submit mock, response mock)."""
        import server.http_server as hs

        async def fake_submit(video_path, output_path, censor_config, early_exit, cache_dir):
            output_path.write_bytes(video_path.read_bytes())
            return MagicMock(success=True, result=str(output_path))

        info = video_info or hs.VideoInfo(frames=300, fps=30.0, width=640, height=360)
        with patch.object(hs, "submit_video_job", AsyncMock(side_effect=fake_submit)) as submit, \
                patch.object(hs, "read_video_info", return_value=info), \
                patch.object(hs.web, "StreamResponse") as stream_response:
            stream_response.return_value.prepare = AsyncMock()
            stream_response.return_value.write = AsyncMock()
            response = await hs.censor_video(self._request(parts))
        return submit, stream_response, response

    @pytest.mark.asyncio
    @pytest.mark.parametrize("parts, output_name, content_type", [
        ([("filename", "clip.mkv")], "clip.mkv", "video/x-matroska"),
        ([("filename", "clip.mkv"), ("format", "webm")], "clip.webm", "video/webm"),
        ([("format", "gif")], "video.gif", "image/gif"),
        ([], "video.mp4", "video/mp4"),
    ])
    async def test_output_path_and_content_type(self, dirs, parts, output_name, content_type):
        _, censored, _ = dirs
        submit, stream_response, _ = await self._run(
            [self._text_part(name, text) for name, text in parts] + [self._video_part()])

        assert submit.call_args.kwargs["output_path"] == censored / output_name
        assert stream_response.call_args.kwargs["headers"]["Content-Type"] == content_type

    @pytest.mark.asyncio
    async def test_headers_describe_the_video(self, dirs):
        _, stream_response, _ = await self._run([self._text_part("filename", "clip.mp4"), self._video_part()])
        headers = stream_response.call_args.kwargs["headers"]

        assert headers["X-Frame-Count"] == "300"
        assert headers["X-Video-Fps"] == "30.000"
        assert headers["X-Video-Duration"] == "10.000"
        assert (headers["X-Video-Width"], headers["X-Video-Height"]) == ("640", "360")
        assert int(headers["X-Processing-Time-Ms"]) >= 0
        assert headers["Content-Disposition"] == 'attachment; filename="clip.mp4"'

    @pytest.mark.asyncio
    async def test_taken_names_get_a_number(self, dirs):
        upload, censored, _ = dirs
        (upload / "clip.mp4").write_bytes(b"older upload")
        (censored / "clip_1.mp4").write_bytes(b"older output")

        submit, _, _ = await self._run([self._text_part("filename", "clip.mp4"), self._video_part(b"new")])

        assert submit.call_args.kwargs["video_path"] == upload / "clip_1.mp4"
        assert (upload / "clip.mp4").read_bytes() == b"older upload"
        assert submit.call_args.kwargs["output_path"] == censored / "clip_1_1.mp4"
        assert (censored / "clip_1.mp4").read_bytes() == b"older output"

    @pytest.mark.asyncio
    async def test_files_are_removed_once_sent(self, dirs):
        upload, censored, _ = dirs
        submit, stream_response, _ = await self._run([self._text_part("filename", "clip.mp4"),
                                                      self._video_part(b"new")])

        stream_response.return_value.write.assert_awaited_with(b"new")  # the 'censored' copy was sent
        assert list(upload.iterdir()) == [] and list(censored.iterdir()) == []

    @pytest.mark.asyncio
    async def test_detections_go_to_the_video_cache(self, dirs):
        # one folder for every video: the backend names each cache after the video's contents
        _, _, cache = dirs
        submit, _, _ = await self._run([self._video_part(b"video a")])
        assert submit.call_args.kwargs["cache_dir"] == cache / "videos"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("parts, error", [
        ([("filename", "clip.mp4")], "No video"),
        ([("config", "{not json")], "not valid JSON"),
    ])
    async def test_bad_requests_are_400(self, dirs, parts, error):
        submit, _, response = await self._run([self._text_part(name, text) for name, text in parts])

        assert response.status == 400
        assert error in json.loads(response.body)["error"]
        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_video_is_400_and_removed(self, dirs):
        upload, _, _ = dirs
        submit, _, response = await self._run([self._video_part(b"")])

        assert response.status == 400
        assert list(upload.iterdir()) == []
        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_job_is_500_and_leaves_no_output(self, dirs):
        import server.http_server as hs
        _, censored, _ = dirs

        with patch.object(hs, "submit_video_job", AsyncMock(return_value=MagicMock(success=False, error="boom"))), \
                patch.object(hs, "read_video_info", return_value=hs.VideoInfo(10, 10.0, 64, 48)):
            response = await hs.censor_video(self._request([self._video_part()]))

        assert response.status == 500
        assert json.loads(response.body)["code"] == "processing_failed"
        assert list(censored.iterdir()) == []


class TestCensorVideoUrl:
    """The 'video_url' part of /censor_video."""

    @pytest.fixture()
    def dirs(self, tmp_path):
        import server.http_server as hs
        upload, censored = tmp_path / "up", tmp_path / "out"
        upload.mkdir(), censored.mkdir()
        with patch.object(hs, "UPLOAD_DIR", upload), patch.object(hs, "CENSORED_PATH", censored), \
                patch.object(hs, "CACHE_DIR", tmp_path / "cache"):
            yield upload, censored

    @staticmethod
    def downloads(content=b"downloaded video"):
        """Patches download_video to write content."""
        def fake(url, path, max_size, digest=None, stop=None):
            path.write_bytes(content)
            if digest is not None:
                digest.update(content)
        return patch("server.http_server.rr.download_video", side_effect=fake)

    async def _run(self, parts, info=None):
        import server.http_server as hs

        async def fake_submit(video_path, output_path, censor_config, early_exit, cache_dir):
            output_path.write_bytes(video_path.read_bytes())
            return MagicMock(success=True, result=str(output_path))

        request = TestCensorVideoFormat._request([TestCensorVideoFormat._text_part(n, t) for n, t in parts])
        with patch.object(hs, "submit_video_job", AsyncMock(side_effect=fake_submit)) as submit, \
                patch.object(hs, "read_video_info", return_value=info or hs.VideoInfo(20, 10.0, 64, 48)), \
                patch.object(hs.web, "StreamResponse") as stream_response:
            stream_response.return_value.prepare = AsyncMock()
            stream_response.return_value.write = AsyncMock()
            response = await hs.censor_video(request)
        return submit, stream_response, response

    @pytest.mark.asyncio
    async def test_name_and_format_come_from_the_url(self, dirs):
        upload, censored = dirs
        with self.downloads() as download:
            submit, stream_response, _ = await self._run([("video_url", "https://example.com/media/my%20anim.gif")])

        assert download.call_args.args[0] == "https://example.com/media/my%20anim.gif"
        assert submit.call_args.kwargs["video_path"] == upload / "my anim.gif"
        assert submit.call_args.kwargs["output_path"] == censored / "my anim.gif"
        assert list(upload.iterdir()) == []  # removed once sent
        assert stream_response.call_args.kwargs["headers"]["Content-Type"] == "image/gif"

    @pytest.mark.asyncio
    async def test_filename_and_format_parts_win_in_any_order(self, dirs):
        upload, censored = dirs
        with self.downloads():
            submit, _, _ = await self._run([("video_url", "https://example.com/watch?v=1"),
                                            ("filename", "clip.mkv"), ("format", "webm")])

        assert submit.call_args.kwargs["video_path"] == upload / "clip.mkv"
        assert submit.call_args.kwargs["output_path"] == censored / "clip.webm"

    @pytest.mark.asyncio
    async def test_url_without_a_file_name_is_video_mp4(self, dirs):
        upload, _ = dirs
        with self.downloads():
            submit, _, _ = await self._run([("video_url", "https://example.com/watch?v=1")])

        assert submit.call_args.kwargs["video_path"] == upload / "video.mp4"

    @pytest.mark.asyncio
    async def test_video_and_video_url_together_is_400(self, dirs):
        import server.http_server as hs
        upload, _ = dirs

        video = MagicMock()
        video.name = "video"
        video.filename = None
        video.read_chunk = AsyncMock(side_effect=[b"data", b""])
        url = TestCensorVideoFormat._text_part("video_url", "https://example.com/a.mp4")
        with self.downloads() as download, patch.object(hs, "submit_video_job") as submit:
            response = await hs.censor_video(TestCensorVideoFormat._request([url, video]))

        assert response.status == 400 and "not both" in json.loads(response.body)["error"]
        download.assert_not_called()
        submit.assert_not_called()
        assert list(upload.iterdir()) == []

    @pytest.mark.asyncio
    async def test_local_urls_are_refused_before_anything_is_read(self, dirs):
        with self.downloads() as download:
            submit, _, response = await self._run([("video_url", "http://127.0.0.1/a.mp4")])

        assert response.status == 400
        assert json.loads(response.body)["code"] == "url_not_allowed"
        download.assert_not_called()
        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_download_is_502_and_leaves_no_file(self, dirs):
        from server.errors import ApiError
        upload, _ = dirs

        def fail(url, path, max_size, digest=None, stop=None):
            path.write_bytes(b"partial")
            raise ApiError("Could not download the video: the server answered 404.", "download_failed", 502)

        with patch("server.http_server.rr.download_video", side_effect=fail):
            submit, _, response = await self._run([("video_url", "https://example.com/a.mp4")])

        assert response.status == 502
        assert json.loads(response.body)["code"] == "download_failed"
        assert list(upload.iterdir()) == []
        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_unreadable_video_is_400_and_removed(self, dirs):
        import server.http_server as hs
        upload, _ = dirs

        with self.downloads(b"<html>not a video</html>"):
            submit, _, response = await self._run([("video_url", "https://example.com/a.mp4")],
                                                  info=hs.VideoInfo(0, 0.0, 0, 0))

        assert response.status == 400
        assert json.loads(response.body)["code"] == "invalid_video"
        assert list(upload.iterdir()) == []
        submit.assert_not_called()


class TestVideoFormats:

    @pytest.mark.asyncio
    async def test_lists_formats(self, fake_request):
        from server.http_server import VIDEO_FORMATS, video_formats

        body = json.loads((await video_formats(fake_request)).body)
        assert body["default"] == "mp4"
        assert {f["format"]: f["content_type"] for f in body["formats"]} == VIDEO_FORMATS


class TestUniquePath:

    def test_reserves_free_names(self, tmp_path):
        from server.helpers import unique_path

        first = unique_path(tmp_path, "a.mp4")
        second = unique_path(tmp_path, "a.mp4")
        third = unique_path(tmp_path, "a.mp4")
        assert [p.name for p in (first, second, third)] == ["a.mp4", "a_1.mp4", "a_2.mp4"]
        assert all(p.exists() for p in (first, second, third))


# ---------------------------------------------------------------------------
# censor_frame: config messages
# ---------------------------------------------------------------------------

VALID_CONFIG = {"features_to_censor": ["FACE_FEMALE"],
                "default_censor_config": {"censor_style": {"type": "blur", "strength": 15}},
                "inverse_censor_style": {"type": "blur", "strength": 31}}


class TestReadConfigMessage:

    def test_valid_config_is_returned_checked(self):
        from server.http_server import read_config_message

        config = read_config_message(json.dumps({"type": "config", "config": dict(VALID_CONFIG)}))
        assert config["features_to_censor"] == ["FACE_FEMALE"]
        assert "stickers_dir" in config  # went through check_client_censor_config

    def test_null_config_resets(self):
        from server.http_server import read_config_message

        assert read_config_message('{"type": "config", "config": null}') is None

    @pytest.mark.parametrize("text, error", [
        ("not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"type": "cancel"}', "Unknown message type"),
        ('{"type": "config"}', "no 'config'"),
        ('{"type": "config", "config": [1]}', "object or null"),
        ('{"type": "config", "config": {"features_to_censor": []}}', "Invalid censor config"),
        ('{"type": "config", "config": {"stickers_dir": "/etc", "features_to_censor": [], '
         '"default_censor_config": {}}}', "not allowed"),
    ])
    def test_invalid_messages_are_rejected(self, text, error):
        from app.config import CensorConfigError
        from server.http_server import read_config_message

        with pytest.raises((ValueError, CensorConfigError), match=error):
            read_config_message(text)


class TestCensorFrameConfig:
    """Drives the WebSocket with a fake pipeline that finishes every job at once."""

    HEADER = bytes([0]) + bytes(12)

    @pytest_asyncio.fixture
    async def ws(self):
        from aiohttp import web as aio_web
        from aiohttp.test_utils import TestClient, TestServer
        import server.http_server as hs

        submitted = []  # the config of every submitted job

        def submit(image_bytes, output_path, config, use_cache=True):
            job = MagicMock()
            job.job_id = str(len(submitted))
            job.result = {"image": image_bytes}
            submitted.append(config)
            handle = MagicMock()
            handle.wait.return_value = job
            return job, handle

        fake_cc = MagicMock()
        fake_cc.wait_any.side_effect = lambda futures: futures[0]

        app = aio_web.Application()
        app.router.add_get('/censor_frame', hs.censor_frame)
        with patch("server.http_server.submit_censoring_job", side_effect=submit), \
                patch("server.http_server.cc", fake_cc), \
                patch("server.http_server.np_to_bytes", side_effect=lambda img, ext: b"censored:" + img):
            client = TestClient(TestServer(app))
            await client.start_server()
            ws = await client.ws_connect('/censor_frame')
            ws.submitted = submitted
            yield ws
            await ws.close()
            await client.close()

    async def frame(self, ws, image: bytes) -> bytes:
        await ws.send_bytes(self.HEADER + image)
        reply = await ws.receive_bytes(timeout=2)
        assert reply[:13] == self.HEADER
        return reply[13:]

    @pytest.mark.asyncio
    async def test_frames_use_the_profile_config_by_default(self, ws):
        assert await self.frame(ws, b"a") == b"censored:a"
        assert ws.submitted == [None]

    @pytest.mark.asyncio
    async def test_config_applies_to_following_frames_and_can_be_reset(self, ws):
        await ws.send_str(json.dumps({"type": "config", "config": dict(VALID_CONFIG)}))
        assert await ws.receive_json(timeout=2) == {"type": "config", "status": "success"}
        await self.frame(ws, b"a")
        assert ws.submitted[-1]["features_to_censor"] == ["FACE_FEMALE"]

        await ws.send_str('{"type": "config", "config": null}')
        assert (await ws.receive_json(timeout=2))["status"] == "success"
        await self.frame(ws, b"b")
        assert ws.submitted[-1] is None

    @pytest.mark.asyncio
    async def test_remembered_frames_are_censored_again_after_a_change(self, ws):
        await self.frame(ws, b"a")
        await self.frame(ws, b"a")
        assert len(ws.submitted) == 1  # the second one came from memory

        await ws.send_str(json.dumps({"type": "config", "config": dict(VALID_CONFIG)}))
        await ws.receive_json(timeout=2)
        await self.frame(ws, b"a")
        assert len(ws.submitted) == 2

    @pytest.mark.asyncio
    async def test_rejected_config_keeps_the_previous_one_and_the_connection(self, ws):
        await ws.send_str(json.dumps({"type": "config", "config": dict(VALID_CONFIG)}))
        await ws.receive_json(timeout=2)

        await ws.send_str('{"type": "config", "config": {"stickers_dir": "/etc"}}')
        reply = await ws.receive_json(timeout=2)
        assert reply["type"] == "error" and "not allowed" in reply["error"]

        await ws.send_str("not json")
        assert (await ws.receive_json(timeout=2))["type"] == "error"

        await self.frame(ws, b"a")
        assert ws.submitted[-1]["features_to_censor"] == ["FACE_FEMALE"]


# ---------------------------------------------------------------------------
# stickers
# ---------------------------------------------------------------------------

@pytest.fixture()
def sticker_root(tmp_path, monkeypatch):
    """data/stickers with plants/fern.png, plants/flowers/rose.JPG, star.webp, a README and a symlink out."""
    import constants

    instance = tmp_path / "instance"
    root = instance / "data" / "stickers"
    (root / "plants" / "flowers").mkdir(parents=True)
    (root / "misc").mkdir()
    (root / "plants" / "fern.png").write_bytes(b"png")
    (root / "plants" / "flowers" / "rose.JPG").write_bytes(b"jpg")
    (root / "star.webp").write_bytes(b"webp")
    (root / "plants" / "README.txt").write_text("not a sticker")
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"secret")
    (root / "misc" / "link.png").symlink_to(outside / "secret.png")
    (root / "out").symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(constants, "instance_dir", instance)
    monkeypatch.setattr(constants, "stickers_root_path", root)
    return root


def make_query_request(query=None, match_info=None):
    request = MagicMock()
    request.query = query or {}
    request.match_info = match_info or {}
    return request


class TestGetStickers:

    def test_lists_categories_and_images_only(self, sticker_root):
        from server.helpers import get_stickers

        result = get_stickers()

        assert result["stickers_dir"] == "data/stickers"
        assert result["categories"] == ["misc", "plants", "plants/flowers"]
        assert result["stickers"] == [
            {"path": "plants/fern.png", "url": "/stickers/plants/fern.png"},
            {"path": "plants/flowers/rose.JPG", "url": "/stickers/plants/flowers/rose.JPG"},
            {"path": "star.webp", "url": "/stickers/star.webp"},
        ]

    def test_subfolder_paths_are_relative_to_it_and_urls_to_the_root(self, sticker_root):
        from server.helpers import get_stickers

        result = get_stickers("data/stickers/plants")

        assert result["stickers_dir"] == "data/stickers/plants"
        assert result["categories"] == ["flowers"]
        assert [s["path"] for s in result["stickers"]] == ["fern.png", "flowers/rose.JPG"]
        assert result["stickers"][0]["url"] == "/stickers/plants/fern.png"

    def test_missing_folder_is_empty(self, sticker_root):
        from server.helpers import get_stickers

        result = get_stickers("data/stickers/not_created_yet")
        assert result["categories"] == [] and result["stickers"] == []

    def test_url_is_quoted(self, sticker_root):
        from server.helpers import get_stickers

        (sticker_root / "my sticker#1.png").write_bytes(b"png")
        urls = [s["url"] for s in get_stickers()["stickers"]]
        assert "/stickers/my%20sticker%231.png" in urls

    @pytest.mark.parametrize("stickers_dir", ["data", "/etc", "data/stickers/.."])
    def test_outside_stickers_root_is_rejected(self, sticker_root, stickers_dir):
        from app.config import CensorConfigError
        from server.helpers import get_stickers

        with pytest.raises(CensorConfigError):
            get_stickers(stickers_dir)


class TestStickersEndpoint:

    @pytest.mark.asyncio
    async def test_returns_listing(self, sticker_root):
        from server.http_server import stickers

        response = await stickers(make_query_request({"dir": "data/stickers/plants"}))

        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "success"
        assert body["categories"] == ["flowers"]

    @pytest.mark.asyncio
    async def test_dir_outside_stickers_root_returns_400(self, sticker_root):
        from server.http_server import stickers

        response = await stickers(make_query_request({"dir": "/etc"}))

        assert response.status == 400
        assert "not allowed" in json.loads(response.body)["error"]


class TestStickerFile:

    @pytest.mark.asyncio
    async def test_serves_sticker(self, sticker_root):
        from aiohttp import web
        from server.http_server import sticker_file

        response = await sticker_file(make_query_request(match_info={"path": "plants/fern.png"}))

        assert isinstance(response, web.FileResponse)
        assert Path(response._path) == (sticker_root / "plants" / "fern.png").resolve()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["plants/README.txt", "missing.png", "../../secret/secret.png",
                                      "misc/link.png", "out/secret.png", "plants"])
    async def test_rejects_non_stickers(self, sticker_root, path):
        from server.http_server import sticker_file

        response = await sticker_file(make_query_request(match_info={"path": path}))

        assert response.status == 404
