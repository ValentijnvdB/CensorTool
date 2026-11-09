"""
Unit tests for server/http_server.py

Covers all three route handlers:
  - censor_image
  - detect_features
  - reset_cache

Run with:
    pytest test_http_server.py -v
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_request():
    """Minimal aiohttp request mock."""
    return AsyncMock()


@pytest.fixture()
def successful_job():
    """A job whose .success is True and carries a fake image."""
    job = MagicMock()
    job.success = True
    job.image = b"censoredimage"
    return job


@pytest.fixture()
def failed_job():
    """A job whose .success is False with a descriptive error."""
    job = MagicMock()
    job.success = False
    job.error = Exception("pipeline exploded")
    return job


def make_detection(id=0, label="FACE_FEMALE", score=0.92, box=(10, 20, 100, 100),
                   polygons=(), model="nudenet", parent=None) -> dict:
    """A detection dict as censor_core 2.0 returns it in result["detections"]."""
    return {"id": id, "label": label, "score": score, "box": box,
            "polygons": list(polygons), "model": model, "parent": parent}


def make_detection_submit(detections: list[dict], cached_models=()):
    """Return value of submit_detection_job: (job, handle), where handle.wait() returns the finished job."""
    job = MagicMock()
    job.success = True
    job.result = {"image": None, "output_path": None,
                  "detections": detections, "cached_models": list(cached_models)}
    handle = MagicMock()
    handle.wait.return_value = job
    return job, handle


@pytest.fixture()
def detection_job():
    """A detect_features submission with one detection."""
    return make_detection_submit([make_detection()])


@pytest.fixture()
def image_path():
    return Path("photo.png")


@pytest.fixture()
def fake_response():
    return MagicMock()


# ---------------------------------------------------------------------------
# censor_image
# ---------------------------------------------------------------------------

class TestCensorImage:

    @pytest.mark.asyncio
    async def test_returns_constructed_response_on_success(
        self, fake_request, successful_job, image_path, fake_response
    ):
        from server.http_server import censor_image

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_censoring_job", return_value=successful_job):
                with patch("server.http_server.construct_response", return_value=fake_response):
                    result = await censor_image(fake_request)

        assert result is fake_response

    @pytest.mark.asyncio
    async def test_calls_construct_response_with_correct_args(
        self, fake_request, successful_job, image_path
    ):
        from server.http_server import censor_image

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_censoring_job", return_value=successful_job):
                with patch("server.http_server.construct_response", return_value=MagicMock()) as mock_construct:
                    await censor_image(fake_request)

        mock_construct.assert_called_once_with(
            expected_response="base64",
            image=successful_job.image,
            extension=image_path.suffix,
            image_path=None,          # no output_path for non-url response
            name=image_path.name,
        )

    @pytest.mark.asyncio
    async def test_output_path_set_for_url_response(
        self, fake_request, successful_job, image_path
    ):
        from server.http_server import censor_image

        censored_path = Path("/data/censored")
        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "url")):
            with patch("server.http_server.CENSORED_PATH", censored_path):
                with patch("server.http_server.submit_censoring_job", return_value=successful_job) as mock_submit:
                    with patch("server.http_server.construct_response", return_value=MagicMock()):
                        await censor_image(fake_request)

        _, output_path_arg, _ = mock_submit.call_args[0]
        assert output_path_arg == censored_path / image_path.name

    @pytest.mark.asyncio
    async def test_output_path_is_none_for_base64_response(
        self, fake_request, successful_job, image_path
    ):
        from server.http_server import censor_image

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_censoring_job", return_value=successful_job) as mock_submit:
                with patch("server.http_server.construct_response", return_value=MagicMock()):
                    await censor_image(fake_request)

        _, output_path_arg, _ = mock_submit.call_args[0]
        assert output_path_arg is None

    @pytest.mark.asyncio
    async def test_forwards_censor_config_to_job(self, fake_request, successful_job, image_path):
        from server.http_server import censor_image

        censor_config = MagicMock()
        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, censor_config, "base64")):
            with patch("server.http_server.submit_censoring_job", return_value=successful_job) as mock_submit:
                with patch("server.http_server.construct_response", return_value=MagicMock()):
                    await censor_image(fake_request)

        _, _, config_arg = mock_submit.call_args[0]
        assert config_arg is censor_config

    @pytest.mark.asyncio
    async def test_returns_500_when_job_fails(self, fake_request, failed_job, image_path):
        from server.http_server import censor_image

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_censoring_job", return_value=failed_job):
                result = await censor_image(fake_request)

        assert result.status == 500
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_returns_500_on_read_request_exception(self, fake_request):
        from server.http_server import censor_image

        with patch("server.http_server.rr.read_request", side_effect=RuntimeError("bad request")):
            result = await censor_image(fake_request)

        assert result.status == 500
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_returns_500_on_unexpected_exception(self, fake_request, image_path):
        from server.http_server import censor_image

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_censoring_job", side_effect=Exception("unexpected")):
                result = await censor_image(fake_request)

        assert result.status == 500


# ---------------------------------------------------------------------------
# detect_features
# ---------------------------------------------------------------------------

class TestDetectFeatures:

    @pytest.mark.asyncio
    async def test_returns_success_status(self, fake_request, detection_job, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert body["status"] == "success"

    @pytest.mark.asyncio
    async def test_returns_correct_file_name(self, fake_request, detection_job):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", Path("face.jpg"), None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert body["file_name"] == "face.jpg"

    @pytest.mark.asyncio
    async def test_returns_correct_detection_fields(self, fake_request, detection_job, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert len(body["detections"]) == 1
        d = body["detections"][0]
        assert d["id"] == 0
        assert d["label"] == "FACE_FEMALE"
        assert d["score"] == pytest.approx(0.92)
        assert d["box"] == [10.0, 20.0, 100.0, 100.0]
        assert d["polygons"] == []
        assert d["model"] == "nudenet"
        assert d["parent"] is None

    @pytest.mark.asyncio
    async def test_box_values_are_floats(self, fake_request, detection_job, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        for val in body["detections"][0]["box"]:
            assert isinstance(val, float)

    @pytest.mark.asyncio
    async def test_polygons_and_parent_are_serialized(self, fake_request, image_path):
        from server.http_server import detect_features

        import numpy as np
        eye = make_detection(id=1, label="EYES_FEMALE", model="landmarks", parent=0,
                             polygons=[np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)])
        submit = make_detection_submit([make_detection(), eye], cached_models=["nudenet"])

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=submit):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert body["detections"][1]["polygons"] == [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        assert body["detections"][1]["parent"] == 0
        assert body["cached_models"] == ["nudenet"]

    @pytest.mark.asyncio
    async def test_returns_empty_detections_list_when_nothing_found(self, fake_request, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=make_detection_submit([])):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert body["detections"] == []

    @pytest.mark.asyncio
    async def test_returns_multiple_detections_in_order(self, fake_request, image_path):
        from server.http_server import detect_features

        submit = make_detection_submit([
            make_detection(id=0, label="BODY", model="people"),
            make_detection(id=1, label="FACE_FEMALE"),
        ])

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=submit):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert [d["label"] for d in body["detections"]] == ["BODY", "FACE_FEMALE"]

    @pytest.mark.asyncio
    async def test_returns_500_when_job_failed(self, fake_request, image_path):
        from server.http_server import detect_features

        job, handle = make_detection_submit([])
        job.success = False
        job.error = "pipeline exploded"

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=(job, handle)):
                result = await detect_features(fake_request)

        assert result.status == 500

    @pytest.mark.asyncio
    async def test_returns_500_on_read_request_exception(self, fake_request):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", side_effect=Exception("bad")):
            result = await detect_features(fake_request)

        assert result.status == 500
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_returns_500_on_job_exception(self, fake_request, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", side_effect=RuntimeError("crash")):
                result = await detect_features(fake_request)

        assert result.status == 500


# ---------------------------------------------------------------------------
# reset_cache
# ---------------------------------------------------------------------------

class TestResetCache:

    @pytest.mark.asyncio
    async def test_returns_success_status(self, fake_request, tmp_path):
        from server.http_server import reset_cache

        with patch("server.http_server.CACHE_DIR", tmp_path):
            result = await reset_cache(fake_request)

        body = json.loads(result.body)
        assert body["status"] == "success"

    @pytest.mark.asyncio
    async def test_deletes_all_files_in_cache_dir(self, fake_request, tmp_path):
        from server.http_server import reset_cache

        (tmp_path / "cache1.bin").write_bytes(b"a")
        (tmp_path / "cache2.bin").write_bytes(b"b")
        (tmp_path / "cache3.bin").write_bytes(b"c")

        with patch("server.http_server.CACHE_DIR", tmp_path):
            await reset_cache(fake_request)

        remaining_files = [p for p in tmp_path.iterdir() if p.is_file()]
        assert remaining_files == []

    @pytest.mark.asyncio
    async def test_removes_subdirectories(self, fake_request, tmp_path):
        from server.http_server import reset_cache

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        with patch("server.http_server.CACHE_DIR", tmp_path):
            await reset_cache(fake_request)

        assert not subdir.exists()

    @pytest.mark.asyncio
    async def test_succeeds_when_cache_dir_does_not_exist(self, fake_request, tmp_path):
        from server.http_server import reset_cache

        missing = tmp_path / "nonexistent"
        with patch("server.http_server.CACHE_DIR", missing):
            result = await reset_cache(fake_request)

        body = json.loads(result.body)
        assert body["status"] == "success"

    @pytest.mark.asyncio
    async def test_returns_500_on_os_error(self, fake_request):
        from server.http_server import reset_cache

        with patch("server.http_server.CACHE_DIR") as mock_dir:
            mock_dir.exists.side_effect = OSError("permission denied")
            result = await reset_cache(fake_request)

        assert result.status == 500
        body = json.loads(result.body)
        assert "error" in body


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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("parts, output_name, content_type", [
        ([("filename", "clip.mkv")], "clip.mkv", "video/x-matroska"),
        ([("filename", "clip.mkv"), ("format", "webm")], "clip.webm", "video/webm"),
        ([("format", "gif")], "video.gif", "image/gif"),
        ([], "video.mp4", "video/mp4"),
    ])
    async def test_output_path_and_content_type(self, tmp_path, parts, output_name, content_type):
        import server.http_server as hs

        output = tmp_path / output_name
        output.write_bytes(b"data")
        result = MagicMock(success=True, result=output)
        request = self._request([self._text_part(name, text) for name, text in parts])

        with patch.object(hs, "submit_video_job", AsyncMock(return_value=result)) as submit, \
                patch.object(hs.web, "StreamResponse") as stream_response:
            stream_response.return_value.prepare = AsyncMock()
            stream_response.return_value.write = AsyncMock()
            await hs.censor_video(request)

        assert submit.call_args.kwargs["output_path"] == hs.CENSORED_PATH / output_name
        assert stream_response.call_args.kwargs["headers"]["Content-Type"] == content_type


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

        def submit(image_bytes, output_path, config):
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
