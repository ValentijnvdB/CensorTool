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


@pytest.fixture()
def detection_job():
    """A job suitable for detect_features, with one mock bounding box."""
    box = MagicMock()
    box.class_id = 1
    box.label = "face"
    box.score = 0.92
    box.polygon.bounds = (10.0, 20.0, 110.0, 120.0)

    job = MagicMock()
    job.result.features = [[box]]
    return job


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
    async def test_returns_correct_feature_fields(self, fake_request, detection_job, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert len(body["features"]) == 1
        feature = body["features"][0]
        assert feature["class"] == 1
        assert feature["label"] == "face"
        assert feature["confidence"] == pytest.approx(0.92)
        assert feature["bbox"] == [10.0, 20.0, 110.0, 120.0]

    @pytest.mark.asyncio
    async def test_all_bbox_values_are_floats(self, fake_request, detection_job, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        for val in body["features"][0]["bbox"]:
            assert isinstance(val, float)

    @pytest.mark.asyncio
    async def test_class_id_is_int(self, fake_request, detection_job, image_path):
        from server.http_server import detect_features

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=detection_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert isinstance(body["features"][0]["class"], int)

    @pytest.mark.asyncio
    async def test_returns_empty_features_list_when_no_boxes(self, fake_request, image_path):
        from server.http_server import detect_features

        empty_job = MagicMock()
        empty_job.result.features = [[]]

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=empty_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert body["features"] == []

    @pytest.mark.asyncio
    async def test_returns_multiple_boxes(self, fake_request, image_path):
        from server.http_server import detect_features

        def make_box(class_id, label, score):
            b = MagicMock()
            b.class_id = class_id
            b.label = label
            b.score = score
            b.polygon.bounds = (0.0, 0.0, 1.0, 1.0)
            return b

        multi_job = MagicMock()
        multi_job.result.features = [[make_box(0, "person", 0.9), make_box(1, "face", 0.8)]]

        with patch("server.http_server.rr.read_request", return_value=(b"img", image_path, None, "base64")):
            with patch("server.http_server.submit_detection_job", return_value=multi_job):
                result = await detect_features(fake_request)

        body = json.loads(result.body)
        assert len(body["features"]) == 2
        assert body["features"][0]["label"] == "person"
        assert body["features"][1]["label"] == "face"

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
