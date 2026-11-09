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
