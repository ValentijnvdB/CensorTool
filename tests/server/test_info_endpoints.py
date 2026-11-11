"""GET /health, GET /censor_config/schema, POST /censor_config/validate and the config checks behind them."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

VALID_CONFIG = {"features_to_censor": ["FACE"],
                "default_censor_config": {"censor_style": {"type": "blur", "strength": 15}},
                "inverse_censor_style": {"type": "blur"}}


def json_request(body):
    request = AsyncMock()
    request.content_type = "application/json"
    request.json = AsyncMock(return_value=body)
    return request


def fake_pipeline(labels: dict, segmentation=()):
    """A started pipeline whose labels come from models; models in segmentation produce outlines."""
    pipeline = MagicMock()
    pipeline.labels = labels
    pipeline.is_running = True
    pipeline.graph.stage.side_effect = lambda name: MagicMock(polygons=name in segmentation)
    return pipeline


# ---------------------------------------------------------------------------
# warnings
# ---------------------------------------------------------------------------

def test_collect_warnings_captures_instead_of_printing(capsys):
    from core.config_utils import collect_warnings, log_warning

    with collect_warnings() as warnings:
        log_warning("one")
    log_warning("two")

    assert warnings == ["one"]
    assert "one" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# inspect_censor_config
# ---------------------------------------------------------------------------

class TestInspectCensorConfig:

    def test_valid_config(self):
        from server.helpers import inspect_censor_config

        with patch("server.helpers._image_pipeline", fake_pipeline({"FACE": "nudenet"})):
            result = inspect_censor_config(VALID_CONFIG)

        assert result.valid and result.errors == [] and result.warnings == []
        assert "stickers_dir" in result.config
        assert "stickers_dir" not in VALID_CONFIG  # the input is not changed

    def test_unknown_labels_and_keys_are_warnings(self):
        from server.helpers import inspect_censor_config

        config = {**VALID_CONFIG, "features_to_censor": ["FACES"], "colour": 1}
        with patch("server.helpers._image_pipeline", fake_pipeline({"FACE": "nudenet"})):
            result = inspect_censor_config(config)

        assert result.valid
        assert any("FACES" in w and "did you mean 'FACE'" in w for w in result.warnings)
        assert any("colour" in w for w in result.warnings)

    def test_warnings_are_reported_every_time(self):
        from server.helpers import inspect_censor_config

        config = {**VALID_CONFIG, "features_to_censor": ["NOPE"]}
        with patch("server.helpers._image_pipeline", fake_pipeline({"FACE": "nudenet"})):
            assert inspect_censor_config(config).warnings == inspect_censor_config(config).warnings != []

    @pytest.mark.parametrize("override, error", [
        ({"clip_to": ["BODYY"]}, "no model produces label 'BODYY'"),
        ({"clip_to": ["HAND"]}, "clip_to needs a segmentation model"),
        ({"shape": "mask"}, "shape MASK for 'FACE' needs a segmentation model"),
    ])
    def test_pipeline_errors(self, override, error):
        from server.helpers import inspect_censor_config

        config = {**VALID_CONFIG, "feature_overrides": {"FACE": override}}
        pipeline = fake_pipeline({"FACE": "nudenet", "HAND": "hands", "BODY": "people"}, segmentation={"people"})
        with patch("server.helpers._image_pipeline", pipeline):
            result = inspect_censor_config(config)

        assert not result.valid and result.code == "invalid_config"
        assert any(error in e for e in result.errors)

    def test_clip_to_a_segmentation_label_is_fine(self):
        from server.helpers import inspect_censor_config

        config = {**VALID_CONFIG, "feature_overrides": {"FACE": {"clip_to": ["BODY"]}}}
        pipeline = fake_pipeline({"FACE": "nudenet", "BODY": "people"}, segmentation={"people"})
        with patch("server.helpers._image_pipeline", pipeline):
            assert inspect_censor_config(config).valid

    @pytest.mark.parametrize("config, code, error", [
        ([1], "invalid_config", "JSON object"),
        ({"features_to_censor": []}, "invalid_config", "Missing required key"),
        ({**VALID_CONFIG, "stickers_dir": "/etc"}, "config_not_allowed", "not allowed"),
    ])
    def test_invalid_configs(self, config, code, error):
        from server.helpers import inspect_censor_config

        result = inspect_censor_config(config)
        assert result.code == code and error in result.errors[0] and result.config is None

    def test_check_censor_config_raises_api_error(self):
        from server.errors import ApiError
        from server.helpers import check_censor_config

        with pytest.raises(ApiError) as e:
            check_censor_config({"stickers_dir": "/etc"})
        assert e.value.status == 400 and e.value.code == "config_not_allowed"


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

class TestValidateEndpoint:

    @pytest.mark.asyncio
    async def test_valid(self):
        from server.http_server import validate_censor_config

        with patch("server.helpers._image_pipeline", fake_pipeline({"FACE": "nudenet"})):
            body = json.loads((await validate_censor_config(json_request({"config": VALID_CONFIG}))).body)

        assert body == {"status": "success", "valid": True, "errors": [], "warnings": []}

    @pytest.mark.asyncio
    async def test_invalid_is_still_200(self):
        from server.http_server import validate_censor_config

        response = await validate_censor_config(json_request({"config": {"features_to_censor": []}}))
        body = json.loads(response.body)

        assert response.status == 200
        assert body["valid"] is False and body["code"] == "invalid_config" and body["errors"]

    @pytest.mark.asyncio
    async def test_missing_config_is_400(self):
        from server.http_server import validate_censor_config

        response = await validate_censor_config(json_request({"features_to_censor": []}))
        assert response.status == 400


class TestHealth:

    @pytest.mark.asyncio
    async def test_running(self):
        from server.http_server import health

        with patch("server.helpers._image_pipeline", fake_pipeline({})), \
                patch("server.helpers._started_at", 0.0), \
                patch("server.helpers.backend.name", return_value="python"), \
                patch("server.helpers.current_profile", return_value="nudity"):
            response = await health(MagicMock())

        body = json.loads(response.body)
        assert response.status == 200
        assert body["status"] == "ok" and body["pipeline_running"] is True
        assert body["backend"] == "python" and body["profile"] == "nudity"
        assert body["uptime_seconds"] > 0
        for key in ("pending_image_jobs", "running_video_jobs", "waiting_video_jobs", "open_frame_streams"):
            assert isinstance(body[key], int)

    @pytest.mark.asyncio
    async def test_not_started_is_503(self):
        from server.http_server import health

        with patch("server.helpers._image_pipeline", None):
            response = await health(MagicMock())

        assert response.status == 503
        assert json.loads(response.body)["status"] == "unavailable"

    def test_pending_image_jobs_drop_finished_ones(self):
        from server.helpers import _Activity

        activity = _Activity(image_jobs={MagicMock(done=lambda: True), MagicMock(done=lambda: False)})
        assert activity.pending_image_jobs() == 1

    def test_pending_image_jobs_survive_a_handle_whose_result_was_taken(self):
        # censor_core < 2.1.1: done() raises after wait() emptied the std::future
        from server.helpers import _Activity

        def raises():
            raise RuntimeError("std::future_error: No associated state")

        activity = _Activity(image_jobs={MagicMock(done=raises), MagicMock(done=lambda: False)})
        assert activity.pending_image_jobs() == 1

    def test_job_result_stops_counting_the_job(self):
        from server import helpers

        handle = MagicMock(done=lambda: False)
        handle.wait.return_value = "job"
        with patch.object(helpers, "activity", helpers._Activity(image_jobs={handle})):
            assert helpers.job_result(handle) == "job"
            assert helpers.activity.image_jobs == set()


class TestSchema:

    @pytest.mark.asyncio
    async def test_is_served(self):
        from server.http_server import censor_config_schema_handler

        body = json.loads((await censor_config_schema_handler(MagicMock())).body)
        assert body["$schema"].startswith("https://json-schema.org/")
        assert body["required"] == ["features_to_censor", "default_censor_config"]

    def test_keys_match_the_parser(self):
        from core.censor_config import CENSOR_BOX_KEYS, ROOT_KEYS
        from server.censor_config_schema import SCHEMA

        assert set(SCHEMA["properties"]) == set(ROOT_KEYS)
        assert set(SCHEMA["$defs"]["censor_box"]["properties"]) == set(CENSOR_BOX_KEYS)

    def test_styles_match_the_parser(self):
        from core.censor_config import _parse_censor_style, CsBlur
        from server.censor_config_schema import CENSOR_STYLES

        for kind, schema in CENSOR_STYLES.items():
            if kind == "ai_remove":
                continue  # its ComfyUI settings come from the server
            style = {"type": kind, **{k: v["default"] for k, v in schema["properties"].items() if k != "type"}}
            parsed = _parse_censor_style(style, "test")
            assert kind == "blur" or not isinstance(parsed, CsBlur), f"{kind} fell back to blur"

    def test_examples_from_the_docs_validate(self):
        jsonschema = pytest.importorskip("jsonschema")
        from server.censor_config_schema import SCHEMA

        config = {
            "features_to_censor": ["face", "BODY"],
            "default_censor_config": {"censor_style": {"type": "bar", "color": [0, 0, 0]}, "min_probability": 0.25},
            "feature_overrides": {
                "face": {"shape": "circle", "clip_to": ["BODY"],
                         "overlay": {"type": "sticker", "probability": 0.5, "stickers": ["plants"]}},
                "BODY": {"censor_style": {"type": "pixel", "factor": 30},
                         "border": {"color": [200, 0, 150], "thickness": 8}},
            },
            "inverse_censor_style": {"type": "blur", "strength": 51},
            "stickers_dir": "data/stickers",
        }
        jsonschema.validate(config, SCHEMA)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**config, "inverse_censor_style": {"type": "debug"}}, SCHEMA)
