"""
Unit tests for server/response_constructor.py

Covers:
  - _construct_bytes_response
  - _base64_body
  - _url_body
  - construct_response  (dispatcher)

Run with:
    pytest test_response_constructor.py -v
"""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def png_bytes():
    """Minimal valid PNG bytes (1×1 black pixel)."""
    import cv2
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return bytes(cv2.imencode(".png", img)[1])


@pytest.fixture()
def image_array():
    return np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture()
def censored_root(tmp_path):
    root = tmp_path / "censored"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# _construct_bytes_response
# ---------------------------------------------------------------------------

class TestConstructBytesResponse:

    def test_content_type_is_image_png(self, png_bytes):
        from server.response_constructor import _construct_bytes_response

        resp = _construct_bytes_response(png_bytes, name="out.png")
        assert resp.content_type == "image/png"

    def test_body_equals_input_bytes(self, png_bytes):
        from server.response_constructor import _construct_bytes_response

        resp = _construct_bytes_response(png_bytes, name="out.png")
        assert resp.body == png_bytes

    def test_content_disposition_contains_filename(self, png_bytes):
        from server.response_constructor import _construct_bytes_response

        resp = _construct_bytes_response(png_bytes, name="portrait.png")
        disposition = resp.headers.get("Content-Disposition", "")
        assert "portrait.png" in disposition

    def test_content_disposition_is_attachment(self, png_bytes):
        from server.response_constructor import _construct_bytes_response

        resp = _construct_bytes_response(png_bytes, name="img.png")
        disposition = resp.headers.get("Content-Disposition", "")
        assert disposition.startswith("attachment")

    def test_different_filenames_reflected_in_header(self, png_bytes):
        from server.response_constructor import _construct_bytes_response

        resp_a = _construct_bytes_response(png_bytes, name="alpha.png")
        resp_b = _construct_bytes_response(png_bytes, name="beta.png")

        assert "alpha.png" in resp_a.headers.get("Content-Disposition", "")
        assert "beta.png" in resp_b.headers.get("Content-Disposition", "")


# ---------------------------------------------------------------------------
# _base64_body
# ---------------------------------------------------------------------------

class TestBase64Body:

    def test_has_image_data_and_mime_type_only(self, png_bytes):
        from server.response_constructor import _base64_body

        assert set(_base64_body(png_bytes, extension=".png")) == {"image_data", "mime_type"}

    def test_image_data_is_valid_base64(self, png_bytes):
        from server.response_constructor import _base64_body

        assert base64.b64decode(_base64_body(png_bytes, extension=".png")["image_data"]) == png_bytes

    @pytest.mark.parametrize("extension, expected", [
        (".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".JPG", "image/jpeg"),
        (".tiff", "image/tiff"), (".webp", "image/webp"), (".avif", "image/avif"), (".gif", "image/gif"),
        ("bmp", "image/bmp"),
    ])
    def test_mime_type_is_a_real_mime_type(self, png_bytes, extension, expected):
        from server.response_constructor import _base64_body

        assert _base64_body(png_bytes, extension=extension)["mime_type"] == expected


# ---------------------------------------------------------------------------
# _url_body
# ---------------------------------------------------------------------------

class TestUrlBody:

    def test_has_image_name_and_url(self, censored_root):
        from server.response_constructor import _url_body

        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            body = _url_body(Path("my image.png"))

        assert body == {"image_name": "my image.png", "url": "/censored/my%20image.png"}

    def test_relative_path_passed_through_unchanged(self, censored_root):
        from server.response_constructor import _url_body

        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            body = _url_body(Path("subfolder/myimage.png"))

        assert body["image_name"] == "subfolder/myimage.png"

    def test_absolute_path_made_relative_to_censored_path(self, censored_root):
        from server.response_constructor import _url_body

        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            body = _url_body(censored_root / "myimage.png")

        assert body["image_name"] == "myimage.png"


# ---------------------------------------------------------------------------
# info fields
# ---------------------------------------------------------------------------

class TestInfo:

    INFO = {"width": 4, "height": 3, "processing_ms": 12, "detections": []}

    def test_json_answers_have_status_and_info(self, png_bytes, censored_root):
        from server.response_constructor import construct_response

        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            for mode in ("base64", "url"):
                body = json.loads(construct_response(mode, png_bytes, ".png", image_path=Path("x.png"),
                                                     info=self.INFO).body)
                assert body["status"] == "success"
                assert {k: body[k] for k in self.INFO} == self.INFO

    def test_bytes_answer_sends_info_as_headers(self, png_bytes):
        from server.response_constructor import construct_response

        resp = construct_response("bytes", png_bytes, ".jpg", name="x.jpg", info=self.INFO)
        assert resp.content_type == "image/jpeg"
        assert resp.headers["X-Image-Width"] == "4"
        assert resp.headers["X-Image-Height"] == "3"
        assert resp.headers["X-Processing-Time-Ms"] == "12"


# ---------------------------------------------------------------------------
# construct_response  (dispatcher)
# ---------------------------------------------------------------------------

class TestConstructResponse:

    def test_bytes_mode_returns_image_png_content_type(self, png_bytes):
        from server.response_constructor import construct_response

        resp = construct_response("bytes", png_bytes, extension=".png", name="x.png")
        assert resp.content_type == "image/png"

    def test_base64_mode_returns_json_with_image_data(self, png_bytes):
        from server.response_constructor import construct_response

        resp = construct_response("base64", png_bytes, extension=".png")
        body = json.loads(resp.body)
        assert "image_data" in body

    def test_url_mode_returns_json_with_image_name(self, png_bytes, censored_root):
        from server.response_constructor import construct_response

        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            resp = construct_response(
                "url", png_bytes, extension=".png",
                image_path=Path("img.png"),
            )

        body = json.loads(resp.body)
        assert "image_name" in body

    def test_numpy_array_is_converted_before_dispatch(self, image_array):
        from server.response_constructor import construct_response

        with patch("server.response_constructor.utils.np_to_bytes", return_value=b"converted") as mock_conv:
            construct_response("bytes", image_array, extension=".png", name="x.png")

        mock_conv.assert_called_once_with(image_array, ".png")

    def test_bytes_input_skips_numpy_conversion(self, png_bytes):
        from server.response_constructor import construct_response

        with patch("server.response_constructor.utils.np_to_bytes") as mock_conv:
            construct_response("bytes", png_bytes, extension=".png", name="x.png")

        mock_conv.assert_not_called()

    def test_unknown_response_type_raises_value_error(self, png_bytes):
        from server.response_constructor import construct_response

        with pytest.raises(ValueError, match="Unexpected response type"):
            construct_response("xml", png_bytes, extension=".png")

    def test_unknown_response_type_message_contains_type(self, png_bytes):
        from server.response_constructor import construct_response

        with pytest.raises(ValueError, match="xml"):
            construct_response("xml", png_bytes, extension=".png")

    @pytest.mark.parametrize("mode", ["bytes", "base64", "url"])
    def test_all_valid_modes_do_not_raise(self, mode, png_bytes, censored_root):
        from server.response_constructor import construct_response

        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            # Should complete without raising
            construct_response(
                mode, png_bytes, extension=".png",
                image_path=Path("img.png"),
                name="img.png",
            )
