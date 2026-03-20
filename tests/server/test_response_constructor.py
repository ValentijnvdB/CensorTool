"""
Unit tests for server/response_constructor.py

Covers:
  - _construct_bytes_response
  - _construct_base64_response
  - _construct_url_response
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
# _construct_base64_response
# ---------------------------------------------------------------------------

class TestConstructBase64Response:

    def test_response_body_has_image_data_key(self, png_bytes):
        from server.response_constructor import _construct_base64_response

        resp = _construct_base64_response(png_bytes, extension=".png")
        body = json.loads(resp.body)
        assert "image_data" in body

    def test_response_body_has_mime_type_key(self, png_bytes):
        from server.response_constructor import _construct_base64_response

        resp = _construct_base64_response(png_bytes, extension=".png")
        body = json.loads(resp.body)
        assert "mime_type" in body

    def test_mime_type_reflects_extension(self, png_bytes):
        from server.response_constructor import _construct_base64_response

        resp = _construct_base64_response(png_bytes, extension=".png")
        body = json.loads(resp.body)
        assert body["mime_type"] == "image/.png"

    def test_image_data_is_valid_base64(self, png_bytes):
        from server.response_constructor import _construct_base64_response

        resp = _construct_base64_response(png_bytes, extension=".png")
        body = json.loads(resp.body)
        decoded = base64.b64decode(body["image_data"])
        assert decoded == png_bytes

    def test_different_extensions_reflected_in_mime(self, png_bytes):
        from server.response_constructor import _construct_base64_response

        resp_jpg = _construct_base64_response(png_bytes, extension=".jpg")
        body = json.loads(resp_jpg.body)
        assert ".jpg" in body["mime_type"]

    def test_no_extra_keys_in_body(self, png_bytes):
        from server.response_constructor import _construct_base64_response

        resp = _construct_base64_response(png_bytes, extension=".png")
        body = json.loads(resp.body)
        assert set(body.keys()) == {"image_data", "mime_type"}


# ---------------------------------------------------------------------------
# _construct_url_response
# ---------------------------------------------------------------------------

class TestConstructUrlResponse:

    def test_body_has_image_name_key(self, png_bytes, censored_root):
        from server.response_constructor import _construct_url_response

        path = Path("myimage.png")
        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            resp = _construct_url_response(png_bytes, image_path=path)

        body = json.loads(resp.body)
        assert "image_name" in body

    def test_relative_path_passed_through_unchanged(self, png_bytes, censored_root):
        from server.response_constructor import _construct_url_response

        path = Path("subfolder/myimage.png")
        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            resp = _construct_url_response(png_bytes, image_path=path)

        body = json.loads(resp.body)
        assert "myimage.png" in body["image_name"]

    def test_absolute_path_made_relative_to_censored_path(self, png_bytes, censored_root):
        from server.response_constructor import _construct_url_response

        abs_path = censored_root / "myimage.png"
        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            resp = _construct_url_response(png_bytes, image_path=abs_path)

        body = json.loads(resp.body)
        # Should not start with the root prefix, only the relative part
        assert str(censored_root) not in body["image_name"]
        assert "myimage.png" in body["image_name"]

    def test_response_is_json(self, png_bytes, censored_root):
        from server.response_constructor import _construct_url_response

        with patch("server.response_constructor.CENSORED_PATH", censored_root):
            resp = _construct_url_response(png_bytes, image_path=Path("img.png"))

        # Should not raise
        json.loads(resp.body)


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
