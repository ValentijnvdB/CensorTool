"""
Unit tests for server/request_reader.py

Covers:
  - get_image_path_from_url
  - get_image_from_base64
  - get_image_from_source
  - get_image_from_json
  - read_request

Run with:
    pytest test_request_reader.py -v
"""

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_extensions():
    return [".jpg", ".jpeg", ".png", ".gif", ".webp"]


@pytest.fixture()
def png_base64_payload(tmp_path):
    """A well-formed base64 payload for a PNG image."""
    raw = b"fake png bytes"
    return {
        "mime_type": "image/png",
        "image_url": "http://example.com/image.png",
        "image_data": base64.b64encode(raw).decode(),
        "_raw": raw,
    }


@pytest.fixture()
def source_payload():
    return {"image_url": "http://example.com/photo.jpg"}


# ---------------------------------------------------------------------------
# get_image_path_from_url
# ---------------------------------------------------------------------------

class TestGetImagePathFromUrl:

    def test_returns_path_object(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                result = get_image_path_from_url("http://example.com/photo.jpg")

        assert isinstance(result, Path)

    def test_extension_inferred_from_url(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                result = get_image_path_from_url("http://example.com/photo.png")

        assert result.suffix == ".png"

    def test_explicit_extension_overrides_url(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                result = get_image_path_from_url("http://example.com/photo", extension=".jpg")

        assert result.suffix == ".jpg"

    def test_extension_dot_added_if_missing(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                result = get_image_path_from_url("http://example.com/photo", extension="png")

        assert result.suffix == ".png"

    def test_path_is_under_upload_dir(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                result = get_image_path_from_url("http://example.com/photo.png")

        assert result.parent == tmp_path

    def test_same_url_produces_same_filename(self, valid_extensions, tmp_path):
        """Hash-based naming must be deterministic."""
        from server.request_reader import get_image_path_from_url

        url = "http://example.com/stable.png"
        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                p1 = get_image_path_from_url(url)
                p2 = get_image_path_from_url(url)

        assert p1 == p2

    def test_different_urls_produce_different_filenames(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                p1 = get_image_path_from_url("http://example.com/a.png")
                p2 = get_image_path_from_url("http://example.com/b.png")

        assert p1 != p2

    def test_invalid_extension_raises(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with pytest.raises(Exception, match="Invalid image extension"):
                    get_image_path_from_url("http://example.com/malware.exe")

    def test_unknown_extension_raises(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with pytest.raises(Exception, match="Invalid image extension"):
                    get_image_path_from_url("http://example.com/archive.zip")


# ---------------------------------------------------------------------------
# get_image_from_base64
# ---------------------------------------------------------------------------

class TestGetImageFromBase64:

    def test_returns_tuple_of_bytes_and_path(self, png_base64_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_base64

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                image_bytes, path = get_image_from_base64(png_base64_payload)

        assert isinstance(image_bytes, bytes)
        assert isinstance(path, Path)

    def test_decodes_base64_correctly(self, png_base64_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_base64

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                image_bytes, _ = get_image_from_base64(png_base64_payload)

        assert image_bytes == png_base64_payload["_raw"]

    def test_path_extension_matches_mime_type(self, png_base64_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_base64

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                _, path = get_image_from_base64(png_base64_payload)

        assert path.suffix == ".png"


# ---------------------------------------------------------------------------
# get_image_from_source
# ---------------------------------------------------------------------------

class TestGetImageFromSource:

    def test_fetches_remote_url(self, source_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_source

        remote_content = b"remote image content"
        mock_resp = MagicMock()
        mock_resp.content = remote_content

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.requests.get", return_value=mock_resp):
                    with patch("pathlib.Path.exists", return_value=False):
                        image_bytes, _ = get_image_from_source(source_payload)

        assert image_bytes == remote_content

    def test_uses_cached_file_without_network_call(self, source_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_source

        cached = b"cached bytes"

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                # Pre-populate the cache
                from server.request_reader import get_image_path_from_url
                with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
                    cache_path = get_image_path_from_url(source_payload["image_url"])
                cache_path.write_bytes(cached)

                with patch("server.request_reader.requests.get") as mock_get:
                    image_bytes, _ = get_image_from_source(source_payload)

                mock_get.assert_not_called()

        assert image_bytes == cached

    def test_raises_on_missing_url_key(self):
        from server.request_reader import get_image_from_source

        with pytest.raises(Exception, match="No image url provided"):
            get_image_from_source({})

    def test_raises_on_localhost_url(self):
        from server.request_reader import get_image_from_source

        with pytest.raises(Exception, match="Invalid image url"):
            get_image_from_source({"image_url": "http://localhost/image.png"})

    def test_raises_on_127_url(self):
        from server.request_reader import get_image_from_source

        with pytest.raises(Exception, match="Invalid image url"):
            get_image_from_source({"image_url": "http://127.0.0.1/image.png"})

    def test_returns_path_under_upload_dir(self, source_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_source

        mock_resp = MagicMock()
        mock_resp.content = b"data"

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.requests.get", return_value=mock_resp):
                    with patch("pathlib.Path.exists", return_value=False):
                        _, path = get_image_from_source(source_payload)

        assert path.parent == tmp_path

    def test_request_uses_five_second_timeout(self, source_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_source

        mock_resp = MagicMock()
        mock_resp.content = b"data"

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.requests.get", return_value=mock_resp) as mock_get:
                    with patch("pathlib.Path.exists", return_value=False):
                        get_image_from_source(source_payload)

        _, kwargs = mock_get.call_args
        assert kwargs.get("timeout") == 5


# ---------------------------------------------------------------------------
# get_image_from_json
# ---------------------------------------------------------------------------

class TestGetImageFromJson:

    def test_raises_when_type_key_missing(self):
        from server.request_reader import get_image_from_json

        with pytest.raises(Exception, match="No message type provided"):
            get_image_from_json({})

    def test_raises_on_unknown_type(self):
        from server.request_reader import get_image_from_json

        with pytest.raises(Exception, match="Unknown image message type"):
            get_image_from_json({"type": "ftp"})

    def test_dispatches_to_base64(self, png_base64_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_json

        payload = {**png_base64_payload, "type": "base64"}
        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                image_bytes, _ = get_image_from_json(payload)

        assert image_bytes == png_base64_payload["_raw"]

    def test_dispatches_to_source(self, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_json

        remote = b"remote"
        mock_resp = MagicMock()
        mock_resp.content = remote
        payload = {"type": "source", "image_url": "http://example.com/img.jpg"}

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.requests.get", return_value=mock_resp):
                    with patch("pathlib.Path.exists", return_value=False):
                        image_bytes, _ = get_image_from_json(payload)

        assert image_bytes == remote

    def test_writes_image_to_disk(self, png_base64_payload, valid_extensions, tmp_path):
        """get_image_from_json must persist the image at the returned path."""
        from server.request_reader import get_image_from_json

        payload = {**png_base64_payload, "type": "base64"}
        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                image_bytes, path = get_image_from_json(payload)

        assert path.exists()
        assert path.read_bytes() == image_bytes


# ---------------------------------------------------------------------------
# read_request
# ---------------------------------------------------------------------------

class TestReadRequest:

    @pytest.mark.asyncio
    async def test_raises_for_non_json_content_type(self):
        from server.request_reader import read_request

        request = AsyncMock()
        request.content_type = "multipart/form-data"

        with pytest.raises(ValueError, match="Only json type is supported"):
            await read_request(request)

    @pytest.mark.asyncio
    async def test_raises_for_text_plain_content_type(self):
        from server.request_reader import read_request

        request = AsyncMock()
        request.content_type = "text/plain"

        with pytest.raises(ValueError, match="Only json type is supported"):
            await read_request(request)

    @pytest.mark.asyncio
    async def test_returns_four_tuple_for_valid_json(self, valid_extensions, tmp_path):
        from server.request_reader import read_request

        raw = b"data"
        payload = {
            "type": "base64",
            "mime_type": "image/png",
            "image_url": "http://example.com/img.png",
            "image_data": base64.b64encode(raw).decode(),
        }
        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=payload)

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                result = await read_request(request)

        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_default_expected_response_is_base64(self, valid_extensions, tmp_path):
        from server.request_reader import read_request

        payload = {
            "type": "base64",
            "mime_type": "image/png",
            "image_url": "http://example.com/img.png",
            "image_data": base64.b64encode(b"x").decode(),
        }
        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=payload)

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                _, _, _, exp_response = await read_request(request)

        assert exp_response == "base64"

    @pytest.mark.asyncio
    async def test_honours_explicit_expected_response(self, valid_extensions, tmp_path):
        from server.request_reader import read_request

        payload = {
            "type": "base64",
            "mime_type": "image/png",
            "image_url": "http://example.com/img.png",
            "image_data": base64.b64encode(b"x").decode(),
            "expected_response": "url",
        }
        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=payload)

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                _, _, _, exp_response = await read_request(request)

        assert exp_response == "url"

    @pytest.mark.asyncio
    async def test_censor_config_is_none_when_absent(self, valid_extensions, tmp_path):
        from server.request_reader import read_request

        payload = {
            "type": "base64",
            "mime_type": "image/png",
            "image_url": "http://example.com/img.png",
            "image_data": base64.b64encode(b"x").decode(),
        }
        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=payload)

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                _, _, censor_config, _ = await read_request(request)

        assert censor_config is None

    @pytest.mark.asyncio
    async def test_censor_config_parsed_when_present(self, valid_extensions, tmp_path):
        from server.request_reader import read_request

        payload = {
            "type": "base64",
            "mime_type": "image/png",
            "image_url": "http://example.com/img.png",
            "image_data": base64.b64encode(b"x").decode(),
            "config": {"threshold": 0.5},
        }
        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=payload)

        fake_config = MagicMock()
        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.construct_censor_config", return_value=fake_config) as mock_cc:
                    _, _, censor_config, _ = await read_request(request)

        mock_cc.assert_called_once_with({"threshold": 0.5})
        assert censor_config is fake_config

    @pytest.mark.asyncio
    async def test_image_bytes_and_path_come_from_get_image_from_json(self, valid_extensions, tmp_path):
        from server.request_reader import read_request

        payload = {
            "type": "base64",
            "mime_type": "image/png",
            "image_url": "http://example.com/img.png",
            "image_data": base64.b64encode(b"actual").decode(),
        }
        request = AsyncMock()
        request.content_type = "application/json"
        request.json = AsyncMock(return_value=payload)

        fake_bytes = b"actual"
        fake_path = Path("/uploads/something.png")

        with patch("server.request_reader.get_image_from_json", return_value=(fake_bytes, fake_path)) as mock_gij:
            image_bytes, image_path, _, _ = await read_request(request)

        mock_gij.assert_called_once_with(payload)
        assert image_bytes is fake_bytes
        assert image_path is fake_path
