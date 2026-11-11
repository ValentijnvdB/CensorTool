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
# get_image_from_source / download_image / check_url_allowed
# ---------------------------------------------------------------------------

def fake_response(content=b"", status=200, location=None):
    """A streamed requests response."""
    resp = MagicMock()
    resp.status_code = status
    resp.is_redirect = location is not None
    resp.headers = {"Location": location} if location else {}
    resp.iter_content.return_value = [content]
    return resp


def addrinfo(*addresses):
    return [(None, None, None, "", (a, 80)) for a in addresses]


@pytest.fixture()
def public_dns():
    """Every host name resolves to a public address."""
    with patch("server.request_reader.socket.getaddrinfo", return_value=addrinfo("93.184.216.34")) as m:
        yield m


class TestCheckUrlAllowed:

    @pytest.mark.parametrize("url", [
        "http://localhost/a.png", "http://LOCALHOST:8443/a.png", "http://foo.localhost/a.png",
        "http://127.0.0.1/a.png", "http://127.1.2.3/a.png", "http://[::1]/a.png", "http://0.0.0.0/a.png",
        "http://192.168.1.10/a.png", "http://10.0.0.1/a.png", "http://172.16.0.1/a.png",
        "http://169.254.169.254/latest/meta-data", "http://[::ffff:127.0.0.1]/a.png", "http://[fe80::1]/a.png",
        "http://100.64.0.1/a.png",
    ])
    def test_refuses_this_machine_and_the_local_network(self, url):
        from server.errors import URL_NOT_ALLOWED
        from server.request_reader import ApiError, check_url_allowed

        with pytest.raises(ApiError) as e:
            check_url_allowed(url)
        assert e.value.code == URL_NOT_ALLOWED and e.value.status == 400

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a.png", "http:///a.png",
                                     "http://example.com:99999/a.png"])
    def test_refuses_other_schemes_and_broken_urls(self, url):
        from server.request_reader import ApiError, check_url_allowed

        with pytest.raises(ApiError, match="Invalid URL"):
            check_url_allowed(url)

    @pytest.mark.parametrize("addresses", [("127.0.0.1",), ("93.184.216.34", "10.0.0.5"), ("::1",)])
    def test_refuses_host_names_that_resolve_locally(self, addresses):
        from server.request_reader import ApiError, check_url_allowed

        with patch("server.request_reader.socket.getaddrinfo", return_value=addrinfo(*addresses)):
            with pytest.raises(ApiError, match="local network"):
                check_url_allowed("http://sneaky.example.com/a.png")

    def test_allows_public_hosts(self, public_dns):
        from server.request_reader import check_url_allowed

        check_url_allowed("https://example.com/a.png")
        check_url_allowed("http://93.184.216.34/a.png")

    def test_unknown_host_is_a_download_error(self):
        import socket
        from server.errors import DOWNLOAD_FAILED
        from server.request_reader import ApiError, check_url_allowed

        with patch("server.request_reader.socket.getaddrinfo", side_effect=socket.gaierror("nope")):
            with pytest.raises(ApiError) as e:
                check_url_allowed("http://does-not-exist.example/a.png")
        assert e.value.code == DOWNLOAD_FAILED and e.value.status == 502


class TestDownloadImage:

    def test_follows_redirects_to_allowed_urls(self, public_dns):
        from server.request_reader import download_image

        responses = [fake_response(location="/other.png"), fake_response(b"image")]
        with patch("server.request_reader.requests.get", side_effect=responses) as get:
            assert download_image("http://example.com/a.png") == b"image"
        assert get.call_args_list[1].args[0] == "http://example.com/other.png"
        assert all(c.kwargs["allow_redirects"] is False for c in get.call_args_list)

    def test_refuses_redirects_to_this_machine(self, public_dns):
        from server.request_reader import ApiError, download_image

        with patch("server.request_reader.requests.get",
                   return_value=fake_response(location="http://127.0.0.1/secret")) as get:
            with pytest.raises(ApiError, match="Invalid URL"):
                download_image("http://example.com/a.png")
        get.assert_called_once()

    def test_http_errors_are_download_errors(self, public_dns):
        from server.request_reader import ApiError, download_image

        with patch("server.request_reader.requests.get", return_value=fake_response(b"<html>", status=404)):
            with pytest.raises(ApiError, match="answered 404") as e:
                download_image("http://example.com/a.png")
        assert e.value.status == 502

    def test_connection_errors_are_download_errors(self, public_dns):
        import requests
        from server.request_reader import ApiError, download_image

        with patch("server.request_reader.requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(ApiError, match="Could not download"):
                download_image("http://example.com/a.png")


class TestDownloads:

    def test_size_limit(self, public_dns):
        from server.request_reader import ApiError, iter_download

        response = fake_response()
        response.iter_content.return_value = [b"x" * 600, b"x" * 600]
        with patch("server.request_reader.requests.get", return_value=response):
            with pytest.raises(ApiError, match="larger than"):
                list(iter_download("http://example.com/a.mp4", "video", max_size=1000))

    def test_download_video_writes_and_hashes(self, public_dns, tmp_path):
        import hashlib
        from server.request_reader import download_video

        response = fake_response()
        response.iter_content.return_value = [b"abc", b"def"]
        digest = hashlib.sha256()
        with patch("server.request_reader.requests.get", return_value=response):
            download_video("http://example.com/a.mp4", tmp_path / "a.mp4", 2**20, digest)

        assert (tmp_path / "a.mp4").read_bytes() == b"abcdef"
        assert digest.hexdigest() == hashlib.sha256(b"abcdef").hexdigest()

    def test_video_errors_say_video(self, public_dns, tmp_path):
        from server.request_reader import ApiError, download_video

        with patch("server.request_reader.requests.get", return_value=fake_response(status=403)):
            with pytest.raises(ApiError, match="Could not download the video: the server answered 403"):
                download_video("http://example.com/a.mp4", tmp_path / "a.mp4", 2**20)

    def test_video_larger_than_max_size_is_refused(self, public_dns, tmp_path):
        from server.request_reader import ApiError, download_video

        response = fake_response()
        response.iter_content.return_value = [b"x" * 600_000, b"x" * 600_000]
        with patch("server.request_reader.requests.get", return_value=response):
            with pytest.raises(ApiError, match="larger than 1 MB"):
                download_video("http://example.com/a.mp4", tmp_path / "a.mp4", 2**20)

    @pytest.mark.parametrize("url, name", [
        ("https://example.com/media/clip.webm", "clip.webm"),
        ("https://example.com/media/my%20clip.gif?x=1#t", "my clip.gif"),
        ("https://example.com/watch?v=123", "video.mp4"),
        ("https://example.com/", "video.mp4"),
        ("https://example.com/noext", "video.mp4"),
        ("https://example.com/a/..%2F..%2Fetc%2Fpasswd.mp4", "video.mp4"),
    ])
    def test_filename_from_url(self, url, name):
        from server.request_reader import filename_from_url

        assert filename_from_url(url) == name


class TestGetImageFromSource:

    def test_fetches_remote_url(self, source_payload, valid_extensions, tmp_path, public_dns):
        from server.request_reader import get_image_from_source

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.requests.get", return_value=fake_response(b"remote image")):
                    image_bytes, path = get_image_from_source(source_payload)

        assert image_bytes == b"remote image"
        assert path.parent == tmp_path

    def test_uses_cached_file_without_network_call(self, source_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_source, get_image_path_from_url

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                get_image_path_from_url(source_payload["image_url"]).write_bytes(b"cached bytes")

                with patch("server.request_reader.requests.get") as mock_get, \
                        patch("server.request_reader.socket.getaddrinfo") as mock_dns:
                    image_bytes, _ = get_image_from_source(source_payload)

                mock_get.assert_not_called()
                mock_dns.assert_not_called()

        assert image_bytes == b"cached bytes"

    def test_raises_on_missing_url_key(self):
        from server.request_reader import ApiError, get_image_from_source

        with pytest.raises(ApiError, match="No 'image_url' provided"):
            get_image_from_source({})

    @pytest.mark.parametrize("url", ["http://localhost/image.png", "http://127.0.0.1/image.png",
                                     "http://[::1]/image.png", "http://0.0.0.0/image.png"])
    def test_raises_on_local_urls(self, url):
        from server.request_reader import get_image_from_source

        with pytest.raises(Exception, match="Invalid URL"):
            get_image_from_source({"image_url": url})

    def test_request_uses_five_second_timeout(self, source_payload, valid_extensions, tmp_path, public_dns):
        from server.request_reader import get_image_from_source

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.requests.get", return_value=fake_response(b"data")) as mock_get:
                    get_image_from_source(source_payload)

        assert mock_get.call_args.kwargs.get("timeout") == 5


# ---------------------------------------------------------------------------
# get_image_from_json
# ---------------------------------------------------------------------------

class TestGetImageFromJson:

    def test_raises_when_type_key_missing(self):
        from server.request_reader import ApiError, get_image_from_json

        with pytest.raises(ApiError, match="No message 'type' provided"):
            get_image_from_json({})

    def test_raises_on_unknown_type(self):
        from server.request_reader import ApiError, get_image_from_json

        with pytest.raises(ApiError, match="Unknown image message type"):
            get_image_from_json({"type": "ftp"})

    def test_dispatches_to_base64(self, png_base64_payload, valid_extensions, tmp_path):
        from server.request_reader import get_image_from_json

        payload = {**png_base64_payload, "type": "base64"}
        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                image_bytes, _ = get_image_from_json(payload)

        assert image_bytes == png_base64_payload["_raw"]

    def test_dispatches_to_source(self, valid_extensions, tmp_path, public_dns):
        from server.request_reader import get_image_from_json

        payload = {"type": "source", "image_url": "http://example.com/img.jpg"}

        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                with patch("server.request_reader.requests.get", return_value=fake_response(b"remote")):
                    image_bytes, _ = get_image_from_json(payload)

        assert image_bytes == b"remote"

    def test_writes_image_to_disk(self, png_base64_payload, valid_extensions, tmp_path):
        """get_image_from_json must persist the image at the returned path."""
        from server.request_reader import get_image_from_json

        payload = {**png_base64_payload, "type": "base64"}
        with patch("server.request_reader.constants.IMAGE_EXT", valid_extensions):
            with patch("server.request_reader.UPLOAD_DIR", tmp_path):
                image_bytes, path = get_image_from_json(payload)

        assert path.exists()
        assert path.read_bytes() == image_bytes

    @pytest.mark.parametrize("payload, error", [
        ({"type": "base64", "mime_type": "image/png"}, "No 'image_data'"),
        ({"type": "base64", "mime_type": "image/png", "image_data": "not base64!"}, "not valid base64"),
        ({"type": "base64", "image_data": "eA=="}, "mime_type"),
        ({"type": "base64", "mime_type": "image/svg+xml", "image_data": "eA=="}, "Invalid image extension"),
        ({"type": "base64", "mime_type": "image/gif", "image_data": "eA=="}, "/censor_video"),
    ])
    def test_bad_base64_requests_are_400(self, payload, error, tmp_path):
        from server.request_reader import ApiError, get_image_from_json

        with patch("server.request_reader.UPLOAD_DIR", tmp_path):
            with pytest.raises(ApiError, match=error) as e:
                get_image_from_json(payload)
        assert e.value.status == 400

    def test_base64_without_url_is_named_after_its_contents(self, tmp_path):
        from server.request_reader import get_image_from_json

        payload = {"type": "base64", "mime_type": "image/jpeg", "image_data": base64.b64encode(b"x").decode()}
        with patch("server.request_reader.UPLOAD_DIR", tmp_path):
            _, path_1 = get_image_from_json(payload)
            _, path_2 = get_image_from_json(dict(payload))

        assert path_1 == path_2 and path_1.suffix == ".jpeg"

    def test_extension_is_case_insensitive(self, tmp_path):
        from server.request_reader import get_image_path_from_url

        with patch("server.request_reader.UPLOAD_DIR", tmp_path):
            assert get_image_path_from_url("http://example.com/PHOTO.JPG").suffix == ".jpg"


# ---------------------------------------------------------------------------
# read_request
# ---------------------------------------------------------------------------

def json_request(payload):
    request = AsyncMock()
    request.content_type = "application/json"
    request.json = AsyncMock(return_value=payload)
    return request


BASE64_PAYLOAD = {
    "type": "base64",
    "mime_type": "image/png",
    "image_url": "http://example.com/img.png",
    "image_data": base64.b64encode(b"data").decode(),
}


class TestReadRequest:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content_type", ["multipart/form-data", "text/plain"])
    async def test_raises_for_non_json_content_type(self, content_type):
        from server.request_reader import ApiError, read_request

        request = AsyncMock()
        request.content_type = content_type

        with pytest.raises(ApiError, match="Only JSON requests") as e:
            await read_request(request)
        assert e.value.status == 400

    @pytest.mark.asyncio
    async def test_raises_for_invalid_json(self):
        import json
        from server.request_reader import ApiError, read_request

        request = json_request(None)
        request.json = AsyncMock(side_effect=json.JSONDecodeError("bad", "x", 0))
        with pytest.raises(ApiError, match="not valid JSON"):
            await read_request(request)

    @pytest.mark.asyncio
    async def test_returns_image_request_with_defaults(self, tmp_path):
        from server.request_reader import ImageRequest, read_request

        with patch("server.request_reader.UPLOAD_DIR", tmp_path):
            result = await read_request(json_request(dict(BASE64_PAYLOAD)))

        assert isinstance(result, ImageRequest)
        assert result.image_bytes == b"data"
        assert result.expected_response == "base64"
        assert result.include_detections is False
        assert result.config is None

    @pytest.mark.asyncio
    async def test_honours_explicit_fields(self, tmp_path):
        from server.request_reader import read_request

        payload = {**BASE64_PAYLOAD, "expected_response": "url", "include_detections": True}
        with patch("server.request_reader.UPLOAD_DIR", tmp_path):
            result = await read_request(json_request(payload))

        assert result.expected_response == "url"
        assert result.include_detections is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fields, error", [
        ({"expected_response": "xml"}, "Unknown expected_response"),
        ({"include_detections": "yes"}, "true or false"),
    ])
    async def test_rejects_bad_fields(self, tmp_path, fields, error):
        from server.request_reader import ApiError, read_request

        with patch("server.request_reader.UPLOAD_DIR", tmp_path):
            with pytest.raises(ApiError, match=error):
                await read_request(json_request({**BASE64_PAYLOAD, **fields}))

    @pytest.mark.asyncio
    async def test_censor_config_is_checked(self, tmp_path):
        from server.request_reader import read_request

        fake_config = MagicMock()
        with patch("server.request_reader.UPLOAD_DIR", tmp_path):
            with patch("server.request_reader.check_censor_config", return_value=fake_config) as mock_check:
                result = await read_request(json_request({**BASE64_PAYLOAD, "config": {"threshold": 0.5}}))

        mock_check.assert_called_once_with({"threshold": 0.5})
        assert result.config is fake_config

    @pytest.mark.asyncio
    async def test_invalid_config_is_rejected_before_the_image_is_read(self):
        from server.errors import INVALID_CONFIG
        from server.request_reader import ApiError, read_request

        with patch("server.request_reader.get_image_from_json") as mock_gij:
            with pytest.raises(ApiError) as e:
                await read_request(json_request({**BASE64_PAYLOAD, "config": {"features_to_censor": []}}))

        assert e.value.code == INVALID_CONFIG
        mock_gij.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_bytes_and_path_come_from_get_image_from_json(self):
        from server.request_reader import read_request

        fake_bytes = b"actual"
        fake_path = Path("/uploads/something.png")

        payload = dict(BASE64_PAYLOAD)
        with patch("server.request_reader.get_image_from_json", return_value=(fake_bytes, fake_path)) as mock_gij:
            result = await read_request(json_request(payload))

        mock_gij.assert_called_once_with(payload)
        assert result.image_bytes is fake_bytes
        assert result.image_path is fake_path
