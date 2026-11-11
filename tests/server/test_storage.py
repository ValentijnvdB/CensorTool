"""Clean-up of the files the server keeps: server/storage.py, /censored/<name> and /storage."""
import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

HOUR = 3600


@pytest.fixture()
def server_dir(tmp_path):
    """data/server/ in tmp_path, with a 24 hour retention."""
    from app.config import CONFIG
    root = tmp_path / "server"
    (root / "uncensored").mkdir(parents=True)
    (root / "censored").mkdir()
    with patch("server.server_config.SERVER_DIR", root), \
            patch("server.server_config.CENSORED_PATH", root / "censored"), \
            patch("server.http_server.CENSORED_PATH", root / "censored"), \
            patch.object(CONFIG, "server_file_retention_hours", 24):
        yield root


def make_file(path: Path, content=b"x", age_hours=0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    when = time.time() - age_hours * HOUR
    os.utime(path, (when, when))
    return path


class TestSweep:

    def test_removes_only_files_unused_too_long(self, server_dir):
        from server import storage

        old = make_file(server_dir / "uncensored" / "old.jpg", b"12345", age_hours=25)
        fresh = make_file(server_dir / "censored" / "fresh.jpg", age_hours=23)
        old_debug = make_file(server_dir / "debug" / "x.png", b"12", age_hours=48)

        assert storage.sweep(set()) == {"deleted_files": 2, "deleted_bytes": 7}
        assert not old.exists() and not old_debug.exists() and fresh.exists()

    def test_keeps_protected_files(self, server_dir):
        from server import storage

        job_input = make_file(server_dir / "uncensored" / "clip.mp4", age_hours=30)
        assert storage.sweep({job_input})["deleted_files"] == 0
        assert job_input.exists()

    def test_zero_retention_keeps_everything(self, server_dir):
        from app.config import CONFIG
        from server import storage

        old = make_file(server_dir / "uncensored" / "old.jpg", age_hours=1000)
        with patch.object(CONFIG, "server_file_retention_hours", 0):
            assert storage.sweep(set())["deleted_files"] == 0
        assert old.exists()

    def test_touch_resets_the_clock(self, server_dir):
        from server import storage

        path = make_file(server_dir / "censored" / "a.jpg", age_hours=30)
        storage.touch(path)
        assert storage.sweep(set())["deleted_files"] == 0

    def test_remove_all(self, server_dir):
        from server import storage

        keep = make_file(server_dir / "censored" / "job.mp4")
        make_file(server_dir / "censored" / "a.jpg")
        make_file(server_dir / "uncensored" / "b.jpg")
        assert storage.remove_files(None, {keep})["deleted_files"] == 2
        assert keep.exists()

    @pytest.mark.asyncio
    async def test_periodic_sweep_runs_at_once(self, server_dir):
        from server import storage

        old = make_file(server_dir / "uncensored" / "old.jpg", age_hours=25)
        task = asyncio.create_task(storage.sweep_periodically(set, interval=3600))
        for _ in range(200):
            if not old.exists():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        assert not old.exists()


def test_reused_download_counts_as_a_use(server_dir):
    from server import request_reader as rr

    url = "https://example.com/a.jpg"
    with patch.object(rr, "UPLOAD_DIR", server_dir / "uncensored"):
        path = make_file(rr.get_image_path_from_url(url), b"jpeg", age_hours=23)
        with patch.object(rr, "download_image") as download:
            assert rr.get_image_from_source({"image_url": url}) == (b"jpeg", path)
        download.assert_not_called()
    assert time.time() - path.stat().st_mtime < 60


def test_storage_info(server_dir):
    from server import storage

    make_file(server_dir / "uncensored" / "a.jpg", b"123")
    make_file(server_dir / "censored" / "b.jpg", b"12")
    make_file(server_dir / "censored" / "c.jpg", b"1")

    info = storage.storage_info()
    assert info["folders"] == {"censored": {"files": 2, "bytes": 3}, "uncensored": {"files": 1, "bytes": 3}}
    assert (info["files"], info["bytes"]) == (3, 6)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(server_dir):
    from aiohttp.test_utils import TestClient, TestServer
    import server.http_server as hs
    from server.video_jobs import VideoJobs

    jobs = VideoJobs()
    with patch.object(hs, "video_jobs", jobs):
        client = TestClient(TestServer(await hs.init_app()))
        await client.start_server()
        client.jobs = jobs
        yield client
        await client.close()


class TestEndpoints:

    @pytest.mark.asyncio
    async def test_censored_file_is_served_and_counts_as_a_use(self, client, server_dir):
        path = make_file(server_dir / "censored" / "a b.jpg", b"jpeg", age_hours=23)

        response = await client.get("/censored/a%20b.jpg")
        assert response.status == 200 and await response.read() == b"jpeg"
        assert time.time() - path.stat().st_mtime < 60

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["missing.jpg", "..%2Funcensored%2Fsecret.jpg"])
    async def test_censored_file_404(self, client, server_dir, name):
        make_file(server_dir / "uncensored" / "secret.jpg")
        response = await client.get(f"/censored/{name}")
        assert response.status == 404
        assert (await response.json())["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_storage_info(self, client, server_dir):
        make_file(server_dir / "censored" / "a.jpg", b"123")
        body = await (await client.get("/storage")).json()
        assert body["status"] == "success" and body["retention_hours"] == 24
        assert body["folders"]["censored"] == {"files": 1, "bytes": 3}

    @pytest.mark.asyncio
    async def test_delete_storage_keeps_job_files(self, client, server_dir):
        from server.video_jobs import VideoWork

        job_file = make_file(server_dir / "uncensored" / "clip.mp4")
        client.jobs.works["x"] = VideoWork("x", "mp4", "video/mp4", "config", state="done", input_path=job_file)
        make_file(server_dir / "censored" / "a.jpg", b"123")

        body = await (await client.delete("/storage")).json()
        assert body == {"status": "success", "deleted_files": 1, "deleted_bytes": 3}
        assert job_file.exists()


# ---------------------------------------------------------------------------
# config.yml
# ---------------------------------------------------------------------------

BASE_CONFIG = """
video_censor_fps: 15
live_cap_top: 0
live_cap_left: 0
live_cap_height: 1080
live_cap_width: 1920
live_show_fps: false
live_cursor_color: [168, 93, 253]
comfy_host: localhost
comfy_port: 8188
comfy_workflow: ./data/workflow.json
"""


@pytest.mark.parametrize("extra, expected", [("", 24), ("server_file_retention_hours: 1.5", 1.5),
                                             ("server_file_retention_hours: 0", 0)])
def test_config_retention(tmp_path, extra, expected):
    from app.config import load_config

    path = tmp_path / "config.yml"
    path.write_text(BASE_CONFIG + extra)
    assert load_config(path).server_file_retention_hours == expected


@pytest.mark.parametrize("value", ["-1", "true", "'a day'"])
def test_config_retention_must_be_a_positive_number(tmp_path, value):
    from app.config import load_config

    path = tmp_path / "config.yml"
    path.write_text(BASE_CONFIG + f"server_file_retention_hours: {value}")
    with pytest.raises(ValueError, match="server_file_retention_hours"):
        load_config(path)


@pytest.mark.parametrize("extra, grace, max_mb", [
    ("", 30, 500),
    ("server_video_job_grace_seconds: 0\nserver_max_video_download_mb: 50", 0, 50),
    ("server_video_job_grace_seconds: 2.5\nserver_max_video_download_mb: 0.5", 2.5, 0.5),
])
def test_config_video_job_settings(tmp_path, extra, grace, max_mb):
    from app.config import load_config

    path = tmp_path / "config.yml"
    path.write_text(BASE_CONFIG + extra)
    config = load_config(path)
    assert (config.server_video_job_grace_seconds, config.server_max_video_download_mb) == (grace, max_mb)


@pytest.mark.parametrize("line, key", [
    ("server_video_job_grace_seconds: -1", "server_video_job_grace_seconds"),
    ("server_video_job_grace_seconds: true", "server_video_job_grace_seconds"),
    ("server_max_video_download_mb: 0", "server_max_video_download_mb"),
    ("server_max_video_download_mb: '500 MB'", "server_max_video_download_mb"),
])
def test_config_video_job_settings_must_be_valid(tmp_path, line, key):
    from app.config import load_config

    path = tmp_path / "config.yml"
    path.write_text(BASE_CONFIG + line)
    with pytest.raises(ValueError, match=key):
        load_config(path)
