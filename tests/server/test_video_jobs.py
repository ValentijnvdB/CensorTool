"""Background video jobs: server/video_jobs.py and the /videos endpoints."""
import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio


class FakeVideoJob:
    """Like the Python backend's VideoJob; with_cancel=False is like censor_core 2.0 (no progress, no cancel)."""

    def __init__(self, video_path, output_path, with_cancel=True):
        self.video_path, self.output_path = str(video_path), str(output_path)
        self.success, self.error, self.result = False, None, None
        self.cancel_called = False
        self._cancelled = threading.Event()
        if with_cancel:
            self.phase, self.frames_done, self.frames_total = None, 0, 0
            self.cancel = self._cancel

    def _cancel(self):
        self.cancel_called = True
        self._cancelled.set()


class FakeVideoPipeline:
    """Runs a job until release() (or its cancel); fail=True makes the next jobs fail."""

    def __init__(self):
        self.released = threading.Event()
        self.started = []
        self.fail = False

    def release(self):
        self.released.set()

    def submit(self, job):
        self.started.append(job)
        if hasattr(job, 'phase'):
            job.phase, job.frames_done, job.frames_total = 'censoring', 5, 10
        while not self.released.is_set() and not job._cancelled.is_set():
            time.sleep(0.005)
        if hasattr(job, 'phase'):
            job.phase = 'done'
        if job._cancelled.is_set():
            job.error = 'cancelled'
        elif self.fail:
            job.error = 'model crashed'
        else:
            Path(job.output_path).write_bytes(b"censored " + Path(job.video_path).read_bytes())
            job.success, job.result = True, job.output_path
        return job


@pytest.fixture(autouse=True)
def fresh_queue():
    """A video lock and queue for this test's event loop."""
    from server import helpers
    with patch.object(helpers, "_video_lock", None), patch.object(helpers, "activity", helpers._Activity()):
        yield


@pytest.fixture()
def pipeline():
    pipeline = FakeVideoPipeline()
    with patch("server.helpers._video_pipeline", pipeline):
        yield pipeline
    pipeline.release()


async def until(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(0.005)


def make_prepare(tmp_path, name="clip", with_cancel=True):
    """A prepare() that 'uploads' a video and reserves its output."""
    from server.http_server import VideoInfo

    async def prepare(record):
        record.input_path = tmp_path / f"{name}.mp4"
        record.input_path.write_bytes(b"video")
        record.output_path = tmp_path / f"{name}_out.mp4"
        record.output_path.write_bytes(b"")
        record.info = VideoInfo(20, 10.0, 64, 48)
        return FakeVideoJob(record.input_path, record.output_path, with_cancel)
    return prepare


# ---------------------------------------------------------------------------
# the manager
# ---------------------------------------------------------------------------

class TestVideoJobs:

    @pytest.mark.asyncio
    async def test_lifecycle(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs()
        record = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path))
        await until(lambda: record.current_state == "running")

        running = record.to_dict()
        assert running["phase"] == "censoring"
        assert (running["frames_done"], running["frames_total"]) == (5, 10)
        assert running["progress"] == pytest.approx(0.825)  # halfway through 0.7-0.95
        assert running["started_at"] is not None and running["result_url"] is None
        assert running["video"] == {"frames": 20, "fps": 10.0, "duration": 2.0, "width": 64, "height": 48}

        pipeline.release()
        await record.task
        done = record.to_dict()
        assert done["state"] == "done" and done["progress"] == 1.0
        assert done["result_url"] == f"/videos/{record.id}/result"
        assert done["filename"] == "clip_out.mp4"
        assert done["processing_ms"] >= 0 and done["expires_at"] is not None
        assert record.output_path.read_bytes() == b"censored video"

    @pytest.mark.asyncio
    async def test_jobs_wait_their_turn_in_order(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs()
        first = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path, "a"))
        await until(lambda: first.current_state == "running")
        second = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path, "b"))
        third = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path, "c"))
        await until(lambda: third.to_dict()["queue_position"] == 2)

        assert second.to_dict()["state"] == "queued" and second.to_dict()["queue_position"] == 1
        assert second.to_dict()["progress"] == 0.0
        assert first.to_dict()["queue_position"] is None

        pipeline.release()
        await asyncio.gather(first.task, second.task, third.task)
        assert [Path(j.video_path).stem for j in pipeline.started] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_delete_waiting_job_never_runs(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs()
        first = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path, "a"))
        await until(lambda: first.current_state == "running")
        second = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path, "b"))
        await until(lambda: second.current_state == "queued" and second.ticket is not None)

        assert jobs.delete(second.id).to_dict()["state"] == "cancelled"
        await asyncio.gather(second.task, return_exceptions=True)
        assert second.id not in jobs.jobs
        assert not second.input_path.exists() and not second.output_path.exists()

        pipeline.release()
        await first.task
        assert len(pipeline.started) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("with_cancel", [True, False])
    async def test_delete_running_job(self, tmp_path, pipeline, with_cancel):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs()
        record = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path, with_cancel=with_cancel))
        await until(lambda: record.current_state == "running")
        job = record.ticket.job

        jobs.delete(record.id)
        assert record.id not in jobs.jobs
        if with_cancel:
            assert job.cancel_called  # the Python backend stops it
        else:
            pipeline.release()  # the C++ backend runs on; its files go once it is done
        await record.task

        assert record.state == "cancelled"
        assert not record.input_path.exists() and not record.output_path.exists()

    @pytest.mark.asyncio
    async def test_without_progress_support_progress_is_unknown(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        record = VideoJobs().create("mp4", "video/mp4", False, make_prepare(tmp_path, with_cancel=False))
        await until(lambda: record.current_state == "running")
        body = record.to_dict()
        assert (body["phase"], body["progress"], body["frames_done"]) == (None, None, None)
        pipeline.release()
        await record.task

    @pytest.mark.asyncio
    async def test_failed_job(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        pipeline.fail = True
        pipeline.release()
        record = VideoJobs().create("mp4", "video/mp4", False, make_prepare(tmp_path))
        await record.task

        body = record.to_dict()
        assert body["state"] == "failed" and body["code"] == "processing_failed"
        assert "model crashed" in body["error"]
        assert body["result_url"] is None
        assert not record.output_path.exists()
        assert record.input_path.exists()  # kept until the job expires or is deleted

    @pytest.mark.asyncio
    async def test_failed_prepare(self, pipeline):
        from server.errors import ApiError
        from server.video_jobs import VideoJobs

        async def prepare(record):
            raise ApiError("Could not download the video: the server answered 404.", "download_failed", 502)

        record = VideoJobs().create("mp4", "video/mp4", True, prepare)
        assert record.to_dict()["state"] == "downloading"
        await record.task
        assert (record.state, record.code) == ("failed", "download_failed")
        assert pipeline.started == []

    @pytest.mark.asyncio
    async def test_expire_removes_old_jobs_and_files(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        pipeline.release()
        jobs = VideoJobs(retention=60)
        record = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path))
        await record.task

        assert jobs.expire(now=record.finished_at + 30) == 0
        assert jobs.expire(now=record.finished_at + 61) == 1
        assert record.id not in jobs.jobs
        assert not record.input_path.exists() and not record.output_path.exists()

    @pytest.mark.asyncio
    async def test_capacity(self, tmp_path, pipeline):
        from server.errors import ApiError
        from server.video_jobs import VideoJobs

        jobs = VideoJobs(max_active=1)
        record = jobs.create("mp4", "video/mp4", False, make_prepare(tmp_path))
        with pytest.raises(ApiError) as e:
            jobs.check_capacity()
        assert (e.value.status, e.value.code) == (429, "queue_full")
        pipeline.release()
        await record.task
        jobs.check_capacity()  # finished jobs do not count

    @pytest.mark.asyncio
    async def test_unknown_job_is_404(self):
        from server.errors import ApiError
        from server.video_jobs import VideoJobs

        with pytest.raises(ApiError) as e:
            VideoJobs().get("nope")
        assert e.value.status == 404


@pytest.mark.parametrize("phase, done, total, expected", [
    ("detecting", 0, 0, 0.0), ("detecting", 50, 100, 0.35), ("censoring", 0, 10, 0.7),
    ("censoring", 10, 10, 0.95), ("encoding", 0, 0, 0.95),
])
def test_progress_per_phase(phase, done, total, expected):
    from server.helpers import VideoTicket
    from server.video_jobs import VideoJobRecord

    job = FakeVideoJob("in.mp4", "out.mp4")
    job.phase, job.frames_done, job.frames_total = phase, done, total
    record = VideoJobRecord("x", "mp4", "video/mp4", False, state="queued", ticket=VideoTicket(job, "running"))
    assert record.to_dict()["progress"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# the endpoints
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(tmp_path, pipeline):
    from aiohttp.test_utils import TestClient, TestServer
    import server.http_server as hs
    from server.video_jobs import VideoJobs

    upload, censored = tmp_path / "up", tmp_path / "out"
    upload.mkdir(), censored.mkdir()
    jobs = VideoJobs()
    with patch.object(hs, "UPLOAD_DIR", upload), patch.object(hs, "CENSORED_PATH", censored), \
            patch.object(hs, "CACHE_DIR", tmp_path / "cache"), patch.object(hs, "video_jobs", jobs), \
            patch.object(hs, "read_video_info", return_value=hs.VideoInfo(20, 10.0, 64, 48)), \
            patch.object(hs, "make_video_job",
                         side_effect=lambda video_path, output_path, config, cache_dir: FakeVideoJob(video_path,
                                                                                                     output_path)):
        client = TestClient(TestServer(await hs.init_app()))
        await client.start_server()
        client.dirs, client.jobs = (upload, censored), jobs
        yield client
        pipeline.release()
        await client.close()


def upload_form(**fields):
    import aiohttp
    form = aiohttp.FormData(default_to_multipart=True)
    for name, value in fields.items():
        if name == "video":
            form.add_field("video", value, filename="clip.mp4")
        else:
            form.add_field(name, value)
    return form


class TestVideoEndpoints:

    @pytest.mark.asyncio
    async def test_upload_poll_and_fetch(self, client, pipeline):
        response = await client.post("/videos", data=upload_form(filename="clip.mp4", video=b"video"))
        assert response.status == 202
        job = (await response.json())["job"]
        assert response.headers["Location"] == f"/videos/{job['id']}"
        assert job["state"] in ("queued", "running")

        early = await client.get(f"/videos/{job['id']}/result")
        assert early.status == 409 and (await early.json())["code"] == "not_ready"

        pipeline.release()
        await until(lambda: client.jobs.jobs[job["id"]].state == "done")
        body = (await (await client.get(f"/videos/{job['id']}")).json())["job"]
        assert body["state"] == "done" and body["result_url"] == f"/videos/{job['id']}/result"

        result = await client.get(body["result_url"])
        assert result.status == 200
        assert await result.read() == b"censored video"
        assert result.headers["Content-Type"] == "video/mp4"
        assert result.headers["X-Frame-Count"] == "20"
        assert "X-Processing-Time-Ms" in result.headers

        listing = await (await client.get("/videos")).json()
        assert [j["id"] for j in listing["jobs"]] == [job["id"]]

    @pytest.mark.asyncio
    async def test_upload_name_is_used_without_a_filename_part(self, client, pipeline):
        upload, censored = client.dirs
        pipeline.release()
        job = (await (await client.post("/videos", data=upload_form(video=b"v", format="webm"))).json())["job"]
        await until(lambda: client.jobs.jobs[job["id"]].state == "done")
        assert (upload / "clip.mp4").exists() and (censored / "clip.webm").exists()

    @pytest.mark.asyncio
    async def test_bad_uploads_are_400_straight_away(self, client):
        response = await client.post("/videos", data=upload_form(filename="clip.mp4"))
        assert response.status == 400
        assert client.jobs.jobs == {}

    @pytest.mark.asyncio
    async def test_video_url_downloads_in_the_background(self, client, pipeline):
        upload, _ = client.dirs
        started, release = threading.Event(), threading.Event()

        def slow_download(url, path, digest=None, stop=None):
            started.set()
            while not release.is_set() and not stop.is_set():
                time.sleep(0.005)
            path.write_bytes(b"downloaded")

        with patch("server.http_server.rr.download_video", side_effect=slow_download):
            response = await client.post("/videos", data=upload_form(video_url="https://example.com/a/anim.gif"))
            assert response.status == 202
            job = (await response.json())["job"]
            assert job["state"] == "downloading" and job["format"] == "gif"

            release.set()
            pipeline.release()
            await until(lambda: client.jobs.jobs[job["id"]].state == "done")
        assert (upload / "anim.gif").read_bytes() == b"downloaded"

    @pytest.mark.asyncio
    async def test_delete_while_downloading_stops_and_cleans_up(self, client):
        upload, censored = client.dirs

        def endless_download(url, path, digest=None, stop=None):
            path.write_bytes(b"partial")
            while not stop.is_set():
                time.sleep(0.005)
            from server.errors import ApiError
            raise ApiError("The download was stopped.", "download_failed", 502)

        with patch("server.http_server.rr.download_video", side_effect=endless_download):
            job = (await (await client.post("/videos", data=upload_form(video_url="https://example.com/a.mp4")))
                   .json())["job"]
            await until(lambda: any(upload.iterdir()))
            record = client.jobs.jobs[job["id"]]

            deleted = await client.delete(f"/videos/{job['id']}")
            assert deleted.status == 200 and (await deleted.json())["job"]["state"] == "cancelled"
            await asyncio.gather(record.task, return_exceptions=True)

        assert list(upload.iterdir()) == [] and list(censored.iterdir()) == []
        assert (await client.get(f"/videos/{job['id']}")).status == 404

    @pytest.mark.asyncio
    async def test_failed_job_result_is_409(self, client, pipeline):
        pipeline.fail = True
        pipeline.release()
        job = (await (await client.post("/videos", data=upload_form(video=b"v"))).json())["job"]
        await until(lambda: client.jobs.jobs[job["id"]].state == "failed")

        result = await client.get(f"/videos/{job['id']}/result")
        body = await result.json()
        assert result.status == 409 and body["code"] == "job_failed" and "model crashed" in body["error"]

    @pytest.mark.asyncio
    async def test_non_multipart_is_400(self, client):
        response = await client.post("/videos", json={"video_url": "https://example.com/a.mp4"})
        assert response.status == 400 and "multipart" in (await response.json())["error"]

    @pytest.mark.asyncio
    async def test_unknown_job_is_404(self, client):
        for method, path in (("GET", "/videos/nope"), ("GET", "/videos/nope/result"), ("DELETE", "/videos/nope")):
            response = await client.request(method, path)
            assert response.status == 404 and (await response.json())["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_full_queue_is_429_before_the_upload_is_read(self, client):
        upload, _ = client.dirs
        client.jobs.max_active = 0
        response = await client.post("/videos", data=upload_form(video=b"video"))
        assert response.status == 429
        assert list(upload.iterdir()) == []
