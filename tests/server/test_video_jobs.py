"""Background video jobs: server/video_jobs.py and the /videos endpoints."""
import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio


class FakeVideoJob:
    """Like a backend's VideoJob: progress and cancel."""

    def __init__(self, video_path, output_path):
        self.video_path, self.output_path = str(video_path), str(output_path)
        self.success, self.error, self.result = False, None, None
        self.phase, self.frames_done, self.frames_total = None, 0, 0
        self.cancel_called = False
        self._cancelled = threading.Event()

    def cancel(self):
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
        job.phase, job.frames_done, job.frames_total = 'censoring', 5, 10
        while not self.released.is_set() and not job._cancelled.is_set():
            time.sleep(0.005)
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


def make_prepare(tmp_path, name="clip"):
    """A prepare() that 'uploads' a video and reserves its output."""
    from server.http_server import VideoInfo

    async def prepare(work):
        work.input_path = tmp_path / f"{name}.mp4"
        work.input_path.write_bytes(b"video")
        work.output_path = tmp_path / f"{name}_out.mp4"
        work.output_path.write_bytes(b"")
        work.info = VideoInfo(20, 10.0, 64, 48)
        return FakeVideoJob(work.input_path, work.output_path)
    return prepare


def create(jobs, tmp_path, name="clip", fmt="mp4", key="config", url=None, digest=None):
    """A job on new work; an upload with digest = name unless a url is given."""
    if url is None and digest is None:
        digest = name
    return jobs.create(fmt, f"video/{fmt}", make_prepare(tmp_path, name), config_key=key, url=url, digest=digest)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

class TestVideoJobs:

    @pytest.mark.asyncio
    async def test_lifecycle(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs()
        record = create(jobs, tmp_path)
        await until(lambda: record.current_state == "running")

        running = record.to_dict()
        assert running["phase"] == "censoring"
        assert (running["frames_done"], running["frames_total"]) == (5, 10)
        assert running["progress"] == pytest.approx(0.825)  # halfway through 0.7-0.95
        assert running["started_at"] is not None and running["result_url"] is None
        assert running["video"] == {"frames": 20, "fps": 10.0, "duration": 2.0, "width": 64, "height": 48}

        pipeline.release()
        await record.work.task
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
        first = create(jobs, tmp_path, "a")
        await until(lambda: first.current_state == "running")
        second = create(jobs, tmp_path, "b")
        third = create(jobs, tmp_path, "c")
        await until(lambda: third.to_dict()["queue_position"] == 2)

        assert second.to_dict()["state"] == "queued" and second.to_dict()["queue_position"] == 1
        assert second.to_dict()["progress"] == 0.0
        assert first.to_dict()["queue_position"] is None

        pipeline.release()
        await asyncio.gather(first.work.task, second.work.task, third.work.task)
        assert [Path(j.video_path).stem for j in pipeline.started] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_delete_waiting_job_never_runs(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs(grace=0)
        first = create(jobs, tmp_path, "a")
        await until(lambda: first.current_state == "running")
        second = create(jobs, tmp_path, "b")
        await until(lambda: second.current_state == "queued" and second.work.ticket is not None)

        assert jobs.delete(second.id).to_dict()["state"] == "cancelled"
        await asyncio.gather(second.work.task, return_exceptions=True)
        assert second.id not in jobs.jobs and second.work.id not in jobs.works
        assert not second.work.input_path.exists() and not second.work.output_path.exists()

        pipeline.release()
        await first.work.task
        assert len(pipeline.started) == 1

    @pytest.mark.asyncio
    async def test_delete_running_job(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs(grace=0)
        record = create(jobs, tmp_path)
        await until(lambda: record.current_state == "running")
        job = record.work.ticket.job

        jobs.delete(record.id)
        assert record.id not in jobs.jobs
        assert job.cancel_called  # the backend stops it
        await record.work.task

        assert record.work.state == "cancelled"
        assert not record.work.input_path.exists() and not record.work.output_path.exists()

    @pytest.mark.asyncio
    async def test_failed_job(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        pipeline.fail = True
        pipeline.release()
        record = create(VideoJobs(), tmp_path)
        await record.work.task

        body = record.to_dict()
        assert body["state"] == "failed" and body["code"] == "processing_failed"
        assert "model crashed" in body["error"]
        assert body["result_url"] is None
        assert not record.output_path.exists()
        assert record.work.input_path.exists()  # kept until the job expires or is deleted

    @pytest.mark.asyncio
    async def test_failed_prepare(self, pipeline):
        from server.errors import ApiError
        from server.video_jobs import VideoJobs

        async def prepare(work):
            raise ApiError("Could not download the video: the server answered 404.", "download_failed", 502)

        record = VideoJobs().create("mp4", "video/mp4", prepare, config_key="config", url="https://example.com/a")
        assert record.to_dict()["state"] == "downloading"
        await record.work.task
        assert (record.work.state, record.work.code) == ("failed", "download_failed")
        assert pipeline.started == []

    @pytest.mark.asyncio
    async def test_expire_removes_old_jobs_and_files(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        pipeline.release()
        jobs = VideoJobs(retention=60)
        record = create(jobs, tmp_path)
        await record.work.task
        finished = record.work.finished_at

        assert jobs.expire(now=finished + 30) == 0
        assert jobs.expire(now=finished + 61) == 1
        assert record.id not in jobs.jobs and record.work.id not in jobs.works
        assert not record.work.input_path.exists() and not record.output_path.exists()

    @pytest.mark.asyncio
    async def test_capacity(self, tmp_path, pipeline):
        from server.errors import ApiError
        from server.video_jobs import VideoJobs

        jobs = VideoJobs(max_active=1)
        record = create(jobs, tmp_path)
        with pytest.raises(ApiError) as e:
            jobs.check_capacity()
        assert (e.value.status, e.value.code) == (429, "queue_full")
        assert jobs.join("config", "mp4", digest="clip") is not None  # sharing work needs no room
        pipeline.release()
        await record.work.task
        jobs.check_capacity()  # finished work does not count

    @pytest.mark.asyncio
    async def test_unknown_job_is_404(self):
        from server.errors import ApiError
        from server.video_jobs import VideoJobs

        with pytest.raises(ApiError) as e:
            VideoJobs().get("nope")
        assert e.value.status == 404


class TestSharedWork:

    @pytest.mark.asyncio
    async def test_jobs_for_the_same_video_share_the_work(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs()
        first = create(jobs, tmp_path, url="https://example.com/a.mp4")
        await until(lambda: first.current_state == "running")
        by_url = jobs.join("config", "mp4", url="https://example.com/a.mp4")
        assert by_url is not None and by_url.work is first.work and by_url.id != first.id
        assert by_url.to_dict()["state"] == "running"

        pipeline.release()
        await first.work.task
        first.work.digest = "abc"   # as prepare() sets it after a download
        by_content = jobs.join("config", "mp4", digest="abc")
        assert by_content.work is first.work and by_content.to_dict()["state"] == "done"
        assert len(pipeline.started) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key, fmt, url", [("other", "mp4", "https://example.com/a.mp4"),
                                               ("config", "webm", "https://example.com/a.mp4"),
                                               ("config", "mp4", "https://example.com/b.mp4")])
    async def test_another_config_format_or_video_gets_its_own_work(self, tmp_path, pipeline, key, fmt, url):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs()
        create(jobs, tmp_path, url="https://example.com/a.mp4")
        assert jobs.join(key, fmt, url=url) is None

    @pytest.mark.asyncio
    async def test_failed_work_is_not_shared(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        pipeline.fail = True
        pipeline.release()
        jobs = VideoJobs()
        record = create(jobs, tmp_path)
        await record.work.task
        assert jobs.join("config", "mp4", digest="clip") is None

    @pytest.mark.asyncio
    async def test_deleting_one_job_leaves_the_others(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs(grace=0)
        first = create(jobs, tmp_path)
        await until(lambda: first.current_state == "running")
        second = jobs.join("config", "mp4", digest="clip")

        assert jobs.delete(first.id).to_dict()["state"] == "cancelled"
        assert not first.work.ticket.job.cancel_called
        pipeline.release()
        await second.work.task
        assert second.to_dict()["state"] == "done" and second.output_path.read_bytes() == b"censored video"

    @pytest.mark.asyncio
    async def test_deleting_the_last_job_of_finished_work_removes_its_files(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        pipeline.release()
        jobs = VideoJobs()
        first = create(jobs, tmp_path)
        await first.work.task
        second = jobs.join("config", "mp4", digest="clip")

        jobs.delete(first.id)
        assert second.output_path.exists()
        jobs.delete(second.id)
        assert not second.output_path.exists() and not second.work.input_path.exists()
        assert jobs.works == {}

    @pytest.mark.asyncio
    async def test_work_without_jobs_runs_on_for_the_grace_period(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs(grace=0.2)
        record = create(jobs, tmp_path)
        await until(lambda: record.current_state == "running")
        jobs.delete(record.id)
        await asyncio.sleep(0.05)
        assert not record.work.ticket.job.cancel_called

        # e.g. the page was reloaded: the new job takes the work over
        again = jobs.join("config", "mp4", digest="clip")
        assert again is not None and again.work is record.work
        await asyncio.sleep(0.3)
        assert not record.work.ticket.job.cancel_called
        pipeline.release()
        await again.work.task
        assert again.to_dict()["state"] == "done"

    @pytest.mark.asyncio
    async def test_work_without_jobs_is_cancelled_after_the_grace_period(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        jobs = VideoJobs(grace=0.05)
        record = create(jobs, tmp_path)
        await until(lambda: record.current_state == "running")
        jobs.delete(record.id)
        await until(lambda: record.work.ticket.job.cancel_called)
        await record.work.task
        assert record.work.state == "cancelled" and jobs.works == {}
        assert not record.work.input_path.exists() and not record.work.output_path.exists()

    @pytest.mark.asyncio
    async def test_grace_period_comes_from_config(self):
        from app.config import CONFIG
        from server.video_jobs import VideoJobs

        with patch.object(CONFIG, "server_video_job_grace_seconds", 12):
            assert VideoJobs().grace == 12

    @pytest.mark.asyncio
    async def test_a_job_joining_finished_work_gets_the_full_retention(self, tmp_path, pipeline):
        from server.video_jobs import VideoJobs

        pipeline.release()
        jobs = VideoJobs(retention=60)
        first = create(jobs, tmp_path)
        await first.work.task
        finished = first.work.finished_at
        late = jobs.join("config", "mp4", digest="clip")
        late.created_at = finished + 50

        assert jobs.expire(now=finished + 61) == 1      # the first job
        assert late.output_path.exists()
        assert jobs.expire(now=finished + 111) == 1     # the late one, and with it the work
        assert not late.output_path.exists() and jobs.works == {}


@pytest.mark.parametrize("phase, done, total, expected", [
    (None, 0, 0, 0.0), ("detecting", 0, 0, 0.0), ("detecting", 50, 100, 0.35), ("censoring", 0, 10, 0.7),
    ("censoring", 10, 10, 0.95), ("encoding", 0, 0, 0.95),
])
def test_progress_per_phase(phase, done, total, expected):
    from server.helpers import VideoTicket
    from server.video_jobs import VideoWork

    job = FakeVideoJob("in.mp4", "out.mp4")
    job.phase, job.frames_done, job.frames_total = phase, done, total
    work = VideoWork("x", "mp4", "video/mp4", "config", state="queued", ticket=VideoTicket(job, "running"))
    assert work.progress()[3] == pytest.approx(expected)


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
    jobs = VideoJobs(grace=0)
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


async def post_job(client, **fields) -> dict:
    response = await client.post("/videos", data=upload_form(**fields))
    assert response.status == 202, await response.text()
    return (await response.json())["job"]


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
        await until(lambda: client.jobs.jobs[job["id"]].current_state == "done")
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
        job = await post_job(client, video=b"v", format="webm")
        await until(lambda: client.jobs.jobs[job["id"]].current_state == "done")
        assert (upload / "clip.mp4").exists() and (censored / "clip.webm").exists()

    @pytest.mark.asyncio
    async def test_bad_uploads_are_400_straight_away(self, client):
        response = await client.post("/videos", data=upload_form(filename="clip.mp4"))
        assert response.status == 400
        assert client.jobs.jobs == {}

    @pytest.mark.asyncio
    async def test_video_url_downloads_in_the_background(self, client, pipeline):
        upload, _ = client.dirs
        release = threading.Event()

        def slow_download(url, path, max_size, digest=None, stop=None):
            while not release.is_set() and not stop.is_set():
                time.sleep(0.005)
            path.write_bytes(b"downloaded")

        with patch("server.http_server.rr.download_video", side_effect=slow_download):
            job = await post_job(client, video_url="https://example.com/a/anim.gif")
            assert job["state"] == "downloading" and job["format"] == "gif"

            release.set()
            pipeline.release()
            await until(lambda: client.jobs.jobs[job["id"]].current_state == "done")
        assert (upload / "anim.gif").read_bytes() == b"downloaded"

    @pytest.mark.asyncio
    async def test_download_size_limit_comes_from_config(self, client, pipeline):
        from app.config import CONFIG

        sizes = []

        def download(url, path, max_size, digest=None, stop=None):
            sizes.append(max_size)
            path.write_bytes(b"downloaded")

        with patch("server.http_server.rr.download_video", side_effect=download), \
                patch.object(CONFIG, "server_max_video_download_mb", 2.5):
            job = await post_job(client, video_url="https://example.com/a.mp4")
            await until(lambda: client.jobs.jobs[job["id"]].work.info is not None)
        assert sizes == [round(2.5 * 2**20)]

    @pytest.mark.asyncio
    async def test_delete_while_downloading_stops_and_cleans_up(self, client):
        upload, censored = client.dirs

        def endless_download(url, path, max_size, digest=None, stop=None):
            path.write_bytes(b"partial")
            while not stop.is_set():
                time.sleep(0.005)
            from server.errors import ApiError
            raise ApiError("The download was stopped.", "download_failed", 502)

        with patch("server.http_server.rr.download_video", side_effect=endless_download):
            job = await post_job(client, video_url="https://example.com/a.mp4")
            await until(lambda: any(upload.iterdir()))
            work = client.jobs.jobs[job["id"]].work

            deleted = await client.delete(f"/videos/{job['id']}")
            assert deleted.status == 200 and (await deleted.json())["job"]["state"] == "cancelled"
            await asyncio.gather(work.task, return_exceptions=True)

        assert list(upload.iterdir()) == [] and list(censored.iterdir()) == []
        assert (await client.get(f"/videos/{job['id']}")).status == 404

    @pytest.mark.asyncio
    async def test_the_same_upload_twice_is_censored_once(self, client, pipeline):
        upload, censored = client.dirs
        first = await post_job(client, video=b"video")
        second = await post_job(client, video=b"video")
        assert first["id"] != second["id"]
        assert [p.name for p in upload.iterdir()] == ["clip.mp4"]   # the second upload is not kept

        pipeline.release()
        await until(lambda: client.jobs.jobs[second["id"]].current_state == "done")
        for job in (first, second):
            assert await (await client.get(f"/videos/{job['id']}/result")).read() == b"censored video"
        assert len(pipeline.started) == 1

        other = await post_job(client, video=b"another video")
        assert client.jobs.jobs[other["id"]].work is not client.jobs.jobs[first["id"]].work

    @pytest.mark.asyncio
    async def test_the_same_url_twice_is_downloaded_once(self, client, pipeline):
        downloads = []

        def download(url, path, max_size, digest=None, stop=None):
            downloads.append(url)
            path.write_bytes(b"downloaded")

        with patch("server.http_server.rr.download_video", side_effect=download):
            first = await post_job(client, video_url="https://example.com/a.mp4")
            second = await post_job(client, video_url="https://example.com/a.mp4")
            pipeline.release()
            await until(lambda: client.jobs.jobs[second["id"]].current_state == "done")
            # another censor config is other work
            config = '{"features_to_censor": ["FACE"], "default_censor_config": {"censor_style": {"type": "bar"}}}'
            third = await post_job(client, video_url="https://example.com/a.mp4", config=config)
            await until(lambda: client.jobs.jobs[third["id"]].current_state == "done")
        assert downloads == ["https://example.com/a.mp4"] * 2
        assert client.jobs.jobs[first["id"]].work is client.jobs.jobs[second["id"]].work

    @pytest.mark.asyncio
    async def test_failed_job_result_is_409(self, client, pipeline):
        pipeline.fail = True
        pipeline.release()
        job = await post_job(client, video=b"v")
        await until(lambda: client.jobs.jobs[job["id"]].current_state == "failed")

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
