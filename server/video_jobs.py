"""
Video jobs that run in the background (POST /videos): their state, progress, result and clean-up.

Every POST /videos gets a job of its own (VideoJobRecord), but the work behind it (VideoWork: the download, the
censoring and the censored video) is shared by the jobs for the same video, censor config and output format. A video
is the same when it has the same URL (checked before downloading) or the same content (an upload, or a download once
it is done). So a reloaded page, or two pages with the same video, have it censored once.

Work is created with a prepare() coroutine that gets the video ready (downloads it for a video_url, checks it) and
returns the backend's VideoJob. It then waits its turn in the video queue (helpers.run_video_job) and runs.

A job is kept for the retention time (JOB_RETENTION) after its work finished, or after the job was created if it
joined work that had finished already. Work is removed, with its files, once no job uses it. Deleting a job only
removes that job; work that is still running when its last job is deleted is cancelled after a grace period
(server_video_job_grace_seconds in config.yml), unless a job for the same video comes along first, e.g. when the page
with the video is reloaded.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from app.config import CONFIG
from .errors import ApiError, INTERNAL_ERROR, NOT_FOUND, PROCESSING_FAILED, QUEUE_FULL
from .helpers import VideoTicket, queue_position, run_video_job

JOB_RETENTION = 3600
"""Seconds a job is kept after its work finished."""
MAX_ACTIVE_WORK = 20
"""Videos that may be downloading, waiting or running at the same time."""
EXPIRE_INTERVAL = 60
"""Seconds between checks for expired jobs."""

FINAL_STATES = ('done', 'failed', 'cancelled')

# Share of the progress each phase of a VideoJob covers.
PHASE_PROGRESS = {'detecting': (0.0, 0.7), 'censoring': (0.7, 0.95), 'encoding': (0.95, 1.0)}

Prepare = Callable[['VideoWork'], Awaitable[Any]]
"""Gets the video ready: sets work.info, work.input_path and work.output_path, and returns the backend VideoJob."""


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


# ------------------------------------------------------------------ the shared work

@dataclass(eq=False)
class VideoWork:
    """Censoring one video with one censor config to one format; shared by the jobs that asked for it."""
    id: str
    video_format: str
    content_type: str
    config_key: str                         # helpers.censor_config_key() of the censor config
    url: str | None = None                  # the video_url it downloads, if any
    digest: str | None = None               # sha256 of the video, once it is there
    state: str = 'new'                      # downloading | queued | done | failed | cancelled (see current_state)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    info: Any = None                        # http_server.VideoInfo, once the video is there
    ticket: VideoTicket | None = None
    error: str | None = None
    code: str | None = None
    job_ids: set[str] = field(default_factory=set)
    removed: bool = False                   # cancelled while it ran: remove the files once it ends
    task: asyncio.Task | None = None
    grace_timer: asyncio.TimerHandle | None = None  # set while it runs without jobs

    @property
    def active(self) -> bool:
        return self.state not in FINAL_STATES

    @property
    def current_state(self) -> str:
        if self.state in FINAL_STATES or self.ticket is None:
            return self.state
        return {'running': 'running', 'finished': 'running'}.get(self.ticket.state, 'queued')

    @property
    def started_at(self) -> float | None:
        return self.ticket.started_at if self.ticket is not None else None

    def matches(self, config_key: str, video_format: str, url: str | None, digest: str | None) -> bool:
        """Whether a job for this video, censor config and format can use this work."""
        if self.removed or self.state in ('failed', 'cancelled'):
            return False
        if self.config_key != config_key or self.video_format != video_format:
            return False
        return (url is not None and self.url == url) or (digest is not None and self.digest == digest)

    def progress(self) -> tuple[str | None, int | None, int | None, float | None]:
        """phase, frames_done, frames_total and progress (0-1) of the backend job; None for what is unknown."""
        state = self.current_state
        if state == 'done':
            return 'done', None, None, 1.0
        if state in ('downloading', 'queued', 'new'):
            return None, None, None, 0.0
        if state != 'running':
            return None, None, None, None
        job = self.ticket.job
        phase = job.phase
        if phase is None:
            return None, 0, 0, 0.0          # picked up, no phase started yet
        done, total = int(job.frames_done), int(job.frames_total)
        low, high = PHASE_PROGRESS.get(phase, (1.0, 1.0))
        fraction = min(1.0, done / total) if total > 0 else 0.0
        return phase, done, total, round(low + (high - low) * fraction, 3)

    def remove_files(self) -> None:
        for path in (self.input_path, self.output_path):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Could not remove {path}: {e}")


# ------------------------------------------------------------------ a client's job

@dataclass(eq=False)
class VideoJobRecord:
    """One POST /videos: a client's view of the work it uses."""
    id: str
    work: VideoWork
    retention: float = JOB_RETENTION
    created_at: float = field(default_factory=time.time)
    deleted: bool = False                   # deleted by its client before the work finished

    # what the endpoints read, from the work
    @property
    def current_state(self) -> str:
        return 'cancelled' if self.deleted else self.work.current_state

    @property
    def error(self) -> str | None:
        return self.work.error

    @property
    def output_path(self) -> Path | None:
        return self.work.output_path

    @property
    def video_format(self) -> str:
        return self.work.video_format

    @property
    def info(self) -> Any:
        return self.work.info

    @property
    def expires_at(self) -> float | None:
        """When the job is removed: the retention time after its work finished (or after it joined)."""
        if self.work.finished_at is None:
            return None
        return max(self.work.finished_at, self.created_at) + self.retention

    def to_dict(self) -> dict:
        work = self.work
        state = self.current_state
        phase, frames_done, frames_total, progress = work.progress() if not self.deleted else (None,) * 4
        body = {
            'id': self.id,
            'state': state,
            'phase': phase,
            'progress': progress,
            'frames_done': frames_done,
            'frames_total': frames_total,
            'queue_position': queue_position(work.ticket) if work.ticket is not None and not self.deleted else None,
            'filename': work.output_path.name if work.output_path is not None else None,
            'format': work.video_format,
            'content_type': work.content_type,
            'video': None,
            'created_at': _iso(self.created_at),
            'started_at': _iso(work.started_at),
            'finished_at': _iso(work.finished_at),
            'processing_ms': None,
            'expires_at': _iso(self.expires_at),
            'url': f'/videos/{self.id}',
            'result_url': f'/videos/{self.id}/result' if state == 'done' else None,
        }
        if work.info is not None:
            body['video'] = {'frames': work.info.frames, 'fps': round(work.info.fps, 3),
                             'duration': round(work.info.duration, 3), 'width': work.info.width,
                             'height': work.info.height}
        if work.started_at is not None and work.finished_at is not None:
            body['processing_ms'] = round((work.finished_at - work.started_at) * 1000)
        if state == 'failed':
            body['error'], body['code'] = work.error, work.code
        return body


# ------------------------------------------------------------------ the registry

class VideoJobs:
    """The background video jobs of the server, and the work they share."""

    def __init__(self, retention: float = JOB_RETENTION, max_active: int = MAX_ACTIVE_WORK,
                 grace: float | None = None):
        """
        :param grace: seconds work without jobs goes on before it is cancelled; None reads
                      server_video_job_grace_seconds from config.yml (each time, so a changed value applies).
        """
        self.retention = retention
        self.max_active = max_active
        self._grace = grace
        self.jobs: dict[str, VideoJobRecord] = {}
        self.works: dict[str, VideoWork] = {}

    @property
    def grace(self) -> float:
        return float(CONFIG.server_video_job_grace_seconds) if self._grace is None else self._grace

    # ---------------- creating jobs ----------------
    def check_capacity(self) -> None:
        """:raises ApiError: (429) when too many videos are being censored."""
        if sum(w.active for w in self.works.values()) >= self.max_active:
            raise ApiError(f"Too many video jobs: at most {self.max_active} videos can wait or run at the same "
                           f"time. Try again later.", QUEUE_FULL, 429)

    def join(self, config_key: str, video_format: str, url: str | None = None,
             digest: str | None = None) -> VideoJobRecord | None:
        """
        A new job on the work for the same video (url or digest), censor config and format, if there is such work;
        None otherwise.
        """
        work = next((w for w in self.works.values() if w.matches(config_key, video_format, url, digest)), None)
        if work is None:
            return None
        record = self._add_job(work)
        logger.info(f"Video job {record.id} shares the work of {work.id} ({work.current_state}).")
        return record

    def create(self, video_format: str, content_type: str, prepare: Prepare, *, config_key: str,
               url: str | None = None, digest: str | None = None) -> VideoJobRecord:
        """
        Start new work in the background, and a job on it.

        :param url: the video_url prepare() downloads (an upload has none).
        :param digest: sha256 of an uploaded video.
        :raises ApiError: (429) when too many videos are being censored.
        """
        self.check_capacity()
        work = VideoWork(uuid.uuid4().hex, video_format, content_type, config_key, url=url, digest=digest)
        work.state = 'downloading' if url is not None else 'queued'
        self.works[work.id] = work
        work.task = asyncio.create_task(self._run(work, prepare))
        record = self._add_job(work)
        logger.info(f"Video job {record.id} created ({'downloads the video' if url else 'uploaded video'}).")
        return record

    def _add_job(self, work: VideoWork) -> VideoJobRecord:
        if work.grace_timer is not None:   # wanted again before its grace period ran out
            work.grace_timer.cancel()
            work.grace_timer = None
        record = VideoJobRecord(uuid.uuid4().hex, work, self.retention)
        work.job_ids.add(record.id)
        self.jobs[record.id] = record
        return record

    async def _run(self, work: VideoWork, prepare: Prepare) -> None:
        try:
            job = await prepare(work)
            work.ticket = VideoTicket(job)
            work.state = 'queued'
            result = await run_video_job(work.ticket)
            if work.removed or result.error == 'cancelled':
                work.state = 'cancelled'
            elif result.success:
                work.state = 'done'
            else:
                work.state, work.code = 'failed', PROCESSING_FAILED
                work.error = f"Censoring failed: {result.error}"
        except asyncio.CancelledError:
            work.state = 'cancelled'
        except ApiError as e:
            work.state, work.code, work.error = 'failed', e.code, str(e)
        except Exception as e:
            logger.exception(f"Video work {work.id} failed: {e}")
            work.state, work.code, work.error = 'failed', INTERNAL_ERROR, str(e) or type(e).__name__
        finally:
            work.finished_at = time.time()
            if work.state == 'failed':
                logger.warning(f"Video work {work.id} failed ({work.code}): {work.error}")
            else:
                logger.info(f"Video work {work.id} {work.state}.")
            if work.state != 'done' and work.output_path is not None:
                work.output_path.unlink(missing_ok=True)  # reserved or partly written
            if work.removed:
                work.remove_files()

    # ---------------- reading ----------------
    def get(self, job_id: str) -> VideoJobRecord:
        """:raises ApiError: (404) if there is no such job."""
        self.expire()
        record = self.jobs.get(job_id)
        if record is None:
            raise ApiError(f"No video job '{job_id}'. Finished jobs are removed after {self.retention // 60:.0f} "
                           f"minutes.", NOT_FOUND, 404)
        return record

    def all(self) -> list[VideoJobRecord]:
        self.expire()
        return list(self.jobs.values())

    def files(self) -> set[Path]:
        """The files of the work the server still has, which file clean-up must leave alone."""
        return {path for work in self.works.values() for path in (work.input_path, work.output_path)
                if path is not None}

    # ---------------- removing ----------------
    def delete(self, job_id: str) -> VideoJobRecord:
        """
        Remove a job. Its work goes on while other jobs use it; otherwise finished work is removed with its files
        at once, and running work after the grace period.
        """
        record = self.get(job_id)
        self._remove_job(record)
        if record.work.active:
            record.deleted = True
        return record

    def _remove_job(self, record: VideoJobRecord) -> None:
        del self.jobs[record.id]
        work = record.work
        work.job_ids.discard(record.id)
        if work.job_ids or work.removed:
            return
        if not work.active:
            self._remove_work(work)
        elif self.grace > 0:
            logger.info(f"Video work {work.id} has no jobs left: cancelled in {self.grace:g} s unless it is "
                        f"wanted again.")
            work.grace_timer = asyncio.get_running_loop().call_later(self.grace, self._end_grace, work)
        else:
            self._cancel_work(work)

    def _end_grace(self, work: VideoWork) -> None:
        work.grace_timer = None
        if work.job_ids or work.removed or self.works.get(work.id) is not work:
            return
        if work.active:
            self._cancel_work(work)
        else:
            self._remove_work(work)   # finished during the grace period, but nobody wants it

    def _remove_work(self, work: VideoWork) -> None:
        self.works.pop(work.id, None)
        work.remove_files()

    def _cancel_work(self, work: VideoWork) -> None:
        """Stop work and remove it; its files are removed once it has stopped."""
        self.works.pop(work.id, None)
        work.removed = True
        if work.current_state == 'running':
            work.ticket.job.cancel()
        elif work.task is not None:
            work.task.cancel()   # downloading or waiting: nothing has run yet
        work.state = 'cancelled'
        logger.info(f"Video work {work.id} cancelled.")

    def expire(self, now: float | None = None) -> int:
        """Remove the jobs whose retention time has passed, and the work nothing uses any more. Returns how many."""
        now = time.time() if now is None else now
        expired = [r for r in self.jobs.values() if r.expires_at is not None and now > r.expires_at]
        for record in expired:
            self._remove_job(record)
        return len(expired)

    async def cancel_all(self) -> None:
        """Stop all work, e.g. when the server shuts down."""
        for work in list(self.works.values()):
            if work.grace_timer is not None:
                work.grace_timer.cancel()
            if work.active and work.task is not None:
                work.task.cancel()
        await asyncio.gather(*(w.task for w in self.works.values() if w.task is not None), return_exceptions=True)

    async def expire_periodically(self, interval: float = EXPIRE_INTERVAL) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                if n := self.expire():
                    logger.info(f"Removed {n} expired video job(s).")
            except Exception as e:
                logger.error(f"Error removing expired video jobs: {e}")


video_jobs = VideoJobs()
