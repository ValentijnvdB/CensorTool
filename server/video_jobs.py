"""
Video jobs that run in the background (POST /videos): their state, progress, result and clean-up.

A job is created with a prepare() coroutine that gets the video ready (downloads it for a video_url, checks it) and
returns the backend's VideoJob. It then waits its turn in the video queue (helpers.run_video_job), runs, and is kept
JOB_RETENTION seconds after it finishes; then the job and its files are removed.

Progress and cancelling a running job need the Python backend (VideoJob.phase/frames_done/frames_total/cancel).
With the C++ backend a job reports its state only, and a running job cannot be stopped: it is removed once done.
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

from .errors import ApiError, INTERNAL_ERROR, NOT_FOUND, PROCESSING_FAILED, QUEUE_FULL
from .helpers import VideoTicket, queue_position, run_video_job

JOB_RETENTION = 3600
"""Seconds a finished job (and its files) is kept."""
MAX_ACTIVE_JOBS = 20
"""Jobs that may be downloading, waiting or running at the same time."""
EXPIRE_INTERVAL = 60
"""Seconds between checks for expired jobs."""

ACTIVE_STATES = ('downloading', 'queued', 'running')
FINAL_STATES = ('done', 'failed', 'cancelled')

# Share of the progress each phase of a (Python backend) VideoJob covers.
PHASE_PROGRESS = {'detecting': (0.0, 0.7), 'censoring': (0.7, 0.95), 'encoding': (0.95, 1.0)}

Prepare = Callable[['VideoJobRecord'], Awaitable[Any]]
"""Gets the video ready: sets record.info and record.output_path, and returns the backend VideoJob."""


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


@dataclass(eq=False)
class VideoJobRecord:
    id: str
    video_format: str
    content_type: str
    downloads: bool                         # whether prepare() downloads the video first
    state: str = 'new'
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    info: Any = None                        # http_server.VideoInfo, once the video is there
    ticket: VideoTicket | None = None
    error: str | None = None
    code: str | None = None
    removed: bool = False                   # deleted by the client while it ran: remove the files once it ends
    task: asyncio.Task | None = None

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

    def _progress(self) -> tuple[str | None, int | None, int | None, float | None]:
        """phase, frames_done, frames_total and progress (0-1) of the backend job; None for what is unknown."""
        state = self.current_state
        if state == 'done':
            return 'done', None, None, 1.0
        if state in ('downloading', 'queued', 'new'):
            return None, None, None, 0.0
        job = self.ticket.job if self.ticket is not None else None
        phase = getattr(job, 'phase', None)
        if phase is None or state != 'running':
            return None, None, None, None
        done, total = int(job.frames_done), int(job.frames_total)
        low, high = PHASE_PROGRESS.get(phase, (1.0, 1.0))
        fraction = min(1.0, done / total) if total > 0 else 0.0
        return phase, done, total, round(low + (high - low) * fraction, 3)

    def to_dict(self) -> dict:
        phase, frames_done, frames_total, progress = self._progress()
        state = self.current_state
        body = {
            'id': self.id,
            'state': state,
            'phase': phase,
            'progress': progress,
            'frames_done': frames_done,
            'frames_total': frames_total,
            'queue_position': queue_position(self.ticket) if self.ticket is not None else None,
            'filename': self.output_path.name if self.output_path is not None else None,
            'format': self.video_format,
            'content_type': self.content_type,
            'video': None,
            'created_at': _iso(self.created_at),
            'started_at': _iso(self.started_at),
            'finished_at': _iso(self.finished_at),
            'processing_ms': None,
            'expires_at': _iso(self.finished_at + JOB_RETENTION) if self.finished_at is not None else None,
            'url': f'/videos/{self.id}',
            'result_url': f'/videos/{self.id}/result' if state == 'done' else None,
        }
        if self.info is not None:
            body['video'] = {'frames': self.info.frames, 'fps': round(self.info.fps, 3),
                             'duration': round(self.info.duration, 3), 'width': self.info.width,
                             'height': self.info.height}
        if self.started_at is not None and self.finished_at is not None:
            body['processing_ms'] = round((self.finished_at - self.started_at) * 1000)
        if state == 'failed':
            body['error'], body['code'] = self.error, self.code
        return body

    def remove_files(self) -> None:
        for path in (self.input_path, self.output_path):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Could not remove {path}: {e}")


class VideoJobs:
    """The background video jobs of the server."""

    def __init__(self, retention: float = JOB_RETENTION, max_active: int = MAX_ACTIVE_JOBS):
        self.retention = retention
        self.max_active = max_active
        self.jobs: dict[str, VideoJobRecord] = {}

    # ---------------- lifecycle ----------------
    def check_capacity(self) -> None:
        """:raises ApiError: (429) when too many jobs are active."""
        if sum(r.active for r in self.jobs.values()) >= self.max_active:
            raise ApiError(f"Too many video jobs: at most {self.max_active} can wait or run at the same time. "
                           f"Try again later.", QUEUE_FULL, 429)

    def create(self, video_format: str, content_type: str, downloads: bool, prepare: Prepare) -> VideoJobRecord:
        """Create a job and start it in the background."""
        self.check_capacity()
        record = VideoJobRecord(uuid.uuid4().hex, video_format, content_type, downloads)
        record.state = 'downloading' if downloads else 'queued'
        self.jobs[record.id] = record
        record.task = asyncio.create_task(self._run(record, prepare))
        return record

    async def _run(self, record: VideoJobRecord, prepare: Prepare) -> None:
        try:
            job = await prepare(record)
            record.ticket = VideoTicket(job)
            record.state = 'queued'
            result = await run_video_job(record.ticket)
            if record.removed or result.error == 'cancelled':
                record.state = 'cancelled'
            elif result.success:
                record.state = 'done'
            else:
                record.state, record.code = 'failed', PROCESSING_FAILED
                record.error = f"Censoring failed: {result.error}"
        except asyncio.CancelledError:
            record.state = 'cancelled'
        except ApiError as e:
            record.state, record.code, record.error = 'failed', e.code, str(e)
        except Exception as e:
            logger.exception(f"Video job {record.id} failed: {e}")
            record.state, record.code, record.error = 'failed', INTERNAL_ERROR, str(e) or type(e).__name__
        finally:
            record.finished_at = time.time()
            if record.state != 'done' and record.output_path is not None:
                record.output_path.unlink(missing_ok=True)  # reserved or partly written
            if record.removed:
                record.remove_files()

    def get(self, job_id: str) -> VideoJobRecord:
        """:raises ApiError: (404) if there is no such job."""
        self.expire()
        record = self.jobs.get(job_id)
        if record is None:
            raise ApiError(f"No video job '{job_id}'. Finished jobs are removed after {self.retention // 60:.0f} "
                           f"minutes.", NOT_FOUND, 404)
        return record

    def files(self) -> set[Path]:
        """The files of every job the server still knows, which file clean-up must leave alone."""
        return {path for record in self.jobs.values() for path in (record.input_path, record.output_path)
                if path is not None}

    def all(self) -> list[VideoJobRecord]:
        self.expire()
        return list(self.jobs.values())

    def delete(self, job_id: str) -> VideoJobRecord:
        """
        Remove a job and its files. A waiting job is taken out of the queue; a running one is stopped (Python
        backend) or left to finish, after which its files are removed.
        """
        record = self.get(job_id)
        del self.jobs[job_id]
        if not record.active:
            record.remove_files()
            return record

        record.removed = True
        state = record.current_state
        if state == 'running':
            job = record.ticket.job
            if hasattr(job, 'cancel'):
                job.cancel()
        elif record.task is not None:
            record.task.cancel()  # downloading or waiting: nothing has run yet
        record.state = 'cancelled'
        return record

    def expire(self, now: float | None = None) -> int:
        """Remove the jobs that finished more than retention seconds ago, with their files. Returns how many."""
        now = time.time() if now is None else now
        expired = [r for r in self.jobs.values()
                   if r.finished_at is not None and now - r.finished_at > self.retention]
        for record in expired:
            del self.jobs[record.id]
            record.remove_files()
        return len(expired)

    async def cancel_all(self) -> None:
        """Stop every job, e.g. when the server shuts down."""
        for record in list(self.jobs.values()):
            if record.active and record.task is not None:
                record.task.cancel()
        await asyncio.gather(*(r.task for r in self.jobs.values() if r.task is not None), return_exceptions=True)

    async def expire_periodically(self, interval: float = EXPIRE_INTERVAL) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                if n := self.expire():
                    logger.info(f"Removed {n} expired video job(s).")
            except Exception as e:
                logger.error(f"Error removing expired video jobs: {e}")


video_jobs = VideoJobs()
