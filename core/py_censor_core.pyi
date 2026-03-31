"""
Type stubs for py_censor_core — C++ Image Censoring Pipeline.

These stubs are auto-generated from censor_core.cpp and provide IDE
auto-completion and static type checking (mypy / pyright / PyCharm).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

# ── TypedDicts for return values ───────────────────────────────────────────────
# The C++ side returns plain dicts; these aliases document their shape.

class DetectionDict:
    """Shape of a single detection returned in Job.result["features"]."""
    label: str
    score: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float

class ProcessedResultDict:
    """Shape of Job.result once a job has completed successfully."""
    image: np.ndarray           # HxWx3 uint8 BGR
    output_path: str | None
    features: list[dict[str, Any]]   # list of DetectionDict
    bodies: list[Any]               # reserved — always empty for now

# ── CensorConfig (opaque) ──────────────────────────────────────────────────────

class CensorConfig:
    """
    Opaque handle to a loaded censor configuration.
    Construct via :func:`load_censor_config` or pass a plain ``dict``
    wherever a config is accepted.
    """
    ...

# ── Job ───────────────────────────────────────────────────────────────────────

class Job:
    """
    Unit of work submitted to :class:`ImagePipeline`.

    Parameters
    ----------
    image:
        Either a HxWx3 uint8 NumPy array (BGR) **or** a path string/Path to
        an image file on disk.  ``None`` means the image will be loaded later.
    job_id:
        Unique identifier.  Auto-generated (UUID4) when left empty.
    output_path:
        Where the censored image should be written.  ``None`` for in-memory
        only.
    cache_base_dir:
        Root directory used for cache look-ups and writes.
    config:
        A plain ``dict`` (converted to ``CensorConfig`` internally) or
        ``None`` to use the default config loaded from
        ``./default_censor_config.yml``.
    timestamp:
        Caller-supplied timestamp (e.g. video frame time in seconds).
    early_exit:
        When ``True`` the pipeline skips censoring and returns the image as-is.
    override_cache:
        When ``True`` a cached result is ignored and re-computed.
    skip_cache_write:
        When ``True`` the result is not written to cache even if caching is
        otherwise enabled.
    """

    def __init__(
        self,
        image: np.ndarray | str | None = None,
        job_id: str = "",
        output_path: str | None = None,
        cache_base_dir: str | None = None,
        config: dict[str, Any] | None = None,
        timestamp: float = 0.0,
        early_exit: bool = False,
        override_cache: bool = False,
        skip_cache_write: bool = False,
    ) -> None: ...

    # ── read/write properties ─────────────────────────────────────────────────

    @property
    def image(self) -> np.ndarray | None:
        """HxWx3 uint8 BGR image array, or ``None`` if not yet loaded."""
        ...
    @image.setter
    def image(self, value: np.ndarray | str | None) -> None: ...

    @property
    def output_path(self) -> str | None: ...
    @output_path.setter
    def output_path(self, value: str | None) -> None: ...

    @property
    def cache_base_dir(self) -> str | None: ...
    @cache_base_dir.setter
    def cache_base_dir(self, value: str | None) -> None: ...

    @property
    def config(self) -> None:
        """Always returns ``None`` — ``CensorConfig`` is opaque on the Python side."""
        ...
    @config.setter
    def config(self, value: dict[str, Any] | None) -> None: ...

    job_id: str
    timestamp: float
    early_exit: bool
    override_cache: bool
    skip_cache_write: bool

    # ── read-only result fields ───────────────────────────────────────────────

    @property
    def success(self) -> bool:
        """``True`` once the pipeline has finished processing without error."""
        ...

    @property
    def error(self) -> str:
        """Error message if ``success`` is ``False``, otherwise empty string."""
        ...

    @property
    def result(self) -> dict[str, Any] | None:
        """
        ``None`` until the job completes.  On success contains a
        :class:`ProcessedResultDict`-shaped dict with keys
        ``image``, ``output_path``, ``features``, and ``bodies``.
        """
        ...

    @property
    def cancelled(self) -> bool:
        """``True`` if :meth:`cancel` has been called."""
        ...

    def cancel(self) -> None:
        """Request cancellation.  Sets :attr:`cancelled` to ``True``."""
        ...

# ── JobHandle ─────────────────────────────────────────────────────────────────

class JobHandle:
    """
    Returned by :meth:`ImagePipeline.submit`.  Holds a reference to the
    :class:`Job` and an internal future that resolves when processing is done.
    """

    @property
    def job(self) -> Job:
        """The :class:`Job` associated with this handle."""
        ...

    def wait(self) -> Job:
        """Block (releasing the GIL) until the job completes, then return it."""
        ...

    def done(self) -> bool:
        """Non-blocking check — ``True`` if the job has already finished."""
        ...

# ── ImagePipeline ─────────────────────────────────────────────────────────────

class ImagePipeline:
    """
    GPU-accelerated image censoring pipeline backed by a C++ thread pool and
    ONNX Runtime / CUDA inference.

    Typical usage::

        with ImagePipeline(max_workers=4) as pipeline:
            handle = pipeline.submit(job)
            finished_job = handle.wait()

    Or manually::

        pipeline = ImagePipeline(max_workers=4)
        pipeline.start()
        try:
            handle = pipeline.submit(job)
            finished_job = handle.wait()
        finally:
            pipeline.stop()

    Parameters
    ----------
    max_workers:
        Number of CPU worker threads in the thread pool (default: 4).
    """

    def __init__(self, max_workers: int = 4) -> None: ...

    def start(self) -> None:
        """Start the worker threads.  Must be called before :meth:`submit`."""
        ...

    def stop(self) -> None:
        """Drain the queue and shut down all worker threads."""
        ...

    @property
    def is_running(self) -> bool:
        """``True`` after :meth:`start` and before :meth:`stop`."""
        ...

    def submit(self, job: Job) -> JobHandle:
        """
        Enqueue *job* for processing and return a :class:`JobHandle` immediately.

        The pipeline must be :meth:`start`-ed before calling this.
        """
        ...

    def process_batch(self, jobs: list[Job]) -> list[Job]:
        """
        Submit all *jobs* concurrently and block until every one has finished.

        Returns the same list with each job's :attr:`~Job.result` populated.
        """
        ...

    def __enter__(self) -> ImagePipeline: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

# ── Module-level helpers ──────────────────────────────────────────────────────

def load_censor_config(path: str) -> CensorConfig:
    """
    Load a censor configuration from *path* (YAML or JSON).

    The returned :class:`CensorConfig` is opaque — pass it as ``job.config``
    via a ``dict`` instead if you need to inspect or modify values.
    """
    ...

def wait_any(handles: list[JobHandle]) -> JobHandle:
    """
    Block (releasing the GIL) until **any** handle in *handles* is done,
    then return that handle.

    Useful for streaming results as they complete rather than waiting for the
    whole batch::

        handles = [pipeline.submit(job) for job in jobs]
        while handles:
            done = wait_any(handles)
            handles.remove(done)
            process(done.job)
    """
    ...
