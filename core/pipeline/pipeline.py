from __future__ import annotations

import threading
from abc import ABC
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Any

from .pipeline_types import Job, GPURequest, GPUIntermediateResult, GPUResult

from ..models import nudenet as nn, human_detection as hd

# ---------------------------------------------------------------------------
# GPU worker threads
# ---------------------------------------------------------------------------

_SENTINEL = object()  # signals the GPU thread to stop


def _nudenet_worker(gpu_queue: Queue[GPURequest], hd_queue: Queue[GPUIntermediateResult]) -> None:
    """
    Runs in a single dedicated thread.
    Pulls GPURequests, runs inference sequentially, pushes GPUResults to the human detection queue.
    """
    while True:
        item = gpu_queue.get()
        if item is _SENTINEL:
            hd_queue.put(_SENTINEL)
            gpu_queue.task_done()
            break

        request: GPURequest = item
        raw_features = None
        error = None

        try:
            if any(request.need_features):
                imgs = [img for (i, img) in enumerate(request.data.adj_images) if request.need_features[i]]
                raw_features = nn.get_raw_nudenet_output(imgs)
        except Exception as exc:
            error = exc
        finally:
            hd_queue.put(GPUIntermediateResult(
                raw_features=raw_features,
                need_bodies=request.need_bodies,
                error=error,
                data=request.data
            ))
            gpu_queue.task_done()


def _hd_worker(hd_queue: Queue[GPUIntermediateResult]) -> None:
    """
    Human detection worker. Depending on the model, runs on either CPU or GPU.
    Pulls GPURequests provided by the nudenet_worker, pushes GPUResults back
    via each request's per-image reply_queue.
    """
    while True:
        item = hd_queue.get()
        if item is _SENTINEL:
            hd_queue.task_done()
            break

        request: GPUIntermediateResult = item
        raw_bodies = None
        error = item.error

        if error is not None:
            raise error

        try:
            if any(request.need_bodies):
                imgs = [img for (i, img) in enumerate(request.data.adj_images) if request.need_bodies[i]]
                raw_bodies = hd.find_human_polygons(imgs)
        except Exception as exc:
            error = exc
        finally:
            request.data.reply_queue.put(GPUResult(
                raw_features=request.raw_features,
                raw_bodies=raw_bodies,
                error=error
            ))
            hd_queue.task_done()


# ---------------------------------------------------------------------------
# Generic pipeline class
# ---------------------------------------------------------------------------

class GenericPipeline(ABC):
    def __init__(self, process_one, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._gpu_queue: Queue = Queue()
        self._hd_queue = Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._nudenet_worker: threading.Thread | None = None
        self._hd_worker: threading.Thread | None = None
        self._process_one = process_one

    # --- lifecycle ---

    def start(self) -> None:
        if self._executor is not None:
            return  # already running
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._nudenet_worker = threading.Thread(
            target=_nudenet_worker,
            args=(self._gpu_queue, self._hd_queue),
            daemon=True,
            name="nudenet-worker",
        )
        self._nudenet_worker.start()
        self._hd_worker = threading.Thread(
            target=_hd_worker,
            args=(self._hd_queue,),
            daemon=True,
            name="hd-worker",
        )
        self._hd_worker.start()

    def stop(self, wait: bool = True) -> None:
        """Graceful shutdown. Waits for in-flight work to finish."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
        # Signal GPU thread to exit after draining the queue
        self._gpu_queue.put(_SENTINEL)
        if self._nudenet_worker:
            self._nudenet_worker.join()
            self._nudenet_worker = None
        if self._hd_worker:
            self._hd_worker.join()
            self._hd_worker = None

    def __enter__(self) -> "ImagePipeline":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # --- submission ---

    def submit(self, job: Job) -> Future[Job]:
        """
        Submit one Job for processing. Returns immediately with a Future.
        Call future.result() to get back the same Job with .success,
        .error, and .result populated.
        """
        if self._executor is None:
            raise RuntimeError("Pipeline is not running. Call start() first.")
        return self._executor.submit(self._process_one, job, self._gpu_queue)

    def submit_batch(self, jobs: list[Job]) -> list[Future[Job]]:
        """Submit multiple jobs; returns a list of Futures in the same order."""
        return [self.submit(job) for job in jobs]

    def process_batch(self, jobs: list[Job]) -> list[Job]:
        """Submit a batch and block until all are done. Never raises — check job.success."""
        return [f.result() for f in self.submit_batch(jobs)]



