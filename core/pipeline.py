"""Image pipeline: Job, JobHandle, ImagePipeline and wait_any. Same API as censor_core 2.2."""
from __future__ import annotations

import hashlib
import sys
import threading
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .boxes import build_boxes
from .cache import (CacheEntries, cached_detections, image_cache_path, read_image_cache, update_cache,
                    write_image_cache)
from .censor_config import CensorConfig, optional_censor_config
from .draw import apply_censor_boxes
from .geometry import Detection
from .model_graph import ModelGraph, StageDetections
from .pipeline_config import PipelineConfig, load_pipeline_config, pipeline_config_from_dict
from .yolo import check_image, detection_to_dict

WRITE_WORKERS = 3


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


class Job:
    """
    Unit of work submitted to ImagePipeline.

    image is a HxWx3 uint8 BGR array, or a path to an image file (loaded on the worker thread).
    Without a config every model runs and the detections are returned uncensored; with a config only the
    models it needs run.
    """

    def __init__(self, image: Any = None, job_id: str = '', output_path: Any = None, cache_base_dir: Any = None,
                 config: Optional[dict[str, Any]] = None, size: int = 0, timestamp: float = 0.0,
                 early_exit: bool = False, override_cache: bool = False, skip_existing: bool = False,
                 use_cache: bool = True):
        self._image: Optional[np.ndarray] = None
        self._input_path: Optional[str] = None
        if isinstance(image, np.ndarray):
            self._image = check_image(image)
        elif image is not None:
            self._input_path = str(image)

        self.job_id: str = job_id or str(uuid.uuid4())
        self.output_path = output_path
        self.cache_base_dir = cache_base_dir
        self._config: Optional[CensorConfig] = optional_censor_config(config)
        self.size = int(size)
        self.timestamp = float(timestamp)
        self.early_exit = bool(early_exit)
        self.override_cache = bool(override_cache)
        self.skip_existing = bool(skip_existing)
        self.use_cache = bool(use_cache)

        self._cancelled = threading.Event()
        self._success = False
        self._skipped = False
        self._error: Optional[str] = None
        self._result: Optional[dict[str, Any]] = None
        self._detections: Optional[StageDetections] = None  # per stage, set on completion

    # ------------------------------------ read/write properties ------------------------------------
    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @image.setter
    def image(self, value: Any) -> None:
        if isinstance(value, np.ndarray):
            self._image = check_image(value)
        elif value is not None:
            loaded = cv2.imread(str(value))
            self._image = loaded

    @property
    def output_path(self) -> Optional[str]:
        return self._output_path

    @output_path.setter
    def output_path(self, value: Any) -> None:
        self._output_path = _optional_str(value)

    @property
    def cache_base_dir(self) -> Optional[str]:
        return self._cache_base_dir

    @cache_base_dir.setter
    def cache_base_dir(self, value: Any) -> None:
        self._cache_base_dir = _optional_str(value)

    @property
    def config(self) -> None:
        """Always returns None: the parsed censor config is opaque."""
        return None

    @config.setter
    def config(self, value: Optional[dict[str, Any]]) -> None:
        self._config = optional_censor_config(value)

    # ------------------------------------ read-only result fields ------------------------------------
    @property
    def success(self) -> bool:
        return self._success

    @property
    def skipped(self) -> bool:
        return self._skipped

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def result(self) -> Optional[dict[str, Any]]:
        return self._result

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def __repr__(self) -> str:
        return f"Job(job_id={self.job_id!r}, success={self._success}, error={self._error!r})"


class JobHandle:
    """Returned by ImagePipeline.submit(): the job and a future that resolves when it is done."""

    def __init__(self, job: Job, future: Future):
        self._job = job
        self._future = future

    @property
    def job(self) -> Job:
        return self._job

    def wait(self) -> Job:
        """Block until the job completes, then return it."""
        self._future.result()
        return self._job

    def done(self) -> bool:
        return self._future.done()


def wait_any(handles: list[JobHandle]) -> JobHandle:
    """Block until any handle in handles is done, then return that handle."""
    if not handles:
        raise ValueError("wait_any() needs at least one handle")
    while True:
        for h in handles:
            if h.done():
                return h
        wait([h._future for h in handles], return_when=FIRST_COMPLETED)


def detections_to_list(stages: StageDetections, order: list[str]) -> list[dict[str, Any]]:
    """All detections as one list in pipeline order, with ids, model names and parent ids."""
    ids: dict[tuple[str, int], int] = {}
    for stage in order:
        for i in range(len(stages.get(stage, []))):
            ids[(stage, i)] = len(ids)

    out = []
    for stage in order:
        for i, d in enumerate(stages.get(stage, [])):
            item = detection_to_dict(d)
            item['id'] = ids[(stage, i)]
            item['model'] = stage
            item['parent'] = ids.get((d.parent_stage, d.parent))
            out.append(item)
    return out


def hash_image(image: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).data)
    return h.hexdigest()


class ImagePipeline:
    """
    Runs the configured models and censors images.

    Load the models from a pipeline.yml with ImagePipeline.from_yaml(path), or pass the same structure directly:
    ImagePipeline(models=[{...}], landmarks={...}).

    mode 's' runs one model at a time (clip_to-only models only when a detection needs clipping);
    'p' runs independent models in parallel.
    """

    def __init__(self, models: list[dict[str, Any]], landmarks: Optional[dict[str, Any]] = None,
                 device: str = 'cpu', max_workers: int = 4, mode: str = 's', threads: Any = 'auto'):
        root = {'device': device, 'threads': threads, 'models': _plain(models)}
        if landmarks is not None:
            root['landmarks'] = _plain(landmarks)
        self._init(pipeline_config_from_dict(root, Path.cwd()), max_workers, mode)

    @classmethod
    def _from_config(cls, config: PipelineConfig, max_workers: int, mode: str) -> 'ImagePipeline':
        self = cls.__new__(cls)
        self._init(config, max_workers, mode)
        return self

    def _init(self, config: PipelineConfig, max_workers: int, mode: str) -> None:
        if mode not in ('s', 'p'):
            print(f"WARNING: mode should be 's' or 'p', got '{mode}'. Using 's'.", file=sys.stderr)
        self._max_workers = max(1, int(max_workers))
        self._graph = ModelGraph(config, mode == 'p')
        self._lock = threading.Lock()
        self._running = False
        self._workers: Optional[ThreadPoolExecutor] = None
        self._writers: Optional[ThreadPoolExecutor] = None

    @staticmethod
    def from_yaml(path: str, max_workers: int = 4, mode: str = 's') -> 'ImagePipeline':
        """Loads the models listed in a pipeline.yml; relative paths are relative to the file."""
        return ImagePipeline._from_config(load_pipeline_config(path), max_workers, mode)

    # ------------------------------------ properties ------------------------------------
    @property
    def models(self) -> list[str]:
        """Stage names in pipeline order ('landmarks' last, if configured)."""
        return list(self._graph.stage_names)

    @property
    def labels(self) -> dict[str, str]:
        """Every label the pipeline can produce, mapped to the model that produces it."""
        return dict(self._graph.labels)

    @property
    def graph(self) -> ModelGraph:
        return self._graph

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------ lifecycle ------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._graph.start()
            self._workers = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix='censor-worker')
            self._writers = ThreadPoolExecutor(max_workers=WRITE_WORKERS, thread_name_prefix='censor-writer')
            self._running = True

    def stop(self) -> None:
        """Finishes the queued jobs and pending writes, then shuts down all threads."""
        with self._lock:
            self._running = False
            if self._workers is not None:
                self._workers.shutdown(wait=True)
                self._workers = None
            self._graph.stop()
            if self._writers is not None:
                self._writers.shutdown(wait=True)
                self._writers = None

    def __enter__(self) -> 'ImagePipeline':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------ submission ------------------------------------
    def submit(self, job: Job) -> JobHandle:
        """Enqueue job for processing and return a JobHandle immediately."""
        workers = self._workers
        if not self._running or workers is None:
            raise RuntimeError("Pipeline is not running. Call start() first.")
        return JobHandle(job, workers.submit(self._run_job, job))

    def process_batch(self, jobs: list[Job]) -> list[Job]:
        """Submit all jobs and block until every one has finished."""
        handles = [self.submit(job) for job in jobs]
        for h in handles:
            h.wait()
        return jobs

    # ------------------------------------ processing ------------------------------------
    def _run_job(self, job: Job) -> Job:
        try:
            self._process_one(job)
        except Exception as e:
            job._success = False
            job._error = str(e) or type(e).__name__
            if not isinstance(e, RuntimeError):
                traceback.print_exc()
        return job

    def _write(self, fn, *args) -> None:
        def run():
            try:
                fn(*args)
            except Exception as e:
                print(f"[writer] failed to write {args[0]}: {e}", file=sys.stderr)
        writers = self._writers
        if writers is not None:
            writers.submit(run)
        else:
            run()

    @staticmethod
    def _write_image(path: str, image: np.ndarray) -> None:
        if not cv2.imwrite(path, image):
            raise RuntimeError("cv2.imwrite failed")

    def _process_one(self, job: Job) -> None:
        if job.skip_existing and job.output_path is not None and Path(job.output_path).exists():
            job._success = True
            job._skipped = True
            return

        # --- Step 1: load image ---
        if job._image is None:
            if job._input_path is None:
                job._error = "no image or input_path provided"
                return
            image = cv2.imread(job._input_path)
            if image is None:
                job._error = f"cv2.imread failed: {job._input_path}"
                return
            job._image = image

        config = job._config
        if config is not None:
            self._graph.validate(config)

        # --- Step 2: check the cache ---
        cache_file = None
        if job.use_cache and job.cache_base_dir is not None:
            cache_file = image_cache_path(Path(job.cache_base_dir), hash_image(job._image))

        fingerprints = self._graph.fingerprints(job.size)
        entries: CacheEntries = {} if job.override_cache else read_image_cache(cache_file)
        detections: dict[str, list[Detection]] = cached_detections(entries, fingerprints)
        cached = list(detections)

        if job.cancelled:
            return

        # --- Step 3: run the models whose results are missing ---
        ran = self._graph.run(job, detections, self._graph.plan(config), config)
        if job.cancelled:
            return

        if ran and cache_file is not None:
            update_cache(entries, detections, fingerprints)
            self._write(write_image_cache, cache_file, entries)

        order = self._graph.stage_names
        job._detections = detections

        # --- Step 4: detection only ---
        if job.early_exit or config is None:
            job._result = self._result(job._image, None, detections, order, cached)
            job._success = True
            return

        # --- Step 5: censor ---
        boxes = build_boxes(detections, order, job.timestamp, config)
        job._image = apply_censor_boxes(job._image, boxes, config)

        # --- Step 6: write ---
        if job.output_path is not None:
            self._write(self._write_image, job.output_path, job._image)

        job._result = self._result(job._image, job.output_path, detections, order, cached)
        job._success = True

    @staticmethod
    def _result(image: np.ndarray, output_path: Optional[str], detections: StageDetections, order: list[str],
                cached: list[str]) -> dict[str, Any]:
        return {
            'image': image,
            'output_path': output_path,
            'detections': detections_to_list(detections, order),
            'cached_models': [s for s in order if s in cached],
        }


def _plain(value: Any) -> Any:
    """Paths and tuples to plain strings and lists, like the JSON round trip of the C++ bindings."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
