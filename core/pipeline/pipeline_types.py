import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Any

import numpy as np
from shapely import Polygon

from ..datatypes import RawBox, CensorConfig

ImageInput = Path | str | np.ndarray | bytes


DEFAULT_CENSOR_CONFIG: CensorConfig|None = None
def set_default_censor_config(config: CensorConfig):
    global DEFAULT_CENSOR_CONFIG
    DEFAULT_CENSOR_CONFIG = config


@dataclass
class Elements:
    features: list[RawBox] | None
    bodies: list[Polygon] | None
    cache_path: Path


@dataclass
class Job:
    """
    Submitted by the caller; travels through the whole pipeline unchanged.

    Attributes:
        image:              The image to process - path on disk or a loaded np.ndarray.
        job_id:             Unique identifier. Auto-generated if not supplied.
        output_path:        Where to write the result image. None → don't write.
        return_bytes:       Whether the result image should be returned as bytes.
        output_extension:   How to encode the image if it is returned as bytes (only used if return_bytes is True).
        override_cache:     Whether to ignore the cached elements for this image
        early_exit:         If True, return after the cache write (step 7) and skip
                            image modification, save, and the full ProcessedResult.
                            The returned ProcessedResult will carry the unmodified
                            original image plus features/bodies instead.
        skip_cache_write:   If set, does not write the intermediate results to the cache.
        cache_base_dir:     The base directory for caching intermediate results.
        sizes:              The sizes to run the image at.
        timestamp:          The timestamp this image was taken at. May be left empty.
        config:             The censor config to use.
        cancelled:          If this event is set, the job is canceled and the job is returned as is..
        data:               Extra data, will not be touched by the pipeline.
        success:            Set by the pipeline after processing. None = not yet done.
        error:              Populated on failure; None on success.
        stacktrace:         In case of an error, the stacktrace will be put here.
        time_taken:         The time taken to finish the job.
        result:             The ProcessedResult, set on completion.
    """
    image: ImageInput = None
    job_id: Any = field(default_factory=lambda: str(uuid.uuid4()))
    output_path: Path | str | None = None

    return_bytes: bool = False
    output_extension: str = '.png'
    override_cache: bool = False
    early_exit: bool = False
    skip_cache_write: bool = False
    sizes: list[int] = field(default_factory=lambda: [1280])
    timestamp: float = 0

    cache_base_dir: Path | None = None

    config: CensorConfig = field(default_factory=lambda: DEFAULT_CENSOR_CONFIG)

    data: dict = None

    cancelled: Event = field(default_factory=lambda: Event())

    # Set by the pipeline
    success: bool | None = None
    error: Exception | None = None
    stacktrace: str | None = None
    result: ProcessedResult | None = None
    time_taken: list[tuple[str, int]] | None = None



@dataclass
class ProcessedResult:
    """
            Returned for every job regardless of early_exit.

            When early_exit=False:
                image        — the censored image
                output_path  — where it was saved, or None
                features     — postprocessed features
                bodies       — postprocessed bodies (may be None)

            When early_exit=True:
                image        — the original unmodified image
                output_path  — always None (save was skipped)
                features     — postprocessed features
                bodies       — postprocessed bodies (may be None)
        """
    image: np.ndarray | bytes
    output_path: Path | None
    features: list[list[RawBox]]
    bodies: list[list[Polygon]] | None


@dataclass
class PreprocessedData:
    """Everything a CPU worker produces before handing off to the GPU thread."""
    adj_images: list[np.ndarray]
    scales: list[float]
    # Filled in by GPU thread, then read back by the CPU worker:
    reply_queue: Queue = field(default_factory=Queue)


@dataclass
class GPURequest:
    data: PreprocessedData
    need_features: list[bool]
    need_bodies: list[bool]

@dataclass
class GPUIntermediateResult:
    data: PreprocessedData
    raw_features: list[RawBox] | None
    need_bodies: list[bool]
    error: Exception | None = None

@dataclass
class GPUResult:
    raw_features: Any | None
    raw_bodies: Any | None
    error: Exception | None = None