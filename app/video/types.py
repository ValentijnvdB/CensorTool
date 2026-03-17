from dataclasses import dataclass
from pathlib import Path

from core import Job


VideoInput = Path | str

@dataclass
class VideoJob(Job):
    """
    Submitted by the caller; travels through the whole pipeline unchanged.

    Attributes:
        video:          The video to process - path on disk
        job_id:         Unique identifier. Auto-generated if not supplied.
        avi_path:       Path to save the temporary video file.
        override_cache: Whether to ignore the cached elements for this image
        early_exit:     If True, return after the cache write (step 7) and skip
                        image modification, save, and the full ProcessedResult.
                        The returned ProcessedResult will carry the unmodified
                        original image plus features/bodies instead.
        sizes:          The sizes to run the image at.
        timestamp:      The timestamp this image was taken at. May be left empty.
        data:           Extra data, will not be touched by the pipeline.
        success:        Set by the pipeline after processing. None = not yet done.
        error:          Populated on failure; None on success.
        result:         The ProcessedResult, set on completion.
    """
    video: VideoInput = None
    avi_path: Path | None = None