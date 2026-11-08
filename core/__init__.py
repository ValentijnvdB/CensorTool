"""
Pure-Python censoring backend with the same API as the C++ censor_core 2.0 module.

The app does not import this package directly; it uses ``core.backend.cc``, which is either this package
(backend: python, the default) or the compiled censor_core module (backend: cpp), as set in pipeline.yml.
"""
from .censor_config import load_censor_config
from .pipeline import ImagePipeline, Job, JobHandle, wait_any
from .video import VideoJob, VideoPipeline
from .yolo import YoloModel

__version__ = "2.0.0"

__all__ = [
    "ImagePipeline", "Job", "JobHandle", "VideoPipeline", "VideoJob", "YoloModel",
    "load_censor_config", "wait_any", "__version__",
]
