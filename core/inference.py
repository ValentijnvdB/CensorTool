"""ONNX Runtime sessions, each run on its own worker thread."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import onnxruntime as ort

from .pipeline_config import Device

# One model output: a float32 array per output node.
DetectorOutput = list[np.ndarray]


def _dims(shape: list[Any]) -> list[int]:
    """Tensor shape with dynamic dimensions as -1."""
    return [d if isinstance(d, int) and d > 0 else -1 for d in shape]


class OnnxSession:
    """Owns one ONNX Runtime session. Created in the constructor, so a bad path fails fast."""

    def __init__(self, model_path: str, device: Device, name: str):
        self.name = name
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, device.threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers: list[Any] = ['CPUExecutionProvider']
        if device.gpu:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = [('CUDAExecutionProvider', {'device_id': device.cuda_device_id}), 'CPUExecutionProvider']
                print(f"[{name}] Using CUDA device {device.cuda_device_id}", flush=True)
            else:
                print(f"[{name}] CUDA unavailable, falling back to CPU: onnxruntime has no CUDAExecutionProvider "
                      f"(install onnxruntime-gpu)", flush=True)
        else:
            print(f"[{name}] Using CPU, {max(1, device.threads)} thread(s).", flush=True)

        try:
            self._session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        except Exception as e:
            raise RuntimeError(f"[{name}] failed to load model '{model_path}': {e}") from e

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_fp16 = inp.type == 'tensor(float16)'
        self.input_shape = _dims(inp.shape)
        self.output_shapes = [_dims(o.shape) for o in self._session.get_outputs()]
        self.metadata: dict[str, str] = dict(self._session.get_modelmeta().custom_metadata_map or {})

    def run(self, tensor: np.ndarray) -> DetectorOutput:
        """Runs the model on one input tensor and returns all outputs as float32."""
        if self._input_fp16:
            tensor = tensor.astype(np.float16)
        outputs = self._session.run(None, {self._input_name: tensor})
        return [np.asarray(o, dtype=np.float32) for o in outputs]


class InferenceWorker:
    """
    Runs one model on a dedicated thread. Requests are a batch of inputs (e.g. all face crops of one image);
    serializing inference per model keeps GPU memory use predictable.
    """

    def __init__(self, session: OnnxSession):
        self.session = session
        self._executor: ThreadPoolExecutor | None = None

    def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"infer-{self.session.name}")

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _run_batch(self, inputs: list[np.ndarray]) -> list[DetectorOutput]:
        try:
            return [self.session.run(t) for t in inputs]
        except Exception as e:
            raise RuntimeError(f"[{self.session.name}] {e}") from e

    def submit(self, inputs: list[np.ndarray]) -> Future:
        if self._executor is None:
            raise RuntimeError(f"[{self.session.name}] the inference worker is not running")
        return self._executor.submit(self._run_batch, inputs)
