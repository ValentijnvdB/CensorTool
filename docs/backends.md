# Backends

Censoring is done by one of two backends. Both have the same features and read the same config files.

| Backend | Install | Notes |
|---------|---------|-------|
| `python` | Included. | The default. Pure Python on top of ONNX Runtime and OpenCV. |
| `cpp` | Build and install `censor_core` 2.2 or newer, e.g. `pip install /path/to/PyCensorCore`. Older versions are refused at startup. | Faster, mostly noticeable in live and video censoring and on large batches of files. |

Pick one with `backend:` in [pipeline.yml](pipeline_config.md#general-settings). If `cpp` is selected but
`censor_core` is not installed, Censor Tool stops with an error instead of switching backends.
`python main.py validate` shows which backend is used.

Results of the two can differ by a pixel here and there, not in what is detected or censored. They share the
`cache/` folder but each keeps its own cache files, so switching backends detects everything again once.

## GPU

Both backends can run the models on an NVIDIA GPU. For the Python backend, install `requirements-gpu.txt` instead of
`requirements.txt`; the `cpp` backend needs a `censor_core` built with CUDA. Then set `device: cuda` in
`pipeline.yml`. The CUDA and cuDNN versions must match
[onnxruntime-gpu's requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
Both backends fall back to the CPU when the GPU can't be used; `python main.py validate` reports whether the GPU is
really used. See also [Troubleshooting](troubleshooting.md#the-gpu-isnt-used).
