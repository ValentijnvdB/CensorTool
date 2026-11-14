# Troubleshooting

Start with `python main.py validate` (see [Checking what works](modes/validate.md)): it checks every mode and gives a
hint for each problem it finds.

### The GPU isn't used

Install `requirements-gpu.txt` (not `requirements.txt`; having both `onnxruntime` and `onnxruntime-gpu` installed
causes problems, so `pip uninstall onnxruntime` first) and set `device: cuda` in `pipeline.yml`.
`python main.py validate` reports whether CUDA works. The CUDA and cuDNN versions must match
[onnxruntime-gpu's requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
See also [GPU](backends.md#gpu).

### "Model ... not found"

The model file isn't at the path in `pipeline.yml`, see [Models](models.md). Paths are relative to the working
directory, see [Instances](profiles.md#instances).

### A label in the censor config is reported as unknown

The censor config uses a label none of the models in `pipeline.yml` produce, for example `BODY` without the
segmentation model, or NudeNet labels with Nevermore.

### Videos fail

Check that `ffmpeg` is installed and on your `PATH` (`ffmpeg -version`).

### The virtual camera doesn't work

Install a virtual camera driver, see
[pyvirtualcam](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras). On Linux, load v4l2loopback
(`sudo modprobe v4l2loopback`).

### The HTTPS certificate is missing or expired

Create one, see [HTTPS](modes/http.md#https), or pass `--ssl-cert` and `--ssl-key`.

### Live censoring is slow

Use a GPU, the `cpp` backend, or a smaller model (e.g. `det_v11_medium`). Fewer labels in `features_to_censor` can
also help, as models whose labels aren't needed don't run.
