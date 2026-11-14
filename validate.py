"""
Check which modes work on this machine with a profile: python main.py validate [--profile <name>] [options]

Every check gives PASS, WARN, FAIL or SKIP with a message and a hint on how to fix it. A check whose prerequisites
failed is skipped. At the end every mode gets a verdict. Nothing is written to data/: test files go to a temporary
folder that is removed afterwards.

    --only file,http    check only these modes (the profile, models and pipeline are always checked)
    --quick             skip the slow checks (videos, GIFs and starting the HTTP server)
    --all-profiles      check every profile in configs/, each in its own process

The exit code is 1 when a check failed (optional extras such as the virtual camera, the GPU or ComfyUI excepted),
0 otherwise.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

import constants

MODES = ('file', 'live', 'webcam', 'overlay', 'http')
REQUIRED_IMAGE_EXT = ('.jpg', '.png')  # the other image types are reported as warnings when they don't work


# ----------------------------------------------------------------------------------------------------------------
# Framework
# ----------------------------------------------------------------------------------------------------------------

class Status(Enum):
    PASS = 'PASS'
    WARN = 'WARN'
    FAIL = 'FAIL'
    SKIP = 'SKIP'


OK_STATUSES = (Status.PASS, Status.WARN)


@dataclass
class Result:
    status: Status
    message: str
    hint: str | None = None
    seconds: float = 0.0
    not_needed: bool = False                # a SKIP because the profile does not use it (not 'not tested')


class CheckOutcome(Exception):
    """Raised by a check to end with a status other than PASS."""
    status: Status = Status.FAIL

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint


class Fail(CheckOutcome):
    status = Status.FAIL


class Warn(CheckOutcome):
    status = Status.WARN


class Skip(CheckOutcome):
    """The check could not be done here (e.g. no camera): the mode is reported as not tested."""
    status = Status.SKIP


class NotNeeded(Skip):
    """The profile or the arguments don't use what the check is about (e.g. no ai_remove style)."""


@dataclass
class Check:
    key: str
    title: str
    fn: Callable[['Context'], str | None]
    mode: str | None = None                 # None: always runs (profile, pipeline); otherwise one of MODES
    requires: tuple[str, ...] = ()          # checks that must pass (or warn) first
    timeout: float | None = None            # run in a worker thread and give up after this many seconds
    slow: bool = False                      # skipped with --quick
    optional: bool = False                  # an optional extra: a failure does not make a mode or the exit code fail


@dataclass
class Context:
    """What the checks share."""
    profile: str | None = None
    config_file: str | None = None
    censor_config_file: str | None = None
    pipeline_file: str | None = None
    camera: int = 0
    host: str = 'localhost'
    port: int = 8443
    use_ssl: bool = False
    ssl_cert: str = str(Path('certs') / 'cert.pem')
    ssl_key: str = str(Path('certs') / 'key.pem')

    tmp: Path | None = None
    profile_dir: Path | None = None

    # filled in by the checks
    config_path: Path | None = None
    censor_config_path: Path | None = None
    pipeline_path: Path | None = None
    pipeline_spec: dict[str, Any] = field(default_factory=dict)
    backend_name: str | None = None
    censor_config: dict[str, Any] | None = None
    parsed_censor_config: Any = None
    pipeline: Any = None                    # an ImagePipeline, started
    pipeline_output: str = ''               # what the backend printed while loading the models

    def close_pipeline(self):
        if self.pipeline is not None:
            pipeline, self.pipeline = self.pipeline, None
            with contextlib.suppress(Exception):
                pipeline.stop()


def run_check(check: Check, ctx: Context) -> Result:
    """Run one check; exceptions become a FAIL, a check that takes too long times out."""
    start = time.perf_counter()

    def call() -> Result:
        try:
            message = check.fn(ctx)
            return Result(Status.PASS, message or 'ok')
        except CheckOutcome as e:
            return Result(e.status, e.message, e.hint, not_needed=isinstance(e, NotNeeded))
        except Exception as e:
            logger.opt(exception=True).debug(f"Check '{check.key}' raised:")
            return Result(Status.FAIL, f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)

    if check.timeout is None:
        result = call()
    else:
        box: list[Result] = []
        thread = threading.Thread(target=lambda: box.append(call()), name=f'validate-{check.key}', daemon=True)
        thread.start()
        thread.join(check.timeout)
        if box:
            result = box[0]
        else:
            result = Result(Status.FAIL, f"did not finish within {check.timeout:.0f} s (it hangs)",
                            "Look in data/logs/log.txt, or run the mode itself to see where it gets stuck.")
    result.seconds = time.perf_counter() - start
    return result


def run_checks(checks: list[Check], ctx: Context, modes: tuple[str, ...], quick: bool) -> dict[str, Result]:
    """Run the checks in order, skipping those of other modes and those whose prerequisites did not pass."""
    results: dict[str, Result] = {}
    for check in checks:
        if check.mode is not None and check.mode not in modes:
            continue
        missing = [key for key in check.requires if key in results and results[key].status not in OK_STATUSES]
        not_run = [key for key in check.requires if key not in results]
        if quick and check.slow:
            result = Result(Status.SKIP, "skipped (--quick)")
        elif missing or not_run:
            names = ', '.join(_title(checks, key) for key in missing + not_run)
            result = Result(Status.SKIP, f"needs: {names}", not_needed=any(
                results[key].not_needed for key in check.requires if key in results))
        else:
            logger.info(f"Checking {check.title}...")
            result = run_check(check, ctx)
        results[check.key] = result
        _log_result(check, result)
    return results


STATUS_COLORS = {Status.PASS: 'green', Status.WARN: 'yellow', Status.FAIL: 'red', Status.SKIP: 'dim'}


def _esc(text: Any) -> str:
    """Escape text for a loguru message with colors=True (a '<' would start a color tag)."""
    return str(text).replace('<', r'\<')


def _colored(status: Status, text: str | None = None) -> str:
    """Text (by default the status name) in the color of the status, as loguru markup."""
    color = STATUS_COLORS[status]
    return f"<{color}><bold>{_esc(status.value if text is None else text)}</bold></{color}>"


def _log(level: str, message: str):
    """Log a message with loguru color markup (escape other text with _esc). The colors are left out in the file."""
    logger.opt(colors=True, depth=1).log(level, message)


def _title(checks: list[Check], key: str) -> str:
    return next((c.title for c in checks if c.key == key), key)


def _log_result(check: Check, result: Result):
    level = {Status.PASS: 'INFO', Status.WARN: 'WARNING', Status.SKIP: 'INFO',
             Status.FAIL: 'WARNING' if check.optional else 'ERROR'}[result.status]
    _log(level, f"[{_colored(result.status)}] {_esc(check.title)}: {_esc(result.message)}")
    if result.hint and result.status != Status.PASS:
        _log(level, f"       hint: {_esc(result.hint)}")


# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------

def test_image(width: int = 320, height: int = 240) -> np.ndarray:
    """A deterministic BGR test image with some structure (a gradient and shapes), so encoders have work to do."""
    import cv2
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    image = np.dstack([np.tile(x, (height, 1)), np.tile(y[:, None], (1, width)),
                       np.full((height, width), 128, np.uint8)])
    cv2.circle(image, (width // 3, height // 2), height // 5, (40, 200, 240), -1)
    cv2.rectangle(image, (width // 2, height // 4), (width * 5 // 6, height * 3 // 4), (200, 60, 60), -1)
    return np.ascontiguousarray(image)


@contextlib.contextmanager
def capture_fd_output(stream=None):
    """
    Capture what is written to stdout (or another stream) at the file descriptor level, also by C++ code
    (censor_core) and other threads. Yields a list that holds the captured text after the block.
    """
    stream = stream or sys.stdout
    captured: list[str] = []
    stream.flush()
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        yield captured  # no real stdout (e.g. an IDE console): nothing to capture
        return
    saved = os.dup(fd)
    with tempfile.TemporaryFile(mode='w+b') as tmp:
        os.dup2(tmp.fileno(), fd)
        try:
            yield captured
        finally:
            stream.flush()
            os.dup2(saved, fd)
            os.close(saved)
            tmp.seek(0)
            captured.append(tmp.read().decode(errors='replace'))


def requested_devices(spec: dict[str, Any]) -> dict[str, str]:
    """The device of every model in a pipeline spec (a model's own device, else the top-level one, else cpu)."""
    default = str(spec.get('device', 'cpu'))
    entries = list(spec.get('models') or [])
    if spec.get('landmarks'):
        entries.append({'name': 'landmarks', **spec['landmarks']})
    return {str(m.get('name', Path(str(m.get('path', '?'))).stem)): str(m.get('device', default)).lower()
            for m in entries}


def is_gpu_device(device: str) -> bool:
    return device in ('gpu', 'cuda') or device.startswith('cuda:')


def module_hint(package: str, extra: str = '') -> str:
    return f"Install it: pip install {package}{extra}"


def _require_import(module: str, package: str | None = None, hint: str | None = None):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise Fail(f"'{module}' can't be imported: {e}", hint or module_hint(package or module)) from None


def _wait_job(job_handle, what: str):
    job = job_handle.wait()
    if not job.success:
        raise Fail(f"{what} failed: {job.error}")
    return job


# ----------------------------------------------------------------------------------------------------------------
# Checks: profile and pipeline (all modes)
# ----------------------------------------------------------------------------------------------------------------

def check_python(ctx: Context) -> str:
    version = sys.version_info
    text = f"Python {version.major}.{version.minor}.{version.micro}"
    if version < (3, 12):
        raise Warn(f"{text}; CensorTool is made for Python 3.12 or newer", "Use Python 3.12 or newer.")
    return text


def check_packages(ctx: Context) -> str:
    """The packages every mode needs. The ones of a single mode are checked with that mode."""
    import importlib
    needed = {'numpy': 'numpy', 'cv2': 'opencv-python', 'yaml': 'PyYAML', 'loguru': 'loguru',
              'onnxruntime': 'onnxruntime', 'shapely': 'Shapely', 'tqdm': 'tqdm', 'requests': 'requests'}
    missing = []
    for module, package in needed.items():
        try:
            importlib.import_module(module)
        except ImportError as e:
            missing.append(f"{package} ({e})")
    if missing:
        raise Fail(f"missing: {', '.join(missing)}", "Install the requirements: pip install -r requirements.txt")
    import cv2
    return f"all {len(needed)} present (OpenCV {cv2.__version__})"


def check_profile(ctx: Context) -> str:
    from app.config import ProfileError, resolve_profile_dir, resolve_profile_files
    try:
        profile_dir = resolve_profile_dir(ctx.profile)
        ctx.config_path, ctx.censor_config_path, ctx.pipeline_path = resolve_profile_files(
            profile_dir, ctx.config_file, ctx.censor_config_file, ctx.pipeline_file)
    except ProfileError as e:
        raise Fail(str(e), "Check --profile, or create the profile in configs/ (see README, Profiles).") from None
    ctx.profile_dir = profile_dir
    def short(path: Path) -> str:
        path = Path(path).absolute()
        return str(path.relative_to(constants.instance_dir)) if path.is_relative_to(constants.instance_dir) \
            else str(path)
    return f"'{profile_dir.name}': {', '.join(short(p) for p in (ctx.config_path, ctx.censor_config_path, ctx.pipeline_path))}"


def check_config(ctx: Context) -> str:
    from app.config import load_config
    try:
        config = load_config(ctx.config_path)
    except (ValueError, OSError) as e:
        raise Fail(str(e), "Fix config.yml; all settings are explained in docs/config.md.") from None
    except Exception as e:  # e.g. a YAML syntax error
        raise Fail(f"{ctx.config_path} can't be read: {e}", "Fix the YAML syntax of config.yml.") from None
    live = config.live
    return (f"live region {live.cap_width}x{live.cap_height} at ({live.cap_left}, {live.cap_top}), "
            f"video_censor_fps {config.video_censor_fps}")


def check_pipeline_config(ctx: Context) -> str:
    from app.config import load_pipeline_runtime, load_pipeline_spec
    try:
        n_workers, max_jobs, ctx.backend_name = load_pipeline_runtime(ctx.pipeline_path)
        ctx.pipeline_spec = load_pipeline_spec(ctx.pipeline_path)
    except ValueError as e:
        raise Fail(str(e), "Fix pipeline.yml; all settings are explained in docs/pipeline_config.md.") from None
    except Exception as e:
        raise Fail(f"{ctx.pipeline_path} can't be read: {e}", "Fix the YAML syntax of pipeline.yml.") from None
    models = ctx.pipeline_spec.get('models') or []
    if not models:
        raise Fail(f"{ctx.pipeline_path} lists no models", "Add at least one model under 'models:'.")
    devices = sorted(set(requested_devices(ctx.pipeline_spec).values()))
    return (f"backend '{ctx.backend_name}', {len(models)} model(s) on {', '.join(devices)}, "
            f"n_workers {n_workers}, max_concurrent_jobs {max_jobs}")


def check_models(ctx: Context) -> str:
    entries = list(ctx.pipeline_spec.get('models') or [])
    if ctx.pipeline_spec.get('landmarks'):
        entries.append({'name': 'landmarks', **ctx.pipeline_spec['landmarks']})
    missing, found = [], []
    for entry in entries:
        name = entry.get('name', '?')
        if 'path' not in entry:
            missing.append(f"'{name}' has no path")
        elif not Path(entry['path']).is_file():
            missing.append(f"'{name}' at {entry['path']}")
        else:
            found.append(f"{name} ({Path(entry['path']).stat().st_size / 2 ** 20:.0f} MB)")
    if missing:
        raise Fail(f"model files not found: {'; '.join(missing)}",
                   "Download the models (see README, Models). Relative paths are relative to the working directory "
                   f"({constants.instance_dir}).")
    return ', '.join(found)


def check_backend(ctx: Context) -> str:
    """Select the backend and load the profile, as the modes do (init_app)."""
    import importlib.metadata
    from app.config import ProfileError, init_configs
    from core import backend
    try:
        init_configs(profile=ctx.profile_dir, config_file=ctx.config_file, default_censor_config=ctx.censor_config_file,
                     pipeline_file=ctx.pipeline_file)
    except ProfileError as e:
        hint = None
        if ctx.backend_name == 'cpp':
            hint = "Install censor_core (pip install /path/to/PyCensorCore), or set 'backend: python' in pipeline.yml."
        raise Fail(str(e), hint) from None
    other = 'cpp' if ctx.backend_name == 'python' else 'python'
    text = f"'{backend.name()}'"
    if backend.name() == 'cpp':
        text += f" (censor_core {importlib.metadata.version('censor_core')})"
    else:
        import onnxruntime
        text += f" (onnxruntime {onnxruntime.__version__})"
    return text + f"; '{other}' is {'also installed' if backend.is_available(other) else 'not installed'}"


def check_censor_config(ctx: Context) -> str:
    from app.config import CensorConfigError, read_censor_config_from_file
    from core.censor_config import censor_config_from_dict
    from core.config_utils import collect_warnings
    try:
        config = read_censor_config_from_file(ctx.censor_config_path)
    except CensorConfigError as e:
        raise Fail(str(e), "Stickers must be inside data/stickers.") from None
    except Exception as e:
        raise Fail(f"{ctx.censor_config_path} can't be read: {e}", "Fix the YAML syntax of censor_config.yml.") \
            from None
    with collect_warnings() as warnings:
        try:
            ctx.parsed_censor_config = censor_config_from_dict(config)
        except Exception as e:
            raise Fail(f"invalid censor config: {e}", "See docs/censor_config.md.") from None
    ctx.censor_config = config
    if warnings:
        raise Warn('; '.join(warnings), "See docs/censor_config.md.")
    return f"{len(ctx.parsed_censor_config.features_to_censor)} feature(s) to censor"


def check_writable(ctx: Context) -> str:
    """The folders the modes write to. Only a test file is created and removed; nothing in them is read."""
    folders = {'cache': constants.cache_root, 'logs': constants.log_path, 'temp': constants.temp_path,
               'output': constants.censored_path}
    problems = []
    for name, folder in folders.items():
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder, prefix='.validate-'):
                pass
        except OSError as e:
            problems.append(f"{name} ({folder}): {e}")
    if problems:
        raise Fail(f"can't write to: {'; '.join(problems)}",
                   "Check the permissions, and run CensorTool from the folder that holds data/ and configs/.")
    return ', '.join(folders)


def check_pipeline(ctx: Context) -> str:
    """Load the models as the modes do and run a detection on a test image."""
    from app.config import make_image_pipeline
    from core.backend import cc

    with capture_fd_output() as output:
        pipeline = make_image_pipeline(mode='p')
        pipeline.start()
        ctx.pipeline = pipeline
        job = _wait_job(pipeline.submit(cc.Job(image=test_image(), use_cache=False)), "detection")
    ctx.pipeline_output = ''.join(output)
    for line in ctx.pipeline_output.splitlines():
        logger.debug(f"backend: {line}")

    if not isinstance(job.result, dict) or 'detections' not in job.result:
        raise Fail(f"the detection gave no detections list: {job.result!r}")
    labels = sorted(pipeline.labels)
    return f"models {pipeline.models} loaded, {len(labels)} label(s): {', '.join(labels)}"


def check_censor_labels(ctx: Context) -> str:
    """The censor config against the labels of the loaded models (as /censor_config/validate does)."""
    from server.helpers import pipeline_errors
    warnings: list[str] = []
    errors = pipeline_errors(ctx.parsed_censor_config, warnings, ctx.pipeline)
    if errors:
        raise Fail('; '.join(errors), "Fix the labels in censor_config.yml; GET /profile lists the labels.")
    if warnings:
        raise Warn('; '.join(warnings), "Use a label the models produce (they are listed under 'Pipeline').")
    return f"all {len(ctx.parsed_censor_config.features_to_censor)} label(s) are produced by the models"


def check_censor(ctx: Context) -> str:
    from core.backend import cc
    image = test_image()
    job = _wait_job(ctx.pipeline.submit(cc.Job(image=image, config=ctx.censor_config, use_cache=False)),
                    "censoring")
    out = job.result.get('image') if isinstance(job.result, dict) else None
    if not isinstance(out, np.ndarray):
        raise Fail(f"censoring returned no image: {type(out).__name__}")
    if out.shape != image.shape or out.dtype != np.uint8:
        raise Fail(f"censoring returned a {out.shape} {out.dtype} image for a {image.shape} uint8 one")
    return f"censored a {image.shape[1]}x{image.shape[0]} test image"


def check_gpu(ctx: Context) -> str:
    """
    With a GPU in pipeline.yml: whether the models really run on it (both backends fall back to the CPU and only
    print that). Without one: whether CUDA would work, with onnxruntime.
    """
    devices = requested_devices(ctx.pipeline_spec)
    gpu_models = [name for name, device in devices.items() if is_gpu_device(device)]
    if gpu_models:
        if ctx.pipeline is None:
            raise Skip("the pipeline did not load")
        fallback = [line.strip() for line in ctx.pipeline_output.splitlines() if 'falling back to CPU' in line]
        if fallback:
            raise Fail(f"pipeline.yml asks for a GPU, but the models run on the CPU: {fallback[0]}",
                       "Install a CUDA build (onnxruntime-gpu, or censor_core with CUDA) with matching CUDA and "
                       "cuDNN versions (README, GPU), or set 'device: cpu'.")
        used = [line.strip() for line in ctx.pipeline_output.splitlines() if 'Using CUDA' in line]
        if not used:
            raise Warn(f"pipeline.yml asks for a GPU for {', '.join(gpu_models)}, but it can't be told whether "
                       f"it is used (the backend printed nothing about it)")
        return f"{', '.join(gpu_models)} run on the GPU"

    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        raise NotNeeded(f"no GPU in pipeline.yml, and onnxruntime has no CUDA support (providers: {', '.join(providers)})",
                   "Optional: install onnxruntime-gpu to use an Nvidia GPU.")
    model = min((m for m in ctx.pipeline_spec.get('models') or [] if Path(m.get('path', '')).is_file()),
                key=lambda m: Path(m['path']).stat().st_size, default=None)
    if model is None:
        raise NotNeeded("no model file to try")
    with capture_fd_output():
        try:
            session = ort.InferenceSession(model['path'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            active = session.get_providers()
        except Exception as e:
            raise Warn(f"CUDA is installed but loading '{model.get('name')}' on it failed: {e}") from None
    if 'CUDAExecutionProvider' not in active:
        raise Warn("onnxruntime lists CUDA, but a model can't use it (CUDA or cuDNN libraries missing?)",
                   "The CUDA and cuDNN versions must match onnxruntime-gpu (README, GPU).")
    return "no GPU in pipeline.yml, but CUDA works: set 'device: cuda' in pipeline.yml to use it"


def check_comfy(ctx: Context) -> str:
    from app.config import _censor_styles
    styles = [s for s in _censor_styles(ctx.censor_config) if s.get('type') == 'ai_remove']
    if not styles:
        raise NotNeeded("the censor config does not use ai_remove")
    import requests
    problems, seen = [], set()
    for style in styles:
        host, port, workflow = str(style['comfy_host']), style['comfy_port'], Path(str(style['comfy_workflow']))
        if not workflow.is_file():
            problems.append(f"workflow {workflow} not found")
        if (host, port) in seen:
            continue
        seen.add((host, port))
        base = host if '://' in host else f"http://{host}"
        url = f"{base.rstrip('/')}:{port}/system_stats"
        try:
            requests.get(url, timeout=5).raise_for_status()
        except Exception as e:
            problems.append(f"ComfyUI at {base}:{port} does not answer ({type(e).__name__})")
    if problems:
        raise Fail('; '.join(problems), "Start ComfyUI and check comfy_host, comfy_port and comfy_workflow in "
                                        "config.yml; without it ai_remove can't censor.")
    return f"ComfyUI answers at {', '.join(f'{h}:{p}' for h, p in seen)}"


def check_stickers(ctx: Context) -> str:
    parsed = ctx.parsed_censor_config
    overlays = {label: box.overlay for label, box in parsed.features_to_censor.items() if box.overlay is not None}
    if not parsed.enable_overlays or not overlays:
        raise NotNeeded("the censor config uses no sticker overlays")
    from core.boxes import _sticker_files
    empty = [label for label, overlay in overlays.items()
             if not _sticker_files(Path(parsed.stickers_dir), overlay.stickers)]
    if empty:
        raise Fail(f"no sticker images for: {', '.join(empty)} (in {parsed.stickers_dir})",
                   "Put sticker images (png/jpg/webp) in data/stickers, or fix the overlay 'stickers' entries.")
    return f"{len(overlays)} overlay(s) have stickers"


# ----------------------------------------------------------------------------------------------------------------
# Checks: file mode
# ----------------------------------------------------------------------------------------------------------------

def check_image_files(ctx: Context) -> str:
    """Censor an image file of every supported type, from file to file, as file mode does."""
    import cv2
    from core.backend import cc
    image = test_image()
    folder = ctx.tmp / 'images'
    (folder / 'out').mkdir(parents=True, exist_ok=True)
    broken: dict[str, str] = {}
    pending: dict[str, Path] = {}
    for ext in constants.IMAGE_EXT:
        source, target = folder / f'test{ext}', folder / 'out' / f'test{ext}'
        try:
            if not cv2.imwrite(str(source), image):
                raise RuntimeError("OpenCV can't write this type")
            job = ctx.pipeline.submit(cc.Job(image=str(source), output_path=str(target), config=ctx.censor_config,
                                             use_cache=False)).wait()
            if not job.success:
                raise RuntimeError(job.error)
            pending[ext] = target
        except Exception as e:
            broken[ext] = str(e) or type(e).__name__

    # The pipeline writes the files in the background (a failed write is only printed), so wait for them.
    deadline = time.monotonic() + 15
    while pending and time.monotonic() < deadline:
        for ext, target in list(pending.items()):
            if target.is_file():
                out = cv2.imread(str(target))
                if out is not None and out.shape == image.shape:
                    del pending[ext]
        if pending:
            time.sleep(0.1)
    for ext, target in pending.items():
        broken[ext] = ("the censored file was not written" if not target.is_file()
                       else "the censored file can't be read back")

    required = [ext for ext in broken if ext in REQUIRED_IMAGE_EXT]
    details = '; '.join(f"{ext}: {error}" for ext, error in broken.items())
    if required:
        raise Fail(details)
    if broken:
        from core import backend
        builder = "censor_core's OpenCV" if backend.name() == 'cpp' else "Your OpenCV"
        raise Warn(f"these image types don't work: {details}",
                   f"{builder} can't read or write them; the other types work.")
    return ', '.join(constants.IMAGE_EXT)


def check_ffmpeg(ctx: Context) -> str:
    versions = []
    for tool in ('ffmpeg', 'ffprobe'):
        path = shutil.which(tool)
        if path is None:
            raise Fail(f"{tool} is not on PATH", "Install ffmpeg (it includes ffprobe), e.g. sudo apt install ffmpeg.")
        out = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=10)
        versions.append(out.stdout.split('\n', 1)[0].split(' Copyright')[0] if out.returncode == 0 else tool)
    return '; '.join(versions)


def _make_test_video(ctx: Context) -> Path:
    """A 1 s 320x240 video at 15 fps with a sound track (testsrc + sine), made by ffmpeg."""
    path = ctx.tmp / 'videos' / 'test.mp4'
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=15',
            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100', '-t', '1', '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(path)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:  # e.g. an ffmpeg without libx264: let it pick the codecs
        args = [a for a in args if a not in ('-c:v', 'libx264', '-c:a', 'aac')]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise Fail(f"ffmpeg can't make a test video: {proc.stderr.strip()[-300:]}")
    return path


def _censor_test_video(ctx: Context, output: Path) -> Any:
    from app.config import CONFIG
    from core.backend import cc
    source = _make_test_video(ctx)
    video_pipeline = cc.VideoPipeline(ctx.pipeline)  # shares the loaded models, as the HTTP server does
    with capture_fd_output(), capture_fd_output(sys.stderr):  # no progress output
        job = video_pipeline.submit(cc.VideoJob(
            job_id='validate', video_path=str(source), avi_path=str(output.with_suffix('.avi')),
            output_path=str(output), use_cache=False, config=ctx.censor_config,
            censor_fps=CONFIG.video_censor_fps))
    if not job.success:
        raise Fail(f"censoring the test video failed: {job.error}")
    if not output.is_file() or output.stat().st_size == 0:
        raise Fail(f"no output file was written ({output.name})")
    return job


def _count_frames(path: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = 0
    while cap.isOpened():
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    cap.release()
    return frames


def check_video_file(ctx: Context) -> str:
    output = ctx.tmp / 'videos' / 'out' / 'test.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    _censor_test_video(ctx, output)
    frames = _count_frames(output)
    if frames == 0:
        raise Fail("the censored video has no readable frames")
    from core.video import file_has_audio
    if not file_has_audio(str(output)):
        raise Warn(f"censored {frames} frames, but the sound track was lost",
                   "Check that your ffmpeg can encode audio (aac).")
    return f"censored a 1 s test video with sound ({frames} frames)"


def check_gif_file(ctx: Context) -> str:
    output = ctx.tmp / 'videos' / 'out' / 'test.gif'
    output.parent.mkdir(parents=True, exist_ok=True)
    _censor_test_video(ctx, output)
    frames = _count_frames(output)
    if frames == 0:
        raise Fail("the censored GIF has no readable frames")
    return f"censored a test video to a GIF ({frames} frames)"


# ----------------------------------------------------------------------------------------------------------------
# Checks: live, webcam and the virtual camera
# ----------------------------------------------------------------------------------------------------------------

def _display_problem() -> str | None:
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        return "no display (DISPLAY and WAYLAND_DISPLAY are not set)"
    return None


def check_screen_capture(ctx: Context) -> str:
    """Capture the live region with the device live mode uses, and the mouse position it draws the cursor at."""
    _require_import('mss')
    problem = _display_problem()
    if problem:
        raise Fail(problem, "Live mode needs a desktop session.")
    from app.live.input_devices.screenshot_device import ScreenshotDevice
    try:
        device = ScreenshotDevice()
    except ValueError as e:
        raise Fail(str(e), "Set live_cap_top/left/width/height in config.yml to a region on your screens.") from None
    try:
        _, frame = device.get_frame()
        region = device.screen_region()
    finally:
        device.close()

    try:
        import pyautogui
        pyautogui.position()
    except Exception as e:
        raise Fail(f"reading the mouse position failed: {e}", module_hint('pyautogui')) from None

    text = f"captured {frame.shape[1]}x{frame.shape[0]} at ({region[0]}, {region[1]})"
    if not frame.any():
        wayland = os.environ.get('XDG_SESSION_TYPE') == 'wayland' or os.environ.get('WAYLAND_DISPLAY')
        raise Warn(f"{text}, but it is completely black",
                   "On Wayland screen capture does not work; log in with an X11 session." if wayland else
                   "Maybe the region shows a black screen; otherwise capturing is not allowed.")
    return text


def check_live_window(ctx: Context) -> str:
    """Whether OpenCV can show the censored window (without opening one)."""
    import cv2
    gui = [line.split(':', 1)[1].strip() for line in cv2.getBuildInformation().splitlines()
           if line.strip().startswith('GUI:')]
    if gui and gui[0].upper() == 'NONE':
        raise Fail("this OpenCV has no window support (opencv-python-headless?)",
                   "pip uninstall opencv-python-headless, then pip install opencv-python. Or use --vcam.")
    problem = _display_problem()
    if problem:
        raise Fail(problem, "The censored window needs a desktop session. Or use --vcam.")
    return f"OpenCV window support: {gui[0] if gui else 'unknown'}"


def check_webcam(ctx: Context) -> str:
    import cv2
    if sys.platform.startswith('linux') and not list(Path('/dev').glob('video*')):
        raise Skip("no camera found (no /dev/video*)", "Connect a webcam to use webcam mode.")
    from app.live.input_devices.camera_device import CameraDevice
    device = CameraDevice(ctx.camera)
    try:
        if not device.device.isOpened():
            raise Fail(f"camera {ctx.camera} can't be opened",
                       "Pick another camera with --device <id>, and close other programs that use it.")
        _, frame = device.get_frame()
        fps = device.device.get(cv2.CAP_PROP_FPS)
    finally:
        device.close()
    return f"camera {ctx.camera}: {frame.shape[1]}x{frame.shape[0]}" + (f" at {fps:.0f} fps" if fps else "")


def check_vcam(ctx: Context) -> str:
    _require_import('pyvirtualcam')
    from app.live.output_devices.vcam_device import VcamDevice
    try:
        device = VcamDevice(320, 240, 15)
    except Exception as e:
        hint = ("Load v4l2loopback: sudo modprobe v4l2loopback devices=1 exclusive_caps=1"
                if sys.platform.startswith('linux') else "Install OBS (its virtual camera is used).")
        raise Fail(f"no virtual camera: {e}", hint + " Only needed with --vcam.") from None
    try:
        device.push_frame(test_image())
        name = getattr(device.device, 'device', '?')
    finally:
        device.close()
    return f"sent a frame to {name}"


# ----------------------------------------------------------------------------------------------------------------
# Checks: overlay mode
# ----------------------------------------------------------------------------------------------------------------

def check_overlay_session(ctx: Context) -> str:
    _require_import('PySide6', hint="Install the overlay requirements: pip install -r requirements-overlay.txt")
    from app.live import windowing
    problem = windowing.session_problem()
    if problem:
        raise Fail(problem)
    return f"window system: {windowing.get().name}"


def check_overlay_capture(ctx: Context) -> str:
    """List the windows and capture one (the focused window, else the top one), as overlay mode does."""
    from app.live import windowing
    ws = windowing.get()
    display = ws.open_display()
    try:
        windows = display.window_list(stacking=True)
        if not windows:
            raise Skip("no windows are open to try capturing")
        focus = ws.focus_tracker()
        try:
            time.sleep(0.5)
            target = focus.active
        finally:
            focus.stop()
        if target is None or display.visible_area(target) is None:
            shown = [w for w in windows if display.visible_area(w) is not None]
            if not shown:
                raise Skip(f"none of the {len(windows)} windows is shown")
            target = shown[-1]
        title = display.title(target)[:40]
    finally:
        display.close()

    grabber = ws.grabber(target)
    try:
        frame = None
        for _ in range(10):
            grabber.refresh()
            image = grabber.grab()
            if image is not None:
                frame = image
                break
            time.sleep(0.05)
    finally:
        grabber.close()
    if frame is None:
        raise Fail(f"capturing window {title!r} gave no image",
                   "Run python -m app.live.windowing.selfcheck for details.")
    text = f"{len(windows)} windows; captured {title!r} ({frame.shape[1]}x{frame.shape[0]})"
    if not frame[..., :3].any():
        raise Warn(f"{text}, but the capture is black",
                   "Run python -m app.live.windowing.selfcheck --pick to try another window.")
    return text


# ----------------------------------------------------------------------------------------------------------------
# Checks: HTTP server
# ----------------------------------------------------------------------------------------------------------------

def check_http_packages(ctx: Context) -> str:
    aiohttp = _require_import('aiohttp')
    _require_import('aiohttp_cors', 'aiohttp_cors')
    return f"aiohttp {aiohttp.__version__}"


def check_http_port(ctx: Context) -> str:
    try:
        infos = socket.getaddrinfo(ctx.host, ctx.port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise Fail(f"host '{ctx.host}' can't be resolved: {e}", "Check --host.") from None
    family, _, _, _, address = infos[0]
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(address)
        except OSError as e:
            raise Fail(f"{ctx.host}:{ctx.port} can't be used: {e}",
                       "Is CensorTool (or another program) already running on it? Pick another with --port.") \
                from None
    return f"{ctx.host}:{ctx.port} is free"


def check_https_cert(ctx: Context) -> str:
    if not ctx.use_ssl:
        raise NotNeeded("--ssl not given")
    for name, path in (('certificate', ctx.ssl_cert), ('key', ctx.ssl_key)):
        if not Path(path).is_file():
            raise Fail(f"{name} not found at {path}",
                       "Create one (README, HTTPS), or pass --ssl-cert and --ssl-key.")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    try:
        context.load_cert_chain(ctx.ssl_cert, ctx.ssl_key)
    except (ssl.SSLError, OSError) as e:
        raise Fail(f"the certificate and key can't be loaded: {e}", "Do they belong together?") from None
    try:
        not_after = ssl._ssl._test_decode_cert(ctx.ssl_cert)['notAfter']  # private, but stable for years
        expires = ssl.cert_time_to_seconds(not_after)
    except Exception:
        return "certificate and key load"
    if expires < time.time():
        raise Warn(f"the certificate expired on {not_after}", "Create a new one (README, HTTPS).")
    return f"certificate and key load (valid until {not_after})"


def check_http_server(ctx: Context) -> str:
    """Start the server as http mode does (on a free port on 127.0.0.1) and send it real requests."""
    import asyncio
    import base64

    import cv2

    if ctx.use_ssl and not all(Path(p).is_file() for p in (ctx.ssl_cert, ctx.ssl_key)):
        raise Skip("the HTTPS certificate or key is missing (see 'HTTPS certificate')")
    # The server loads the models itself: free the ones of the earlier checks first.
    ctx.close_pipeline()

    from server import helpers, http_server

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    scheme = 'https' if ctx.use_ssl else 'http'
    base = f"{scheme}://127.0.0.1:{port}"
    image = test_image()
    ok, png = cv2.imencode('.png', image)
    assert ok

    async def exercise() -> str:
        import aiohttp
        from aiohttp import web
        app = await http_server.init_app()
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            ssl_context = None
            if ctx.use_ssl:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(ctx.ssl_cert, ctx.ssl_key)
            await web.TCPSite(runner, host='127.0.0.1', port=port, ssl_context=ssl_context).start()

            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async def call(method: str, path: str, **kwargs) -> dict:
                    async with session.request(method, base + path, ssl=False, **kwargs) as response:
                        body = await response.json(content_type=None)
                        if response.status != 200:
                            raise Fail(f"{method} {path} answered {response.status}: {body}")
                        return body

                health = await call('GET', '/health')
                if not health.get('pipeline_running'):
                    raise Fail(f"/health says the pipeline does not run: {health}")
                await call('GET', '/profile')
                answer = await call('POST', '/censor_image', json={
                    'type': 'base64', 'image_data': base64.b64encode(png.tobytes()).decode(),
                    'mime_type': 'image/png', 'expected_response': 'base64'})
                data = base64.b64decode(answer['image_data'])
                censored = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if censored is None or censored.shape != image.shape:
                    raise Fail(f"/censor_image returned an unreadable or wrongly sized image ({answer.get('width')}x"
                               f"{answer.get('height')})")
                return f"{scheme} server started; /health, /profile and /censor_image answered " \
                       f"(backend {health.get('backend')}, {answer.get('processing_ms')} ms)"
        finally:
            await runner.cleanup()

    with capture_fd_output():
        helpers.start_pipeline(debug=False)
    try:
        return asyncio.run(exercise())
    finally:
        helpers.stop_pipeline()


# ----------------------------------------------------------------------------------------------------------------
# The checks, the modes and the report
# ----------------------------------------------------------------------------------------------------------------

CORE = ('profile', 'config', 'pipeline_config', 'models', 'backend', 'censor_config', 'pipeline', 'censor')

CHECKS: list[Check] = [
    # every mode
    Check('python', 'Python', check_python),
    Check('packages', 'Packages', check_packages),
    Check('profile', 'Profile', check_profile),
    Check('config', 'config.yml', check_config, requires=('profile',)),
    Check('pipeline_config', 'pipeline.yml', check_pipeline_config, requires=('profile',)),
    Check('models', 'Model files', check_models, requires=('pipeline_config',)),
    Check('backend', 'Backend', check_backend, requires=('packages', 'config', 'pipeline_config')),
    Check('censor_config', 'censor_config.yml', check_censor_config, requires=('backend',)),
    Check('writable', 'Writable folders', check_writable),
    Check('pipeline', 'Pipeline (load models, detect)', check_pipeline, requires=('backend', 'models')),
    Check('censor_labels', 'Censor config labels', check_censor_labels, requires=('pipeline', 'censor_config')),
    Check('censor', 'Censoring', check_censor, requires=('pipeline', 'censor_config')),
    Check('stickers', 'Stickers', check_stickers, requires=('censor_config',)),
    Check('comfy', 'ComfyUI (ai_remove)', check_comfy, requires=('censor_config',), timeout=30),
    Check('gpu', 'GPU (CUDA)', check_gpu, requires=('pipeline_config',), optional=True, timeout=120),
    # file
    Check('image_files', 'Image files', check_image_files, mode='file', requires=('censor',)),
    Check('ffmpeg', 'ffmpeg', check_ffmpeg, mode='file', timeout=30),
    Check('video_file', 'Video file', check_video_file, mode='file', requires=('censor', 'ffmpeg'),
          timeout=300, slow=True),
    Check('gif_file', 'GIF file', check_gif_file, mode='file', requires=('censor', 'ffmpeg'),
          timeout=300, slow=True),
    # live and webcam
    Check('screen_capture', 'Screen capture', check_screen_capture, mode='live', requires=('backend',),
          timeout=20),
    Check('live_window', 'Censored window', check_live_window, mode='live'),
    Check('webcam', 'Webcam', check_webcam, mode='webcam', timeout=20),
    Check('webcam_window', 'Censored window (webcam)', check_live_window, mode='webcam'),
    Check('vcam', 'Virtual camera (--vcam)', check_vcam, mode='vcam', optional=True, timeout=20),
    # overlay
    Check('overlay_session', 'Overlay session', check_overlay_session, mode='overlay', timeout=20),
    Check('overlay_capture', 'Window capture', check_overlay_capture, mode='overlay',
          requires=('overlay_session',), timeout=30),
    # http
    Check('http_packages', 'Server packages', check_http_packages, mode='http'),
    Check('http_port', 'Server port', check_http_port, mode='http'),
    Check('https_cert', 'HTTPS certificate', check_https_cert, mode='http'),
    Check('http_server', 'Server requests', check_http_server, mode='http',
          requires=('http_packages', 'censor'), timeout=300, slow=True),
]

# what each mode needs; optional extras are reported but do not make a mode fail
MODE_CHECKS: dict[str, tuple[str, ...]] = {
    'File: images': CORE + ('censor_labels', 'stickers', 'comfy', 'writable', 'image_files'),
    'File: videos': CORE + ('censor_labels', 'stickers', 'comfy', 'writable', 'ffmpeg', 'video_file'),
    'File: GIFs': CORE + ('censor_labels', 'stickers', 'comfy', 'writable', 'ffmpeg', 'gif_file'),
    'Live (screen)': CORE + ('censor_labels', 'stickers', 'screen_capture', 'live_window'),
    'Webcam': CORE + ('censor_labels', 'stickers', 'webcam', 'webcam_window'),
    'Overlay': CORE + ('censor_labels', 'stickers', 'overlay_session', 'overlay_capture'),
    'HTTP server': CORE + ('censor_labels', 'stickers', 'comfy', 'writable', 'http_packages', 'http_port',
                           'https_cert', 'http_server'),
}
MODE_OF_VERDICT = {'File: images': 'file', 'File: videos': 'file', 'File: GIFs': 'file', 'Live (screen)': 'live',
                   'Webcam': 'webcam', 'Overlay': 'overlay', 'HTTP server': 'http'}
EXTRAS = {'gpu': 'GPU', 'vcam': 'Virtual camera (--vcam)'}


def mode_verdict(keys: tuple[str, ...], results: dict[str, Result], checks: list[Check]) -> tuple[Status, str]:
    """(status, text) for a mode from the results of the checks it needs."""
    ran = [(key, results[key]) for key in keys if key in results]
    failed = [(key, r) for key, r in ran if r.status == Status.FAIL]
    if failed:
        key, result = failed[0]
        return Status.FAIL, f"NOT WORKING: {_title(checks, key)}: {result.message}"
    # a check skipped because something it needs failed is covered above; other skips mean "not tested"
    untested = [(key, r) for key, r in ran if r.status == Status.SKIP and not r.not_needed]
    if untested:
        key, result = untested[0]
        return Status.SKIP, f"not tested: {_title(checks, key)}: {result.message}"
    warned = [_title(checks, key) for key, r in ran if r.status == Status.WARN]
    if warned:
        return Status.WARN, f"should work, with warnings ({', '.join(warned)})"
    return Status.PASS, "should work"


def report(results: dict[str, Result], checks: list[Check], modes: tuple[str, ...], profile: str | None) -> bool:
    """Log the overview. Returns whether everything required passed."""
    line = '=' * 78
    logger.info(line)
    logger.info(f"Validation of profile '{profile}':")
    width = max(len(c.title) for c in checks)
    for check in checks:
        if check.key in results:
            r = results[check.key]
            _log('INFO', f"  {_colored(r.status)}  {_esc(f'{check.title:<{width}}')}  {_esc(r.message)}")
    logger.info(line)
    logger.info("Modes:")
    for name, keys in MODE_CHECKS.items():
        if MODE_OF_VERDICT[name] not in modes:
            continue
        status, text = mode_verdict(keys, results, checks)
        level = 'ERROR' if status == Status.FAIL else 'WARNING' if status == Status.WARN else 'INFO'
        _log(level, f"  {_esc(f'{name:<14}')} {_colored(status, text)}")
    extras = [(label, results[key]) for key, label in EXTRAS.items() if key in results]
    if extras:
        logger.info("Optional extras:")
        for label, result in extras:
            _log('INFO', f"  {_esc(f'{label:<24}')} {_colored(result.status)}: {_esc(result.message)}")
    logger.info(line)
    failed = [c for c in checks if c.key in results and not c.optional and results[c.key].status == Status.FAIL]
    return not failed


def validate_install(profile: str | None = None, modes: tuple[str, ...] = MODES, quick: bool = False,
                     **options) -> int:
    """
    Run the checks of the given modes with a profile and log a report.

    :param options: the Context fields: config_file, censor_config_file, pipeline_file, camera, host, port,
                    use_ssl, ssl_cert, ssl_key.
    :return: the exit code: 1 when a (non-optional) check failed, else 0.
    """
    modes = tuple(modes)
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown mode(s) {', '.join(sorted(unknown))}; choose from {', '.join(MODES)}")
    run_modes = modes + (('vcam',) if 'live' in modes or 'webcam' in modes else ())
    ctx = Context(profile=profile, **options)
    with tempfile.TemporaryDirectory(prefix='censortool-validate-') as tmp:
        ctx.tmp = Path(tmp)
        try:
            results = run_checks(CHECKS, ctx, run_modes, quick)
        finally:
            ctx.close_pipeline()
    ok = report(results, CHECKS, modes, ctx.profile_dir.name if ctx.profile_dir else ctx.profile)
    return 0 if ok else 1


def validate_all_profiles(argv: list[str]) -> int:
    """Validate every profile, each in its own process (a process can only use one backend and one profile)."""
    from app.config import list_profiles
    profiles = list_profiles()
    if not profiles:
        logger.error(f"No profiles found in '{constants.profiles_path}'.")
        return 1

    args, skip = [], False
    for arg in argv:  # pass the arguments on, without --all-profiles and --profile
        if skip:
            skip = False
        elif arg == '--all-profiles':
            continue
        elif arg in ('-p', '--profile'):
            skip = True
        elif not arg.startswith('--profile='):
            args.append(arg)

    main_py = constants.install_dir / 'main.py'
    codes = {}
    for profile in profiles:
        logger.info(f"########## Profile '{profile}' ##########")
        codes[profile] = subprocess.run([sys.executable, str(main_py), *args, '--profile', profile]).returncode
    logger.info("Profiles:")
    for profile, code in codes.items():
        status = Status.PASS if code == 0 else Status.FAIL
        _log('INFO' if code == 0 else 'ERROR',
             f"  {_esc(f'{profile:<24}')} {_colored(status, 'ok' if code == 0 else 'problems found')}")
    return 0 if all(code == 0 for code in codes.values()) else 1
