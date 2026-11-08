"""
Shared fixtures for the backend test suite. Every test runs once per backend:

    python   the pure-Python implementation in core/
    cpp      the compiled censor_core module (skipped when it is not installed)

The tests and goldens are the ones of the C++ censor_core repository, so both backends are held to the same
expectations.

Environment
-----------
CENSOR_CORE_MODELS        Directory with the ONNX models in the layout below. Default: the project's models/
                          folder (models/nudenet/640m.onnx, models/eye_detection/pfld.onnx, models/yolo/...).
                              640m.onnx                   NudeNet v3
                              pfld.onnx                   68-point face landmarks
                              yolo/yolo26n-seg.onnx       COCO segmentation (end2end)
                              yolo/8-11/yolo11n-seg.onnx  COCO segmentation (v8 layout; optional)
CENSOR_CORE_TEST_IMAGES   Directory containing bus.jpg and zidane.jpg
                          (default: the assets folder of the installed `ultralytics` package).

Run ``pytest tests/core --update-goldens`` to (re)generate the files in tests/core/golden/ (from the C++ backend
when it is installed: the goldens are its reference output).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
TEST_IMAGES = ("bus.jpg", "zidane.jpg")
REQUIRED_MODELS = ("640m.onnx", "pfld.onnx", "yolo/yolo26n-seg.onnx")

# test layout -> location in the project's models/ folder
_PROJECT_MODELS = {
    "640m.onnx": "nudenet/640m.onnx",
    "pfld.onnx": "eye_detection/pfld.onnx",
    "yolo/yolo26n-seg.onnx": "yolo/yolo26n-seg.onnx",
    "yolo/8-11/yolo11n-seg.onnx": "yolo/8-11/yolo11n-seg.onnx",
}

BACKENDS = ("python", "cpp")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Overwrite the golden files in tests/core/golden/ with the current output.",
    )


# ------------------------------------------------------------------ discovery


def _images_dir() -> Path | None:
    if env := os.environ.get("CENSOR_CORE_TEST_IMAGES"):
        return Path(env).resolve()
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent / "assets"


@pytest.fixture(scope="session")
def update_goldens(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-goldens"))


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if env := os.environ.get("CENSOR_CORE_MODELS"):
        d = Path(env).resolve()
    else:
        # Symlink the project's models into the layout the tests expect.
        d = tmp_path_factory.mktemp("models")
        for rel, src in _PROJECT_MODELS.items():
            source = PROJECT_ROOT / "models" / src
            if source.is_file():
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).symlink_to(source)
    missing = [m for m in REQUIRED_MODELS if not (d / m).is_file()]
    if missing:
        pytest.skip(f"Models missing in {d}: {missing}. Set CENSOR_CORE_MODELS.")
    return d


@pytest.fixture(scope="session")
def images_dir() -> Path:
    d = _images_dir()
    if d is None or not all((d / n).is_file() for n in TEST_IMAGES):
        pytest.skip("Test images not found. Install ultralytics or set CENSOR_CORE_TEST_IMAGES.")
    return d


# ------------------------------------------------------------------ backend


@pytest.fixture(scope="session", params=BACKENDS)
def cc(request: pytest.FixtureRequest):
    """The backend module under test."""
    if request.param == "python":
        import core

        return core
    if importlib.util.find_spec("censor_core") is None:
        pytest.skip("censor_core (C++ backend) is not installed")
    import censor_core

    return censor_core


@pytest.fixture(scope="session")
def workdir(tmp_path_factory: pytest.TempPathFactory, models_dir: Path) -> Path:
    """
    Run everything from a scratch directory with a ``models`` symlink.

    Some paths are relative to the CWD (e.g. ./data/stickers), so tests must not depend on where pytest is started.
    """
    d = tmp_path_factory.mktemp("cwd")
    (d / "models").symlink_to(models_dir, target_is_directory=True)
    old = Path.cwd()
    os.chdir(d)
    yield d
    os.chdir(old)


@pytest.fixture(scope="session")
def pipeline_config(workdir: Path, models_dir: Path) -> dict:
    """CPU pipeline equivalent to the v1 setup: NudeNet, person segmentation as BODY, eye landmarks."""
    return {
        "models": [
            {
                "name": "nudenet",
                "path": str(models_dir / "640m.onnx"),
                "resize": "rect",
                "agnostic_nms": True,
                "nms_iou": 0.75,
                "max_det": 0,
                "min_confidence": 0.05,
            },
            {
                "name": "people",
                "path": str(models_dir / "yolo" / "yolo26n-seg.onnx"),
                "resize": "stretch",
                "classes": ["person"],
                "labels": {"person": "BODY"},
                "min_confidence": 0.05,
            },
        ],
        "landmarks": {
            "path": str(models_dir / "pfld.onnx"),
            "run_on": {"FACE_FEMALE": "EYES_FEMALE", "FACE_MALE": "EYES_MALE"},
        },
    }


@pytest.fixture(scope="session")
def pipelines(cc, pipeline_config):
    """One sequential ('s') and one parallel ('p') pipeline per backend for the whole session."""
    seq = cc.ImagePipeline(**pipeline_config, max_workers=2, mode="s")
    par = cc.ImagePipeline(**pipeline_config, max_workers=2, mode="p")
    seq.start()
    par.start()
    yield {"s": seq, "p": par}
    seq.stop()
    par.stop()


@pytest.fixture(scope="session")
def example_configs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """
    The default profile's pipeline.yml and censor_config.yml. The app-only keys (backend, n_workers, ...) are
    removed and model paths resolved against the project root, as app/config.py does.
    """
    import yaml

    from app.config import PIPELINE_RUNTIME_KEYS

    profile = PROJECT_ROOT / "configs" / "default"
    spec = yaml.safe_load((profile / "pipeline.yml").read_text())
    for key in PIPELINE_RUNTIME_KEYS:
        spec.pop(key, None)
    for entry in spec.get("models", []) + ([spec["landmarks"]] if spec.get("landmarks") else []):
        entry["path"] = str(PROJECT_ROOT / entry["path"])
        if not Path(entry["path"]).is_file():
            pytest.skip(f"model {entry['path']} not available")
    out = tmp_path_factory.mktemp("example") / "pipeline.yml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    return out, profile / "censor_config.yml"
