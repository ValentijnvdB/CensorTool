"""
Golden tests: compare pipeline output against files recorded from a known build.

* golden/detections_<image>.json   raw detections (features >= MIN_SCORE, bodies)
* golden/<scenario>__<image>.png   1-bit mask of the pixels the censor changed

Regenerate with ``pytest --update-goldens`` and review the diff.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
TEST_IMAGES = ("bus.jpg", "zidane.jpg")
from scenarios import DETECTION_CONFIG, SCENARIOS

MIN_SCORE = 0.05     # detections below this are noise and not recorded
BOX_TOL_PX = 2       # allowed box coordinate drift
SCORE_TOL = 0.01     # allowed score drift
MASK_MIN_IOU = 0.99  # censored-pixel masks must overlap at least this much


def _run(pipeline, cc, image_path: Path, config: dict | None, early_exit: bool = False) -> tuple[np.ndarray, dict]:
    original = cv2.imread(str(image_path))
    assert original is not None, image_path
    job = cc.Job(image=original, config=config, use_cache=False, early_exit=early_exit)
    job = pipeline.submit(job).wait()
    assert job.success, job.error
    return original, job.result


# ------------------------------------------------------------------ detections


def _summarise(result: dict) -> dict:
    features = sorted(
        (
            {"label": d["label"], "score": round(float(d["score"]), 4), "box": [round(float(v)) for v in d["box"]]}
            for d in result["detections"]
            if d["model"] == "nudenet" and d["score"] >= MIN_SCORE
        ),
        key=lambda d: (d["label"], -d["score"], d["box"]),
    )
    bodies = []
    for b in result["detections"]:
        if b["label"] != "BODY" or b["score"] < MIN_SCORE:
            continue
        pts = np.concatenate([np.array(p, dtype=np.int32) for p in b["polygons"]])
        x, y, w, h = cv2.boundingRect(pts)
        bodies.append({"score": round(float(b["score"]), 4), "bbox": [x, y, w, h]})
    bodies.sort(key=lambda b: -b["score"])
    return {"features": features, "bodies": bodies}


def _assert_close(actual: list[dict], expected: list[dict], key: str, what: str) -> None:
    assert len(actual) == len(expected), f"{what}: {len(actual)} found, {len(expected)} expected"
    for a, e in zip(actual, expected):
        if "label" in e:
            assert a["label"] == e["label"], f"{what}: {a} != {e}"
        assert abs(a["score"] - e["score"]) <= SCORE_TOL, f"{what}: score {a} != {e}"
        assert np.abs(np.subtract(a[key], e[key])).max() <= BOX_TOL_PX, f"{what}: box {a} != {e}"


@pytest.mark.parametrize("image", TEST_IMAGES)
def test_detections(cc, pipelines, images_dir, update_goldens, image: str) -> None:
    # Without a censor config every model runs and the raw detections are returned.
    _, result = _run(pipelines["s"], cc, images_dir / image, DETECTION_CONFIG)
    actual = _summarise(result)
    golden = GOLDEN_DIR / f"detections_{Path(image).stem}.json"

    if update_goldens:
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden.write_text(json.dumps(actual, indent=1) + "\n")
        pytest.skip(f"updated {golden.name}")

    expected = json.loads(golden.read_text())
    _assert_close(actual["features"], expected["features"], "box", f"{image} features")
    _assert_close(actual["bodies"], expected["bodies"], "bbox", f"{image} bodies")


# ------------------------------------------------------------------ censor masks


def _changed_mask(original: np.ndarray, censored: np.ndarray) -> np.ndarray:
    assert original.shape == censored.shape
    return (np.abs(original.astype(np.int16) - censored.astype(np.int16)).max(axis=2) > 0).astype(np.uint8) * 255


@pytest.mark.parametrize("image", TEST_IMAGES)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_censor_mask(cc, pipelines, images_dir, update_goldens, scenario: str, image: str) -> None:
    mode, config = SCENARIOS[scenario]
    original, result = _run(pipelines[mode], cc, images_dir / image, config)
    actual = _changed_mask(original, result["image"])
    golden = GOLDEN_DIR / f"{scenario}__{Path(image).stem}.png"

    if update_goldens:
        GOLDEN_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(golden), actual, [cv2.IMWRITE_PNG_BILEVEL, 1])
        pytest.skip(f"updated {golden.name}")

    expected = cv2.imread(str(golden), cv2.IMREAD_GRAYSCALE)
    assert expected is not None, f"missing golden {golden.name}; run pytest --update-goldens"
    a, e = actual > 0, expected > 0
    union = np.logical_or(a, e).sum()
    iou = 1.0 if union == 0 else np.logical_and(a, e).sum() / union
    assert iou >= MASK_MIN_IOU, f"{scenario}/{image}: IoU {iou:.4f} < {MASK_MIN_IOU} ({a.sum()} vs {e.sum()} px)"
