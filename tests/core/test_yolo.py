"""
YoloModel tests.

* Synthetic models: tiny ONNX graphs whose outputs are constants, so decoding
  (layouts, letterbox mapping, NMS, class filter, masks, metadata) can be
  checked against exact expected values without a real network.
* Real models: the YOLOv8-layout and end-to-end exports must agree with each
  other and, when Ultralytics is installed, with Ultralytics' own predict.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


# ------------------------------------------------------------------ synthetic models


def make_model(
    path: Path,
    outputs: list[np.ndarray],
    input_hw: tuple[int, int] | None = (64, 64),
    metadata: dict[str, str] | None = None,
) -> str:
    """Writes an ONNX model with a (1, 3, H, W) input whose outputs are the given constants."""
    h, w = input_hw if input_hw else ("height", "width")
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])
    nodes, outs = [], []
    for i, value in enumerate(outputs):
        name = f"output{i}"
        nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value.astype(np.float32))))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)))
    model = helper.make_model(
        helper.make_graph(nodes, "synthetic", [inp], outs),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    for k, v in (metadata or {}).items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(model, str(path))
    return str(path)


def e2e_rows(*rows: list[float], nm: int = 0, n: int = 4) -> np.ndarray:
    """End-to-end output [1, n, 6 + nm]; unused rows are zero."""
    out = np.zeros((1, n, 6 + nm), np.float32)
    for i, r in enumerate(rows):
        out[0, i, : len(r)] = r
    return out


def v8_anchors(*anchors: list[float], nc: int) -> np.ndarray:
    """V8 output [1, 4 + nc, A] from per-anchor [cx, cy, w, h, score_0..score_nc-1]."""
    return np.array(anchors, np.float32).T[None]


# 128x64 image letterboxed into 64x64: scale 0.5, 16 px padding top and bottom.
WIDE = np.full((64, 128, 3), 50, np.uint8)


def boxes(dets: list[dict]) -> list[list[float]]:
    return [[round(v, 3) for v in d["box"]] for d in dets]


def test_end2end_detect_letterbox_mapping(cc, tmp_path):
    path = make_model(tmp_path / "m.onnx", [e2e_rows([10, 20, 30, 40, 0.9, 1])],
                      metadata={"names": "{0: 'cat', 1: 'dog'}", "end2end": "True"})
    m = cc.YoloModel(path)
    assert (m.task, m.layout, m.names, m.input_size) == ("detect", "end2end", ["cat", "dog"], (64, 64))

    dets = m.detect(WIDE)
    assert [d["label"] for d in dets] == ["dog"]
    assert dets[0]["score"] == pytest.approx(0.9)
    # x: 10 / 0.5 = 20 .. 60; y: (20 - 16) / 0.5 = 8 .. (40 - 16) / 0.5 = 48
    assert boxes(dets) == [[20, 8, 40, 40]]
    assert dets[0]["polygons"] == []


def test_stretch_mapping(cc, tmp_path):
    path = make_model(tmp_path / "m.onnx", [e2e_rows([10, 20, 30, 40, 0.9, 0])], metadata={"end2end": "True"})
    dets = cc.YoloModel(path, resize="stretch").detect(WIDE)
    # x scale 0.5, y scale 1
    assert boxes(dets) == [[20, 20, 40, 20]]


def test_end2end_without_metadata_is_detected_from_shape(cc, tmp_path):
    path = make_model(tmp_path / "m.onnx", [e2e_rows([0, 16, 64, 48, 0.8, 3])])
    m = cc.YoloModel(path)
    assert (m.layout, m.names) == ("end2end", [])
    assert [d["label"] for d in m.detect(WIDE)] == ["class_3"]


def test_end2end_confidence_and_class_filters(cc, tmp_path):
    path = make_model(tmp_path / "m.onnx", [e2e_rows(
        [0, 16, 32, 48, 0.9, 0],
        [32, 16, 64, 48, 0.4, 1],
        [8, 20, 9, 20, 0.9, 0],    # degenerate (zero height)
    )], metadata={"end2end": "True"})
    assert len(cc.YoloModel(path).detect(WIDE)) == 2
    assert len(cc.YoloModel(path, min_confidence=0.5).detect(WIDE)) == 1
    assert [d["label"] for d in cc.YoloModel(path, classes=[1]).detect(WIDE)] == ["class_1"]
    assert len(cc.YoloModel(path, max_det=1).detect(WIDE)) == 1


def test_v8_nms_per_class_and_agnostic(cc, tmp_path):
    path = make_model(tmp_path / "m.onnx", [v8_anchors(
        [32, 32, 20, 20, 0.9, 0.0],   # class 0
        [33, 32, 20, 20, 0.8, 0.0],   # class 0, overlaps the first -> suppressed
        [32, 33, 20, 20, 0.0, 0.7],   # class 1, overlaps but other class
        [10, 40, 4, 4, 0.1, 0.0],     # below the default 0.25
        nc=2,
    )], metadata={"names": "{0: 'a', 1: 'b'}"})
    m = cc.YoloModel(path)
    assert (m.layout, m.task) == ("v8", "detect")
    assert sorted((d["label"], round(d["score"], 2)) for d in m.detect(WIDE)) == [("a", 0.9), ("b", 0.7)]
    assert [d["label"] for d in cc.YoloModel(path, agnostic_nms=True).detect(WIDE)] == ["a"]
    assert len(cc.YoloModel(path, min_confidence=0.05).detect(WIDE)) == 3


def test_segment_mask_follows_box_and_letterbox(cc, tmp_path):
    # One prototype channel, all +10 -> mask is "on" everywhere; the box crops it.
    protos = np.full((1, 1, 16, 16), 10, np.float32)
    path = make_model(tmp_path / "m.onnx", [e2e_rows([16, 24, 48, 40, 0.9, 0, 1.0], nm=1), protos],
                      metadata={"end2end": "True", "task": "segment"})
    m = cc.YoloModel(path)
    assert m.task == "segment"
    (det,) = m.detect(WIDE)
    assert boxes([det]) == [[32, 16, 64, 32]]
    assert len(det["polygons"]) == 1
    x, y, w, h = cv2.boundingRect(np.array(det["polygons"][0], np.int32))
    assert abs(x - 32) <= 2 and abs(y - 16) <= 2 and abs(w - 64) <= 3 and abs(h - 32) <= 3


def _two_blob_model(tmp_path) -> str:
    # 16x16 prototypes over the 64x64 input (4 input px = 8 image px each). Two
    # separate "on" regions inside one box covering the whole image content.
    protos = np.full((1, 1, 16, 16), -10, np.float32)
    protos[0, 0, 5:11, 1:5] = 10    # left blob
    protos[0, 0, 5:11, 11:15] = 10  # right blob
    return make_model(tmp_path / "blobs.onnx", [e2e_rows([0, 16, 64, 48, 0.9, 0, 1.0], nm=1), protos],
                      metadata={"end2end": "True", "names": "{0: 'blob'}"})


def test_segment_keeps_detached_parts(cc, tmp_path):
    (det,) = cc.YoloModel(_two_blob_model(tmp_path)).detect(WIDE)
    assert len(det["polygons"]) == 2
    xs = sorted(cv2.boundingRect(np.array(p, np.int32))[0] for p in det["polygons"])
    assert xs[0] < 30 and xs[1] > 80  # one part on each side of the gap


def test_pipeline_censors_every_part(cc, tmp_path):
    path = _two_blob_model(tmp_path)
    config = {
        "features_to_censor": ["blob"],
        "default_censor_config": {"censor_style": {"type": "bar", "color": [0, 0, 0]}, "min_probability": 0.5},
        "inverse_censor_style": {"type": "blur", "strength": 31},
    }
    with cc.ImagePipeline(models=[{"path": path}], max_workers=1) as p:
        job = p.submit(cc.Job(image=WIDE.copy(), config=config, use_cache=False)).wait()
    assert job.success, job.error
    changed = (job.result["image"] != WIDE).any(axis=2)
    assert changed[20:50, 10:35].mean() > 0.8    # left part censored
    assert changed[20:50, 92:118].mean() > 0.8   # right part censored
    assert not changed[:, 55:75].any()           # the gap between them is not


def test_segment_negative_coefficient_gives_no_polygon(cc, tmp_path):
    protos = np.full((1, 1, 16, 16), 10, np.float32)
    path = make_model(tmp_path / "m.onnx", [e2e_rows([16, 24, 48, 40, 0.9, 0, -1.0], nm=1), protos],
                      metadata={"end2end": "True"})
    (det,) = cc.YoloModel(path).detect(WIDE)
    assert det["polygons"] == []


def test_rect_pads_dynamic_input_to_multiple_of_32(cc, tmp_path):
    # Dynamic model: 128x64 at size 64 -> 64x32, already a multiple of 32, no padding.
    path = make_model(tmp_path / "m.onnx", [e2e_rows([0, 0, 64, 32, 0.9, 0])], input_hw=None,
                      metadata={"end2end": "True"})
    m = cc.YoloModel(path, resize="rect")
    assert m.dynamic_input
    assert boxes(m.detect(WIDE, 64)) == [[0, 0, 128, 64]]


def test_names_override(cc, tmp_path):
    path = make_model(tmp_path / "m.onnx", [e2e_rows([0, 16, 64, 48, 0.9, 1])],
                      metadata={"names": "{0: 'cat', 1: 'dog'}", "end2end": "True"})
    assert [d["label"] for d in cc.YoloModel(path, names=["x", "plate"]).detect(WIDE)] == ["plate"]


@pytest.mark.parametrize(
    ("kwargs", "metadata", "message"),
    [
        ({}, {"task": "pose"}, "unsupported task 'pose'"),
        ({"resize": "rect"}, {}, "dynamic input"),
        ({"resize": "squash"}, {}, "unknown resize mode"),
        ({"layout": "v5"}, {}, "unknown layout"),
    ],
)
def test_rejected_configurations(cc, tmp_path, kwargs, metadata, message):
    path = make_model(tmp_path / "m.onnx", [e2e_rows([0, 0, 1, 1, 0.9, 0])], metadata=metadata)
    with pytest.raises(Exception, match=message):
        cc.YoloModel(path, **kwargs)


def test_v8_name_count_mismatch(cc, tmp_path):
    path = make_model(tmp_path / "m.onnx", [v8_anchors([32, 32, 20, 20, 0.9, 0.1, 0.1], nc=3)],
                      metadata={"names": "{0: 'a', 1: 'b'}"})
    with pytest.raises(Exception, match="3 classes but 2 names"):
        cc.YoloModel(path)


# ------------------------------------------------------------------ real models

SEG_MODELS = {"end2end": "yolo/yolo26n-seg.onnx", "v8": "yolo/8-11/yolo11n-seg.onnx"}


def _model(models_dir: Path, rel: str) -> str:
    path = models_dir / rel
    if not path.exists():
        pytest.skip(f"{rel} not available")
    return str(path)


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


@pytest.mark.parametrize("layout", SEG_MODELS)
def test_real_seg_model_introspection(cc, models_dir, layout):
    m = cc.YoloModel(_model(models_dir, SEG_MODELS[layout]))
    assert (m.task, m.layout, len(m.names), m.names[0], m.input_size) == ("segment", layout, 80, "person", (640, 640))


def test_layouts_agree_on_people(cc, models_dir, images_dir):
    img = cv2.imread(str(images_dir / "bus.jpg"))
    found = {
        layout: cc.YoloModel(_model(models_dir, rel), classes=[0], min_confidence=0.5).detect(img)
        for layout, rel in SEG_MODELS.items()
    }
    assert len(found["v8"]) == len(found["end2end"]) >= 3
    for d in found["end2end"]:
        assert max(_iou(d["box"], o["box"]) for o in found["v8"]) > 0.9
        assert d["polygons"] and all(len(p) >= 3 for p in d["polygons"])


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="ultralytics not installed")
@pytest.mark.parametrize(
    ("rel", "task", "kwargs"),
    [
        ("yolo/yolo26n-seg.onnx", "segment", {}),
        ("yolo/8-11/yolo11n-seg.onnx", "segment", {}),
        ("640m.onnx", "detect", {"resize": "rect"}),  # dynamic input: Ultralytics uses rect letterbox
    ],
)
@pytest.mark.parametrize("image", ["bus.jpg", "zidane.jpg"])
def test_matches_ultralytics(cc, models_dir, images_dir, rel, task, kwargs, image):
    from ultralytics import YOLO

    path = _model(models_dir, rel)
    img = cv2.imread(str(images_dir / image))
    ref = YOLO(path, task=task).predict(img, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
    expected = [
        (ref.names[int(c)], s, [x1, y1, x2 - x1, y2 - y1])
        for (x1, y1, x2, y2), s, c in zip(ref.boxes.xyxy.tolist(), ref.boxes.conf.tolist(), ref.boxes.cls.tolist())
    ]
    actual = cc.YoloModel(path, **kwargs).detect(img, 640)

    assert len(actual) == len(expected)
    for d in actual:
        same = [e for e in expected if e[0] == d["label"]]
        assert same, f"{d['label']} not found by Ultralytics"
        best = max(same, key=lambda e: _iou(e[2], d["box"]))
        assert _iou(best[2], d["box"]) > 0.99
        assert abs(best[1] - d["score"]) < 0.03
