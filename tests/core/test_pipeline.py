"""
Config-driven pipelines: pipeline.yml, labels, on-demand models, clip_to,
cascades (run_on), landmarks and the per-model cache.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import pytest

from scenarios import SCENARIOS, _cfg


def _image(images_dir: Path, name: str = "zidane.jpg"):
    image = cv2.imread(str(images_dir / name))
    assert image is not None
    return image


def _run(pipeline, cc, image, config=None, **kwargs) -> dict:
    job = pipeline.submit(cc.Job(image=image, config=config, use_cache=False, **kwargs)).wait()
    assert job.success, job.error
    return job.result


def _models_in(result: dict) -> set[str]:
    return {d["model"] for d in result["detections"]}


def _error(pipeline, cc, image, config) -> str:
    job = pipeline.submit(cc.Job(image=image, config=config, use_cache=False)).wait()
    assert not job.success
    return job.error


# ------------------------------------------------------------------ construction


def test_labels_and_models(pipelines) -> None:
    p = pipelines["s"]
    assert p.models == ["nudenet", "people", "landmarks"]
    labels = p.labels
    assert labels["BODY"] == "people"
    assert labels["FACE_MALE"] == "nudenet"
    assert labels["EYES_MALE"] == "landmarks"
    assert "person" not in labels and "bus" not in labels  # classes filter + rename


def test_from_yaml_resolves_paths_relative_to_file(cc, models_dir, tmp_path) -> None:
    (tmp_path / "m").symlink_to(models_dir, target_is_directory=True)
    yml = tmp_path / "pipeline.yml"
    yml.write_text(
        "device: cpu\n"
        "models:\n"
        "  - name: people\n"
        "    path: m/yolo/yolo26n-seg.onnx\n"
        "    classes: [person]\n"
        "    labels: {person: BODY}\n"
    )
    p = cc.ImagePipeline.from_yaml(str(yml), max_workers=1)
    assert p.models == ["people"] and list(p.labels) == ["BODY"]


def test_label_collision_is_an_error(cc, pipeline_config) -> None:
    people = pipeline_config["models"][1]
    with pytest.raises(RuntimeError, match="produced by both 'people' and 'people2'.*labels: or label_prefix:"):
        cc.ImagePipeline(models=[people, {**people, "name": "people2"}])


def test_label_prefix_and_class_names(cc, pipeline_config) -> None:
    people = pipeline_config["models"][1]
    coco = {"name": "coco", "path": people["path"], "label_prefix": "coco_", "classes": ["bus", "tie"]}
    p = cc.ImagePipeline(models=[people, coco])
    assert sorted(p.labels) == ["BODY", "coco_bus", "coco_tie"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"classes": ["persn"]}, "no class 'persn'.*did you mean 'person'"),
        ({"labels": {"persn": "BODY"}}, "no class 'persn'.*did you mean 'person'"),
        ({"run_on": ["FACE"]}, "runs on label 'FACE', which no model produces"),
    ],
)
def test_bad_model_settings(cc, pipeline_config, change, message) -> None:
    people = {**pipeline_config["models"][1], **change}
    with pytest.raises(RuntimeError, match=message):
        cc.ImagePipeline(models=[people])


def test_reserved_and_duplicate_names(cc, pipeline_config) -> None:
    people = pipeline_config["models"][1]
    with pytest.raises(RuntimeError, match="reserved"):
        cc.ImagePipeline(models=[{**people, "name": "landmarks"}])
    with pytest.raises(RuntimeError, match="two models are named"):
        cc.ImagePipeline(models=[people, {**people, "labels": {}, "label_prefix": "x_"}])


# ------------------------------------------------------------------ censor config validation


def test_unknown_label_warns_once(cc, pipelines, images_dir, capfd) -> None:
    config = _cfg(["FACE_MAEL"])
    for _ in range(2):
        _run(pipelines["s"], cc, _image(images_dir), config)
    err = capfd.readouterr().err
    assert err.count("No model produces label 'FACE_MAEL'") == 1
    assert "did you mean 'FACE_MALE'" in err


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"FACE_MALE": {"clip_to": ["BODDY"]}}, "no model produces label 'BODDY'.*did you mean 'BODY'"),
        ({"FACE_MALE": {"clip_to": ["FEET_COVERED"]}}, "comes from detect model 'nudenet'"),
        ({"FACE_MALE": {"shape": "mask"}}, "needs a segmentation model"),
    ],
)
def test_invalid_censor_config_fails_the_job(cc, pipelines, images_dir, overrides, message) -> None:
    import re

    error = _error(pipelines["s"], cc, _image(images_dir), _cfg(["FACE_MALE"], overrides))
    assert re.search(message, error), error


def test_removed_keys_explain_replacement(cc, capfd) -> None:
    cc.Job(config=_cfg(["FACE_MALE"], {"FACE_MALE": {"intersect_human": True}}, body_seg_override=True))
    err = capfd.readouterr().err
    assert "'intersect_human'" in err and "clip_to" in err
    assert "'body_seg_override'" in err and "no longer supported" in err

    cc.Job(config=_cfg(["FACE_MALE"], merge_overlapping_censor_boxes=True))
    assert "'merge_overlapping_censor_boxes'" in capfd.readouterr().err


# ------------------------------------------------------------------ on-demand models


def test_only_needed_models_run(cc, pipelines, images_dir) -> None:
    image = _image(images_dir)
    assert _models_in(_run(pipelines["s"], cc, image, SCENARIOS["faces_bar"][1])) == {"nudenet"}
    assert _models_in(_run(pipelines["s"], cc, image, SCENARIOS["full_body"][1])) == {"people"}
    assert _models_in(_run(pipelines["s"], cc, image, SCENARIOS["eyes"][1])) == {"nudenet", "landmarks"}
    assert _models_in(_run(pipelines["s"], cc, image)) == {"nudenet", "people", "landmarks"}


def test_clip_only_model_is_lazy_in_sequential_mode(cc, pipelines, images_dir) -> None:
    image = _image(images_dir)
    config = deepcopy(SCENARIOS["faces_intersect"][1])
    # A face is found, so the segmentation model runs.
    assert "people" in _models_in(_run(pipelines["s"], cc, image, config))
    # No face can reach min_probability 0.99: nothing to clip, segmentation is skipped.
    config["default_censor_config"]["min_probability"] = 0.99
    assert _models_in(_run(pipelines["s"], cc, image, config)) == {"nudenet"}
    # Parallel mode runs it up front regardless.
    assert "people" in _models_in(_run(pipelines["p"], cc, image, config))


# ------------------------------------------------------------------ cascades and landmarks


def test_landmark_parents(cc, pipelines, images_dir) -> None:
    detections = _run(pipelines["s"], cc, _image(images_dir))["detections"]
    by_id = {d["id"]: d for d in detections}
    eyes = [d for d in detections if d["model"] == "landmarks"]
    assert eyes
    for eye in eyes:
        face = by_id[eye["parent"]]
        assert face["label"] == {"EYES_MALE": "FACE_MALE", "EYES_FEMALE": "FACE_FEMALE"}[eye["label"]]
        assert eye["score"] == face["score"]
    assert all(d["parent"] is None for d in detections if d["model"] != "landmarks")


def test_cascade_runs_on_crops(cc, pipeline_config, images_dir) -> None:
    people, nudenet = pipeline_config["models"][1], pipeline_config["models"][0]
    cascade = {**nudenet, "name": "body_parts", "run_on": ["BODY"], "label_prefix": "crop_", "crop_padding": 0.1}
    p = cc.ImagePipeline(models=[people, cascade], max_workers=1)
    with p:
        detections = _run(p, cc, _image(images_dir, "bus.jpg"))["detections"]

    by_id = {d["id"]: d for d in detections}
    parts = [d for d in detections if d["model"] == "body_parts"]
    assert any(d["label"] == "crop_FACE_MALE" for d in parts)
    for d in parts:
        body = by_id[d["parent"]]
        assert body["label"] == "BODY"
        bx, by, bw, bh = body["box"]
        x, y, w, h = d["box"]
        slack = 0.06 * max(bw, bh) + 1  # crop_padding 0.1 -> 5% per side
        assert bx - slack <= x and x + w <= bx + bw + slack
        assert by - slack <= y and y + h <= by + bh + slack


# ------------------------------------------------------------------ per-model cache


def test_cache_is_per_model(cc, pipeline_config, images_dir, tmp_path) -> None:
    image = _image(images_dir)
    cache = str(tmp_path / "cache")

    def run(config, models):
        with cc.ImagePipeline(models=models, landmarks=pipeline_config["landmarks"], max_workers=1) as p:
            job = p.submit(cc.Job(image=image, config=config, cache_base_dir=cache)).wait()
        assert job.success, job.error
        return job.result

    models = deepcopy(pipeline_config["models"])
    first = run(None, models)
    assert first["cached_models"] == []

    assert run(None, models)["cached_models"] == ["nudenet", "people", "landmarks"]

    # Changing one model's options invalidates only that model (and nothing downstream of it).
    models[1]["min_confidence"] = 0.6
    third = run(None, models)
    assert third["cached_models"] == ["nudenet", "landmarks"]
    assert all(d["score"] >= 0.6 for d in third["detections"] if d["model"] == "people")

    # One entry per model: switching back reruns that model once.
    assert run(None, deepcopy(pipeline_config["models"]))["cached_models"] == ["nudenet", "landmarks"]


def test_cache_key_includes_input_size_for_dynamic_models(cc, pipeline_config, images_dir, tmp_path) -> None:
    image = _image(images_dir)
    cache = str(tmp_path / "cache")
    with cc.ImagePipeline(models=pipeline_config["models"], max_workers=1) as p:
        p.submit(cc.Job(image=image, cache_base_dir=cache, size=640)).wait()
    with cc.ImagePipeline(models=pipeline_config["models"], max_workers=1) as p:
        result = p.submit(cc.Job(image=image, cache_base_dir=cache, size=320)).wait().result
    # NudeNet has a dynamic input, so its result depends on size; the people model does not.
    assert result["cached_models"] == ["people"]


def test_size_defaults_to_model_input_size(cc, pipeline_config, images_dir, tmp_path) -> None:
    image = _image(images_dir)
    nudenet = pipeline_config["models"][0]
    with cc.ImagePipeline(models=[nudenet], max_workers=1) as p:
        default = _run(p, cc, image)["detections"]
        explicit = _run(p, cc, image, size=640)["detections"]
    assert default == explicit  # NudeNet has no imgsz metadata: 640

    # imgsz in the pipeline config now applies to whole-image runs too.
    with cc.ImagePipeline(models=[{**nudenet, "imgsz": 320}], max_workers=1) as p:
        small = _run(p, cc, image)["detections"]
        assert small == _run(p, cc, image, size=320)["detections"]
    assert small != default

    # size=0 and the model's own size share one cache entry.
    cache = str(tmp_path / "cache")
    with cc.ImagePipeline(models=[nudenet], max_workers=1) as p:
        p.submit(cc.Job(image=image, cache_base_dir=cache)).wait()
        assert p.submit(cc.Job(image=image, cache_base_dir=cache, size=640)).wait().result["cached_models"] == ["nudenet"]


def test_example_pipeline_yml_loads(cc, example_configs) -> None:
    # The nudity profile runs only Nevermore big.
    pipeline_yml, _ = example_configs
    p = cc.ImagePipeline.from_yaml(str(pipeline_yml), max_workers=1)
    assert p.models == ["nevermore"]
    assert {"face", "female_breast_exposed", "person"} <= set(p.labels)
    assert set(p.labels.values()) == {"nevermore"}


def test_example_censor_config_is_valid(cc, images_dir, example_configs, capfd) -> None:
    pipeline_yml, censor_yml = example_configs
    config = cc.load_censor_config(str(censor_yml))
    with cc.ImagePipeline.from_yaml(str(pipeline_yml), max_workers=1) as p:
        job = p.submit(cc.Job(image=_image(images_dir), config=config, use_cache=False)).wait()
    assert job.success, job.error
    assert "WARNING" not in capfd.readouterr().err  # every key and label is known


def test_version(cc) -> None:
    assert cc.__version__.split(".")[0] == "2"
