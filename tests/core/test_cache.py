"""Image detection cache."""
from __future__ import annotations

import cv2

from scenarios import SCENARIOS


def _features(result: dict) -> list[tuple]:
    return sorted((d["label"], round(d["score"], 4), tuple(round(v) for v in d["box"])) for d in result["detections"])


def test_cache_roundtrip_creates_missing_dirs(cc, pipeline_config, images_dir, tmp_path) -> None:
    cache_dir = tmp_path / "does" / "not" / "exist"
    image = cv2.imread(str(images_dir / "zidane.jpg"))
    _, config = SCENARIOS["faces_bar"]

    # First run writes the cache. stop() drains the writer threads.
    with cc.ImagePipeline(**pipeline_config, max_workers=1) as p:
        first = p.submit(cc.Job(image=image, config=config, cache_base_dir=str(cache_dir))).wait()
    assert first.success, first.error
    cache_files = list(cache_dir.iterdir())
    assert len(cache_files) == 1

    # Second run (fresh pipeline) reads it back.
    with cc.ImagePipeline(**pipeline_config, max_workers=1) as p:
        second = p.submit(cc.Job(image=image, config=config, cache_base_dir=str(cache_dir))).wait()
    assert second.success, second.error
    assert _features(second.result) == _features(first.result)


def test_no_cache_dir_means_no_cache(cc, pipelines, images_dir, workdir) -> None:
    image = cv2.imread(str(images_dir / "zidane.jpg"))
    _, config = SCENARIOS["faces_bar"]
    job = pipelines["s"].submit(cc.Job(image=image, config=config, use_cache=True)).wait()
    assert job.success, job.error
    assert not (workdir / "cache").exists()
