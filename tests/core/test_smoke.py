"""Import and API-surface checks that need no models."""
from __future__ import annotations


def test_import(cc) -> None:
    for name in ("ImagePipeline", "Job", "JobHandle", "VideoPipeline", "VideoJob", "load_censor_config", "wait_any"):
        assert hasattr(cc, name), name


def test_job_defaults(cc) -> None:
    job = cc.Job()
    assert job.job_id  # auto-generated uuid
    assert job.size == 0  # each model's own input size
    assert job.use_cache is True
    assert job.result is None
    assert job.cancelled is False
    job.cancel()
    assert job.cancelled is True


def test_bad_model_path_fails_fast(cc, tmp_path) -> None:
    try:
        cc.ImagePipeline(models=[{"path": str(tmp_path / "does_not_exist.onnx")}])
    except RuntimeError as e:
        assert "does_not_exist.onnx" in str(e)
    else:
        raise AssertionError("ImagePipeline accepted a missing model")
