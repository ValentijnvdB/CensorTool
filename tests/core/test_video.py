"""Video pipeline smoke test (requires ffmpeg/ffprobe on PATH)."""
from __future__ import annotations

import shutil

import cv2
import pytest

from scenarios import SCENARIOS

N_FRAMES = 30
FPS = 15


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed"
)


def _make_clip(images_dir, path) -> None:
    frame = cv2.imread(str(images_dir / "zidane.jpg"))
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for _ in range(N_FRAMES):
        writer.write(frame)
    writer.release()


@needs_ffmpeg
def test_video_roundtrip(cc, pipelines, images_dir, tmp_path) -> None:
    # Shell metacharacters in the file names must be passed to ffmpeg literally.
    weird = tmp_path / "it's a $HOME `x` dir"
    weird.mkdir()
    src = weird / "in put.mp4"
    _make_clip(images_dir, src)

    out = weird / "out put.mp4"
    _, config = SCENARIOS["faces_bar"]
    vp = cc.VideoPipeline(pipelines["s"])
    job = vp.submit(cc.VideoJob(str(src), str(out), config=config, censor_fps=FPS, use_cache=False))

    assert job.success, job.error
    assert out.is_file()
    cap = cv2.VideoCapture(str(out))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert abs(count - N_FRAMES) <= 1, count


@needs_ffmpeg
def test_video_ffmpeg_failure_is_reported(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    out = tmp_path / "missing_dir" / "out.mp4"  # ffmpeg cannot write here

    _, config = SCENARIOS["faces_bar"]
    vp = cc.VideoPipeline(pipelines["s"])
    job = vp.submit(cc.VideoJob(str(src), str(out), config=config, censor_fps=FPS, use_cache=False))

    assert not job.success
    assert "ffmpeg failed" in (job.error or "")
    assert not any(p.suffix == ".avi" for p in tmp_path.rglob("*")), "temporary AVI left behind"


@needs_ffmpeg
def test_video_cache_roundtrip(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    cache = tmp_path / "cache"
    vp = cc.VideoPipeline(pipelines["s"])

    _, faces = SCENARIOS["faces_bar"]
    for i in range(2):  # second run reads the cache
        out = tmp_path / f"out{i}.mp4"
        job = vp.submit(cc.VideoJob(str(src), str(out), config=faces, censor_fps=FPS, cache_base_dir=str(cache)))
        assert job.success, job.error
        assert out.is_file()
    assert any(cache.rglob("*"))

    # A config needing another model fills in the missing stage on the cached frames.
    _, body = SCENARIOS["full_body"]
    job = vp.submit(cc.VideoJob(str(src), str(tmp_path / "body.mp4"), config=body, censor_fps=FPS,
                                cache_base_dir=str(cache)))
    assert job.success, job.error


@needs_ffmpeg
def test_video_without_config_needs_early_exit(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    vp = cc.VideoPipeline(pipelines["s"])
    job = vp.submit(cc.VideoJob(str(src), str(tmp_path / "out.mp4"), censor_fps=FPS, use_cache=False))
    assert not job.success and "censor config" in job.error
    job = vp.submit(cc.VideoJob(str(src), str(tmp_path / "out.mp4"), censor_fps=FPS, use_cache=False, early_exit=True))
    assert job.success, job.error
