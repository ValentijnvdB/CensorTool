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


# ------------------------------------------------------------------ progress and cancel (Python backend only)

def _python_only(cc) -> None:
    if not hasattr(cc.VideoJob, "cancel"):
        pytest.skip("progress and cancel are only in the Python backend")


@needs_ffmpeg
def test_video_progress_ends_done(cc, pipelines, images_dir, tmp_path) -> None:
    _python_only(cc)
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    _, config = SCENARIOS["faces_bar"]
    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, censor_fps=FPS, use_cache=False)
    assert job.phase is None and not job.cancelled

    seen = []
    import core.video as video
    original = video.apply_censor_boxes

    def spy(frame, boxes, config):
        seen.append((job.phase, job.frames_done, job.frames_total))
        return original(frame, boxes, config)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(video, "apply_censor_boxes", spy)
        cc.VideoPipeline(pipelines["s"]).submit(job)

    assert job.success, job.error
    assert job.phase == "done"
    assert seen[0] == ("censoring", 0, N_FRAMES)
    assert seen[-1] == ("censoring", N_FRAMES - 1, N_FRAMES)


@needs_ffmpeg
def test_video_cancel_before_start(cc, pipelines, images_dir, tmp_path) -> None:
    _python_only(cc)
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    _, config = SCENARIOS["faces_bar"]
    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, use_cache=False)
    job.cancel()

    cc.VideoPipeline(pipelines["s"]).submit(job)
    assert not job.success and job.error == "cancelled" and job.phase == "done"
    assert not (tmp_path / "out.mp4").exists()


@needs_ffmpeg
@pytest.mark.parametrize("during", ["detecting", "censoring"])
def test_video_cancel_while_running(cc, pipelines, images_dir, tmp_path, during) -> None:
    _python_only(cc)
    import core.video as video

    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    _, config = SCENARIOS["faces_bar"]
    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, censor_fps=FPS, use_cache=False)
    pipeline = pipelines["s"]
    vp = cc.VideoPipeline(pipeline)

    with pytest.MonkeyPatch.context() as mp:
        if during == "detecting":
            original_submit = pipeline.submit

            def submit(frame_job):
                job.cancel()  # cancelled once the first frame is on its way
                return original_submit(frame_job)
            mp.setattr(pipeline, "submit", submit)
        else:
            original = video.apply_censor_boxes

            def censor(frame, boxes, config):
                if job.frames_done == 5:
                    job.cancel()
                return original(frame, boxes, config)
            mp.setattr(video, "apply_censor_boxes", censor)
        vp.submit(job)

    assert not job.success and job.error == "cancelled"
    assert job.phase == "done"
    assert not (tmp_path / "out.mp4").exists()
    assert not any(p.suffix == ".avi" for p in tmp_path.rglob("*")), "temporary AVI left behind"
