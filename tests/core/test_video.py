"""Video pipeline smoke test (requires ffmpeg/ffprobe on PATH)."""
from __future__ import annotations

import shutil
import threading
import time

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


def _detecting_totals(seen) -> list[int]:
    """The frames_total of every detecting state: how many frames a job had to detect."""
    return [total for phase, _, total in seen if phase == "detecting"]


@needs_ffmpeg
def test_video_cache_is_found_by_content(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "a" / "in.mp4"
    src.parent.mkdir()
    _make_clip(images_dir, src)
    cache = tmp_path / "cache"
    vp = cc.VideoPipeline(pipelines["s"])
    _, config = SCENARIOS["faces_bar"]
    job = vp.submit(cc.VideoJob(str(src), str(tmp_path / "out0.mp4"), config=config, censor_fps=FPS,
                                cache_base_dir=str(cache)))
    assert job.success, job.error
    assert len(list(cache.iterdir())) == 1

    # The same video under another name, in another folder: every frame comes from the cache.
    copy = tmp_path / "b" / "other name.mp4"
    copy.parent.mkdir()
    copy.write_bytes(src.read_bytes())
    job = cc.VideoJob(str(copy), str(tmp_path / "out1.mp4"), config=config, censor_fps=FPS, cache_base_dir=str(cache))
    thread, seen = _watch(job)
    vp.submit(job)
    thread.join(5)
    assert job.success, job.error
    assert all(total == 0 for total in _detecting_totals(seen)), f"detected frames again: {seen}"
    assert len(list(cache.iterdir())) == 1

    # Another video with the same name gets its own cache.
    other = tmp_path / "c" / "in.mp4"
    other.parent.mkdir()
    frame = cv2.imread(str(images_dir / "bus.jpg"))
    writer = cv2.VideoWriter(str(other), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (frame.shape[1], frame.shape[0]))
    for _ in range(5):
        writer.write(frame)
    writer.release()
    job = vp.submit(cc.VideoJob(str(other), str(tmp_path / "out2.mp4"), config=config, censor_fps=FPS,
                                cache_base_dir=str(cache)))
    assert job.success, job.error
    assert len(list(cache.iterdir())) == 2


@needs_ffmpeg
def test_video_cancelled_while_detecting_keeps_its_detections(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    cache = tmp_path / "cache"
    vp = cc.VideoPipeline(pipelines["s"])
    _, config = SCENARIOS["faces_bar"]

    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, censor_fps=FPS, cache_base_dir=str(cache))
    cancelled_at = []

    def cancel_after_some_frames(state) -> None:
        phase, done, _ = state
        if not cancelled_at and phase == "detecting" and done >= 5:
            job.cancel()
            cancelled_at.append(done)

    thread, seen = _watch(job, cancel_after_some_frames)
    vp.submit(job)
    thread.join(5)
    assert cancelled_at, f"never saw 5 detected frames: {seen}"
    assert job.error == "cancelled"
    full = max(_detecting_totals(seen))
    assert any(cache.iterdir()), "the detections were not saved"

    # The next job only detects the frames that were still missing.
    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, censor_fps=FPS, cache_base_dir=str(cache))
    thread, seen = _watch(job)
    vp.submit(job)
    thread.join(5)
    assert job.success, job.error
    assert max(_detecting_totals(seen), default=0) <= full - cancelled_at[0], seen


@needs_ffmpeg
def test_video_without_config_needs_early_exit(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    vp = cc.VideoPipeline(pipelines["s"])
    job = vp.submit(cc.VideoJob(str(src), str(tmp_path / "out.mp4"), censor_fps=FPS, use_cache=False))
    assert not job.success and "censor config" in job.error
    job = vp.submit(cc.VideoJob(str(src), str(tmp_path / "out.mp4"), censor_fps=FPS, use_cache=False, early_exit=True))
    assert job.success, job.error


# ------------------------------------------------------------------ progress and cancel

def _watch(job, on_progress=None) -> tuple[threading.Thread, list]:
    """
    Records (phase, frames_done, frames_total) whenever it changes, from another thread while submit() runs (as the
    HTTP server does). on_progress(state) is called with each new state and may cancel the job.
    """
    seen: list = []

    def run() -> None:
        while True:
            state = (job.phase, job.frames_done, job.frames_total)
            if not seen or state != seen[-1]:
                seen.append(state)
                if on_progress is not None:
                    on_progress(state)
            if state[0] == "done":
                return
            time.sleep(0.0005)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, seen


@needs_ffmpeg
def test_video_progress(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    _, config = SCENARIOS["faces_bar"]
    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, censor_fps=FPS, use_cache=False)
    assert job.phase is None and not job.cancelled
    assert (job.frames_done, job.frames_total) == (0, 0)

    thread, seen = _watch(job)
    cc.VideoPipeline(pipelines["s"]).submit(job)
    thread.join(5)

    assert job.success, job.error
    assert job.phase == "done" and seen[-1][0] == "done"

    phases = [phase for phase, _, _ in seen if phase is not None]
    order = ["detecting", "censoring", "encoding", "done"]
    assert phases == sorted(phases, key=order.index), f"phases out of order: {seen}"
    assert "censoring" in phases, f"never saw the censoring phase: {seen}"
    for phase in ("detecting", "censoring"):
        states = [(done, total) for p, done, total in seen if p == phase]
        dones = [done for done, _ in states]
        assert dones == sorted(dones), f"frames_done went back in {phase}: {seen}"
        assert all(0 <= done <= total for done, total in states), f"frames_done out of range in {phase}: {seen}"
    assert {total for p, _, total in seen if p == "censoring"} == {N_FRAMES}


@needs_ffmpeg
def test_video_cancel_before_start(cc, pipelines, images_dir, tmp_path) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    _, config = SCENARIOS["faces_bar"]
    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, use_cache=False)
    job.cancel()
    assert job.cancelled

    cc.VideoPipeline(pipelines["s"]).submit(job)
    assert not job.success and job.error == "cancelled" and job.phase == "done"
    assert not (tmp_path / "out.mp4").exists()


@needs_ffmpeg
@pytest.mark.parametrize("during", ["detecting", "censoring"])
def test_video_cancel_while_running(cc, pipelines, images_dir, tmp_path, during) -> None:
    src = tmp_path / "in.mp4"
    _make_clip(images_dir, src)
    _, config = SCENARIOS["faces_bar"]
    job = cc.VideoJob(str(src), str(tmp_path / "out.mp4"), config=config, censor_fps=FPS, use_cache=False)
    cancelled_at = []

    def cancel_once_in_phase(state) -> None:
        phase, done, _ = state
        if not cancelled_at and phase == during and (during == "detecting" or done >= 5):
            job.cancel()
            cancelled_at.append(state)

    thread, seen = _watch(job, cancel_once_in_phase)
    cc.VideoPipeline(pipelines["s"]).submit(job)
    thread.join(5)

    assert cancelled_at, f"never saw the job in '{during}': {seen}"
    assert [p for p, _, _ in seen][-1] == "done"
    assert "encoding" not in [p for p, _, _ in seen], "the job went on to encode after it was cancelled"
    assert not job.success and job.error == "cancelled"
    assert job.phase == "done"
    assert not (tmp_path / "out.mp4").exists()
    assert not any(p.suffix == ".avi" for p in tmp_path.rglob("*")), "temporary AVI left behind"
