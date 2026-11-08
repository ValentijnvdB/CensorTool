"""Video pipeline: VideoJob and VideoPipeline. Same API as censor_core 2.0. Needs ffmpeg/ffprobe on PATH."""
from __future__ import annotations

import math
import subprocess
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from typing import Any, Optional

import cv2

from .boxes import Box, build_boxes
from .cache import (VideoCache, cached_detections, read_video_cache, update_cache, video_cache_path,
                    write_video_cache)
from .censor_config import CensorConfig, optional_censor_config
from .draw import apply_censor_boxes
from .pipeline import ImagePipeline, Job
from .pipeline_config import load_pipeline_config


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


class VideoJob:
    """Unit of work submitted to VideoPipeline. A censor config is required unless early_exit."""

    def __init__(self, video_path: str, output_path: str, avi_path: Any = None, cache_base_dir: Any = None,
                 config: Optional[dict[str, Any]] = None, job_id: str = '', size: int = 0, censor_fps: float = 15,
                 early_exit: bool = False, override_cache: bool = False, skip_existing: bool = False,
                 use_cache: bool = True):
        self.video_path = str(video_path)
        self.output_path = str(output_path)
        self.avi_path = _optional_str(avi_path)
        self.cache_base_dir = _optional_str(cache_base_dir)
        self._config: Optional[CensorConfig] = optional_censor_config(config)
        self.job_id: str = job_id or str(uuid.uuid4())
        self.size = int(size)
        self.censor_fps = float(censor_fps)
        self.early_exit = bool(early_exit)
        self.override_cache = bool(override_cache)
        self.skip_existing = bool(skip_existing)
        self.use_cache = bool(use_cache)

        self._success = False
        self._skipped = False
        self._error: Optional[str] = None
        self._result: Optional[str] = None

    @property
    def config(self) -> None:
        """Always returns None: the parsed censor config is opaque."""
        return None

    @config.setter
    def config(self, value: Optional[dict[str, Any]]) -> None:
        self._config = optional_censor_config(value)

    @property
    def success(self) -> bool:
        return self._success

    @property
    def skipped(self) -> bool:
        return self._skipped

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def result(self) -> Optional[str]:
        """The output path on success."""
        return self._result

    def __repr__(self) -> str:
        return f"VideoJob(video_path={self.video_path!r}, success={self._success}, error={self._error!r})"


def censor_frames(censor_fps: float, video_fps: float, max_frames: int) -> list[int]:
    """The frame numbers to analyse, honoring censor_fps."""
    factor = video_fps / censor_fps
    frames = []
    for i in range(max_frames):
        frame = int(math.floor(i * factor))
        if frame >= max_frames:
            break
        frames.append(frame)
    return frames


def _run_ffmpeg(args: list[str]) -> None:
    try:
        proc = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg failed: {args[0]} not found on PATH") from e
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (status {proc.returncode}): {' '.join(args)}\n"
                           f"{proc.stderr.decode(errors='replace').strip()}")


def file_has_audio(video_path: str) -> bool:
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                              'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return False
    return 'audio' in out.stdout.decode(errors='replace')


class VideoPipeline:
    """
    Synchronous video censoring backed by an ImagePipeline for per-frame detection.

    VideoPipeline.from_yaml(...) creates and owns its own (started) ImagePipeline; VideoPipeline(pipeline) wraps
    an existing, already-started one, which the caller keeps ownership of.
    """

    def __init__(self, pipeline: ImagePipeline, max_concurrent_jobs: int = 128):
        self._pipeline = pipeline
        self._owned = False
        self._max_concurrent_jobs = int(max_concurrent_jobs)

    @staticmethod
    def from_yaml(path: str, max_workers: int = 4, max_concurrent_jobs: int = 128, mode: str = 's') -> 'VideoPipeline':
        """Creates (and starts) its own ImagePipeline from a pipeline.yml."""
        pipeline = ImagePipeline._from_config(load_pipeline_config(path), max_workers, mode)
        pipeline.start()
        vp = VideoPipeline(pipeline, max_concurrent_jobs)
        vp._owned = True
        return vp

    def close(self) -> None:
        if self._owned:
            self._pipeline.stop()

    def __enter__(self) -> 'VideoPipeline':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        if getattr(self, '_owned', False):
            try:
                self._pipeline.stop()
            except Exception:
                pass

    # ------------------------------------ submit ------------------------------------
    def submit(self, job: VideoJob) -> VideoJob:
        """Process job synchronously; returns it with success, error and result populated."""
        if not Path(job.video_path).is_file():
            job._error = f"Video file not found: {job.video_path}"
            return job

        if job.skip_existing and Path(job.output_path).exists():
            job._skipped = True
            job._success = True
            return job

        output = Path(job.output_path)
        avi_path = Path(job.avi_path) if job.avi_path else output.parent / f"{output.stem}_tmp.avi"
        try:
            self._submit(job, avi_path)
        except Exception as e:
            job._success = False
            job._error = str(e) or type(e).__name__
        finally:
            try:
                avi_path.unlink(missing_ok=True)  # never leave the temporary AVI behind
            except OSError:
                pass
        return job

    def _submit(self, job: VideoJob, avi_path: Path) -> None:
        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {job.video_path}")
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_fps <= 0:
                raise RuntimeError(f"Could not read the frame rate of {job.video_path}")

            graph = self._pipeline.graph
            if job._config is None and not job.early_exit:
                raise RuntimeError("a VideoJob needs a censor config (or early_exit=True)")
            if job._config is not None:
                graph.validate(job._config)

            fingerprints = graph.fingerprints(job.size)
            cache: VideoCache = {}
            cache_path = None
            if job.use_cache and job.cache_base_dir is not None:
                cache_path = video_cache_path(Path(job.video_path), Path(job.cache_base_dir))
                if not job.override_cache:
                    cache = read_video_cache(cache_path)

            self._run_detection(job, cap, video_fps, num_frames, cache, fingerprints)

            if cache_path is not None:
                write_video_cache(cache_path, cache)

            if job.early_exit:
                job._success = True
                return

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._apply_censor(job, cap, avi_path, cache, fingerprints, num_frames, vid_w, vid_h, video_fps)
        finally:
            cap.release()

        if Path(job.output_path).suffix.lower() == '.gif':
            self._write_gif(avi_path, Path(job.output_path))
        else:
            self._write_video(avi_path, job.video_path, job.output_path, file_has_audio(job.video_path))

        job._result = job.output_path
        job._success = True

    # ------------------------------------ detection pass ------------------------------------
    def _run_detection(self, job: VideoJob, cap: cv2.VideoCapture, video_fps: float, num_frames: int,
                       cache: VideoCache, fingerprints: dict[str, str]) -> None:
        censor_fps = job.censor_fps
        if censor_fps <= 0 or censor_fps > video_fps:
            censor_fps = video_fps

        graph = self._pipeline.graph
        config = job._config
        plan = graph.plan(config)

        # Frames whose cached detections lack a stage this config needs.
        needed = [fid for fid in censor_frames(censor_fps, video_fps, num_frames)
                  if graph.missing(plan, cached_detections(cache.get(fid, {}), fingerprints), config)]
        if not needed:
            return

        limit = self._max_concurrent_jobs if self._max_concurrent_jobs > 0 else len(needed)
        in_flight: dict[Any, int] = {}  # handle -> frame number
        ptr, read_frame, exhausted = 0, 0, False

        def submit_pending() -> None:
            nonlocal ptr, read_frame, exhausted
            # Frames are read sequentially (seeking is slower); only the needed ones are submitted.
            while not exhausted and ptr < len(needed) and len(in_flight) < limit:
                target = needed[ptr]
                while read_frame <= target:
                    ok, frame = cap.read()
                    if not ok:
                        exhausted = True
                        return
                    if read_frame == target:
                        frame_job = Job(image=frame, job_id=str(target), size=job.size,
                                        timestamp=target / video_fps, early_exit=True, use_cache=False)
                        frame_job._config = config
                        in_flight[self._pipeline.submit(frame_job)] = target
                    read_frame += 1
                ptr += 1

        print("Running models", flush=True)
        submit_pending()

        step = max(1, num_frames * 5 // 100)
        next_update, completed = step, 0
        while in_flight:
            done, _ = wait([h._future for h in in_flight], return_when=FIRST_COMPLETED)
            for handle in [h for h in in_flight if h._future in done]:
                fid = in_flight.pop(handle)
                frame_job = handle.wait()
                if not frame_job.success or frame_job._detections is None:
                    for h in in_flight:
                        h.wait()  # let in-flight frames finish before failing
                    raise RuntimeError(f"detection failed on frame {fid}: {frame_job.error or 'unknown error'}")
                update_cache(cache.setdefault(fid, {}), frame_job._detections, fingerprints)
                completed += 1
                if completed >= next_update:
                    print(f"Frames done: {completed}/{len(needed)}", flush=True)
                    next_update += step
            submit_pending()

    # ------------------------------------ censor pass ------------------------------------
    def _apply_censor(self, job: VideoJob, cap: cv2.VideoCapture, avi_path: Path, cache: VideoCache,
                      fingerprints: dict[str, str], num_frames: int, vid_w: int, vid_h: int,
                      video_fps: float) -> None:
        config = job._config
        order = self._pipeline.graph.stage_names
        args = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f"{vid_w}x{vid_h}", '-pix_fmt', 'bgr24', '-r', f"{video_fps:.6f}", '-i', '-',
                '-an', '-c:v', 'mpeg4', '-qscale:v', '1', str(avi_path)]
        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg failed: ffmpeg not found on PATH") from e

        frames = sorted(cache)
        cursor = 0
        live: deque[Box] = deque()
        broken = False
        try:
            for i in range(num_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                now = i / video_fps

                while live and live[0].end < now:
                    live.popleft()
                while cursor < len(frames) and frames[cursor] / video_fps <= now:
                    fid = frames[cursor]
                    live.extend(build_boxes(cached_detections(cache[fid], fingerprints), order, fid / video_fps,
                                            config))
                    cursor += 1

                frame = apply_censor_boxes(frame, list(live), config)
                if frame.shape[1] != vid_w or frame.shape[0] != vid_h:
                    frame = cv2.resize(frame, (vid_w, vid_h))
                try:
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    broken = True
                    break
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                broken = True
            proc.wait()

        if proc.returncode != 0 or broken:
            err = proc.stderr.read().decode(errors='replace').strip() if proc.stderr else ''
            raise RuntimeError(f"ffmpeg failed (status {proc.returncode}) writing {avi_path}: {err}")

    # ------------------------------------ encoding ------------------------------------
    @staticmethod
    def _write_video(avi_path: Path, video_path: str, output_path: str, has_audio: bool) -> None:
        args = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(avi_path)]
        if has_audio:
            args += ['-i', video_path, '-c:a', 'copy']
        args += ['-c:v', 'libx264', '-crf', '21', '-preset', 'veryfast', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
        if has_audio:
            args += ['-map', '0:0', '-map', '1:a', '-shortest']
        args.append(output_path)
        _run_ffmpeg(args)

    @staticmethod
    def _write_gif(avi_path: Path, output_path: Path) -> None:
        palette = output_path.parent / 'palette_tmp.png'
        try:
            _run_ffmpeg(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(avi_path),
                         '-vf', 'fps=15,scale=480:-1:flags=lanczos,palettegen', str(palette)])
            _run_ffmpeg(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(avi_path), '-i', str(palette),
                         '-filter_complex', 'fps=12,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse',
                         '-loop', '0', str(output_path)])
        finally:
            palette.unlink(missing_ok=True)
