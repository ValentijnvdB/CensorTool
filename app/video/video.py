import subprocess as sp
import time
from concurrent.futures._base import wait, FIRST_COMPLETED

from tqdm import tqdm

from app.config import CONFIG, load_censor_config_from_file

from .helpers import *
from .types import VideoJob
from .utils import video_file_has_audio

from . import ffmpeg
from ..image.image import ImagePipeline



MAX_ANALYZES_JOBS = 128

def censor_videos(
        video_paths: list[Path],
        output_dir: Path|None = None,
        override_cache: bool = False,
        skip_existing: bool = False,
        only_analyze: bool = False,
        censor_config: dict[str, Any] | Path | str = None) -> None:
    """
    Censor all images

    :param video_paths: the videos to censor.
    :param output_dir: the output directory, if None images are not written to file.
    :param override_cache: whether to overwrite cached versions.
    :param skip_existing: whether to skip already existing censored images (in the output directory).
    :param only_analyze: whether to only analyze the images for features and bodies.
    :param censor_config: the censoring configuration.
    """

    if isinstance(censor_config, dict):
        censor_config = CensorConfig(**censor_config)
    else:
        censor_config = load_censor_config_from_file(censor_config)


    for video_path in video_paths:
        vid_name = video_path.stem

        logger.info(f"Censoring video: {vid_name}")

        extension = video_path.suffix
        avi_path: Path = output_dir / (vid_name + '.avi')
        output_path: Path = output_dir / (vid_name + extension)
        if output_path.exists():
            if skip_existing:
                logger.info(f"{vid_name} already exists and skip_existing is True. Skipping...")
                continue
            suf = 1
            while output_path.exists():
                output_path = output_dir / f'{vid_name}_{suf}.{extension}'
                avi_path = output_dir / f'{vid_name}_{suf}.avi'
                suf += 1

        censor_video(VideoJob(
            video=video_path,
            avi_path=avi_path,
            output_path=output_path,
            override_cache=override_cache,
            sizes=CONFIG.picture_sizes,
            early_exit=only_analyze,
            config=censor_config
        ))


def censor_video(job: VideoJob):
    output_path = Path(job.output_path) if isinstance(job.output_path, str) else job.output_path
    avi_path = Path(job.avi_path) if isinstance(job.avi_path, str) else job.avi_path

    # test pre-conditions
    if job.video is None:
        raise ValueError("No video provided.")

    if not job.video.is_file():
        raise FileNotFoundError(f"Video file {job.video} not found")

    cap = cv2.VideoCapture()
    cap.open(str(job.video))
    if not cap.isOpened():
        raise RuntimeError("Cap was closed!")

    # retrieve video properties
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    vid_h, vid_w, ch = frame.shape
    num_frames = get_frame_count(job.video)
    if num_frames == -1:
        logger.warning(f"ffprobe failed to get the number of frames. Using estimate.")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    censor_fps = CONFIG.video_censor_fps

    ##########################################################################################
    ###################  Check cache and construct what we need to detect  ###################
    ##########################################################################################

    if 0 >= censor_fps or censor_fps > vid_fps:
        censor_fps = vid_fps

    raw_boxes_per_pass, bodies_per_pass, cache_paths = check_cache(job.video, job.sizes, job.override_cache)

    assert len(bodies_per_pass) == len(job.sizes)
    assert len(raw_boxes_per_pass) == len(job.sizes)
    assert len(cache_paths) == len(job.sizes)

    # On which frames we need to run the models and at which size.
    # If the frame is in needs_to_detect,then we need to run the model at its value.
    # frame_id -> list[sizes]
    needs_to_detect: dict[int, list[int]] = {}
    for size, raw_boxes, bodies in zip(job.sizes, raw_boxes_per_pass, bodies_per_pass):
        # iterate over all frames that need to be analyzed
        for frame_id in censor_frames(censor_fps, vid_fps, num_frames):
            if frame_id not in raw_boxes or frame_id not in bodies:
                if frame_id not in needs_to_detect:
                    needs_to_detect[frame_id] = [size]
                else:
                    needs_to_detect[frame_id].append(size)

    # size -> index
    sizes_dict = {}
    for i, size in enumerate(job.sizes):
        sizes_dict[size] = i

    ##########################################################################################
    ##############  Analyze the frames and apply censor as the results come in  ##############
    ##########################################################################################

    # setup temp video ffmpeg command
    if CONFIG.debug:
        command_base = [ffmpeg.get_ffmpeg(), '-y']
    else:
        command_base = [ffmpeg.get_ffmpeg(), '-y', '-loglevel', 'error']

    logger.info(f"Analyzing frames...")

    if needs_to_detect != {}:
        # run detection
        raw_boxes_per_pass, bodies_per_pass = run_detection(
            needs_to_detect=needs_to_detect,
            sizes_dict=sizes_dict,
            raw_boxes_per_pass=raw_boxes_per_pass,
            bodies_per_pass=bodies_per_pass,
            cap=cap,
            vid_fps=vid_fps)

        for raw_boxes, bodies, path in zip(raw_boxes_per_pass, bodies_per_pass, cache_paths):
            write_cache(path, raw_boxes, bodies)

    if job.early_exit:
        return job

    logger.info("Applying censor...")

    apply_censor(
        num_frames=num_frames,
        vid_w=vid_w,
        vid_h=vid_h,
        vid_fps=vid_fps,
        avi_path=avi_path,
        cap=cap,
        command_base=command_base,
        raw_boxes_per_pass=raw_boxes_per_pass,
        bodies_per_pass=bodies_per_pass,
        censor_config=job.config
    )

    ##################################################################################
    ##############  Re-encode the censored video to mp4 and with sound  ##############
    ##################################################################################

    logger.info("Censor applied, re-encoding to final output...")

    has_audio = video_file_has_audio(job.video)

    extension = output_path.suffix
    if extension == ".gif":
        write_gif(command_base=command_base,
                  avi_path=avi_path,
                  output_path=output_path)
    else:
        write_video(command_base=command_base,
                    avi_path=avi_path,
                    output_path=output_path,
                    video_path=job.video,
                    has_audio=has_audio)

    if avi_path.exists():
        avi_path.unlink()

    job.result = output_path
    return job


def run_detection(
        needs_to_detect: dict[int, list[int]],
        sizes_dict: dict[int, int],
        raw_boxes_per_pass: list[dict[int, list[RawBox]]],
        bodies_per_pass: list[dict[int, list[Polygon]]],
        cap: cv2.VideoCapture,
        vid_fps: float):
    """ Run the detection pipeline on all required frames. """
    with (ImagePipeline(max_workers=4) as pipeline):
        futures = {}

        # create job for each frame.
        frames: deque[int] = deque()
        for f in needs_to_detect.keys():
            frames.append(f)
        num_frames_to_detect = len(needs_to_detect.keys())
        current_frame = 0

        def add_frame(frame_num: int) -> bool:
            """Submit a frame to the pipeline."""
            # This function may look a bit weird, because we are reading every frame from the video, even if it is not
            # needed for analyzes. I found this approach was as much as 2.5x as fast vs skipping through the
            # frames. Probably something to do with how the cv2.VideoCapture works under the hood.
            _ret, _frame = cap.read()
            if not _ret:
                return False

            next_expected = frames.popleft()
            if frame_num != next_expected:
                frames.appendleft(next_expected)
                return True

            sizes = needs_to_detect[frame_num]

            frame_job = Job(
                job_id=frame_num,
                image=_frame,
                early_exit=True,
                sizes=sizes,
                timestamp=frame_num / vid_fps,
                skip_cache_write=True
            )
            futures[frame_num] = pipeline.submit(frame_job)
            return True

        def create_jobs():
            nonlocal current_frame
            while frames and len(futures) < MAX_ANALYZES_JOBS:
                if not cap.isOpened():
                    break
                success = add_frame(current_frame)
                current_frame += 1
                if not success:
                    break

        # create the initial jobs
        create_jobs()

        # --- wait for the image analyzes to come back ---
        pbar = tqdm(total=num_frames_to_detect)
        per_job_times = []
        while futures:

            # Block until at least one analyzes job is done
            done, _ = wait(futures.values(), return_when=FIRST_COMPLETED)

            # look for the finished job with the lowest frame_number
            for job_id, future in futures.items():
                if future in done:
                    completed: Job = future.result()
                    del futures[job_id]
                    break

            # create new jobs
            create_jobs()

            if not completed.success:
                # If the job failed, the frame is added the buffered, but no boxes are added.
                logger.warning(f"Frame {completed.job_id} failed.")
                continue

            frame_id = completed.job_id
            result = completed.result
            assert len(result.features) == len(needs_to_detect[frame_id])
            assert len(result.bodies) == len(needs_to_detect[frame_id])

            for raw_boxes, bodies, size in zip(result.features, result.bodies, needs_to_detect[frame_id]):
                index = sizes_dict[size]
                raw_boxes_per_pass[index][frame_id] = raw_boxes
                bodies_per_pass[index][frame_id] = bodies

            pbar.update(1)
            per_job_times.append(('Add to data struct', time.perf_counter_ns()))

    pbar.close()

    return raw_boxes_per_pass, bodies_per_pass


def apply_censor(num_frames: int,
                 vid_w: int,
                 vid_h: int,
                 vid_fps: float,
                 avi_path: Path,
                 cap: cv2.VideoCapture,
                 command_base: list[str],
                 raw_boxes_per_pass: list[dict[int, list[RawBox]]],
                 bodies_per_pass: list[dict[int, list[Polygon]]],
                 censor_config: CensorConfig):
    """ Convert the raw_boxes as outputted by the detection loop and apply them to the frames """

    boxes = process_raw_data(bodies_per_pass, raw_boxes_per_pass, censor_config)

    command = command_base + [
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{vid_w}x{vid_h}',
        '-pix_fmt', 'bgr24',
        '-r', '%.6f' % vid_fps,
        '-i', '-',
        '-an',
        '-c:v', 'mpeg4',
        '-qscale:v', '1',
        str(avi_path)
    ]

    proc = sp.Popen(command, stdin=sp.PIPE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    i = 0
    live_boxes: deque[Box] = deque()
    pbar = tqdm(total=num_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = i / vid_fps
        while live_boxes and live_boxes[0].end < curr_time:
            live_boxes.popleft()

        while boxes and boxes[0].start <= curr_time:
            live_boxes.append(boxes.popleft())

        censored_frame = censor.censor_image_from_boxes(frame, live_boxes, censor_config)

        proc.stdin.write(censored_frame.tobytes())
        i += 1
        pbar.update(1)

    pbar.close()
    proc.stdin.close()
    proc.wait()


def write_video(
        command_base: list[str],
        has_audio: bool,
        avi_path: Path,
        video_path: Path,
        output_path: Path):
    """Convert avi_path into a video (format depends on output_path.suffix)"""

    if has_audio:
        command = command_base + [
            '-stats',
            '-i', str(avi_path),
            '-i', str(video_path),
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-crf', '21',
            '-preset', 'veryfast',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-map', '0:0',
            '-map', '1:a',
            '-shortest',
            str(output_path)
        ]
    else:
        command = command_base + [
            '-stats',
            '-i', str(avi_path),
            '-c:v', 'libx264',
            '-crf', '21',
            '-preset', 'veryfast',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            str(output_path)
        ]

    logger.debug(command)

    proc2 = sp.Popen(command)
    proc2.wait()


def write_gif(
        command_base: list[str],
        avi_path: Path,
        output_path: Path):
    """Convert avi_path into a GIF."""

    palette_file = constants.temp_path / "palette.png"

    # first generate a palette for a nicer end result
    command = command_base + [
        '-i', str(avi_path),
        '-vf', 'fps=15,scale=480:-1:flags=lanczos,palettegen', str(palette_file)
    ]

    proc2 = sp.Popen(command)
    proc2.wait()

    # convert the video
    command = command_base  + [
        '-i', str(avi_path),
        '-i', str(palette_file),
        '-filter_complex', 'fps=12,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse',
        '-loop', '0',
        str(output_path)
    ]

    proc3 = sp.Popen(command)
    proc3.wait()

    # clean up
    if palette_file.exists():
        palette_file.unlink()

