import time
import traceback

import constants

from .helpers import *
from .pipeline_types import *
from .pipeline import GenericPipeline

def _process_one(job: Job, gpu_queue: Queue[GPURequest]) -> Job:
    """
    Runs inside a ThreadPoolExecutor worker. Mutates job.success / .error /
    .result in-place and also returns job so the Future carries it.

    Steps 1–3:  load → cache check → prepare variants
    Steps 4–5:  hand off to GPU thread (blocks on reply_queue)
    Step  6-7:  post-process → write to cache
    [early_exit=True → return here with unmodified image + features/bodies]
    Steps 8–10: modify image → save → return full result
    """
    times = [('start', time.perf_counter_ns())]

    def cancel_job():
        job.success = False
        job.time_taken = times
        return job

    output_path = None
    if job.output_path is not None:
        output_path = Path(job.output_path)

    if job.image is None:
        raise ValueError("No image provided.")

    try:
        # --- Step 1: load ---
        image = job.image
        if isinstance(image, bytes):
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        elif isinstance(image, str|Path):
            image = load_image(job.image)

        times.append( ('load_image', time.perf_counter_ns()) )
        if job.cancelled.is_set():
            return cancel_job()

        # --- Step 2: cache check + light preprocessing ---
        cache_base_dir: Path = job.cache_base_dir if job.cache_base_dir is not None else constants.image_cache_path
        cached = check_cache(image, job.sizes, cache_base_dir, job.override_cache)

        # check what we need to do
        need_features: list[bool] = []
        need_bodies: list[bool] = []
        for c in cached:
            need_features.append(c.features is None)
            need_bodies.append(c.bodies is None)

        times.append( ('check_cache', time.perf_counter_ns()) )
        if job.cancelled.is_set():
            return cancel_job()

        # --- Step 3: prepare image variants (only if we need inference) ---
        if any(need_features) or any(need_bodies):
            adj_images, scales = prepare_image_variants(image, job.sizes)

            times.append( ('prepare image', time.perf_counter_ns()) )
            if job.cancelled.is_set():
                return cancel_job()

        # --- Steps 4–5: GPU inference (skipped if fully cached) ---
            data = PreprocessedData(adj_images=adj_images, scales=scales)
            request = GPURequest(data=data, need_features=need_features, need_bodies=need_bodies)
            gpu_queue.put(request)
            gpu_result: GPUResult = data.reply_queue.get()  # blocks until GPU thread responds

            if gpu_result.error:
                raise gpu_result.error

            features = gpu_result.raw_features
            bodies = gpu_result.raw_bodies

            times.append( ('run models', time.perf_counter_ns()) )
            if job.cancelled.is_set():
                return cancel_job()

        # --- Step 6: post-process and Step 7 write to cache---
            cached = postprocess(image=image,
                                 adj_images=adj_images,
                                 scales=scales,
                                 timestamp=job.timestamp,
                                 needs_to_detect_features=need_features,
                                 needs_to_detect_bodies=need_bodies,
                                 raw_nn_output=features,
                                 raw_hd_output=bodies,
                                 cached_items=cached,
                                 skip_cache_write=job.skip_cache_write)


            times.append( ('post process model output', time.perf_counter_ns()) )

        # --- Early exit: return processed data, skip image modification + save ---
        if job.early_exit:
            job.result = ProcessedResult(
                image=image,  # original, unmodified
                output_path=None,  # save was skipped
                features=[c.features for c in cached],
                bodies=[c.bodies for c in cached],
            )
            job.success = True
            job.time_taken = times
            return job

        if job.cancelled.is_set():
            return cancel_job()

        # --- Steps 8–9: modify image + write to disk ---
        modified_image = apply_censor(image, cached, job.output_path, job.config)

        if job.return_bytes:
            modified_image = bytes(cv2.imencode(ext=job.output_extension, img=modified_image)[1])

        times.append(('apply censor', time.perf_counter_ns()))

        # --- Step 10: return ---
        job.result = ProcessedResult(
            image=modified_image,
            output_path=output_path,
            features=[c.features for c in cached],
            bodies=[c.bodies for c in cached],
        )
        job.success = True
        job.time_taken = times

    except Exception as exc:
        job.success = False
        job.error = exc
        job.stacktrace = traceback.format_exc()

    return job


class ImagePipeline(GenericPipeline):
    """
        Thread-safe image processing pipeline.

        Usage (batch, paths known upfront):
            with ImagePipeline(max_workers=4) as pipeline:
                done = pipeline.process_batch(jobs)

        Usage (streaming):
            pipeline = ImagePipeline(max_workers=4)
            pipeline.start()
            ...
            future = pipeline.submit(job)
            done_job = future.result()   # block, or store future and check later
            ...
            pipeline.stop()
        """
    def __init__(self, max_workers: int = 4) -> None:
        super().__init__(_process_one, max_workers)
