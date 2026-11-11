# config.yml

General settings: video sampling, the live capture region and the ComfyUI connection for `ai_remove`.
Every profile needs one; copy it from `configs/nudity/config.yml` when you make a new profile.

While live or webcam censoring runs, changes to this file are applied as soon as it is saved (press `r` in the
window to force a reload). An invalid file is reported and the previous settings are kept.

Models, devices, workers and input sizes are not set here but in [pipeline.yml](pipeline_config.md).

## Video

| Key                | Default | Description |
|--------------------|---------|-------------|
| `video_censor_fps` | `15`    | How many frames per second are run through the models. The censor boxes of an analysed frame are used for the frames after it (see `time_safety` in the [censor config](censor_config.md#censor-box)). Higher is more accurate but slower. `0` analyses every frame. |

## Live and webcam

| Key                     | Default           | Description |
|-------------------------|-------------------|-------------|
| `live_cap_top`          | `0`               | Top of the captured screen region, in pixels. |
| `live_cap_left`         | `0`               | Left of the captured screen region. |
| `live_cap_width`        | `1920`            | Width of the captured region. |
| `live_cap_height`       | `1080`            | Height of the captured region. |
| `live_show_fps`         | `false`           | Show an fps counter in the output. |
| `live_cursor_color`     | `[168, 93, 253]`  | Color (RGB) of the replacement cursor drawn in live mode. |
| `live_input_size`       | `0`               | Model input size for live frames, a multiple of 32. `0` uses the model's own size. Smaller (e.g. `480`, `416`) is faster but finds small body parts less reliably. Only affects models with a dynamic input size (`resize: rect`), such as NudeNet. Optional. |
| `live_frames_in_flight` | `2`               | Frames censored at the same time. `2` overlaps capturing, censoring and displaying (about 1.5-2x the fps on a GPU) at the cost of one frame of latency. `1` gives the lowest latency. Optional. |

Live mode captures the region above and shows it censored in a window (or a virtual camera with `--vcam`).
It works best with two displays: the content on one, the censored window on the other.

## ai_remove (ComfyUI)

Used by the `ai_remove` censor style, which inpaints the censored area with a ComfyUI workflow. These values are
filled in for every `ai_remove` style in the censor config that doesn't set them itself.

| Key              | Default                          | Description |
|------------------|----------------------------------|-------------|
| `comfy_host`     | `"localhost"`                    | Host of the ComfyUI server. |
| `comfy_port`     | `8188`                           | Port of the ComfyUI server. |
| `comfy_workflow` | `"./data/workflow.json"`         | Workflow in ComfyUI's API format. It needs a node titled `IMAGE` and one titled `MASK` (both image loaders); the censored image and the mask of the censored area are put in those. Not included with Censor Tool. |

## HTTP server

| Key                           | Default | Description |
|-------------------------------|---------|-------------|
| `server_file_retention_hours` | `24`    | Files the HTTP server keeps in `data/server/` (uploaded, downloaded and censored images, video files) are removed once they have not been used for this many hours. Fetching a censored image counts as a use. `0` keeps them forever. Optional. See [Stored files](http_api.md#stored-files). |
| `server_video_job_grace_seconds` | `30` | A [video job](http_api.md#video-jobs) shares its work with the other jobs for the same video. Work that is still running when its last job is deleted (e.g. the page with the video was closed) goes on for this many seconds, so a reloaded page can take it over. `0` stops it at once. Optional. |
| `server_max_video_download_mb` | `500` | The largest video the server downloads for a `video_url` (`/censor_video` and `/videos`), in MB. Larger videos fail with `download_failed`. Optional. |
