# HTTP API

Start the server with

```
python main.py http [--host localhost] [--port 8443] [--ssl --ssl-cert cert.pem --ssl-key key.pem]
```

It listens on `http://localhost:8443` by default (`https://` with `--ssl`). All endpoints allow cross-origin requests
(CORS), so they can be called from a browser extension such as
[CensorExtension](https://github.com/ValentijnvdB/CensorExtension). Requests can be up to 100 MB.

The models in the profile's `pipeline.yml` are loaded once at startup. Requests that don't send their own censor
config use the profile's `censor_config.yml`.

| Endpoint                                              | Method           | Purpose                                            |
|-------------------------------------------------------|------------------|----------------------------------------------------|
| [`/censor_image`](#post-censor_image)                 | POST (JSON)      | Censor one image.                                  |
| [`/censor_images`](#post-censor_images)               | POST (JSON)      | Censor several images in one request.              |
| [`/detect_features`](#post-detect_features)           | POST (JSON)      | Detect features without censoring.                 |
| [`/censor_video`](#post-censor_video)                 | POST (multipart) | Censor a video or GIF and wait for it.             |
| [`/videos`](#post-videos)                             | POST (multipart) | Start censoring a video or GIF in the background.  |
| [`/videos`](#get-videos)                              | GET              | All background video jobs.                         |
| [`/videos/<id>`](#get-videosid)                       | GET              | The state and progress of a video job.             |
| [`/videos/<id>/result`](#get-videosidresult)          | GET              | The censored video of a finished job.              |
| [`/videos/<id>`](#delete-videosid)                    | DELETE           | Cancel a video job and remove it.                  |
| [`/video_formats`](#get-video_formats)                | GET              | The formats `/censor_video` can return.            |
| [`/censor_frame`](#get-censor_frame)                  | GET (WebSocket)  | Censor a stream of frames in real time.            |
| [`/health`](#get-health)                              | GET              | Whether the server can censor, and how busy it is. |
| [`/cache`](#get-cache)                                | GET              | Size of the server's detection cache.              |
| [`/cache`](#delete-cache)                             | DELETE           | Delete the server's detection cache.               |
| [`/profile`](#get-profile)                            | GET              | The loaded profile and the labels it can detect.   |
| [`/censor_config`](#get-censor_config)                | GET              | The censor config used when a request has none.    |
| [`/censor_config/schema`](#get-censor_configschema)   | GET              | A JSON Schema of the censor config.                |
| [`/censor_config/validate`](#post-censor_configvalidate) | POST (JSON)   | Check a censor config without censoring.           |
| [`/stickers`](#get-stickers)                          | GET              | The sticker categories and images.                 |
| [`/stickers/<path>`](#get-stickerspath)               | GET              | One sticker image.                                 |
| `/censored/<name>`                                    | GET              | Censored images saved with `expected_response: url`. |
| [`/storage`](#get-storage)                            | GET              | Files and bytes the server keeps.                  |
| [`/storage`](#delete-storage)                         | DELETE           | Remove the files the server keeps.                 |
| `/assets/<name>`                                      | GET              | Static assets (status images).                     |

## Errors

Errors are returned as JSON with a status code, a machine-readable `code` and a message:

```json
{"status": "error", "code": "invalid_config", "error": "Invalid censor config: Missing required key: features_to_censor"}
```

| `code`               | Status | Meaning                                                                          |
|----------------------|--------|----------------------------------------------------------------------------------|
| `invalid_request`    | 400    | The request is malformed: not JSON, a missing or wrong field.                    |
| `invalid_image`      | 400    | The image is empty, not valid base64, not an image, or of an unsupported type.   |
| `invalid_video`      | 400    | The video cannot be read.                                                        |
| `use_censor_video`   | 400    | A GIF was sent to an image endpoint; send it to [`/censor_video`](#post-censor_video). |
| `invalid_config`     | 400    | The censor config is not valid (see [`/censor_config/validate`](#post-censor_configvalidate)). |
| `config_not_allowed` | 400    | The censor config has a `stickers_dir` or sticker outside `data/stickers`.       |
| `url_not_allowed`    | 400    | The image or video URL is not `http(s)`, or points to this machine or the local network. |
| `not_found`          | 404    | No such sticker or video job.                                                    |
| `not_ready`          | 409    | The video job has not finished yet.                                              |
| `job_failed`         | 409    | The video job failed or was cancelled, so it has no result.                      |
| `queue_full`         | 429    | Too many video jobs are waiting or running; try again later.                     |
| `download_failed`    | 502    | The image or video URL could not be downloaded (unknown host, timeout, an HTTP error, larger than 100 MB). |
| `processing_failed`  | 500    | Censoring or detection failed.                                                   |
| `internal_error`     | 500    | Anything else.                                                                   |
| `unavailable`        | 503    | The pipeline is not running.                                                     |

In a `config` sent with a request, the ComfyUI settings of `ai_remove` styles (`comfy_host`, `comfy_port`,
`comfy_workflow`) are always replaced by the ones in `config.yml`.

## Image requests

`/censor_image` and `/detect_features` take a JSON body (`Content-Type: application/json`) in one of two forms.

**The image itself, base64 encoded**

```json
{
  "type": "base64",
  "image_data": "<base64 encoded image>",
  "image_url": "https://example.com/picture.jpg",
  "mime_type": "image/jpeg"
}
```

`image_url` is used to name the file on the server (and to find the image type when `mime_type` is missing); the
server does not download it. It can be left out when `mime_type` is given; the file is then named after its contents.

**A URL the server downloads**

```json
{
  "type": "source",
  "image_url": "https://example.com/picture.jpg"
}
```

The image type is taken from the extension in the URL. Only `http` and `https` URLs on the internet are downloaded:
URLs that point to this machine (`localhost`, `127.0.0.1`, `[::1]`, `0.0.0.0`, ...) or the local network
(`192.168.x.x`, `10.x.x.x`, link-local addresses, ...) are refused with `url_not_allowed`, also when a host name
resolves to such an address or a redirect leads there. Downloaded images are kept in `data/server/uncensored/` and
reused when the same URL is requested again, until they are [removed](#stored-files) for not being used.

Supported image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp` and `.avif`. Extensions are
case-insensitive.

GIFs are censored as videos: send them to [`/censor_video`](#post-censor_video), as a file or by URL
(`video_url`). The image endpoints answer a GIF with `400` and code `use_censor_video`, also when it is sent with
another extension or `mime_type`. A `source` URL ending in `.gif` is refused before it is downloaded.

**Optional fields**

| Field                | Default    | Description                                                                 |
|----------------------|------------|-----------------------------------------------------------------------------|
| `config`             | profile's  | A censor config as JSON, same structure as `censor_config.yml` (see [censor_config.md](censor_config.md)). It is checked before the image is read. |
| `expected_response`  | `base64`   | `/censor_image` only: `base64`, `bytes` or `url`, see below.                |
| `include_detections` | `false`    | `/censor_image` only: also return what was detected, see below.             |

## POST /censor_image

Censors the image and returns it in the format set by `expected_response`:

| `expected_response` | Response                                                                                      |
|---------------------|-----------------------------------------------------------------------------------------------|
| `base64`            | JSON with `image_data` (base64) and `mime_type` (e.g. `image/jpeg`).                          |
| `bytes`             | The image file itself, as an attachment, with its own Content-Type.                           |
| `url`               | JSON with `image_name` and `url`. The image is saved on the server before the answer is sent; fetch it from `url` (`/censored/<name>`). It is [removed](#stored-files) once it has not been fetched for `server_file_retention_hours` (24 by default); after that the URL gives a `404`. |

The JSON answers look like this:

```json
{
  "status": "success",
  "image_data": "<base64>",
  "mime_type": "image/jpeg",
  "width": 1280,
  "height": 720,
  "processing_ms": 42
}
```

- `width` and `height` are the size of the returned image in pixels.
- `processing_ms` is the time the server spent on the request, including downloading the image.
- The `bytes` answer sends these as the headers `X-Image-Width`, `X-Image-Height` and `X-Processing-Time-Ms`.

With `"include_detections": true` the JSON answers also have `detections` and `cached_models`, as in
[`/detect_features`](#post-detect_features). `include_detections` cannot be combined with `bytes` (`400`).

Example:

```
curl -X POST http://localhost:8443/censor_image \
     -H "Content-Type: application/json" \
     -d '{"type": "source", "image_url": "https://example.com/picture.jpg", "expected_response": "url"}'
```

## POST /censor_images

Censors up to 50 images in one request; they are downloaded and censored at the same time. `images` is a list of
[image requests](#image-requests). `config`, `expected_response` and `include_detections` can be given once for all
images, and per image to override that. `expected_response` must be `base64` or `url`.

```json
{
  "images": [
    {"type": "source", "image_url": "https://example.com/a.jpg"},
    {"type": "source", "image_url": "https://example.com/b.png", "include_detections": true}
  ],
  "expected_response": "url",
  "config": {"features_to_censor": ["face"], "default_censor_config": {"censor_style": {"type": "blur"}}}
}
```

The answer has one result per image, in the same order. Each is the answer `/censor_image` would give, or an
[error](#errors) for that image only; one failing image does not fail the others:

```json
{
  "status": "success",
  "processing_ms": 230,
  "results": [
    {"status": "success", "image_name": "<name>.jpg", "url": "/censored/<name>.jpg", "width": 800, "height": 600, "processing_ms": 180},
    {"status": "error", "code": "download_failed", "error": "Could not download the image: the server answered 404."}
  ]
}
```

A request that is wrong as a whole (no `images` list, more than 50 images, an invalid shared `config`) gets a `400`.

## POST /detect_features

Runs the models without censoring and returns what was found:

```json
{
  "status": "success",
  "file_name": "<name of the stored image>",
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "id": 0,
      "label": "face",
      "score": 0.92,
      "box": [x, y, width, height],
      "polygons": [[[x, y], [x, y], ...]],
      "model": "nevermore",
      "parent": null
    }
  ],
  "cached_models": ["nevermore"]
}
```

- `width` and `height` are the size of the image in pixels; `box` and `polygons` are in these pixels.
- `polygons` holds the outline (largest part first) for segmentation labels such as `BODY` and for eyes; it is empty
  for box-only detections.
- `model` is the model that found the detection (its `name` in `pipeline.yml`, or `landmarks`).
- `parent` is the `id` of the detection it was found in (eyes in a face, cascades), otherwise `null`.
- `cached_models` lists the models whose results came from the cache.

Without a `config` every model in `pipeline.yml` runs. With a `config` only the models that config needs run.

## POST /censor_video

Censors a video or a GIF. A `multipart/form-data` request with these parts:

| Part        | Required | Description                                       |
|-------------|----------|---------------------------------------------------|
| `video`     | one of   | The video file.                                   |
| `video_url` | these    | A URL the server downloads the video from, instead of sending the file. |
| `filename`  | no       | Name of the video, e.g. `clip.mp4`, without folders (`400` otherwise). With `video`, send it before the `video` part. Default: the name the file was uploaded with, or the file name at the end of `video_url`, else `video.mp4`. |
| `config`   | no       | A censor config as JSON text.                     |
| `format`   | no       | Format of the censored video: `mp4`, `m4v`, `mov`, `mkv`, `webm`, `avi` or `gif` (a leading `.` and capitals are fine; `400` otherwise). Default: the format of `filename`, or `mp4` when that is not one of these. See also [`/video_formats`](#get-video_formats). |

The request must be `multipart/form-data`. Sending both `video` and `video_url`, or neither, is a `400`, as is an empty video or one that cannot be read
(`invalid_video`).

`video_url` is downloaded like a `source` image (see [Image requests](#image-requests)): only `http` and `https`
URLs on the internet, never this machine or the local network (`url_not_allowed`), up to 100 MB. A failed download
is a `502` with code `download_failed`. Downloaded videos are not reused: the same URL is downloaded again, but its
detections come from the cache.

The response is the censored video, streamed back with the Content-Type of its format (e.g. `video/mp4`, `video/webm`,
`image/gif`). The request returns when the whole video is done. Videos are censored one at a time; meanwhile the
server keeps answering other requests.

The response has these headers:

| Header                 | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| `Content-Disposition`  | `attachment; filename="<name>"`, the name of the censored video on the server. |
| `X-Processing-Time-Ms` | The time the server spent on the request, including the upload.     |
| `X-Frame-Count`        | Frames in the uploaded (or downloaded) video.                       |
| `X-Video-Fps`          | Frame rate of the uploaded video.                                   |
| `X-Video-Duration`     | Duration of the uploaded video in seconds.                          |
| `X-Video-Width`, `X-Video-Height` | Size of the uploaded video in pixels. A `gif` output is scaled to 480 pixels wide and 12 fps, so it can differ. |

The uploaded (or downloaded) video is saved in `data/server/uncensored/` and the censored one in
`data/server/censored/`, under a unique name: when a name is taken, `_1`, `_2`, ... is added (`clip.mp4`,
`clip_1.mp4`, ...), so requests never overwrite each other's files. Both files are removed once the response is sent
(or the request fails). Detections are cached per video content, so sending the same video again is faster.

The video is encoded as H.264 (VP9 for `webm`). The audio is kept as it is when the format does not change (and always
for `mkv`); otherwise it is re-encoded to AAC (`mp4`, `m4v`, `mov`), Opus (`webm`) or MP3 (`avi`). A `gif` has no audio.

```
curl -X POST http://localhost:8443/censor_video \
     -F filename=clip.mp4 -F video=@clip.mp4 -o clip_censored.mp4

curl -X POST http://localhost:8443/censor_video \
     -F filename=clip.mp4 -F format=webm -F video=@clip.mp4 -o clip_censored.webm

curl -X POST http://localhost:8443/censor_video \
     -F video_url=https://example.com/funny.gif -o funny_censored.gif
```

## Video jobs

`/censor_video` keeps the request open until the whole video is done. For long videos, censor them in the
background instead: start a job, poll its progress, then fetch the result.

```
curl -X POST http://localhost:8443/videos -F video=@clip.mp4          # -> {"job": {"id": "3f2a...", ...}}
curl http://localhost:8443/videos/3f2a...                             # poll until "state": "done"
curl http://localhost:8443/videos/3f2a.../result -o clip_censored.mp4
```

Videos are censored one at a time, in the order they were submitted, from both `/censor_video` and `/videos`.
A finished job (done, failed or cancelled) is kept for **1 hour**, then the job and its files (the uploaded and the
censored video) are removed. Jobs are kept in memory: restarting the server forgets them; their files are then
removed like other [unused files](#stored-files).

### POST /videos

Takes the same parts as [`/censor_video`](#post-censor_video) and answers `202 Accepted` straight away with the
[job](#the-job) and a `Location` header pointing to it:

```json
{"status": "success", "job": {"id": "3f2a9c...", "state": "queued", "url": "/videos/3f2a9c...", ...}}
```

- An uploaded video is checked before the answer: a missing, empty or unreadable video is a `400`, as with
  `/censor_video`.
- A `video_url` is downloaded in the background (state `downloading`); a failed download makes the job `failed`
  with code `download_failed` or `url_not_allowed`.
- At most 20 jobs can be downloading, waiting or running at the same time. Beyond that, the request is refused with
  `429` and code `queue_full` before the video is received.

### The job

```json
{
  "id": "3f2a9c...",
  "state": "running",
  "phase": "censoring",
  "progress": 0.82,
  "frames_done": 480,
  "frames_total": 600,
  "queue_position": null,
  "filename": "clip.mp4",
  "format": "mp4",
  "content_type": "video/mp4",
  "video": {"frames": 600, "fps": 30.0, "duration": 20.0, "width": 1280, "height": 720},
  "created_at": "2026-11-10T11:04:57Z",
  "started_at": "2026-11-10T11:04:58Z",
  "finished_at": null,
  "processing_ms": null,
  "expires_at": null,
  "url": "/videos/3f2a9c...",
  "result_url": null
}
```

| Field            | Description |
|------------------|-------------|
| `state`          | `downloading` (a `video_url`), `queued` (waiting for its turn), `running`, `done`, `failed` or `cancelled`. |
| `phase`          | While running: `detecting` (running the models), `censoring` (drawing the censor on every frame) or `encoding` (writing the file); `done` once done. |
| `progress`       | From `0` to `1`, estimated from the phase and its frames: detecting is 0-0.7, censoring 0.7-0.95, encoding 0.95-1. |
| `frames_done`, `frames_total` | Frames done in the current phase, and how many it has. Detection only counts the frames that are analysed (`video_censor_fps`) and not cached; `frames_total` is `0` when there is nothing to detect. |
| `queue_position` | While queued: `1` for the next video to run, `2` after that, ...; otherwise `null`. |
| `filename`       | Name of the censored video, once the video is there. |
| `video`          | Frame count, fps, duration (seconds) and size of the uploaded video, once it is there. |
| `processing_ms`  | Time from start to finish of the censoring, once finished. |
| `expires_at`     | When the finished job and its files are removed. |
| `result_url`     | Where to fetch the censored video, once done. |
| `error`, `code`  | Only when `failed`: why, with an [error code](#errors) such as `processing_failed` or `download_failed`. |

`phase`, `progress`, `frames_done` and `frames_total` need the Python backend. With the C++ backend (`censor_core`
2.0) they are `null` while the job runs; `state` and `queue_position` still work.

### GET /videos

Every job, oldest first: `{"status": "success", "jobs": [<job>, ...]}`.

### GET /videos/&lt;id&gt;

One job: `{"status": "success", "job": <job>}`. An unknown (or expired) id is a `404`.

### GET /videos/&lt;id&gt;/result

The censored video, streamed back with the same headers as [`/censor_video`](#post-censor_video). It can be fetched
any number of times until the job expires. Before the job is done the answer is `409` with code `not_ready`; for a
failed or cancelled job it is `409` with code `job_failed`.

### DELETE /videos/&lt;id&gt;

Removes the job and its files, and answers with the job as it was (`state` is `cancelled` for a job that had not
finished):

- A job that is downloading or waiting is stopped straight away; it never runs.
- A running job is stopped within a moment with the Python backend. The C++ backend cannot stop a running video: it
  runs to the end in the background (other videos wait for it), and its files are removed then.
- A finished job is removed with its files.

## GET /video_formats

The formats [`/censor_video`](#post-censor_video) can return, with their Content-Type:

```json
{
  "status": "success",
  "default": "mp4",
  "formats": [{"format": "mp4", "content_type": "video/mp4"}, {"format": "webm", "content_type": "video/webm"}, ...]
}
```

## GET /censor_frame

A WebSocket for real-time censoring, e.g. of a video playing in the browser. Each binary message is one frame:

```
[13-byte header][image bytes]
```

- The header is sent back unchanged with the censored frame, so the client can use it to match frames (e.g. a
  frame number and timestamp). The server only looks at the first byte: `1` means "cancel", any other value is a
  frame.
- The image is any encoded image (e.g. JPEG or WebP). Censored frames are returned as `[header][WebP bytes]`.
- Frames are censored with the profile's censor config, several at a time, and may come back out of order.
  Frames identical to one seen earlier are answered from memory.
- **Cancel**: a message whose first byte is `1` cancels every frame still in progress. Those frames are not sent
  back. Frames sent after it are censored as usual.
- **Errors**: a frame that cannot be censored (not an image, or censoring failed) is answered with a text message
  that has its header, as hex:

  ```json
  {"type": "error", "code": "processing_failed", "error": "Censoring failed: ...", "header": "00000000000000000000000007"}
  ```

Text messages are JSON objects with a `type`. Invalid JSON or an unknown `type` gets an `error` answer (without a
`header`); the connection stays open.

### Changing the censor config

Send a text message to censor the following frames with another censor config (same structure as the `config` of
the other endpoints). It can be sent at any time and applies to this connection only:

```json
{"type": "config", "config": {"features_to_censor": ["FACE_FEMALE"], "default_censor_config": {"censor_style": {"type": "blur"}}}}
```

Send `"config": null` to go back to the profile's censor config. The server answers with a text message:

```json
{"type": "config", "status": "success"}
{"type": "error", "code": "invalid_config", "error": "<message>"}
```

- The config is checked when it arrives. If it is not valid or not allowed (see [Errors](#errors)), the server
  answers with an `error` and keeps the previous config.
- Frames sent before the config message may still be in progress and come back censored with the previous config,
  also after the `success` answer. To tell them apart, put a counter in the header.
- After a change, frames seen earlier are censored again instead of answered from memory.

### Detections

To also get what was detected in each frame, send:

```json
{"type": "options", "include_detections": true}
```

The server answers `{"type": "options", "status": "success", "include_detections": true}`. From then on, every
censored frame is preceded by a text message with its detections (as in [`/detect_features`](#post-detect_features))
and the frame's header, as hex:

```json
{"type": "detections", "header": "00000000000000000000000007", "width": 1280, "height": 720, "detections": [...]}
```

Send `"include_detections": false` to stop.

### Statistics

Send `{"type": "stats"}` to get the statistics of this connection:

```json
{
  "type": "stats",
  "fps": 14.2,
  "active_jobs": 3,
  "remembered_frames": 120,
  "frames_received": 430,
  "frames_sent": 425,
  "frames_from_memory": 12,
  "frames_failed": 0,
  "frames_cancelled": 2
}
```

- `fps` is the number of frames sent back per second, over the last 10 seconds.
- `active_jobs` is the number of frames being censored now. A client can send fewer frames when it grows.

## GET /health

Whether the server can censor, and what it is busy with. Status `200` when the pipeline runs, `503` otherwise.

```json
{
  "status": "ok",
  "pipeline_running": true,
  "profile": "nudity",
  "backend": "python",
  "uptime_seconds": 3600.5,
  "pending_image_jobs": 2,
  "running_video_jobs": 1,
  "waiting_video_jobs": 0,
  "open_frame_streams": 1
}
```

- `status` is `ok` or `unavailable`.
- `backend` is `python` or `cpp`.
- `pending_image_jobs` counts the images and frames being censored or waiting for a worker.
- Videos are censored one at a time (from `/censor_video` and [video jobs](#video-jobs)): `running_video_jobs` is 0 or 1,
  `waiting_video_jobs` counts the ones waiting.
- `open_frame_streams` is the number of open [`/censor_frame`](#get-censor_frame) connections.

## Stored files

The server keeps files in `data/server/`: uploaded and downloaded originals (`uncensored/`), censored images
(`censored/`), the files of [video jobs](#video-jobs), and debug output (`debug/`, with `--debug`). The detection
cache is separate (see [`/cache`](#get-cache)).

A file is removed once it has not been used for `server_file_retention_hours` (in
[config.yml](config.md#http-server), 24 by default; `0` keeps files forever). Using a file means writing it,
requesting the same image again, or fetching a censored image from `/censored/<name>`, so an image a page keeps
showing is kept. The server looks for unused files when it starts and every 10 minutes. Files of video jobs are left
to the jobs, which remove them 1 hour after they finish; files that a restart left behind are removed like any other.

`/censor_video` removes its files straight away, once the video is sent.

## GET /storage

The files the server keeps, per folder of `data/server/`:

```json
{
  "status": "success",
  "server_dir": "data/server",
  "retention_hours": 24,
  "folders": {"censored": {"files": 120, "bytes": 31457280}, "uncensored": {"files": 118, "bytes": 30408704}},
  "files": 238,
  "bytes": 61865984
}
```

## DELETE /storage

Removes every file in `data/server/` straight away, except the files of video jobs the server still knows (remove
those with [`DELETE /videos/<id>`](#delete-videosid)). Censored images that pages still show then give a `404`.

```json
{"status": "success", "deleted_files": 238, "deleted_bytes": 61865984}
```

## GET /cache

The size of the server's detection cache (`cache/server/`):

```json
{"status": "success", "cache_dir": "cache/server", "files": 120, "bytes": 482133}
```

## DELETE /cache

Deletes everything in the server's detection cache and returns how much was deleted:

```json
{"status": "success", "deleted_files": 120, "deleted_bytes": 482133}
```

This replaces `GET /reset_cache`.

## GET /profile

Returns the name of the loaded profile and lists every label (class name) the models in the profile's `pipeline.yml` can produce, after `names`, `labels`,
`label_prefix` and `classes` are applied. These are the labels a censor config can refer to.

```json
{
  "status": "success",
  "profile": "nudity",
  "labels": ["EYES_FEMALE", "FACE_FEMALE", "FACE_MALE"],
  "models": {"EYES_FEMALE": "landmarks", "FACE_FEMALE": "nudenet", "FACE_MALE": "nudenet"}
}
```

- `profile` is the name of the profile directory the server was started with.
- `labels` is sorted alphabetically.
- `models` maps each label to the model that produces it (its `name` in `pipeline.yml`, or `landmarks`).

## GET /censor_config

Returns the censor config the server uses when a request does not provide one: the profile's `censor_config.yml`
as it was written (not parsed, so left-out settings are not filled in with their defaults). It can be changed and
sent back as `config` in a request.

```json
{
  "status": "success",
  "profile": "nudity",
  "config": {
    "features_to_censor": ["FACE_FEMALE", "FEMALE_BREAST_EXPOSED"],
    "default_censor_config": {"censor_style": {"type": "bar", "color": [0, 0, 0]}, "min_probability": 0.25},
    "feature_overrides": {"FACE_FEMALE": {"censor_style": {"type": "pixel", "factor": 20}}}
  }
}
```

- `stickers_dir` is always included, relative to the working directory (`data/stickers` when the file leaves it out).
- The file is read once, when the server starts. Changes to it take effect after a restart.
- The ComfyUI settings (`comfy_host`, `comfy_port`, `comfy_workflow`) are left out of `ai_remove` censor styles.
  They are not needed when the config is sent back: the server always uses its own.

## GET /censor_config/schema

A [JSON Schema](https://json-schema.org/) (draft 2020-12) of the censor config: every setting with its type,
`default`, allowed values and a `description`. A client can use it to build a settings form, or to check a config
with any JSON Schema library. The answer is the schema itself (no `status` field).

The schema describes the format; [`/censor_config/validate`](#post-censor_configvalidate) is the authority on
whether a config works with the loaded models. The schema is stricter in two places: shapes are listed in lower
case (the server accepts any case), and unknown keys are not allowed (the server ignores them with a warning).

## POST /censor_config/validate

Checks a censor config the same way a censor request does, without censoring anything:

```json
{"config": {"features_to_censor": ["fcae"], "default_censor_config": {"censor_style": {"type": "pixel"}}}}
```

```json
{
  "status": "success",
  "valid": true,
  "errors": [],
  "warnings": ["No model produces label 'fcae' (did you mean 'face'?); it is never censored."]
}
```

- `valid` is `false` when the config would be rejected by a censor request; `errors` then says why and `code` is
  `invalid_config` or `config_not_allowed`. The status is `200` either way.
- `warnings` lists what the server ignores or fills in: unknown keys, labels no model produces, left-out settings
  that get a default.
- `clip_to` labels that no model produces are errors. With the Python backend, `shape: mask` and `clip_to` on a
  label of a detect (not segmentation) model are errors too; the C++ backend reports those only when censoring.

## GET /stickers

Lists the sticker categories and images in `data/stickers`. A category is any subfolder, nested ones included; a
category also picks the images in its own subfolders. Only images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`,
`.tiff`) are listed. Other files are never used as stickers.

```json
{
  "status": "success",
  "stickers_dir": "data/stickers",
  "categories": ["misc", "plants", "plants/flowers"],
  "stickers": [
    {"path": "misc/star.png", "url": "/stickers/misc/star.png"},
    {"path": "plants/fern.png", "url": "/stickers/plants/fern.png"},
    {"path": "plants/flowers/rose.png", "url": "/stickers/plants/flowers/rose.png"}
  ]
}
```

- `?dir=<stickers_dir>` lists a subfolder instead, as it would be written as `stickers_dir` in a censor config, e.g.
  `/stickers?dir=data/stickers/plants`. `categories` and `path` are then relative to that folder, so they can be used
  as `overlay.stickers` entries with that `stickers_dir`. A `dir` outside `data/stickers` is rejected with status `400`.
- `url` is where [`GET /stickers/<path>`](#get-stickerspath) serves the image (always relative to `data/stickers`).
- The folder is read on every request, so stickers added while the server runs are listed straight away.
- A folder that does not exist (yet) gives empty lists.

## GET /stickers/&lt;path&gt;

Returns one sticker image, by its path relative to `data/stickers` (the `url` in [`GET /stickers`](#get-stickers)).
Anything that is not a sticker image inside `data/stickers` gets status `404`.
