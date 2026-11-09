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

| Endpoint                                | Method          | Purpose                                         |
|-----------------------------------------|-----------------|-------------------------------------------------|
| [`/censor_image`](#post-censor_image)   | POST (JSON)     | Censor one image or GIF.                        |
| [`/detect_features`](#post-detect_features) | POST (JSON) | Detect features without censoring.             |
| [`/censor_video`](#post-censor_video)   | POST (multipart)| Censor a video.                                 |
| [`/censor_frame`](#get-censor_frame)    | GET (WebSocket) | Censor a stream of frames in real time.         |
| [`/reset_cache`](#get-reset_cache)      | GET             | Delete the server's detection cache.            |
| [`/profile`](#get-profile)              | GET             | The loaded profile and the labels it can detect. |
| [`/censor_config`](#get-censor_config)  | GET             | The censor config used when a request has none. |
| `/censored/<name>`                      | GET             | Censored images saved with `expected_response: url`. |
| `/assets/<name>`                        | GET             | Static assets (status images).                  |

Errors are returned with status `500` and a JSON body `{"error": "<message>"}`. A `config` that is not allowed
(a `stickers_dir` or sticker category outside `data/stickers`) is rejected with status `400`. In a `config` sent
with a request, the ComfyUI settings of `ai_remove` styles (`comfy_host`, `comfy_port`, `comfy_workflow`) are always
replaced by the ones in `config.yml`.

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
server does not download it.

**A URL the server downloads**

```json
{
  "type": "source",
  "image_url": "https://example.com/picture.jpg"
}
```

The image type is taken from the extension in the URL. URLs pointing to `localhost`/`127.0.0.1` are refused.
Downloaded images are kept in `data/server/uncensored/` and reused when the same URL is requested again.

Supported image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, `.avif` and `.gif` (GIFs are censored as
videos).

**Optional fields**

| Field               | Default    | Description                                                                 |
|---------------------|------------|-----------------------------------------------------------------------------|
| `config`            | profile's  | A censor config as JSON, same structure as `censor_config.yml` (see [censor_config.md](censor_config.md)). |
| `expected_response` | `base64`   | `/censor_image` only: `base64`, `bytes` or `url`, see below.                |

## POST /censor_image

Censors the image and returns it in the format set by `expected_response`:

| `expected_response` | Response                                                                                      |
|---------------------|-----------------------------------------------------------------------------------------------|
| `base64`            | JSON `{"image_data": "<base64>", "mime_type": "image/<ext>"}`                                 |
| `bytes`             | The image file itself, as an attachment.                                                      |
| `url`               | JSON `{"image_name": "<name>"}`. The image is saved on the server; fetch it from `/censored/<name>`. |

Example:

```
curl -X POST http://localhost:8443/censor_image \
     -H "Content-Type: application/json" \
     -d '{"type": "source", "image_url": "https://example.com/picture.jpg", "expected_response": "url"}'
```

## POST /detect_features

Runs the models without censoring and returns what was found:

```json
{
  "status": "success",
  "file_name": "<name of the stored image>",
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

- `box` is in image pixels. `polygons` holds the outline (largest part first) for segmentation labels such as
  `BODY` and for eyes; it is empty for box-only detections.
- `model` is the model that found the detection (its `name` in `pipeline.yml`, or `landmarks`).
- `parent` is the `id` of the detection it was found in (eyes in a face, cascades), otherwise `null`.
- `cached_models` lists the models whose results came from the cache.

Without a `config` every model in `pipeline.yml` runs. With a `config` only the models that config needs run.

## POST /censor_video

A `multipart/form-data` request with these parts:

| Part       | Required | Description                                       |
|------------|----------|---------------------------------------------------|
| `filename` | no       | Name of the video, e.g. `clip.mp4`, without folders (`400` otherwise). Send it before the `video` part. Default `video.mp4`. |
| `video`    | yes      | The video file.                                   |
| `config`   | no       | A censor config as JSON text.                     |

The response is the censored video, streamed back as `video/mp4`. The request returns when the whole video is done.

```
curl -X POST http://localhost:8443/censor_video \
     -F filename=clip.mp4 -F video=@clip.mp4 -o clip_censored.mp4
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

## GET /reset_cache

Deletes everything in the server's detection cache (`cache/server/`). Returns `{"status": "success"}`.

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
