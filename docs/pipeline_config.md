# pipeline.yml

Which backend runs, which models it loads and where they run. `pipeline.yml` is only read at startup.

Relative model paths are relative to the working directory (see [Instances](profiles.md#instances)).

```yaml
backend: python
device: cpu
n_workers: 4
max_concurrent_jobs: 128

models:
  - name: nevermore
    path: models/nevermore/det_v11_big/det_v11_big_640.onnx
    agnostic_nms: true
    nms_iou: 0.75
    max_det: 0
    min_confidence: 0.15
```

## General settings

| Key                   | Default  | Description |
|-----------------------|----------|-------------|
| `backend`             | `python` | `python` or `cpp`. See [Backends](backends.md). |
| `device`              | `cpu`    | Where the models run: `cpu`, `cuda` (first GPU) or `cuda:N`. Default for every model; a model can set its own `device`. |
| `threads`             | `auto`   | ONNX Runtime CPU threads per model: a number, or `auto` (the CPU models split the cores; a GPU model uses one). |
| `n_workers`           | `4`      | Worker threads that process images. |
| `max_concurrent_jobs` | `128`    | Maximum number of video frames in flight. Lower it to use less RAM; keep it at least `n_workers`. `0` = unlimited. |
| `models`              | -        | The detection models, see below. At least one is required. |
| `landmarks`           | none     | The optional eye detection stage, see [Landmarks](#landmarks-eye-bars). |

## Models

Each entry in `models` is a YOLO model exported to ONNX: YOLOv8/v11 or the end-to-end YOLOv10/YOLO26 exports, in
detect or segment variants. Class names, input size and output format are read from the model file.

| Key              | Default        | Description |
|------------------|----------------|-------------|
| `path`           | -              | Path to the `.onnx` file. Required. |
| `name`           | file name      | Name of the model, used in logs, results and the cache. Must be unique; `landmarks` is reserved. |
| `device`         | top-level      | `cpu`, `cuda` or `cuda:N` for this model. |
| `threads`        | top-level      | CPU threads for this model. |
| `resize`         | `letterbox`    | How images are fitted to the model input. `letterbox`: keep the aspect ratio and pad. `stretch`: resize, ignoring the aspect ratio. `rect`: keep the aspect ratio and pad only to a multiple of 32 (models with a dynamic input size only, e.g. NudeNet). |
| `imgsz`          | from the model, else `640` | Input size of a model with a dynamic input size. Larger finds smaller body parts (e.g. in group photos) but is slower: `640`, `1280`, `2560` (a multiple of 32). Ignored by models with a fixed input size. |
| `min_confidence` | `0.25`         | Score floor: detections below it are dropped before anything else and never cached. Keep it at or below the lowest `min_probability` in your censor configs. |
| `nms_iou`        | `0.7`          | Overlap above which boxes are merged (non-maximum suppression). YOLOv8/v11 exports only. |
| `agnostic_nms`   | `false`        | Merge overlapping boxes across classes (one box per region) instead of per class. |
| `max_det`        | `300`          | Maximum number of detections per image. `0` = no limit. |
| `mask_threshold` | `0.5`          | Segmentation models: mask probability that counts as inside the outline. |
| `classes`        | all            | Keep only these classes, by name or id, e.g. `[person]`. |
| `labels`         | -              | Rename classes, e.g. `{person: BODY}`. |
| `label_prefix`   | -              | Prefix added to every class that is not renamed with `labels`. |
| `names`          | from the model | Class names, for models without names in their metadata. |
| `layout`         | `auto`         | Output format: `auto`, `v8` or `end2end`. Only needed when detection from the file fails. |
| `run_on`         | -              | Cascade: run this model on crops of other models' detections instead of on the whole image, e.g. `run_on: [person]`. |
| `crop_padding`   | `0.1`          | Cascade only: grow each crop by this fraction of its size. |

**Labels.** Every label must be unique across all models; if two models produce the same label, rename one with
`labels` or `label_prefix`. These labels are what you use in the [censor config](censor_config.md#labels).

**On demand.** A model only runs when the censor config needs one of its labels. For example, the body segmentation
model only runs when `BODY` is censored or used in `clip_to`.

## Landmarks (eye bars)

The landmark stage runs the PFLD 68-point face landmark model on every detected face and produces an eye bar
outline.

```yaml
landmarks:
  path: models/eye_detection/pfld.onnx
  run_on:
    face: eyes            # face label -> eye label
  min_face_score: 0.15
```

| Key              | Default   | Description |
|------------------|-----------|-------------|
| `path`           | -         | Path to the PFLD `.onnx` file. Required. |
| `run_on`         | -         | Maps face labels to the eye label to produce. Required. NudeNet has two face labels: `{FACE_FEMALE: EYES_FEMALE, FACE_MALE: EYES_MALE}`. |
| `min_face_score` | `0.05`    | Faces below this score are skipped. |
| `device`         | top-level | Device for this model. |
| `threads`        | top-level | CPU threads for this model. |

## Example setups

**Body outlines.** Add a COCO segmentation model and rename its `person` class to `BODY`. Then `BODY` can be censored
and used in `clip_to`:

```yaml
  - name: people
    path: models/yolo/yolo26n-seg.onnx
    resize: stretch
    classes: [person]
    labels: {person: BODY}
    min_confidence: 0.25
```

**NudeNet** instead of Nevermore (it has a dynamic input size, so use `resize: rect` and pick a size with `imgsz`):

```yaml
  - name: nudenet
    path: models/nudenet/640m.onnx
    resize: rect
    imgsz: 640
    agnostic_nms: true
    nms_iou: 0.75
    max_det: 0
    min_confidence: 0.05
```

**Cascade.** Run a second model on every detected person, e.g. a detector that works better on close-ups:

```yaml
  - name: parts
    path: models/my_detector.onnx
    run_on: [person]
    label_prefix: crop_
```
