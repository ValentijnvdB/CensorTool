# censor_config.yml

What is censored and how. Every profile has one; `--censor-config` overrides it for a single run. The HTTP API
accepts the same structure as JSON (see [http_api.md](http_api.md)).

The censor config can be changed without reloading the models: during live and webcam censoring it is reloaded as
soon as it is saved (press `r` in the window to force a reload). Unknown labels and misspelled keys are reported as
warnings when the file is loaded; an invalid file is reported and the previous config is kept.

```yaml
features_to_censor: [
    "face",
    "female_breast_exposed",
    "vagina_exposed",
]

default_censor_config:
    censor_style:
        type: bar
        color: [0, 0, 0]
    min_probability: 0.25
    width_area_safety: 0.1
    height_area_safety: 0.1

feature_overrides:
    face:
        censor_style:
            type: pixel
            factor: 20
```

## Labels

`features_to_censor` lists the labels to censor. These are the labels the models in
[pipeline.yml](pipeline_config.md) produce. A model only runs when the censor config needs one of its labels.

**Nevermore** (the default model):

| Label                   | Label                   | Label             |
|-------------------------|-------------------------|-------------------|
| `face`                  | `male_breast_exposed`   | `vagina_covered`  |
| `person`                | `male_breast_covered`   | `buttocks_exposed`|
| `female_breast_exposed` | `penis_exposed`         | `buttocks_covered`|
| `female_breast_covered` | `penis_covered`         | `foot_exposed`    |
| `nipple_exposed`        | `vagina_exposed`        | `foot_covered`    |

`person` is the box around a whole person.

**Optional stages** (they have to be enabled in `pipeline.yml`):

| Label  | Comes from | Description |
|--------|------------|-------------|
| `BODY` | the body segmentation model | The outline of a person. Censor it to cover the whole body, or use it in `clip_to`. |
| `eyes` | the landmark model | An eye bar over both eyes, for every detected face. |

**NudeNet** (if you use it instead of Nevermore): `FACE_FEMALE`, `FACE_MALE`, `FEMALE_BREAST_EXPOSED`,
`FEMALE_BREAST_COVERED`, `MALE_BREAST_EXPOSED`, `FEMALE_GENITALIA_EXPOSED`, `FEMALE_GENITALIA_COVERED`,
`MALE_GENITALIA_EXPOSED`, `BUTTOCKS_EXPOSED`, `BUTTOCKS_COVERED`, `ANUS_EXPOSED`, `ANUS_COVERED`, `BELLY_EXPOSED`,
`BELLY_COVERED`, `FEET_EXPOSED`, `FEET_COVERED`, `ARMPITS_EXPOSED`, `ARMPITS_COVERED`, and with landmarks
`EYES_FEMALE`/`EYES_MALE`.

## Top-level settings

| Key                         | Default            | Description |
|-----------------------------|--------------------|-------------|
| `features_to_censor`        | -                  | The labels to censor. Required. |
| `default_censor_config`     | -                  | The [censor box](#censor-box) used for every feature. Required. |
| `feature_overrides`         | -                  | Per label, the censor box settings that differ from the default. See [Overrides](#overrides). |
| `inverse_censor_style`      | `blur` (31)        | The [style](#censor-styles) used for [inverse censoring](#inverse-censoring): `blur`, `pixel` or `bar`. |
| `force_inverse_censor`      | `false`            | Apply the inverse censor even when no inverse feature is found (the whole image is censored). |
| `merge_overlapping_borders` | `false`            | Draw one border around overlapping censored areas instead of one per area. With different colors or thicknesses, one is picked. |
| `enable_overlays`           | `true`             | Master switch for [overlays](#overlays). |
| `enable_watermark`          | `false`            | Add a small "Censored using CensorTool" watermark. |
| `stickers_dir`              | `./data/stickers`  | Folder with the sticker images. Must be `data/stickers` or one of its subfolders; a relative path is relative to the working directory. |

## Censor box

A censor box describes how one feature is censored. `default_censor_config` must contain at least `censor_style`.

| Key                  | Default   | Description |
|----------------------|-----------|-------------|
| `censor_style`       | -         | How the area is hidden, see [Censor styles](#censor-styles). Required in the default. |
| `min_probability`    | `0.5`     | Minimum score a detection needs to be censored. Lower censors more (including things that aren't there), higher misses more. It can't go below the model's `min_confidence` in `pipeline.yml`. |
| `shape`              | `default` | The shape of the censored area, see [Shapes](#shapes). |
| `width_area_safety`  | `0.0`     | Extra width, as a fraction: `0.1` makes the area 10% wider. |
| `height_area_safety` | `0.0`     | Extra height, as a fraction. |
| `time_safety`        | `0.0`     | Videos only: keep the censor on screen this many seconds after the feature was last detected. Covers frames the model misses and the frames between analysed frames (`video_censor_fps` in [config.yml](config.md)). |
| `border`             | `null`    | A colored outline around the area: `{color: [R, G, B], thickness: <int>}`. |
| `overlay`            | `null`    | Text or a sticker on top of the area, see [Overlays](#overlays). |
| `inverse`            | `false`   | Keep this feature visible and censor everything else, see [Inverse censoring](#inverse-censoring). |
| `clip_to`            | `[]`      | Clip the area to the outline of one or more segmentation labels, see [clip_to](#clip_to). |

## Censor styles

```yaml
censor_style:
    type: <type>
    <settings>
```

| Type        | Settings                    | Description |
|-------------|-----------------------------|-------------|
| `bar`       | `color: [R, G, B]`          | A solid color. |
| `blur`      | `strength: <int>` (15)      | Blurs the area; higher is stronger. |
| `pixel`     | `factor: <int>` (10)        | Pixelates the area. `10` turns a 200x400 region into 20x40 blocks. |
| `ai_remove` | -                           | Inpaints the area with a ComfyUI workflow (e.g. to add clothing). Needs a running ComfyUI server and workflow, set in [config.yml](config.md#ai_remove-comfyui). Slow. |
| `debug`     | `color: [R, G, B]`          | Draws the outline, label and score instead of censoring. For tuning configs. |

Overlapping areas are drawn in the order `ai_remove`, `blur`, `pixel`, `bar`, so bars always end up on top.

## Shapes

| Shape       | Description |
|-------------|-------------|
| `default`   | The detection's own shape: the outline for `BODY` and `eyes`, the box for everything else. |
| `rectangle` | The box. |
| `ellipse`   | An ellipse filling the box. |
| `circle`    | A circle in the middle of the box. |
| `mask`      | The segmentation outline. Only for labels from segmentation models, such as `BODY`. |

## clip_to

`clip_to` intersects the censored area with the outline of other (segmentation) labels, so a censor doesn't spill
over the background:

```yaml
feature_overrides:
    face:
        shape: circle
        clip_to: [BODY]
```

The listed labels are detected even when they are not in `features_to_censor`. Their `min_probability` (from
`feature_overrides`, otherwise the default) decides which outlines count. If no outline overlaps a detection, the
detection is censored unclipped. `BODY` requires the body segmentation model in `pipeline.yml`.

## Overlays

Overlays are drawn on top of the censored area. `enable_overlays` must be `true`.

**Text**: one of the values, picked at random:

```yaml
overlay:
    type: text
    values: ['Hidden', 'Censored']
    probability: 1.0      # chance the overlay is placed (0-1)
    color: [255, 255, 255]
    thickness: 1
    font_scale: 1.0
```

**Sticker**: a random image from `data/stickers/<category>/`:

```yaml
overlay:
    type: sticker
    probability: 0.5
    categories: ['plants', 'misc']   # [] picks from all stickers
```

Stickers are scaled to fit the censored area; PNGs with transparency work best.

Stickers are only read from `data/stickers`: a `stickers_dir` or category that points outside it (an absolute path
elsewhere, `..`, or a symlink out of the folder) is an error.

## Inverse censoring

A feature with `inverse: true` stays visible while the rest of the image is censored with `inverse_censor_style`.
Other (normal) features are censored on top of that. Use `force_inverse_censor: true` to censor the whole image
when no inverse feature is found.

```yaml
inverse_censor_style:
    type: blur
    strength: 51

feature_overrides:
    face:
        inverse: true
```

## Overrides

`feature_overrides` changes the censor box for specific labels. Missing settings come from
`default_censor_config`; nested settings such as `censor_style` are replaced as a whole.

```yaml
feature_overrides:
    female_breast_exposed:
        shape: ellipse
        width_area_safety: 0.2

    BODY:
        min_probability: 0.5     # also used when BODY is only a clip_to target
        censor_style:
            type: pixel
            factor: 30
        border:
            color: [200, 0, 150]
            thickness: 8
```
