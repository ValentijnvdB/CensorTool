# Censor Tool

Censor Tool finds things in images, videos and GIFs, and on your screen or webcam in real time, and censors them.
What it looks for is up to the [YOLO](https://docs.ultralytics.com/) models you give it, so it works for more than
one purpose:

- **Nudity**: censor nudity in pictures, videos, your screen or a video call (the `nudity` profile).
- **Privacy**: blur faces and license plates in photos, dashcam or street footage before you share it
  (the `privacy` profile).
- **Screen recordings and streams**: hide phones, laptops and monitors that may show private information.
- **Anything else**: any object a YOLO model is trained on, such as logos, weapons or cigarettes.
  See [Other models](docs/models.md#other-models).

How it is censored (bars, blur, pixelation, stickers, ...) is fully configurable.

- Censor images, videos and GIFs from a file or folder.
- Censor your screen or webcam in real time, in a window, in place over a window, or as a virtual camera.
- Use any YOLO detection or segmentation model (ONNX), or several at once.
- Choose per label what is censored and how: style, shape, margins, borders, text or sticker overlays,
  inverse censoring, clipping to the body outline.
- Group settings in profiles and switch between them.
- An HTTP server to censor images, videos and video streams from other programs, such as the browser extension
  [CensorExtension](https://github.com/ValentijnvdB/CensorExtension).
- Runs on the CPU or an NVIDIA GPU, with a pure-Python backend or an optional, faster C++ backend.

## Quick start

```
git clone https://github.com/ValentijnvdB/CensorTool.git
cd CensorTool
pip install -r requirements.txt
python main.py init
```

Download [`det_v11_big_640.onnx`](https://huggingface.co/kinkyrvn/nevermore/blob/main/det_v11_big/det_v11_big_640.onnx)
to `models/nevermore/det_v11_big/`, put your images in `data/uncensored/` and run

```
python main.py file --profile nudity
```

The censored images appear in `data/censored/`.

## Installation

Requirements:

- Python 3.12 or newer and [git](https://git-scm.com/downloads).
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`, for video and GIF censoring.
- Optional: an NVIDIA GPU with [CUDA](https://developer.nvidia.com/cuda-downloads) and cuDNN, for faster
  censoring. Real-time censoring works on a CPU, but a GPU makes it a lot smoother.

Steps:

1. Clone the repository.
    ```
    git clone https://github.com/ValentijnvdB/CensorTool.git
    cd CensorTool
    ```
2. Create and activate a virtual environment (optional, but recommended), for example with
   [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install):
    ```
    conda create -n CensorTool python=3.12
    conda activate CensorTool
    ```
3. Install the dependencies.
    ```
    # CPU only
    pip install -r requirements.txt

    # NVIDIA GPU (check that your CUDA version is supported by onnxruntime-gpu first)
    pip install -r requirements-gpu.txt
    ```
   For the GPU, also set `device: cuda` in `configs/<profile>/pipeline.yml`, see [GPU](docs/backends.md#gpu).
4. Create the data folders.
    ```
    python main.py init
    ```
5. Download the models of the profile you want to use. The required ones:

   | Profile | Model | Path |
   |---------|-------|------|
   | `nudity` | [Nevermore big](https://huggingface.co/kinkyrvn/nevermore/tree/main/det_v11_big) | `models/nevermore/det_v11_big/det_v11_big_640.onnx` |
   | `privacy` | [License plates](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) and [faces](https://huggingface.co/AdamCodd/YOLOv11n-face-detection) | `models/license_plates/license-plate-finetune-v1n.onnx`, `models/faces/yolov11n-face.onnx` |

   Optional models (body outlines, eye bars), other sizes and other models are described in
   [docs/models.md](docs/models.md).
6. Check what works.
    ```
    python main.py validate
    ```
   This loads the profile's models, tries every mode and prints which modes should work. See
   [Checking what works](docs/modes/validate.md).

Optionally, install the faster [C++ backend](docs/backends.md).

## Usage

```
python main.py <mode> [arguments]
```

| Mode | Description | Docs |
|------|-------------|------|
| `file` (`f`, `files`) | Censor the images and videos in the input folder (or a single file). | [file](docs/modes/file.md) |
| `live` | Censor a region of your screen in real time. | [live](docs/modes/live.md) |
| `webcam` | Censor a webcam in real time. | [webcam](docs/modes/live.md) |
| `overlay` | Censor a window in place, in real time (Linux with X11). | [overlay](docs/modes/overlay.md) |
| `http` (`https`) | Start the HTTP server. | [http](docs/modes/http.md) |
| `init` | Create the data folders. | |
| `validate` | Check which features work on this machine. | [validate](docs/modes/validate.md) |

Arguments for every mode:

| Argument | Default | Description |
|----------|---------|-------------|
| `-p`, `--profile` | asks | Profile: a folder name in `configs/`, or a path to a profile folder. |
| `--config` | from profile | Use this `config.yml` instead of the profile's. |
| `-cc`, `--censor-config` | from profile | Use this `censor_config.yml` instead of the profile's. |
| `-pl`, `--pipeline` | from profile | Use this `pipeline.yml` instead of the profile's. |
| `--debug` | off | More logging and debug output. |

The arguments of each mode are listed in its docs; `python main.py --help` lists them all.

## Configuration

Settings are grouped in profiles: folders in `configs/` with three files each.

| File | Contents | Reference |
|------|----------|-----------|
| `config.yml` | General settings: video sampling rate, live capture region, ComfyUI connection, server. | [docs/config.md](docs/config.md) |
| `pipeline.yml` | The backend, which models run, and on which device. | [docs/pipeline_config.md](docs/pipeline_config.md) |
| `censor_config.yml` | What is censored and how. | [docs/censor_config.md](docs/censor_config.md) |

Two profiles are included: `nudity` and `privacy`. To make your own, copy one to `configs/<name>/` and edit the
files; your own profiles are ignored by git. Without `--profile` you are asked to pick one. See
[docs/profiles.md](docs/profiles.md) for more, and for the folders Censor Tool uses.

`censor_config.yml` lists the labels to censor and how each one is censored:

```yaml
features_to_censor: ["face", "female_breast_exposed", "vagina_exposed", "penis_exposed"]

default_censor_config:
    censor_style: {type: bar, color: [0, 0, 0]}
    min_probability: 0.25

feature_overrides:
    face:
        censor_style: {type: pixel, factor: 20}
        shape: circle
```

The labels depend on the models in `pipeline.yml`. All labels, styles, shapes, overlays and other options are
described in [docs/censor_config.md](docs/censor_config.md).

## Documentation

- Modes: [file](docs/modes/file.md), [live and webcam](docs/modes/live.md), [overlay](docs/modes/overlay.md),
  [HTTP server](docs/modes/http.md), [validate](docs/modes/validate.md)
- [Models](docs/models.md): the models per profile, and how to add other models
- [Profiles, folders and instances](docs/profiles.md)
- [Backends](docs/backends.md): Python or C++, CPU or GPU
- Config reference: [config.yml](docs/config.md), [pipeline.yml](docs/pipeline_config.md),
  [censor_config.yml](docs/censor_config.md)
- [HTTP API](docs/http_api.md)
- [Troubleshooting](docs/troubleshooting.md)

## Development

```
app/        the modes: file (image, video), live, webcam and overlay
core/       the Python backend: models, detection, censoring, cache
server/     the HTTP server
configs/    the profiles
docs/       documentation and reference
tests/      tests (tests/core: backend tests, run against both backends)
```

Run the tests with

```
python -m pytest tests
```

The backend tests in `tests/core` run against every installed backend, so against `cpp` too when `censor_core` is
installed. Tests that need a model are skipped when the model is missing from `models/`.
