# Censor Tool

Censor Tool finds things in images, videos and GIFs, and on your screen or webcam in real time, and censors them.
What it looks for is up to the [YOLO](https://docs.ultralytics.com/) models you give it, so it works for more than
one purpose:

- **Nudity**: censor nudity in pictures, videos, your screen or a video call (the `nudity` profile).
- **Privacy**: blur faces and license plates in photos, dashcam or street footage before you share it
  (the `privacy` profile).
- **Screen recordings and streams**: hide phones, laptops and monitors that may show private information.
- **Anything else**: any object a YOLO model is trained on, such as logos, weapons or cigarettes.
  See [Other models](#other-models).

How it is censored (bars, blur, pixelation, stickers, ...) is fully configurable.

- Censor images, videos and GIFs from a file or folder.
- Censor your screen or webcam in real time, in a window or as a virtual camera.
- Use any YOLO detection or segmentation model (ONNX), or several at once.
- Choose per label what is censored and how: style, shape, margins, borders, text or sticker overlays,
  inverse censoring, clipping to the body outline.
- Group settings in profiles and switch between them.
- An HTTP server to censor images, videos and video streams from other programs, such as the browser extension
  [CensorExtension](https://github.com/ValentijnvdB/CensorExtension).
- Runs on the CPU or an NVIDIA GPU, with a pure-Python backend or an optional, faster C++ backend.

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Backends](#backends)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

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

### Requirements

- Python 3.12 or newer.
- [git](https://git-scm.com/downloads).
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`, for video and GIF censoring.
- Optional: an NVIDIA GPU with [CUDA](https://developer.nvidia.com/cuda-downloads) and cuDNN, for faster
  censoring. Real-time censoring works on a CPU, but a GPU makes it a lot smoother.
- Optional, for the virtual camera (`--vcam`): a virtual camera driver, see
  [pyvirtualcam](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras) (OBS on Windows and macOS,
  v4l2loopback on Linux).

### Steps

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
   For the GPU, also set `device: cuda` in `configs/<profile>/pipeline.yml`.
4. Create the data folders.
    ```
    python main.py init
    ```
5. Download the models, see [Models](#models).
6. Check what works.
    ```
    python main.py validate
    ```
   This loads the models on the CPU and the GPU and tests the screen capture and the virtual camera, then prints
   which modes should work.

Optionally, install the faster [C++ backend](#backends).

### Models

The models are not included in the repository. Each profile lists the models it uses in its `pipeline.yml`; download
them to the paths below, or change the paths in `pipeline.yml`.

#### Nudity (`nudity` profile)

| Model | Used for | Path | Download |
|-------|----------|------|----------|
| Nevermore big | Faces, breasts, genitals, buttocks, feet and people. **Required.** | `models/nevermore/det_v11_big/det_v11_big_640.onnx` | [Hugging Face](https://huggingface.co/kinkyrvn/nevermore/tree/main/det_v11_big) |
| YOLO segmentation | Optional: body outlines (`BODY`), for full-body censoring and `clip_to`. | `models/yolo/yolo26n-seg.onnx` | Export with [Ultralytics](https://docs.ultralytics.com/modes/export/), see below |
| PFLD 68-point landmarks | Optional: eye bars (`eyes`). | `models/eye_detection/pfld.onnx` | e.g. [PFLD_68points_Pytorch](https://github.com/github-luffy/PFLD_68points_Pytorch), exported to ONNX (112x112 input) |

Nevermore comes in more sizes (`tiny` to `huge`) and fp16 versions, see its
[model card](https://huggingface.co/kinkyrvn/nevermore). To use another size, change `path` in `pipeline.yml`.

The optional models are commented out in `configs/nudity/pipeline.yml`; uncomment the `people` model for body
outlines and the `landmarks` section for eye bars. To make the segmentation model, install Ultralytics (only needed
for the export) and export a YOLO segmentation model to ONNX:

```
pip install ultralytics
yolo export model=yolo26n-seg.pt format=onnx
```

`n` (nano) is the fastest; `s`, `m`, `l` and `x` are more accurate.

[NudeNet](https://github.com/notAI-tech/NudeNet/releases) (`640m.onnx`) works as an alternative to Nevermore. It has
different labels (`FACE_FEMALE`, `FEMALE_BREAST_EXPOSED`, ...), so change both `pipeline.yml` and
`censor_config.yml`; see the NudeNet examples in [docs/pipeline_config.md](docs/pipeline_config.md#example-setups) and
[docs/censor_config.md](docs/censor_config.md#labels).

#### Privacy (`privacy` profile)

Blurs faces and puts a bar over license plates, e.g. in photos, dashcam or street footage.

| Model | Used for | Path | Download |
|-------|----------|------|----------|
| YOLO11 license plate detection | License plates (`license_plate`). | `models/license_plates/license-plate-finetune-v1n.onnx` | [Hugging Face](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) (AGPL-3.0) |
| YOLO11 face detection | Faces (`face`). | `models/faces/yolov11n-face.onnx` | [Hugging Face](https://huggingface.co/AdamCodd/YOLOv11n-face-detection), download `model.onnx` and rename it (Apache-2.0) |

Both come in larger, more accurate versions (`v1s` to `v1x` for plates, `YOLOv11x` for faces).

```
python main.py file --profile privacy -i <photos or videos>
```

#### Other models

Censor Tool runs any YOLO model exported to ONNX (YOLOv8, YOLO11, YOLO26 and other Ultralytics exports, detection or
segmentation), so you can censor anything a model is trained on. Some places to look:

- Any COCO model, such as the segmentation model above, finds 80 everyday objects: `person`, `car`, `laptop`, `tv`,
  `cell phone`, ... Useful to hide screens and phones in screen recordings.
- [Hugging Face](https://huggingface.co/models?search=yolo) and [Roboflow Universe](https://universe.roboflow.com/)
  have models for logos, weapons, smoking, ID cards and much more.

To add a model:

1. Get the model as an `.onnx` file. Models published as `.pt` files have to be exported first:
   `yolo export model=<file>.pt format=onnx` (needs `pip install ultralytics`).
2. Put it in `models/`, e.g. `models/logos/logos.onnx`.
3. Add it to `models:` in the profile's `pipeline.yml` (or make a new profile, see [Profiles](#profiles)):
    ```yaml
    models:
      - name: logos                       # any unique name
        path: models/logos/logos.onnx
        min_confidence: 0.25              # at or below the min_probability in censor_config.yml
        # classes: [logo]                 # optional: keep only these classes
        # labels: {logo: brand_logo}      # optional: rename classes
    ```
4. Add the model's labels to `features_to_censor` in `censor_config.yml`. The labels are the model's class names;
   they are usually on the model card, or print them with
   `python -c "import onnxruntime as o, sys; print(o.InferenceSession(sys.argv[1]).get_modelmeta().custom_metadata_map.get('names'))" <model.onnx>`.
5. Run `python main.py validate --profile <name>` to check that the model loads.

Some things to keep in mind:

- Labels must be unique across the models in one `pipeline.yml`. If two models both have a `face` label (e.g.
  Nevermore and the face model), rename one with `labels:`.
- A model only runs when the censor config needs one of its labels.
- Models with a dynamic input size need `resize: rect` (and optionally `imgsz`), see
  [docs/pipeline_config.md](docs/pipeline_config.md#models) for all settings.
- Check a model's license before you use it; many YOLO models are AGPL-3.0.

## Usage

```
python main.py <mode> [arguments]
```

| Mode               | Description |
|--------------------|-------------|
| `file` (`f`, `files`) | Censor the images and videos in the input folder (or a single file). |
| `live`             | Censor a region of your screen in real time. |
| `webcam`           | Censor a webcam in real time. |
| `overlay`          | Censor a window in place, in real time (Linux with X11). |
| `http` (`https`)   | Start the HTTP server. |
| `init`             | Create the data folders. |
| `validate`         | Check which features work on this machine. |

Every mode uses a profile. Without `--profile` you are asked to pick one; see
[Profiles](#profiles).

### Censoring files

```
python main.py file [-i <input>] [-o <output>] [-r]
```

Censors every image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, `.avif`) and video (`.mp4`, `.avi`, `.gif`)
in `data/uncensored/` and writes the results to `data/censored/`. `-i` can be a folder or a single file; `-r`
includes subfolders.

Detections are cached in `cache/`, so censoring the same file again with a different censor config only redoes the
drawing, not the detection. Use `--override-cache` to detect again, `--skip-existing` to skip files that already
exist in the output folder, and `--only-analyze` to fill the cache without censoring.

### Live and webcam censoring

```
python main.py live      # your screen
python main.py webcam    # webcam 0, or --device <id>
```

`live` captures the region set in `config.yml` (`live_cap_*`, the full 1920x1080 screen by default) and shows it
censored in a window. This works best with two displays: the content on one, the censored window on the other.
`webcam` does the same with a camera.

With `--vcam` the output goes to a virtual camera instead of a window, so it can be used in video calls or OBS
(`--vcam-w`, `--vcam-h` and `--vcam-fps` set its size and frame rate).

In the window, press `q` to quit and `r` to reload the configs. Changes to `config.yml` and `censor_config.yml`
are also picked up as soon as they are saved, so you can tune the censoring while it runs. `pipeline.yml` is only
read at startup.

### Overlay: censoring a window in place

```
pip install -r requirements-overlay.txt   # once: installs PySide6
python main.py overlay                    # then click the window to censor
python main.py overlay --window 0x3a00007 # or give its id (from xwininfo or wmctrl -l)
```

`overlay` captures one window and shows it censored in an overlay that lies exactly over it, so you don't need a
second display. The overlay lets clicks, scrolling and typing through to the window below, so you can use it as
usual. It follows the window when it is moved or resized and hides with it when it is minimised. Other windows,
menus and panels stay on top of it.

The overlay is shown before the first frame is censored, so the window is never visible uncensored. Like live mode, the
censored image lags slightly behind the window (the time it takes to censor a frame), which you notice when
scrolling. Menus, dropdowns and tooltips of the censored window are separate windows and are not censored.

A tray icon has *Select another window*, *Reload configs* and *Quit*; without a system tray, stop it with Ctrl+C.
Overlay mode stops when the censored window is closed. It uses the same `config.yml` settings as live mode
(`live_input_size`, `live_frames_in_flight`, `live_show_fps`); the capture region `live_cap_*` is not used.

Overlay mode works on Linux with an X11 session and a compositing window manager. It has been tested on Cinnamon
(Linux Mint); Wayland, Windows and macOS are not supported yet.

### HTTP server

```
python main.py http [--host localhost] [--port 8443] [--ssl --ssl-cert certs/cert.pem --ssl-key certs/key.pem]
```

Starts a server with endpoints to censor images (`/censor_image`), videos (`/censor_video`) and a stream of
frames over a WebSocket (`/censor_frame`), and to only detect (`/detect_features`). The models are loaded once at
startup; a request can send its own censor config. This is what
[CensorExtension](https://github.com/ValentijnvdB/CensorExtension) uses.

The endpoints and their request and response formats are described in [docs/http_api.md](docs/http_api.md).

### Arguments

| Group | Argument | Default | Description |
|-------|----------|---------|-------------|
| General | `mode` | - | The mode, see above. |
| | `-i`, `--input` | `data/uncensored` | Input file or folder (`file` mode). |
| | `-o`, `--output` | `data/censored` | Output folder (`file` mode). |
| | `-r`, `--recursive` | off | Include subfolders of the input folder. |
| | `--override-cache` | off | Detect again instead of using the cache. |
| | `--skip-existing` | off | Skip files that already exist in the output folder. |
| | `--only-analyze` | off | Only detect (fills the cache), don't censor. |
| | `--debug` | off | More logging and debug output. |
| Configuration | `-p`, `--profile` | asks | Profile: a folder name in `configs/`, or a path to a profile folder. |
| | `--config` | from profile | Use this `config.yml` instead of the profile's. |
| | `-cc`, `--censor-config` | from profile | Use this `censor_config.yml` instead of the profile's. |
| | `-pl`, `--pipeline` | from profile | Use this `pipeline.yml` instead of the profile's. |
| Live & webcam | `--device` | `-1` (live), `0` (webcam) | Input: `-1` is the screen, `0` or higher a camera id. |
| | `--vcam` | off | Output to a virtual camera instead of a window. |
| | `--vcam-w`, `--vcam-h` | `1920`, `1080` | Virtual camera size. |
| | `--vcam-fps` | `10` | Virtual camera frame rate. |
| Overlay | `--window` | click | X11 id of the window to censor, in hex (`0x3a00007`) or decimal. |
| Server | `--host` | `localhost` | Address to listen on. Use `0.0.0.0` to accept connections from other devices. |
| | `--port` | `8443` | Port to listen on. |
| | `--ssl` | off | Use HTTPS. |
| | `--ssl-cert`, `--ssl-key` | `certs/cert.pem`, `certs/key.pem` | Certificate and key for HTTPS. |

### Folders

All folders below are relative to the working directory, see [Instances](#instances).

| Folder | Contents |
|--------|----------|
| `data/uncensored/` | Default input. |
| `data/censored/` | Default output. |
| `data/stickers/<category>/` | Sticker images for [overlays](docs/censor_config.md#overlays). |
| `data/server/` | Files uploaded to and produced by the HTTP server. |
| `data/logs/` | Log files. |
| `cache/` | Cached detections (`images/`, `videos/`, `server/`). Safe to delete. |
| `models/` | The models. |
| `configs/` | The profiles. |
| `certs/` | The HTTPS certificate and key. |

### Instances

Your data and settings (the folders above) are read from and written to the working directory, not the folder
Censor Tool is installed in. Running `main.py` from another folder gives a separate instance with its own profiles,
models, cache and output, using the same install:

```
mkdir ~/other-instance && cd ~/other-instance
cp -r /path/to/CensorTool/configs .        # profiles to start from
ln -s /path/to/CensorTool/models models    # or share the models
python /path/to/CensorTool/main.py file
```

Relative paths in the config files (models in `pipeline.yml`, `stickers_dir`, `comfy_workflow`) are relative to the
working directory too. Only `data/assets/`, which ships with the program, is always read from the install.

## Configuration

### Profiles

Settings are grouped in profiles. Each profile is a folder in `configs/` with three files:

```
configs/
├── nudity/
│   ├── config.yml
│   ├── censor_config.yml
│   └── pipeline.yml
├── privacy/
│   └── ...
└── <your-profile>/
    └── ...
```

| File | Contents | Reference |
|------|----------|-----------|
| `config.yml` | General settings: video sampling rate, live capture region, ComfyUI connection. | [docs/config.md](docs/config.md) |
| `pipeline.yml` | The backend, which models run, and on which device. | [docs/pipeline_config.md](docs/pipeline_config.md) |
| `censor_config.yml` | What is censored and how. | [docs/censor_config.md](docs/censor_config.md) |

Pick a profile with `--profile <name>` or `--profile <path/to/folder>`. Without it you are asked to choose one
from a list (arrow keys and enter). If there is only one profile it is used without asking; when Censor Tool
doesn't run in an interactive terminal, the first profile alphabetically is used, with a warning.
`--config`, `--censor-config` and `--pipeline` replace a single file of the profile, e.g.
`python main.py file --profile nudity --censor-config ./strict.yml`.

To make your own profile, copy one of the included profiles to `configs/<name>/` and edit the files. Profiles
other than `nudity` and `privacy` are ignored by git, so your own profiles stay out of the repository.

The included profiles:

| Profile | What it censors | Models |
|---------|-----------------|--------|
| `nudity` | Nudity and faces | Nevermore big, see [Nudity](#nudity-nudity-profile) |
| `privacy` | Faces and license plates | License plate and face detection, see [Privacy](#privacy-privacy-profile) |

Both use the Python backend on the CPU; set `backend` and `device` in their `pipeline.yml` to change that.

### Censor config

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

## Backends

Censoring is done by one of two backends. Both have the same features and read the same config files.

| Backend | Install | Notes |
|---------|---------|-------|
| `python` | Included. | The default. Pure Python on top of ONNX Runtime and OpenCV. |
| `cpp` | Build and install `censor_core` 2.2 or newer, e.g. `pip install /path/to/PyCensorCore`. Older versions are refused at startup. | Faster, mostly noticeable in live and video censoring and on large batches of files. |

Pick one with `backend:` in `pipeline.yml`. If `cpp` is selected but `censor_core` is not installed, Censor Tool
stops with an error instead of switching backends. `python main.py validate` shows which backend is used.

Results of the two can differ by a pixel here and there, not in what is detected or censored. They share the
`cache/` folder but each keeps its own cache files, so switching backends detects everything again once.

## Troubleshooting

**The GPU isn't used.** Install `requirements-gpu.txt` (not `requirements.txt`; having both `onnxruntime` and
`onnxruntime-gpu` installed causes problems, so `pip uninstall onnxruntime` first) and set `device: cuda` in
`pipeline.yml`. `python main.py validate` reports whether CUDA works. The CUDA and cuDNN versions must match
[onnxruntime-gpu's requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

**"Model ... not found".** The model file isn't at the path in `pipeline.yml`, see [Models](#models). Paths are
relative to the working directory, see [Instances](#instances).

**A label in the censor config is reported as unknown.** The censor config uses a label none of the models in
`pipeline.yml` produce, for example `BODY` without the segmentation model, or NudeNet labels with Nevermore.

**Videos fail.** Check that `ffmpeg` is installed and on your `PATH` (`ffmpeg -version`).

**The virtual camera doesn't work.** Install a virtual camera driver, see
[pyvirtualcam](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras). On Linux, load v4l2loopback
(`sudo modprobe v4l2loopback`).

**Live censoring is slow.** Use a GPU, the `cpp` backend, or a smaller model (e.g. `det_v11_medium`). Fewer
labels in `features_to_censor` can also help, as models whose labels aren't needed don't run.

## Development

```
app/        the modes: file (image, video), live and webcam
core/       the Python backend: models, detection, censoring, cache
server/     the HTTP server
configs/    the profiles
docs/       config and API reference
tests/      tests (tests/core: backend tests, run against both backends)
```

Run the tests with

```
python -m pytest tests
```

The backend tests in `tests/core` run against every installed backend, so against `cpp` too when `censor_core` is
installed. Tests that need a model are skipped when the model is missing from `models/`.
