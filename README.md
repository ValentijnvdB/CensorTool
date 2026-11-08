# Censor Tool

Censor Tool is collection of programs to censor images and videos.


*This readme is still under construction*

## Feature overview

---
- Censor images and videos.
- Censor your screen or webcam in real-time (if you have a dedicated GPU).
- Customize what and how features are censored.
- Censor images remotely via the HTTP endpoint
- Browser extension to automatically censor images: [CensorExtension](https://github.com/ValentijnvdB/CensorExtension.git)

### Planned
- [ ] Video censor via HTTP
- [x] ~~Automatic model downloads~~ (Expect for nudenet, because it requires to login)

## Installation Guide

### Requirements
You will need two programs for the easiest installation:
1. [git](https://git-scm.com/downloads)
2. Your favorite program to create virtual environments. I prefer [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation), so that is what will be used in this guide.

For GPU support you'll need a Nvidia GPU and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
Then set `device: cuda` in `configs/<profile>/pipeline.yml`.

### Installation:
1. Open a terminal or command prompt in the location you want to store the program.
2. Clone this repo. 
    ```
    git clone https://github.com/ValentijnvdB/CensorTool.git
    cd CensorTool
    ```
3. Create and activate a virtual environment (optional step, but highly recommend).
    ```
   conda create -n CensorTool python=3.14
   conda activate CensorTool
   ```
4. Install the dependencies.
    ```
   # Basic install
   pip install -r requirements.txt
   
   # For GPU support (make sure your GPU is supported by onnxruntime before you install)
   pip install -r requirements-gpu.txt
   ```
   Optional: install the faster C++ backend (`censor_core`) and set `backend: cpp` in
   `configs/<profile>/pipeline.yml`, see [Backends](#backends). Without it the Python backend is used.
5. Run the init process. This will create all required directories.
    ```
    python main.py init
    ```
6. Download the models and place them in the ```./models/``` folder:
    - NudeNet ```640m.onnx``` from https://github.com/notAI-tech/NudeNet/releases (as ```models/nudenet/640m.onnx```)
    - a YOLO segmentation model for body outlines, e.g. ```yolo26n-seg.onnx``` (as ```./models/yolo/yolo26n-seg.onnx```)
    - the PFLD landmark model for eye bars (as ```models/eye_detection/pfld.onnx```)

   The models are listed in ```configs/default/pipeline.yml```, see [Pipeline config](#pipeline-config).
7. Run the validate process. This will run a few tests to see what features work.
    ```
    python main.py validate
    ```


You are now setup to censors images, videos and use real-time censoring.
To censor images, place your images in the ```./data/uncensored``` folder then run ```python main.py images```
The censored pictures will appear in ```./data/censored```.
To run real time censoring, use ```python main.py live``` or ```python main.py webcam```.
For all modes and arguments, check the table below.

## Running Censor Tool

Running censor tool requires you to provided the mandatory 'mode' argument, follwed by zero or more optional arguments. They are all explained below.
```
python main.py <mode> <args>
```

### Modes


| **Mode**   | **Description**                                                          |
|------------|--------------------------------------------------------------------------|
| **image**  | Censors images in the specified input directory or file.                 |
| **video**  | Censors videos in the specified input directory or file.                 |
| **all**    | Censors both images and videos in the specified input directory or file. |
| **live**   | Real-time censoring of your screen or video stream.                      |
| **webcam** | Real-time censoring of webcam feed.                                      |
| **http**   | Starts an HTTP server for serving censored content via a web interface.  |


### Parameters Overview

| **Category**      | **Parameter**          | **Type** | **Description**                                                                     | **Default Value**                        |
|-------------------|------------------------|----------|-------------------------------------------------------------------------------------|------------------------------------------|
| **Core**          | `mode`                 | str      | Mode to use.                                                                        | -                                        |
|                   | `-i, --input`          | Path     | Path to input file or directory.                                                    | `./uncensored/`                          |
|                   | `-o, --output`         | Path     | Path to output directory.                                                           | `./censored/`                            |
| **Configuration** | `-p, --profile`        | str/Path | Config profile: a directory name in `./configs/`, or a path to a profile directory. | asks, or `default` if not in a terminal   |
|                   | `--config`             | Path     | Path to the main config file. Overrides the profile's one.                          | from profile                             |
|                   | `-cc, --censor-config` | Path     | Path to the censor config file. Overrides the profile's one.                        | from profile                             |
|                   | `-pl, --pipeline`      | Path     | Path to the pipeline config file (which models are loaded). Overrides the profile's. | from profile                             |
| **Live & Webcam** | `--live-mode`          | str      | Live mode: quick or precise. Quick is recommend.                                    | `quick`                                  |
|                   | `--device`             | int      | Input device to use. -1 means screenshots, >=0 needs to be a valid video device id. | `-1` on Live mode and `0` on Webcam mode |
|                   | `--vcam`               | bool     | Whether to output to a virtual camera.                                              | `False`                                  |
|                   | `--vcam-w`             | int      | Virtual camera width.                                                               | `1920`                                   |
|                   | `--vcam-h`             | int      | Virtual camera height.                                                              | `1080`                                   |
|                   | `--vcam-fps`           | int      | Virtual camera target fps.                                                          | `10`                                     |
| **General Flags** | `--override-cache`     | bool     | Recompute features by overriding the cache.                                         | `False`                                  |
|                   | `--skip-existing`      | bool     | Skip files that already exist in the output directory.                              | `False`                                  |
|                   | `--only-analyze`       | bool     | Only analyze images/videos (no censoring).                                          | `False`                                  |
|                   | `--debug`              | bool     | Enable debug mode.                                                                  | `False`                                  |
| **Server**        | `--port`               | int      | Port to run the server on.                                                          | `8443`                                   |
|                   | `--host`               | str      | Host to run the server on.                                                          | `localhost`                              |
|                   | `--ssl-cert`           | Path     | Path to certificate file.                                                           | `cert.pem`                               |
|                   | `--ssl-key`            | Path     | Path to key file.                                                                   | `key.pem`                                |


## Configuration

Configs are grouped in profiles. Each profile is a directory in `./configs/`:

```
configs/
├── default/
│   ├── config.yml
│   ├── censor_config.yml
│   └── pipeline.yml
└── <your-profile>/
    ├── censor_config.yml
    ├── pipeline.yml
    └── config.yml          (optional)
```

| **File**              | **Contents**                                                                    | **Required** |
|-----------------------|---------------------------------------------------------------------------------|--------------|
| `config.yml`          | General settings: image size, video fps, live capture region, cursor, etc.     | No, falls back to `configs/default/config.yml` |
| `pipeline.yml`        | Which models are loaded, on which device, workers and detection thresholds.     | Yes          |
| `censor_config.yml`   | What is censored and how. Can be swapped per run without reloading the models. | Yes          |

Select a profile with `--profile <name>` (or `--profile <path/to/profile/dir>`). Without it, you are asked to pick
one from a list (arrow keys + enter, `default` on top). When Censor Tool is not run in a terminal (e.g. in Docker),
the `default` profile is used.
`--config`, `--censor-config` and `--pipeline` override a single file of the selected profile, e.g.
`python main.py file --profile strict --pipeline ./my_pipeline.yml`.

During live and webcam censoring, changes to `config.yml` and `censor_config.yml` are applied as soon as the file
is saved (press `r` in the window to force a reload). An invalid file is reported and the previous settings are kept.
`pipeline.yml` is only read at startup.

### Pipeline config

`pipeline.yml` lists the (YOLO) models that are run. Relative model paths are relative to the working directory
(run Censor Tool from the project root).

- `backend`: `python` (default) or `cpp`, see [Backends](#backends).
- `device`: `cpu`, `cuda` or `cuda:N`. Set it at the top for all models, or per model.
- `min_confidence`: score floor per model. Detections below it are dropped before anything else,
  so keep it at or below the lowest `min_probability` in your censor configs.
- `labels`: renames labels. The segmentation model's `person` is renamed to `BODY`.
- `imgsz`: input size of a model with a dynamic input size (NudeNet). Larger finds smaller body parts but is slower
  (`640`, `1280`, `2560`, a multiple of 32). Models with a fixed input size ignore it.
- `n_workers`: number of worker threads that run the models (default `4`).
- `max_concurrent_jobs`: maximum number of jobs active at once, limit it to reduce RAM usage (default `128`, `0` = unlimited).

### Backends

Censoring is done by one of two backends with the same features and config files:

| Backend  | Install                               | Notes                                                                 |
|----------|---------------------------------------|-----------------------------------------------------------------------|
| `python` | included (`pip install -r requirements.txt`) | The default. Pure Python on top of onnxruntime and OpenCV.     |
| `cpp`    | build and install `censor_core` 2.0, e.g. `pip install /path/to/PyCensorCore` | Faster, mostly noticeable in live and video censoring. |

Pick one with `backend:` in `pipeline.yml`. If `cpp` is selected but `censor_core` is not installed, Censor Tool
stops with an error instead of silently switching backends. `python main.py validate` shows which backend is used.
Results of the two can differ slightly (a pixel here and there), not in what is detected or censored.
Both can share the cache folder; each keeps its own cache files.

### Censor config

`features_to_censor` lists the labels to censor. These are the labels the models in `pipeline.yml` produce:
the NudeNet labels (`FACE_FEMALE`, `FEMALE_BREAST_EXPOSED`, ...), `BODY` (the full body outline) and
`EYES_FEMALE`/`EYES_MALE` (eye bars). Unknown labels and misspelled keys are reported as warnings at startup.

Use `clip_to: [BODY]` on a feature to clip its censor area to the body outline.
