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
- [ ] Automatic model downloads

## Installation Guide

### Requirements
You will need two programs for the easiest installation:
1. [git](https://git-scm.com/downloads)
2. Your favorite program to create virtual environments. I prefer [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation), so that is what will be used in this guide.

For GPU support you'll need a Nvidia GPU and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.

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
5. Run the init process. This will create all required directories.
    ```
    python main.py init
    ```
6. Download the detection models.
   * Download NudeNet from https://github.com/notAI-tech/NudeNet/releases/tag/v0. 
     Download ```detector_v2_default_checkpoint.onnx``` and place it in the ```./data/models/``` folder.
   * Download the eye detection model (only needed if you want to enable eye detection) from
     https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_eye.xml
     and place it in the ```./data/models/``` folder.
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
| **Configuration** | `--config`             | Path     | Path to the main config file.                                                       | `./config.yml`                           |
|                   | `-cc, --censor-config` | Path     | Path to the censor config file.                                                     | `./default_censor_config.yml`            |
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
