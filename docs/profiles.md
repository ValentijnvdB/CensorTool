# Profiles, folders and instances

## Profiles

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
| `config.yml` | General settings: video sampling rate, live capture region, ComfyUI connection, server. | [config.md](config.md) |
| `pipeline.yml` | The backend, which models run, and on which device. | [pipeline_config.md](pipeline_config.md) |
| `censor_config.yml` | What is censored and how. | [censor_config.md](censor_config.md) |

A profile needs all three files. Changes to `config.yml` and `censor_config.yml` are picked up while live, webcam
or overlay censoring runs; `pipeline.yml` is only read at startup.

### Picking a profile

Pick a profile with `--profile <name>` or `--profile <path/to/folder>`. Without it you are asked to choose one
from a list (arrow keys and enter). If there is only one profile it is used without asking; when Censor Tool
doesn't run in an interactive terminal, the first profile alphabetically is used, with a warning.

`--config`, `--censor-config` and `--pipeline` replace a single file of the profile, e.g.
`python main.py file --profile nudity --censor-config ./strict.yml`.

### Included profiles

| Profile | What it censors | Models |
|---------|-----------------|--------|
| `nudity` | Nudity and faces | Nevermore big, see [Nudity](models.md#nudity-nudity-profile) |
| `privacy` | Faces and license plates | License plate and face detection, see [Privacy](models.md#privacy-privacy-profile) |

Both use the Python backend on the CPU; set `backend` and `device` in their `pipeline.yml` to change that (see
[Backends](backends.md)).

### Your own profiles

To make your own profile, copy one of the included profiles to `configs/<name>/` and edit the files. Profiles
other than `nudity` and `privacy` are ignored by git, so your own profiles stay out of the repository.

## Folders

All folders below are relative to the working directory, see [Instances](#instances).

| Folder | Contents |
|--------|----------|
| `data/uncensored/` | Default input. |
| `data/censored/` | Default output. |
| `data/stickers/<category>/` | Sticker images for [overlays](censor_config.md#overlays). |
| `data/server/` | Files uploaded to and produced by the HTTP server. |
| `data/logs/` | Log files. |
| `cache/` | Cached detections (`images/`, `videos/`, `server/`). Safe to delete. |
| `models/` | The [models](models.md). |
| `configs/` | The profiles. |
| `certs/` | The [HTTPS certificate and key](modes/http.md#https). |

`python main.py init` creates the data folders.

## Instances

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
