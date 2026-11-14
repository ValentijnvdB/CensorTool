# Checking what works (`validate`)

```
python main.py validate [--profile <name> | --all-profiles] [--only file,live,webcam,overlay,http] [--quick]
```

Runs the checks of every mode with a profile and prints `PASS`, `WARN`, `FAIL` or `SKIP` for each check with a
hint on how to fix it, then a verdict per mode ("should work", "NOT WORKING: <reason>" or "not tested: <reason>",
e.g. without a camera). The profile, the models and the censoring pipeline are always checked; per mode:

| Mode | Checks |
|------|--------|
| `file` | Every image type from file to file, `ffmpeg`/`ffprobe`, a test video with sound and a GIF. |
| `live` | Capturing the `live_cap_*` region (it must be on a screen) and OpenCV window support. |
| `webcam` | Reading a frame from camera `--device` (default 0). |
| `overlay` | The window system, and capturing the focused window. |
| `http` | `--host`/`--port` are free, the `--ssl` certificate loads, and a real server answers `/health`, `/profile` and `/censor_image`. |

Optional extras are reported separately and never make a mode fail: whether the GPU is really used (with
`device: cuda`; both backends fall back to the CPU silently) or would work, and the virtual camera (`--vcam`).
ComfyUI and the stickers are only checked when the censor config uses them. It also shows which
[backend](../backends.md) is used.

Test files go to a temporary folder; nothing in `data/` is changed. The exit code is 1 when a check failed.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--only MODES` | all | Only check these modes, comma separated (`file,live,webcam,overlay,http`). |
| `--quick` | off | Skip the slow checks (videos, GIFs, starting the server). |
| `--all-profiles` | off | Check every profile in `configs/` instead of one, each in its own process. |

The arguments of the other modes (`--device`, `--host`, `--port`, `--ssl`, `--ssl-cert`, `--ssl-key`) set what is
checked, and the [general arguments](../../README.md#usage) work as well.
