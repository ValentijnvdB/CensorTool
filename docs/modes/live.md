# Live and webcam censoring (`live`, `webcam`)

```
python main.py live      # your screen
python main.py webcam    # webcam 0, or --device <id>
```

`live` captures the region set in `config.yml` (`live_cap_*`, the full 1920x1080 screen by default) and shows it
censored in a window. This works best with two displays: the content on one, the censored window on the other.
`webcam` does the same with a camera. To censor a window in place, on a single display, use
[overlay mode](overlay.md) instead.

With `--vcam` the output goes to a virtual camera instead of a window, so it can be used in video calls or OBS
(`--vcam-w`, `--vcam-h` and `--vcam-fps` set its size and frame rate). This needs a virtual camera driver, see
[pyvirtualcam](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras) (OBS on Windows and macOS,
v4l2loopback on Linux).

In the window, press `q` to quit and `r` to reload the configs. Changes to `config.yml` and `censor_config.yml`
are also picked up as soon as they are saved, so you can tune the censoring while it runs. `pipeline.yml` is only
read at startup.

Real-time censoring works on a CPU, but a GPU makes it a lot smoother. The capture region, input size and other live
settings are described in [config.md](../config.md#live-webcam-and-overlay). If it is slow, see
[Troubleshooting](../troubleshooting.md).

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `-1` (live), `0` (webcam) | Input: `-1` is the screen, `0` or higher a camera id. |
| `--vcam` | off | Output to a virtual camera instead of a window. |
| `--vcam-w`, `--vcam-h` | `1920`, `1080` | Virtual camera size. |
| `--vcam-fps` | `10` | Virtual camera frame rate. |

The [general arguments](../../README.md#usage) (`--profile`, `--debug`, ...) work as well.
