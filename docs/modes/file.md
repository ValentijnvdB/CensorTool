# Censoring files (`file`)

```
python main.py file [-i <input>] [-o <output>] [-r]
```

`f` and `files` work as aliases of `file`.

Censors every image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, `.avif`) and video (`.mp4`, `.avi`, `.gif`)
in `data/uncensored/` and writes the results to `data/censored/`. `-i` can be a folder or a single file; `-r`
includes subfolders. Videos and GIFs need [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`.

Detections are cached in `cache/`, so censoring the same file again with a different censor config only redoes the
drawing, not the detection. Use `--override-cache` to detect again, `--skip-existing` to skip files that already
exist in the output folder, and `--only-analyze` to fill the cache without censoring.

How many frames per second of a video are run through the models is set with `video_censor_fps` in
[config.yml](../config.md#video).

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i`, `--input` | `data/uncensored` | Input file or folder. |
| `-o`, `--output` | `data/censored` | Output folder. |
| `-r`, `--recursive` | off | Include subfolders of the input folder. |
| `--override-cache` | off | Detect again instead of using the cache. |
| `--skip-existing` | off | Skip files that already exist in the output folder. |
| `--only-analyze` | off | Only detect (fills the cache), don't censor. |

The [general arguments](../../README.md#usage) (`--profile`, `--debug`, ...) work as well.
