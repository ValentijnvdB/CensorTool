# Models

The models are not included in the repository. Each profile lists the models it uses in its `pipeline.yml`; download
them to the paths below, or change the paths in `pipeline.yml`. Relative paths are relative to the working directory,
see [Instances](profiles.md#instances). All model settings are described in [pipeline_config.md](pipeline_config.md).

## Nudity (`nudity` profile)

| Model | Used for | Path | Download |
|-------|----------|------|----------|
| Nevermore big | Faces, breasts, genitals, buttocks, feet and people. **Required.** | `models/nevermore/det_v11_big/det_v11_big_640.onnx` | [Hugging Face](https://huggingface.co/kinkyrvn/nevermore/tree/main/det_v11_big) |
| YOLO segmentation | Optional: body outlines (`BODY`), for full-body censoring and `clip_to`. | `models/yolo/yolo26n-seg.onnx` | Export with [Ultralytics](https://docs.ultralytics.com/modes/export/), see [below](#body-segmentation) |
| PFLD 68-point landmarks | Optional: eye bars (`eyes`). | `models/eye_detection/pfld.onnx` | e.g. [PFLD_68points_Pytorch](https://github.com/github-luffy/PFLD_68points_Pytorch), exported to ONNX (112x112 input) |

Nevermore comes in more sizes (`tiny` to `huge`) and fp16 versions, see its
[model card](https://huggingface.co/kinkyrvn/nevermore). To use another size, change `path` in `pipeline.yml`.

The optional models are commented out in `configs/nudity/pipeline.yml`; uncomment the `people` model for body
outlines and the `landmarks` section for eye bars.

### Body segmentation

To make the segmentation model, install Ultralytics (only needed for the export) and export a YOLO segmentation model
to ONNX:

```
pip install ultralytics
yolo export model=yolo26n-seg.pt format=onnx
```

`n` (nano) is the fastest; `s`, `m`, `l` and `x` are more accurate.

### NudeNet

[NudeNet](https://github.com/notAI-tech/NudeNet/releases) (`640m.onnx`) works as an alternative to Nevermore. It has
different labels (`FACE_FEMALE`, `FEMALE_BREAST_EXPOSED`, ...), so change both `pipeline.yml` and
`censor_config.yml`; see the NudeNet examples in [pipeline_config.md](pipeline_config.md#example-setups) and
[censor_config.md](censor_config.md#labels).

## Privacy (`privacy` profile)

Blurs faces and puts a bar over license plates, e.g. in photos, dashcam or street footage.

| Model | Used for | Path | Download |
|-------|----------|------|----------|
| YOLO11 license plate detection | License plates (`license_plate`). | `models/license_plates/license-plate-finetune-v1n.onnx` | [Hugging Face](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) (AGPL-3.0) |
| YOLO11 face detection | Faces (`face`). | `models/faces/yolov11n-face.onnx` | [Hugging Face](https://huggingface.co/AdamCodd/YOLOv11n-face-detection), download `model.onnx` and rename it (Apache-2.0) |

Both come in larger, more accurate versions (`v1s` to `v1x` for plates, `YOLOv11x` for faces).

```
python main.py file --profile privacy -i <photos or videos>
```

## Other models

Censor Tool runs any YOLO model exported to ONNX (YOLOv8, YOLO11, YOLO26 and other Ultralytics exports, detection or
segmentation), so you can censor anything a model is trained on. Some places to look:

- Any COCO model, such as the [segmentation model](#body-segmentation) above, finds 80 everyday objects: `person`,
  `car`, `laptop`, `tv`, `cell phone`, ... Useful to hide screens and phones in screen recordings.
- [Hugging Face](https://huggingface.co/models?search=yolo) and [Roboflow Universe](https://universe.roboflow.com/)
  have models for logos, weapons, smoking, ID cards and much more.

To add a model:

1. Get the model as an `.onnx` file. Models published as `.pt` files have to be exported first:
   `yolo export model=<file>.pt format=onnx` (needs `pip install ultralytics`).
2. Put it in `models/`, e.g. `models/logos/logos.onnx`.
3. Add it to `models:` in the profile's `pipeline.yml` (or make a new profile, see [Profiles](profiles.md)):
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
  [pipeline_config.md](pipeline_config.md#models) for all settings.
- Check a model's license before you use it; many YOLO models are AGPL-3.0.
