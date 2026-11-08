"""
All detection stages of a pipeline: the YOLO models (whole-image or cascaded on crops of other labels) and the
optional landmark stage. Stages run on demand: plan() works out which stages a censor config needs, run() executes
the ones whose results are not known yet.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from .censor_config import CensorConfig, Shape
from .config_utils import did_you_mean, log_warning
from .geometry import Detection, bounding_rect, intersect_rects, rect_empty
from .inference import InferenceWorker, OnnxSession
from .landmarks import postprocess_landmarks, preprocess_face_crop
from .pipeline_config import LANDMARKS_STAGE, PipelineConfig
from .yolo import PendingYolo, YoloRunner

if TYPE_CHECKING:
    from .pipeline import Job

# Detections per stage (model name, or "landmarks"). A stage that ran has an entry, even if empty.
StageDetections = dict[str, list[Detection]]


def _file_fingerprint(path: str) -> str:
    """Size plus the first and last MiB: cheap, and changes whenever the model is replaced."""
    try:
        size = os.path.getsize(path)
        chunk = 1 << 20
        with open(path, 'rb') as f:
            head = f.read(chunk)
            f.seek(max(0, size - chunk))
            tail = f.read(chunk)
    except OSError:
        return 'missing'
    return f"{size}:{hashlib.blake2b(head + tail, digest_size=8).hexdigest()}"


@dataclass
class Stage:
    name: str
    yolo: Optional[YoloRunner] = None           # YOLO stage
    landmarks: Optional[InferenceWorker] = None  # landmark stage
    run_on: list[str] = field(default_factory=list)          # YOLO cascade: parent labels
    crop_padding: float = 0.0
    eye_labels: dict[str, str] = field(default_factory=dict)  # landmarks: face label -> eye label
    min_face_score: float = 0.0

    outputs: list[str] = field(default_factory=list)   # labels it produces
    polygons: bool = False
    parents: list[str] = field(default_factory=list)   # stages whose detections it needs
    base_fingerprint: str = ''
    dynamic_input: bool = False  # result depends on the job's input size
    default_size: int = 0        # input size when the job asks for none


@dataclass
class Pending:
    yolo: list[PendingYolo] = field(default_factory=list)
    crops: list[tuple[str, int, tuple[int, int]]] = field(default_factory=list)  # parent stage, parent, offset
    batch: Optional[Future] = None
    faces: list[tuple[str, int, tuple[int, int, int, int], str, float]] = field(default_factory=list)


@dataclass
class Plan:
    stages: list[str] = field(default_factory=list)  # run when missing, in dependency order
    lazy: list[str] = field(default_factory=list)    # clip_to-only stages, run when a detection needs clipping


class ModelGraph:
    def __init__(self, config: PipelineConfig, parallel: bool):
        self._parallel = parallel
        self._stages: list[Stage] = []

        # ------------------------------------ stages ------------------------------------
        for m in config.models:
            runner = YoloRunner(m.name, m.path, m.device, m.options)
            model = runner.model
            s = Stage(name=m.name, yolo=runner)
            s.outputs = model.output_labels()
            if not s.outputs:
                raise RuntimeError(f"model '{m.name}' has no class names; list them with names: [...]")
            s.polygons = model.task == 'segment'
            s.run_on = list(m.run_on)
            s.crop_padding = m.crop_padding
            s.dynamic_input = model.dynamic and not m.run_on
            s.default_size = max(model.input_size)
            o = model.options
            s.base_fingerprint = '|'.join(str(v) for v in (
                'yolo', _file_fingerprint(m.path), o.resize, model.layout,
                f"{model.input_size[0]}x{model.input_size[1]}", o.min_confidence, o.nms_iou, o.agnostic_nms,
                o.max_det, o.mask_threshold, ','.join(model.names), ','.join(s.outputs), ','.join(m.run_on),
                m.crop_padding))
            self._stages.append(s)

        if config.landmarks:
            lm = config.landmarks
            s = Stage(name=LANDMARKS_STAGE,
                      landmarks=InferenceWorker(OnnxSession(lm.path, lm.device, LANDMARKS_STAGE)),
                      eye_labels=dict(lm.run_on), min_face_score=lm.min_face_score, polygons=True)
            for eye in lm.run_on.values():
                if eye not in s.outputs:
                    s.outputs.append(eye)
            s.base_fingerprint = f"landmarks|{_file_fingerprint(lm.path)}|{lm.min_face_score}|" + \
                ''.join(f"{face}>{eye}," for face, eye in sorted(lm.run_on.items()))
            self._stages.append(s)

        self._by_name = {s.name: s for s in self._stages}
        self.stage_names: list[str] = [s.name for s in self._stages]

        # ------------------------------------ labels ------------------------------------
        self.labels: dict[str, str] = {}
        for s in self._stages:
            for label in s.outputs:
                if label in self.labels:
                    raise RuntimeError(f"label '{label}' is produced by both '{self.labels[label]}' and '{s.name}'; "
                                       f"rename one with labels: or label_prefix:")
                self.labels[label] = s.name
        self.labels = dict(sorted(self.labels.items()))

        # ------------------------------------ parents ------------------------------------
        for s in self._stages:
            for label in list(s.run_on) + list(s.eye_labels):
                owner = self.labels.get(label)
                if owner is None:
                    raise RuntimeError(f"stage '{s.name}' runs on label '{label}', which no model produces"
                                       f"{did_you_mean(label, list(self.labels))}")
                if owner == s.name:
                    raise RuntimeError(f"model '{s.name}' cannot run on its own label '{label}'")
                if owner not in s.parents:
                    s.parents.append(owner)

        # ------------------------------------ dependency order ------------------------------------
        self._topo: list[str] = []
        placed: set[str] = set()
        while len(self._topo) < len(self._stages):
            progress = False
            for s in self._stages:
                if s.name not in placed and all(p in placed for p in s.parents):
                    self._topo.append(s.name)
                    placed.add(s.name)
                    progress = True
            if not progress:
                raise RuntimeError("the models' run_on settings form a cycle")

        self._warn_lock = threading.Lock()
        self._warned: set[str] = set()

    # ------------------------------------ lifecycle ------------------------------------
    def start(self) -> None:
        for s in self._stages:
            (s.yolo or s.landmarks).start()

    def stop(self) -> None:
        for s in self._stages:
            (s.yolo or s.landmarks).stop()

    def stage(self, name: str) -> Stage:
        return self._by_name[name]

    # ------------------------------------ fingerprints ------------------------------------
    def _fingerprint(self, s: Stage, size: int) -> str:
        text = s.base_fingerprint
        if s.dynamic_input:
            text += f"|size={size if size > 0 else s.default_size}"
        for p in s.parents:
            text += '|' + self._fingerprint(self.stage(p), size)
        return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()

    def fingerprints(self, size: int) -> dict[str, str]:
        """Cache fingerprint of every stage for a job input size (covers model file, options and parents)."""
        return {s.name: self._fingerprint(s, size) for s in self._stages}

    # ------------------------------------ planning ------------------------------------
    def plan(self, config: Optional[CensorConfig]) -> Plan:
        """The stages a censor config needs; every stage when config is None."""
        plan = Plan()
        if config is None:
            plan.stages = list(self._topo)
            return plan

        needed: set[str] = set()

        def require(name: str) -> None:
            if name in needed:
                return
            needed.add(name)
            for p in self.stage(name).parents:
                require(p)

        clip_targets: set[str] = set()
        for label, box in config.features_to_censor.items():
            if label in self.labels:
                require(self.labels[label])
            clip_targets.update(box.clip_to)

        clip_stages = {self.labels[t] for t in clip_targets if t in self.labels}
        for name in self._topo:
            if name not in clip_stages or name in needed:
                continue
            if self._parallel or self.stage(name).parents:
                require(name)
            else:
                plan.lazy.append(name)

        plan.stages = [name for name in self._topo if name in needed]
        return plan

    def _lazy_needed(self, s: Stage, dets: StageDetections, config: Optional[CensorConfig]) -> bool:
        if config is None:
            return False
        for detections in dets.values():
            for d in detections:
                box = config.features_to_censor.get(d.label)
                if box is None or d.score < box.min_probability:
                    continue
                if any(target in s.outputs for target in box.clip_to):
                    return True
        return False

    def missing(self, plan: Plan, dets: StageDetections, config: Optional[CensorConfig]) -> list[str]:
        """The planned stages that dets lacks (lazy ones only when a detection needs them)."""
        out = [name for name in plan.stages if name not in dets]
        out += [name for name in plan.lazy if name not in dets and self._lazy_needed(self.stage(name), dets, config)]
        return out

    # ------------------------------------ validation ------------------------------------
    def validate(self, config: CensorConfig) -> None:
        """
        Checks a censor config against the pipeline: errors for clip_to targets and shape MASK that cannot work,
        a warning (once per label) for labels no model produces.
        """
        all_labels = list(self.labels)
        for label, box in config.features_to_censor.items():
            owner = self.labels.get(label)
            if owner is None:
                with self._warn_lock:
                    if label not in self._warned:
                        self._warned.add(label)
                        log_warning(f"No model produces label '{label}'{did_you_mean(label, all_labels)}; "
                                    f"it is never censored.")
                continue
            if box.shape == Shape.MASK and not self.stage(owner).polygons:
                raise RuntimeError(f"shape MASK for '{label}' needs a segmentation model, but '{owner}' "
                                   f"is a detect model")
            for target in box.clip_to:
                t = self.labels.get(target)
                if t is None:
                    raise RuntimeError(f"clip_to of '{label}': no model produces label '{target}'"
                                       f"{did_you_mean(target, all_labels)}")
                if not self.stage(t).polygons:
                    raise RuntimeError(f"clip_to of '{label}': '{target}' comes from detect model '{t}'; "
                                       f"clip_to needs a segmentation model")

    # ------------------------------------ execution ------------------------------------
    def _submit(self, s: Stage, job: 'Job', dets: StageDetections) -> Pending:
        p = Pending()
        image = job.image
        image_rect = (0, 0, image.shape[1], image.shape[0])

        if s.yolo is not None and not s.run_on:
            p.yolo.append(s.yolo.submit(image, job.size))
            p.crops.append(('', -1, (0, 0)))
            return p

        if s.yolo is not None:
            # Cascade: run on a padded crop of every parent detection with a run_on label.
            for parent_name in s.parents:
                for i, d in enumerate(dets[parent_name]):
                    if d.label not in s.run_on:
                        continue
                    pad_x = d.box[2] * s.crop_padding / 2.0
                    pad_y = d.box[3] * s.crop_padding / 2.0
                    x1, y1 = math.floor(d.box[0] - pad_x), math.floor(d.box[1] - pad_y)
                    x2 = math.ceil(d.box[0] + d.box[2] + pad_x)
                    y2 = math.ceil(d.box[1] + d.box[3] + pad_y)
                    rx, ry, rw, rh = intersect_rects((x1, y1, x2 - x1, y2 - y1), image_rect)
                    if rw < 2 or rh < 2:
                        continue
                    p.yolo.append(s.yolo.submit(image[ry:ry + rh, rx:rx + rw], 0))
                    p.crops.append((parent_name, i, (rx, ry)))
            return p

        # Landmarks: one batch with a crop of every face that has an eye label.
        inputs = []
        for face_label, eye_label in s.eye_labels.items():
            owner = self.labels[face_label]
            for i, f in enumerate(dets[owner]):
                if f.label != face_label or f.score < s.min_face_score:
                    continue
                rect = intersect_rects(tuple(int(v) for v in f.box), image_rect)
                if rect_empty(rect):
                    continue
                x, y, w, h = rect
                inputs.append(preprocess_face_crop(image[y:y + h, x:x + w]))
                p.faces.append((owner, i, rect, eye_label, f.score))
        if inputs:
            p.batch = s.landmarks.submit(inputs)
        return p

    def _collect(self, s: Stage, p: Pending) -> list[Detection]:
        out: list[Detection] = []
        if s.yolo is not None:
            for pending, (parent_stage, parent, (ox, oy)) in zip(p.yolo, p.crops):
                for d in s.yolo.collect(pending):
                    if ox or oy:
                        d.box = (d.box[0] + ox, d.box[1] + oy, d.box[2], d.box[3])
                        d.polygons = [part + np.array([ox, oy], dtype=np.int32) for part in d.polygons]
                    d.parent_stage, d.parent = parent_stage, parent
                    out.append(d)
            return out

        if not p.faces:
            return out
        outputs = p.batch.result()
        if len(outputs) != len(p.faces):
            raise RuntimeError(f"[{s.name}] expected {len(p.faces)} outputs")
        for output, (parent_stage, parent, rect, eye_label, score) in zip(outputs, p.faces):
            shape = postprocess_landmarks(output, rect)
            if len(shape) == 0:
                continue
            x, y, w, h = bounding_rect(shape)
            out.append(Detection(label=eye_label, score=score, box=(float(x), float(y), float(w), float(h)),
                                 polygons=[shape], parent_stage=parent_stage, parent=parent))
        return out

    def run(self, job: 'Job', dets: StageDetections, plan: Plan, config: Optional[CensorConfig]) -> list[str]:
        """
        Runs the planned stages missing from dets and adds their results. Stops early (without error) when the
        job is cancelled. Raises on failure. Returns the stages that ran.
        """
        ran: list[str] = []

        def run_wave(wave: list[str]) -> None:
            if self._parallel:
                pending = [self._submit(self.stage(name), job, dets) for name in wave]
                for name, p in zip(wave, pending):
                    dets[name] = self._collect(self.stage(name), p)
            else:
                for name in wave:
                    dets[name] = self._collect(self.stage(name), self._submit(self.stage(name), job, dets))
            ran.extend(wave)

        todo = [name for name in plan.stages if name not in dets]
        while todo:
            if job.cancelled:
                return ran
            wave = [n for n in todo if all(p in dets for p in self.stage(n).parents)]
            rest = [n for n in todo if n not in wave]
            if not wave:
                raise RuntimeError("ModelGraph: a planned stage is missing its parent")
            run_wave(wave)
            todo = rest

        for name in plan.lazy:
            if job.cancelled:
                return ran
            if name not in dets and self._lazy_needed(self.stage(name), dets, config):
                run_wave([name])
        return ran
