"""YOLO detection and tracking.

Every inference parameter is passed explicitly. The legacy call site invoked
``model.track()`` with no ``conf``, ``imgsz``, ``device`` or ``half`` and relied
on whatever defaults the installed Ultralytics version happened to use, while
the ``--conf`` and ``--fps`` command line flags were validated, logged, and then
never passed to anything. Tuning those knobs changed nothing, which made every
past tuning result uninterpretable.

Coordinate translation from ROI space back to full-frame space happens in
exactly one place, :meth:`Detector._to_full_frame`, so it can be tested.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ModelConfig, RoiConfig

BBox = tuple[float, float, float, float]


def _fixed_input_size(model_path: Path) -> int | None:
    """The one input size a CoreML export will accept, if it is fixed.

    A ``.mlpackage`` is exported at a single input resolution and, unless it
    was given size flexibility (none of this repo's exports were), CoreML
    refuses anything else outright:

        RuntimeError: Image size 640 x 640 not in allowed set of image sizes

    Returns None when the model is flexible, is not CoreML, or cannot be
    inspected -- in all of which cases ``imgsz`` means what it says.
    """
    if model_path.suffix != ".mlpackage":
        return None
    try:
        import coremltools as ct

        spec = ct.models.MLModel(
            str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY
        ).get_spec()
    except Exception:
        return None  # best effort: never block startup on an inspection failure
    for descriptor in spec.description.input:
        image = descriptor.type.imageType
        width, height = int(image.width), int(image.height)
        if width and height and not image.WhichOneof("SizeFlexibility"):
            return width
    return None


@dataclass
class Detection:
    """One detected object, in full-frame pixel coordinates."""

    xyxy: BBox
    conf: float
    cls: int
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Roi:
    """A rectangular crop of the frame, derived from a normalized polygon."""

    def __init__(self, config: RoiConfig, frame_width: int, frame_height: int):
        self.enabled = config.enabled and bool(config.polygon)
        self.scale = config.scale
        if not self.enabled:
            self.x1, self.y1 = 0, 0
            self.x2, self.y2 = frame_width, frame_height
        else:
            xs = [p[0] * frame_width for p in config.polygon]
            ys = [p[1] * frame_height for p in config.polygon]
            self.x1 = max(0, int(min(xs)))
            self.y1 = max(0, int(min(ys)))
            self.x2 = min(frame_width, int(max(xs)))
            self.y2 = min(frame_height, int(max(ys)))
            if self.x2 <= self.x1 or self.y2 <= self.y1:
                raise ValueError(
                    "ROI polygon collapses to an empty rectangle; recalibrate it."
                )

    def crop(self, image: np.ndarray) -> np.ndarray:
        region = image[self.y1 : self.y2, self.x1 : self.x2]
        if self.scale != 1.0:
            region = cv2.resize(
                region,
                (int(region.shape[1] * self.scale), int(region.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
        return region

    def to_full_frame(self, box: BBox) -> BBox:
        """Map a box from cropped/scaled space back to full-frame pixels."""
        x1, y1, x2, y2 = box
        if self.scale != 1.0:
            x1, y1, x2, y2 = (c / self.scale for c in (x1, y1, x2, y2))
        return (x1 + self.x1, y1 + self.y1, x2 + self.x1, y2 + self.y1)


class Detector:
    """Stateful YOLO tracker over a fixed frame geometry."""

    def __init__(
        self,
        model_config: ModelConfig,
        roi_config: RoiConfig,
        frame_width: int,
        frame_height: int,
    ):
        from ultralytics import YOLO  # imported lazily: heavy and optional for tests

        model_path = Path(model_config.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        tracker_path = Path(model_config.tracker)
        if not tracker_path.exists():
            raise FileNotFoundError(f"Tracker config not found: {tracker_path}")

        self.config = model_config
        self.roi = Roi(roi_config, frame_width, frame_height)
        self.model = YOLO(str(model_path))

        # Second stage may run its own, smaller export. Sharing one YOLO object
        # across the two passes would also be a threading hazard if the second
        # stage ever moves off the frame loop, so keep them separate objects.
        self.second_stage = None
        self.crops_skipped = 0
        self.two_stage_errors = 0
        self.two_stage_last_error: str | None = None
        self.warnings: list[str] = []

        # Effective sizes, which are not necessarily the configured ones. A
        # CoreML export accepts exactly one input size and *raises* on any
        # other, so an imgsz the model cannot honour is not a slow path or a
        # quality trade -- it is a hard failure on every inference. Resolve it
        # here, once, and say so, rather than discovering it per frame.
        self.imgsz = model_config.imgsz
        self.two_stage_imgsz = model_config.two_stage_imgsz

        fixed = _fixed_input_size(model_path)
        if fixed is not None and fixed != self.imgsz:
            self.warnings.append(
                f"imgsz={self.imgsz} but {model_path.name} is exported at a fixed "
                f"{fixed}x{fixed} and rejects any other size. Running at {fixed}. "
                f"Set model.imgsz={fixed} so the config describes what happens."
            )
            self.imgsz = fixed

        if model_config.two_stage:
            second_path = Path(model_config.two_stage_model or model_config.path)
            if not second_path.exists():
                raise FileNotFoundError(f"Two-stage model not found: {second_path}")
            self.second_stage = (
                self.model if second_path == model_path else YOLO(str(second_path))
            )
            second_fixed = (
                fixed if second_path == model_path else _fixed_input_size(second_path)
            )
            if second_fixed is not None and second_fixed != self.two_stage_imgsz:
                self.warnings.append(
                    f"two_stage_imgsz={self.two_stage_imgsz} but {second_path.name} "
                    f"is exported at a fixed {second_fixed}x{second_fixed} and "
                    f"rejects any other size. Running crops at {second_fixed}, which "
                    f"is not the speedup two-stage is for -- export a "
                    f"{self.two_stage_imgsz}px model and set model.two_stage_model "
                    f"to it."
                )
                self.two_stage_imgsz = second_fixed

    def warmup(self, frame_width: int, frame_height: int) -> float:
        """Run one throwaway inference so the first real frame isn't slow.

        Loading the CoreML model and running its first inference costs a few
        seconds; paying that on the first frame of a race means the pipeline
        is blind while it happens. Uses ``predict`` rather than ``track`` so
        the tracker's state isn't seeded with detections from a blank frame.
        """
        started = time.time()
        blank = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        try:
            self.model.predict(
                blank,
                conf=self.config.conf,
                imgsz=self.imgsz,
                device=self.config.device,
                half=self.config.half,
                verbose=False,
            )
        except Exception:
            pass
        if self.second_stage is not None and self.second_stage is not self.model:
            # Person-shaped, not frame-shaped: the second stage never sees a
            # full frame, and the first inference cost is paid per input shape.
            crop = np.zeros((frame_height // 3, frame_height // 6, 3), dtype=np.uint8)
            try:
                self.second_stage.predict(
                    crop,
                    conf=self.config.conf,
                    imgsz=self.two_stage_imgsz,
                    device=self.config.device,
                    half=self.config.half,
                    verbose=False,
                )
            except Exception:
                pass
        return time.time() - started

    def track(self, image: np.ndarray) -> list[Detection]:
        """Run detection + tracking on one frame.

        Returns detections in full-frame coordinates. Untracked detections keep
        ``track_id=None`` rather than being dropped, so bib boxes (which are not
        tracked) still reach the OCR stage.
        """
        region = self.roi.crop(image)
        results = self.model.track(
            region,
            persist=True,
            tracker=self.config.tracker,
            classes=[self.config.person_class, self.config.bib_class],
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.imgsz,
            device=self.config.device,
            half=self.config.half,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for box in boxes:
            raw = tuple(float(v) for v in box.xyxy[0].tolist())
            track_id = None if box.id is None else int(box.id.item())
            detections.append(
                Detection(
                    xyxy=self.roi.to_full_frame(raw),
                    conf=float(box.conf.item()),
                    cls=int(box.cls.item()),
                    track_id=track_id,
                )
            )
        return detections

    @staticmethod
    def merge(primary: list[Detection], extra: list[Detection], iou: float = 0.5):
        """Add `extra` detections, dropping ones that duplicate `primary`.

        The two-stage pass re-examines a region the full-frame pass already
        looked at, so any bib visible at both scales gets found twice. Left
        unmerged that produces stacked near-identical boxes: noise for a human
        reviewer, and duplicated supervision if the output is trained on.
        """
        def iou_of(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        merged = list(primary)
        for candidate in extra:
            if any(
                candidate.cls == kept.cls and iou_of(candidate.xyxy, kept.xyxy) >= iou
                for kept in merged
            ):
                continue
            merged.append(candidate)
        return merged

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect on a single image with no tracking state.

        ``track()`` runs ByteTrack with ``persist=True``, which is right for
        consecutive video frames and badly wrong for a folder of unrelated
        stills: the tracker tries to associate each image's detections with
        tracks from the previous, unrelated image and drops the ones that don't
        match. Anything iterating over independent images wants this instead.
        """
        region = self.roi.crop(image)
        results = self.model.predict(
            region,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.imgsz,
            device=self.config.device,
            half=self.config.half,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        return [
            Detection(
                xyxy=self.roi.to_full_frame(
                    tuple(float(v) for v in box.xyxy[0].tolist())
                ),
                conf=float(box.conf.item()),
                cls=int(box.cls.item()),
                track_id=None,
            )
            for box in boxes
        ]

    def bibs_in_people(
        self, image: np.ndarray, people: list[Detection]
    ) -> list[Detection]:
        """Second-stage pass: look for a bib inside each person's crop.

        A bib is tiny once a 1920px frame is squeezed into the model's 640px
        input -- median 15px wide in finish-line footage. Cropping to a single
        runner and feeding that to the same input gives the bib hundreds of
        pixels instead, which is a far larger effective upscale than raising
        the whole frame's resolution, at a fraction of the cost.

        Coordinates come back translated to full-frame space. Returns bib
        detections only; person boxes from the second pass are discarded
        because the first pass already owns tracking.

        Cost is one inference *per person*, so ``two_stage_max_crops`` bounds
        the worst case. When a pack exceeds it the largest boxes win -- they
        are the runners nearest the camera, hence nearest the line, hence the
        ones about to finish. Taking the list in detector order instead would
        make the choice arbitrary.
        """
        if not people:
            return []
        crops, origins = [], []
        height, width = image.shape[:2]
        ranked = sorted(
            people,
            key=lambda p: (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1]),
            reverse=True,
        )
        self.crops_skipped += max(0, len(ranked) - self.config.two_stage_max_crops)
        for person in ranked[: self.config.two_stage_max_crops]:
            x1, y1, x2, y2 = person.xyxy
            pad_x = (x2 - x1) * self.config.two_stage_padding
            pad_y = (y2 - y1) * self.config.two_stage_padding
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(width, int(x2 + pad_x))
            cy2 = min(height, int(y2 + pad_y))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crops.append(image[cy1:cy2, cx1:cx2])
            origins.append((cx1, cy1, cx2 - cx1, cy2 - cy1))
        if not crops:
            return []

        # One crop at a time rather than one batched call. The CoreML export is
        # fixed at batch=1, and handing it a list of differently-sized crops
        # makes ultralytics return fewer results than inputs and then index off
        # the end. Batching bought nothing here anyway: a batch-1 backend runs
        # them sequentially regardless.
        model = self.second_stage or self.model
        found: list[Detection] = []
        for crop, (ox, oy, cw, ch) in zip(crops, origins):
            try:
                results = model.predict(
                    crop,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    imgsz=self.two_stage_imgsz,
                    device=self.config.device,
                    half=self.config.half,
                    verbose=False,
                )
            except Exception as exc:
                # Never silent. This swallowed a RuntimeError on *every* crop
                # when two_stage_imgsz did not match the CoreML export's fixed
                # input, so the second stage found nothing at all while the
                # config, the runbook and second_stage_bibs=0 all read as
                # "enabled, just not finding extra bibs". A feature that fails
                # closed and reports success is worse than one that is off.
                self.two_stage_errors += 1
                self.two_stage_last_error = f"{type(exc).__name__}: {exc}"
                continue
            if not results:
                continue
            boxes = getattr(results[0], "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                if int(box.cls.item()) != self.config.bib_class:
                    continue
                bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0].tolist())
                found.append(
                    Detection(
                        xyxy=(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                        conf=float(box.conf.item()),
                        cls=self.config.bib_class,
                        track_id=None,
                    )
                )
        return found

    def split(
        self, detections: list[Detection]
    ) -> tuple[list[Detection], list[Detection]]:
        """Partition detections into (tracked people, bibs).

        Unlike the legacy pipeline there is no geometric gate here. Every person
        in frame is tracked; the finish line alone decides who finished. The old
        code filtered ``tracked_persons`` to a hardcoded wedge, so a racer
        outside it could never finish no matter how clearly they were detected.
        """
        people = [
            d
            for d in detections
            if d.cls == self.config.person_class and d.track_id is not None
        ]
        bibs = [d for d in detections if d.cls == self.config.bib_class]
        return people, bibs
