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
                imgsz=self.config.imgsz,
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
            imgsz=self.config.imgsz,
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
            imgsz=self.config.imgsz,
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

        Crops go through in one batched call, and coordinates come back
        translated to full-frame space. Returns bib detections only; person
        boxes from the second pass are discarded because the first pass already
        owns tracking.
        """
        if not people:
            return []
        crops, origins = [], []
        height, width = image.shape[:2]
        for person in people[: self.config.two_stage_max_crops]:
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
        found: list[Detection] = []
        for crop, (ox, oy, cw, ch) in zip(crops, origins):
            try:
                results = self.model.predict(
                    crop,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    imgsz=self.config.two_stage_imgsz,
                    device=self.config.device,
                    half=self.config.half,
                    verbose=False,
                )
            except Exception:
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
