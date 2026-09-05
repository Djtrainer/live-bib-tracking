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
