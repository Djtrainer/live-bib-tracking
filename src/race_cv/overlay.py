"""Frame annotation for preview, calibration and replay debug video.

Drawing is deliberately separate from the pipeline: it is presentation, it is
expensive, and on race day it must be possible to turn it off entirely without
touching detection or timing.
"""

from __future__ import annotations

import cv2
import numpy as np

from .boundary import CourseBoundary
from .detect import Detection
from .finish import FinishLine
from .pipeline import FrameResult, PipelineStats

FINISH_COLOR = (0, 0, 255)
ROI_COLOR = (0, 200, 0)
PERSON_COLOR = (255, 128, 0)
BIB_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
BOUNDARY_COLOR = (0, 165, 255)
EXCLUDED_COLOR = (110, 110, 110)


def draw_finish_line(image: np.ndarray, line: FinishLine) -> np.ndarray:
    p1, p2 = line.pixel_endpoints()
    thickness = max(2, image.shape[1] // 400)
    cv2.line(
        image,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        FINISH_COLOR,
        thickness,
    )
    return image


def draw_roi(image: np.ndarray, roi) -> np.ndarray:
    if not roi.enabled:
        return image
    cv2.rectangle(image, (roi.x1, roi.y1), (roi.x2, roi.y2), ROI_COLOR, 2)
    return image


def _dashed_line(image: np.ndarray, p1, p2, color, thickness: int, dash_length: int = 14) -> None:
    length = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    if length < 1e-6:
        return
    dashes = max(1, int(length / dash_length))
    for i in range(dashes):
        start = i / dashes
        end = (i + 0.5) / dashes
        x1 = int(p1[0] + start * (p2[0] - p1[0]))
        y1 = int(p1[1] + start * (p2[1] - p1[1]))
        x2 = int(p1[0] + end * (p2[0] - p1[0]))
        y2 = int(p1[1] + end * (p2[1] - p1[1]))
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def draw_course_boundary(image: np.ndarray, boundary: CourseBoundary) -> np.ndarray:
    """Draw the left/right course lines so a miscalibrated boundary is
    something an operator can see and fix, not a silent drop.
    """
    if not boundary.enabled:
        return image
    thickness = max(2, image.shape[1] // 480)
    for p1, p2 in boundary.pixel_lines():
        _dashed_line(
            image,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            BOUNDARY_COLOR,
            thickness,
        )
    return image


def _label(image: np.ndarray, text: str, x: int, y: int, color) -> None:
    scale = max(0.5, image.shape[1] / 2200)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = max(th + 6, y)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 8, y + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_detections(
    image: np.ndarray,
    people: list[Detection],
    bibs: list[Detection],
    bib_labels: dict[int, str] | None = None,
    excluded_people: list[Detection] | None = None,
) -> np.ndarray:
    bib_labels = bib_labels or {}
    for bib in bibs:
        x1, y1, x2, y2 = (int(c) for c in bib.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), BIB_COLOR, 2)
    for person in excluded_people or []:
        x1, y1, x2, y2 = (int(c) for c in person.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), EXCLUDED_COLOR, 1)
        _label(image, "outside course", x1, y1 - 8, EXCLUDED_COLOR)
    for person in people:
        x1, y1, x2, y2 = (int(c) for c in person.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        label = bib_labels.get(person.track_id) or "?"
        _label(image, f"#{person.track_id} bib {label}", x1, y1 - 8, PERSON_COLOR)
    return image


def draw_hud(image: np.ndarray, stats: PipelineStats, extra: str = "") -> np.ndarray:
    """Operator health: a stalled pipeline must be obvious at a glance."""
    lines = [
        f"processed {stats.frames_processed} / seen {stats.frames_seen}"
        f"  ({stats.processed_fps:.1f} fps)",
        f"finishers {stats.events_emitted}   unknown bibs {stats.unknown_bib_events}",
    ]
    if extra:
        lines.append(extra)
    for i, text in enumerate(lines):
        _label(image, text, 12, 30 + i * 34, TEXT_COLOR)
    return image


def annotate(
    result: FrameResult,
    line: FinishLine,
    roi,
    stats: PipelineStats,
    bib_labels: dict[int, str] | None = None,
    extra: str = "",
    boundary: CourseBoundary | None = None,
) -> np.ndarray:
    """Produce an annotated copy of a processed frame."""
    image = result.frame.image.copy()
    draw_roi(image, roi)
    if boundary is not None:
        draw_course_boundary(image, boundary)
    draw_finish_line(image, line)
    draw_detections(image, result.people, result.bibs, bib_labels, result.excluded_people)
    draw_hud(image, stats, extra)
    return image
