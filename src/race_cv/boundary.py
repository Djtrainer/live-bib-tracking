"""Course boundary gating: restrict tracking to a driveway/course region.

Reproduces the 2025 setup's guide-line gate. A person's bounding box is kept
only if at least one of its four corners falls between the left and right
boundary lines, interpolated at that corner's own y -- a perspective-correct
region that narrows or widens with distance the way a driveway does, not a
fixed rectangle. A detection that fails this gate is never tracked for OCR
or finish-line purposes: this is what keeps someone walking past on the
sidewalk off the leaderboard.

The 2025 implementation of this same idea is not what caused the race-day
failures documented in RACE_DAY_ANALYSIS.md -- unlabeled magic numbers with no
way to confirm they matched the camera, and zero visibility into what they
were excluding, did. This version keeps the exact same geometric concept
(and can reproduce the exact same numbers) but as named config with an
overlay to confirm it and a count of everything it drops.
"""

from __future__ import annotations

from .config import CourseBoundaryConfig

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def _interpolate_x_at_y(p1: Point, p2: Point, y: float) -> float:
    """x-position of the line through p1/p2 at a given y.

    Falls back to p1's x for a (near-)horizontal line rather than dividing by
    a near-zero delta.
    """
    x1, y1 = p1
    x2, y2 = p2
    if abs(y2 - y1) < 1e-9:
        return x1
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


class CourseBoundary:
    """A perspective-correct left/right boundary in pixel space."""

    def __init__(self, config: CourseBoundaryConfig, frame_width: int, frame_height: int):
        self.enabled = config.enabled
        self.left_p1: Point = (config.left_p1[0] * frame_width, config.left_p1[1] * frame_height)
        self.left_p2: Point = (config.left_p2[0] * frame_width, config.left_p2[1] * frame_height)
        self.right_p1: Point = (config.right_p1[0] * frame_width, config.right_p1[1] * frame_height)
        self.right_p2: Point = (config.right_p2[0] * frame_width, config.right_p2[1] * frame_height)

    def contains_point(self, point: Point) -> bool:
        px, py = point
        x_left = _interpolate_x_at_y(self.left_p1, self.left_p2, py)
        x_right = _interpolate_x_at_y(self.right_p1, self.right_p2, py)
        low, high = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
        return low <= px <= high

    def contains_box(self, bbox: BBox) -> bool:
        """True if at least one corner of the box is within the boundary.

        Matches the 2025 semantics: a person straddling the edge still
        counts, rather than requiring the whole box inside.
        """
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        return any(self.contains_point(corner) for corner in corners)

    def pixel_lines(self) -> tuple[tuple[Point, Point], tuple[Point, Point]]:
        """The two boundary lines in pixel space, for drawing an overlay."""
        return (self.left_p1, self.left_p2), (self.right_p1, self.right_p2)
