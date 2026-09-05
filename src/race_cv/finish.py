"""Finish-line geometry and crossing detection.

Two deliberate departures from the legacy implementation:

1. **Crossings are transitions, not states.** The old code asked "is any corner
   of the box past the line?" and fired the first time the answer was yes. A
   track that was re-acquired *after* the line therefore fired immediately,
   inventing a finisher, while a wide bounding box fired on whichever corner
   happened to lead. Here we track the signed distance of one stable reference
   point and fire on a sign change.

2. **Crossing times are interpolated.** Because we know the signed distance
   before and after the transition, the exact instant is linear-interpolated
   between the two capture timestamps. At a 10 fps processing rate this turns
   100 ms of quantisation error into roughly 10 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import FinishLineConfig

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass
class Crossing:
    """A confirmed finish-line transition."""

    track_id: int
    capture_ts: float
    frame_index: int
    interpolated: bool


def reference_point(bbox: BBox, mode: str) -> Point:
    """Reduce a bounding box to the point whose crossing defines the finish.

    ``bottom_center`` approximates the racer's feet and is the most stable
    choice: it does not move when the box grows because an arm swings out.
    """
    x1, y1, x2, y2 = bbox
    if mode == "bottom_center":
        return ((x1 + x2) / 2.0, y2)
    if mode == "center":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if mode == "top_center":
        return ((x1 + x2) / 2.0, y1)
    raise ValueError(
        f"Unknown reference_point '{mode}'. "
        "Expected 'bottom_center', 'center' or 'top_center'."
    )


class FinishLine:
    """A finish line in pixel space, built from normalized config."""

    def __init__(self, config: FinishLineConfig, frame_width: int, frame_height: int):
        if config.side not in ("below", "above"):
            raise ValueError(
                f"Unknown finish_line side '{config.side}'. Expected 'below' or 'above'."
            )
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.p1: Point = (config.p1[0] * frame_width, config.p1[1] * frame_height)
        self.p2: Point = (config.p2[0] * frame_width, config.p2[1] * frame_height)

        vx = self.p2[0] - self.p1[0]
        vy = self.p2[1] - self.p1[1]
        length = (vx * vx + vy * vy) ** 0.5
        if length < 1e-9:
            raise ValueError("Finish line p1 and p2 are the same point.")
        self._vx, self._vy, self._length = vx, vy, length

        # Orient the cross product so its sign does not depend on which end the
        # operator clicked first: with this factor applied, positive always
        # means "below the line" in image coordinates.
        self._orientation = 1.0 if (vx if abs(vx) > 1e-9 else vy) > 0 else -1.0
        self._finished_sign = 1.0 if config.side == "below" else -1.0

    def signed_distance(self, point: Point) -> float:
        """Perpendicular distance in pixels, positive on the finished side."""
        wx = point[0] - self.p1[0]
        wy = point[1] - self.p1[1]
        cross = self._vx * wy - self._vy * wx
        return (cross / self._length) * self._orientation * self._finished_sign

    def is_past(self, point: Point) -> bool:
        return self.signed_distance(point) >= 0.0

    def pixel_endpoints(self) -> tuple[Point, Point]:
        return self.p1, self.p2


class CrossingDetector:
    """Per-track finish-line crossing state machine.

    Feeding an observation returns a :class:`Crossing` exactly once per track,
    on the frame where the reference point transitions to the finished side.
    """

    def __init__(self, line: FinishLine, config: FinishLineConfig):
        self.line = line
        self.config = config
        self._last: dict[int, tuple[float, float]] = {}
        self._fired: set[int] = set()
        # Diagnostics: tracks first seen already past the line are suppressed
        # rather than fired, but they are counted so the suppression is visible
        # in the replay report instead of silently eating finishers.
        self.suppressed_first_seen_past: set[int] = set()

    def update(
        self,
        track_id: int,
        bbox: BBox,
        capture_ts: float,
        frame_index: int,
    ) -> Crossing | None:
        """Record one observation of a track; return a Crossing if it just finished."""
        point = reference_point(bbox, self.config.reference_point)
        distance = self.line.signed_distance(point)

        previous = self._last.get(track_id)
        self._last[track_id] = (distance, capture_ts)

        if track_id in self._fired:
            return None

        if previous is None:
            # First sighting of this track.
            if distance >= 0.0:
                if self.config.require_approach:
                    self.suppressed_first_seen_past.add(track_id)
                    return None
                self._fired.add(track_id)
                return Crossing(track_id, capture_ts, frame_index, interpolated=False)
            return None

        previous_distance, previous_ts = previous
        if previous_distance >= 0.0 or distance < 0.0:
            return None

        # Sign change: interpolate the instant the reference point touched zero.
        span = distance - previous_distance
        fraction = (-previous_distance / span) if span > 1e-9 else 1.0
        fraction = min(max(fraction, 0.0), 1.0)
        crossing_ts = previous_ts + fraction * (capture_ts - previous_ts)

        self._fired.add(track_id)
        return Crossing(track_id, crossing_ts, frame_index, interpolated=True)

    def has_finished(self, track_id: int) -> bool:
        return track_id in self._fired

    def forget(self, track_id: int) -> None:
        """Drop per-track state once a result has been delivered."""
        self._last.pop(track_id, None)
