"""Keep the operator's preview from taxing the frame loop.

Measured on the real service, realtime, same clip: headless 29.9 fps with 7
frames dropped; with ``--preview`` 19.4 fps and 762 dropped over the same
minute. ``cv2.imshow`` plus ``waitKey(1)`` on a 1920x1080 frame costs
~19ms, and it ran on every frame, on the frame loop -- as much again as the
whole tracking loop (20ms). A preview that costs a third of the detector's
throughput is a preview that loses racers.

It cannot simply move to a thread: on macOS ``imshow`` has to run on the
main thread. So instead it runs less: at most ``fps`` times a second, on a
downscaled frame (cost is per pixel, so half the width is a quarter the
cost), and the annotation pass is skipped entirely on frames no consumer
will see. An operator cannot tell 10 fps from 30 in a preview window; the
detector underneath very much can.
"""

from __future__ import annotations

import cv2
import numpy as np


class RateGate:
    """Admit at most ``fps`` events per second, without catch-up bursts.

    Deadline-based like ``Pipeline.should_process``: when a caller falls
    behind, the next deadline is set from *now*, never from the missed
    deadline, so a stall is followed by one event rather than a flurry.
    """

    def __init__(self, fps: float):
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self._next: float | None = None

    def due(self, now: float) -> bool:
        if self.interval == 0.0:
            return True
        if self._next is None or now >= self._next:
            self._next = now + self.interval
            return True
        return False


def downscale(image: np.ndarray, scale: float) -> np.ndarray:
    """Shrink a frame for display. ``scale >= 1`` returns it untouched."""
    if scale >= 1.0 or scale <= 0.0:
        return image
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
