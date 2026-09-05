"""Frame sources.

The single most important property here is that **every frame carries the
timestamp of the moment it was grabbed**. The legacy pipeline stamped
``time.time()` when a frame finished *processing* and read
``cap.get(CAP_PROP_POS_MSEC)`` from a capture object that a reader thread had
already advanced past, so finish times drifted by however far behind the
pipeline was running.

Two sources:

* :class:`VideoFileSource` -- deterministic. Every frame is delivered, and the
  timestamp comes from the frame index and the file's frame rate, so a replay
  produces byte-identical results on every run.
* :class:`CameraSource` -- live. A reader thread keeps only the newest frame so
  the pipeline never works on stale data, and the timestamp is taken in that
  thread immediately after the grab.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    """One captured frame and the instant it was captured."""

    image: np.ndarray
    capture_ts: float
    index: int


class VideoFileSource:
    """Deterministic replay of a video file.

    Timestamps are synthesised from the frame index so that a replay is
    reproducible: given the same file and ``start_epoch`` the Nth frame always
    carries the same timestamp.
    """

    def __init__(self, path: str | Path, start_epoch: float = 0.0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_epoch = start_epoch
        self.is_live = False

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self.cap.read()
            if not ok or image is None:
                break
            yield Frame(
                image=image,
                capture_ts=self.start_epoch + index / self.fps,
                index=index,
            )
            index += 1

    def release(self) -> None:
        self.cap.release()


class CameraSource:
    """Live camera capture that always yields the most recent frame.

    OpenCV buffers frames internally, so a pipeline that reads sequentially
    falls progressively further behind real time. The reader thread here
    discards anything the pipeline did not keep up with, bounding latency to a
    single frame at the cost of dropping frames -- which is the correct trade
    for live timing, and is reported rather than hidden.
    """

    def __init__(
        self,
        index: int,
        width: int | None = None,
        height: int | None = None,
        warmup_seconds: float = 2.0,
    ):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(
                f"Could not open camera index {index}. "
                "Try a different index (0 = built-in, 1 = external) and confirm "
                "the terminal has macOS camera permission."
            )
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        # Ask the driver for a shallow buffer where the backend supports it.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = 0
        self.is_live = True

        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._consumed_index = -1
        self._running = False
        self._grabbed = 0
        self._dropped = 0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._warmup_seconds = warmup_seconds

    @property
    def grabbed(self) -> int:
        return self._grabbed

    @property
    def dropped(self) -> int:
        """Frames captured but never processed because a newer one arrived."""
        return self._dropped

    def _read_loop(self) -> None:
        index = 0
        while self._running:
            ok, image = self.cap.read()
            captured_at = time.time()
            if not ok or image is None:
                time.sleep(0.005)
                continue
            with self._lock:
                if self._latest is not None and self._latest.index > self._consumed_index:
                    self._dropped += 1
                self._latest = Frame(image=image, capture_ts=captured_at, index=index)
                self._grabbed += 1
            index += 1
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread.start()
        deadline = time.time() + self._warmup_seconds
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return
            time.sleep(0.01)

    def frames(self) -> Iterator[Frame]:
        if not self._running:
            self.start()
        while self._running:
            with self._lock:
                frame = self._latest
                if frame is not None and frame.index > self._consumed_index:
                    self._consumed_index = frame.index
                else:
                    frame = None
            if frame is None:
                time.sleep(0.002)
                continue
            yield frame

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def release(self) -> None:
        self.stop()
        self.cap.release()


def open_source(spec: str, start_epoch: float = 0.0) -> VideoFileSource | CameraSource:
    """Open a frame source from a CLI spec.

    A bare integer means a camera index; anything else is treated as a path.
    """
    if spec.isdigit():
        return CameraSource(int(spec))
    return VideoFileSource(spec, start_epoch=start_epoch)
