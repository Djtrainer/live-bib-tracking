"""Publish annotated frames to the API server for browser viewing.

This exists only for presentation. It has a strictly weaker contract than
sink.py's finish-event delivery, and that's deliberate: a dropped preview
frame is nothing, a dropped finish event is a lost racer. So this uses
drop-oldest semantics on a single-slot queue -- the same policy
``capture.CameraSource`` uses for reading frames -- and a background thread,
so a slow or hung network POST can never add latency to detection or
finish-line timing. That coupling (the browser's connection affecting race
timing) is exactly the architectural fault this project replaced.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import StreamConfig


@dataclass
class StreamStats:
    """Operator-visible health of the preview relay."""

    sent: int = 0
    dropped: int = 0
    errors: int = 0
    last_error: str | None = None


class FrameStreamer:
    """Best-effort background publisher of the latest annotated frame."""

    def __init__(self, api_url: str, config: StreamConfig, session=None):
        self.url = api_url.rstrip("/") + "/api/frame"
        self.config = config
        self._session = session
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._stats = StreamStats()
        self._lock = threading.Lock()

    @property
    def stats(self) -> StreamStats:
        with self._lock:
            return StreamStats(
                sent=self._stats.sent,
                dropped=self._stats.dropped,
                errors=self._stats.errors,
                last_error=self._stats.last_error,
            )

    def _ensure_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def submit(self, frame_bgr: np.ndarray) -> None:
        """Offer a frame for publishing. Never blocks the caller.

        If the worker hasn't consumed the previous frame yet, that frame is
        dropped in favor of this newer one -- the queue never grows and the
        pipeline never waits on it.
        """
        try:
            self._queue.get_nowait()
            with self._lock:
                self._stats.dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            with self._lock:
                self._stats.dropped += 1

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        min_interval = (
            1.0 / self.config.target_fps if self.config.target_fps > 0 else 0.0
        )
        last_sent = 0.0
        while self._running:
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            now = time.time()
            if min_interval and (now - last_sent) < min_interval:
                continue  # arrived too soon after the last publish; drop it

            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            )
            if not ok:
                continue

            try:
                self._ensure_session().post(
                    self.url,
                    data=buffer.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                    timeout=self.config.timeout_seconds,
                )
                with self._lock:
                    self._stats.sent += 1
                last_sent = now
            except Exception as exc:
                with self._lock:
                    self._stats.errors += 1
                    self._stats.last_error = f"{type(exc).__name__}: {exc}"
