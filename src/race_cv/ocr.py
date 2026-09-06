"""Bib number reading and vote aggregation.

Three fixes over the legacy implementation:

* **Single-digit bibs are readable.** The old vote filter required
  ``2 <= len(text) <= 5``, so bibs 1-9 were discarded by the voting path and
  could only survive the ``>0.99`` lock. Length bounds are now config, defaulting
  to 1.
* **Votes are weighted by both confidences.** A bib box the detector was unsure
  about contributes less than a crisp one; the old code summed OCR confidence
  alone and ignored the detector entirely.
* **Optional roster snapping.** When the start list is known, a read that is not
  a real bib number is worth far less than one that is. This is the cheapest
  large accuracy win available on race day.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import OcrConfig

BBox = tuple[float, float, float, float]


@dataclass
class BibRead:
    """One OCR observation of a bib."""

    text: str
    ocr_conf: float
    yolo_conf: float


@dataclass
class BibVerdict:
    """The resolved bib for a track."""

    text: str | None
    score: float
    votes: int
    locked: bool
    in_roster: bool = False


def crop_with_padding(image: np.ndarray, bbox: BBox, padding: int) -> np.ndarray:
    """Crop a bib region from the **full-resolution** frame."""
    x1, y1, x2, y2 = (int(c) for c in bbox)
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0), dtype=image.dtype)
    return image[y1:y2, x1:x2]


class BibReader:
    """Wraps EasyOCR with bib-specific preprocessing."""

    def __init__(self, config: OcrConfig):
        self.config = config
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr  # lazy: model load is slow and unwanted in unit tests

            self._reader = easyocr.Reader(["en"])
        return self._reader

    def warmup(self) -> float:
        """Load and exercise the OCR model up front. Returns seconds taken.

        Without this, the ~3-8s cost of importing easyocr, building the
        Reader, and running its first inference is paid lazily -- on the first
        bib that clears ``min_bib_yolo_conf``, which is to say at the exact
        moment the first racer becomes readable near the line. On a live
        camera the capture thread keeps grabbing and dropping frames during
        that stall, so the pipeline goes blind at the worst possible time.

        Call this before the frame loop, when a stall costs nothing.
        """
        started = time.time()
        reader = self._ensure_reader()
        # Real inferences, not just construction: torch defers a large part of
        # the cost to the first call *at each input shape*. Measured here, a
        # first call at an unseen width costs 160-370ms while a repeat at a
        # seen width costs ~35ms, so warming a single shape leaves most of the
        # stall in place. preprocess() scales crops to target_height keeping
        # aspect, so sweep the widths that produces for roughly 1:1 to 3:1
        # bibs.
        height = self.config.target_height
        for width in (height, int(height * 1.7), int(height * 2.3), int(height * 3.0)):
            probe = np.full((height, width), 255, dtype=np.uint8)
            cv2.putText(probe, "123", (10, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
            try:
                reader.readtext(probe, allowlist="0123456789")
            except Exception:
                pass
        return time.time() - started

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Grayscale, upscale to a consistent height, equalise, threshold."""
        if crop is None or crop.size == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return gray
        scale = max(1.0, self.config.target_height / float(h))
        if scale > 1.0:
            gray = cv2.resize(
                gray,
                (int(w * scale), self.config.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def read(self, crop: np.ndarray) -> tuple[str | None, float]:
        """Return the highest-confidence digit string in a crop."""
        if crop is None or crop.size == 0 or crop.ndim < 2:
            return None, 0.0
        try:
            results = self._ensure_reader().readtext(crop, allowlist="0123456789")
        except Exception:
            return None, 0.0
        if not results:
            return None, 0.0
        best = max(results, key=lambda r: r[2] if len(r) >= 3 else 0.0)
        text = str(best[1]).strip()
        return (text, float(best[2])) if text else (None, 0.0)


class BibVoter:
    """Accumulates bib reads per track and resolves the most likely number.

    Every method takes ``_lock``. Reads are produced on the OCR worker thread
    (see :class:`AsyncBibReader`) and consumed on the frame loop when a finish
    is built, so ``add`` and ``resolve`` genuinely run concurrently. CPython's
    GIL happens to make the individual dict and list operations here atomic,
    but nothing guarantees that, and the cost of being explicit is an
    uncontended lock acquisition per bib read.
    """

    def __init__(self, config: OcrConfig, roster: set[str] | None = None):
        self.config = config
        self.roster = roster or set()
        self._reads: dict[int, list[BibRead]] = {}
        self._locked: dict[int, BibRead] = {}
        self._lock = threading.Lock()

    def add(self, track_id: int, read: BibRead) -> None:
        """Record a read, locking the track if OCR is essentially certain.

        A lock ends the search: ``resolve`` returns it directly and the
        pipeline stops running OCR on that racer entirely. So when a roster is
        loaded, only a number that is actually in the race may lock. Measured
        on real footage, a racer wearing 120 was read as "20" at 0.999
        confidence -- high confidence in a number nobody is wearing, which
        locked the wrong answer in and foreclosed every later frame that might
        have recovered the missing digit. Certainty about an impossible bib is
        exactly when the search should continue, not stop.

        Non-roster reads are still recorded as votes: they may be the only
        evidence there is, and ``resolve`` prefers roster candidates when any
        exist without discarding the rest.
        """
        if not self._is_plausible(read):
            return
        with self._lock:
            self._reads.setdefault(track_id, []).append(read)
            if track_id in self._locked or read.ocr_conf < self.config.lock_conf:
                return
            if self.roster and read.text not in self.roster:
                return
            self._locked[track_id] = read

    def _is_plausible(self, read: BibRead) -> bool:
        if not read.text or not read.text.isdigit():
            return False
        return self.config.min_len <= len(read.text) <= self.config.max_len

    def is_locked(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self._locked

    def resolve(self, track_id: int) -> BibVerdict:
        """Resolve the winning bib for a track.

        Votes are summed per candidate, weighted by ``ocr_conf * yolo_conf``.
        When a roster is loaded, candidates that are real bib numbers are given
        precedence over ones that are not, regardless of raw score.
        """
        with self._lock:
            locked = self._locked.get(track_id)
            if locked is not None:
                return BibVerdict(
                    text=locked.text,
                    score=locked.ocr_conf,
                    votes=len(self._reads.get(track_id, [])),
                    locked=True,
                    in_roster=locked.text in self.roster,
                )
            reads = [
                r
                for r in self._reads.get(track_id, [])
                if r.ocr_conf >= self.config.min_ocr_conf
            ]
        if not reads:
            return BibVerdict(text=None, score=0.0, votes=0, locked=False)

        scores: dict[str, float] = {}
        for read in reads:
            scores[read.text] = scores.get(read.text, 0.0) + (
                read.ocr_conf * max(read.yolo_conf, 1e-3)
            )

        if self.roster:
            known = {t: s for t, s in scores.items() if t in self.roster}
            if known:
                scores = known
            else:
                # Nothing read is a real bib. An off-roster number may still
                # be the right answer (the roster is incomplete more often
                # than never), but a *single* such read is not evidence of
                # that -- it is what a bibless racer looks like after a few
                # hundred frames of OCR on a logo. Demand agreement.
                counts: dict[str, int] = {}
                for read in reads:
                    counts[read.text] = counts.get(read.text, 0) + 1
                scores = {
                    t: s for t, s in scores.items()
                    if counts[t] >= self.config.min_votes_off_roster
                }
                if not scores:
                    return BibVerdict(text=None, score=0.0, votes=len(reads), locked=False)

        winner = max(scores, key=scores.get)
        return BibVerdict(
            text=winner,
            score=scores[winner],
            votes=len(reads),
            locked=False,
            in_roster=winner in self.roster,
        )

    def reads_for(self, track_id: int) -> list[BibRead]:
        with self._lock:
            return list(self._reads.get(track_id, []))

    def forget(self, track_id: int) -> None:
        with self._lock:
            self._reads.pop(track_id, None)
            self._locked.pop(track_id, None)

    def transfer(self, source: int, target: int) -> None:
        """Move one track's evidence onto another, for a crossing hand-off.

        The reads were of the same racer; only the tracker's id changed. A
        lock carries over too -- certainty about the bib does not expire
        because ByteTrack lost the box for three frames.
        """
        with self._lock:
            reads = self._reads.pop(source, None)
            if reads:
                self._reads.setdefault(target, []).extend(reads)
            locked = self._locked.pop(source, None)
            if locked is not None and target not in self._locked:
                self._locked[target] = locked


@dataclass
class AsyncOcrStats:
    """Health of the OCR worker. Every field is something an operator can act on."""

    submitted: int = 0
    completed: int = 0
    dropped_backlog: int = 0     # queue was full; oldest crop discarded
    skipped_inflight: int = 0    # this track already had enough reads queued
    errors: int = 0
    waits: int = 0
    wait_timeouts: int = 0
    wait_seconds: float = 0.0

    @property
    def mean_wait_ms(self) -> float:
        return (self.wait_seconds / self.waits * 1000.0) if self.waits else 0.0


class AsyncBibReader:
    """Runs OCR on a background thread so the frame loop never waits for it.

    Why this exists
    ---------------
    OCR was called inline, once per tracked runner, from inside
    ``Pipeline.process``. Measured on race footage: full-frame inference at
    1280 costs ~49ms and an EasyOCR read costs ~27ms. A frame with nobody
    readable therefore costs 49ms and fits comfortably inside a 15fps budget
    of 66ms -- but a frame where a runner's bib is legible costs 76ms and
    blows straight through it.

    That is the worst possible distribution of cost. The expensive frames are
    exactly the frames at the finish line, so the pipeline fell behind
    precisely while a racer was crossing, and the camera dropped the frames
    that mattered most. Live runs on the 14-48-12 clip missed a finisher and
    dropped ~1100 frames at 15fps; the same clip offline, where nothing is
    ever dropped, found all four.

    Why it is safe to answer late
    -----------------------------
    A crossing is timestamped when it happens, interpolated between capture
    timestamps, and the finish event is already held for ``confirm_frames``
    afterwards so voting can continue. The bib number is the only thing OCR
    contributes, and it is read off an event that has not been emitted yet.
    Resolving a bib a few hundred milliseconds late therefore changes nothing
    about the recorded finish time -- it only has to arrive before the event
    is built, which :meth:`wait_for` guarantees.

    Queue policy
    ------------
    Unlike ``stream.FrameStreamer``, which keeps only the newest frame, bib
    reads are *evidence* and dropping one costs accuracy. So the queue is
    generous and two separate bounds keep it from growing without limit:
    a per-track in-flight cap, so one runner in a long approach cannot crowd
    out everyone else, and a total size cap that discards the oldest crop.
    Oldest-first is the right thing to lose: a runner's earliest crops are
    from farthest away, where the bib is smallest and least legible.
    """

    def __init__(
        self,
        reader: BibReader,
        voter: BibVoter,
        max_queue: int = 48,
        max_inflight_per_track: int = 3,
    ):
        self._reader = reader
        self._voter = voter
        self._max_inflight_per_track = max_inflight_per_track
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._cond = threading.Condition()
        self._inflight: dict[int, int] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._stats = AsyncOcrStats()
        self._stats_lock = threading.Lock()

    @property
    def stats(self) -> AsyncOcrStats:
        with self._stats_lock:
            return AsyncOcrStats(**vars(self._stats))

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="bib-ocr"
        )
        self._thread.start()

    def submit(self, track_id: int, crop: np.ndarray, yolo_conf: float) -> bool:
        """Offer a bib crop for reading. Never blocks the frame loop.

        ``crop`` must be safe to hold: a numpy slice keeps its whole base frame
        alive, so callers pass a copy rather than pinning a 1080p buffer per
        queued read.
        """
        if not self._running:
            self.start()

        with self._cond:
            if self._inflight.get(track_id, 0) >= self._max_inflight_per_track:
                with self._stats_lock:
                    self._stats.skipped_inflight += 1
                return False
            self._inflight[track_id] = self._inflight.get(track_id, 0) + 1

        item = (track_id, crop, yolo_conf)
        while True:
            try:
                self._queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    stale_id, _, _ = self._queue.get_nowait()
                except queue.Empty:
                    continue  # worker drained it between the two calls; retry
                self._retire(stale_id)
                with self._stats_lock:
                    self._stats.dropped_backlog += 1
        with self._stats_lock:
            self._stats.submitted += 1
        return True

    def wait_for(self, track_id: int, timeout: float) -> bool:
        """Block until this track has no reads in flight. Returns False on timeout.

        Called once per finish event, never per frame. The bound matters: if
        the worker is wedged, this must degrade to "resolve from the votes we
        have" rather than becoming the new stall it was built to remove.
        """
        if timeout <= 0:
            with self._cond:
                return self._inflight.get(track_id, 0) == 0
        started = time.monotonic()
        deadline = started + timeout
        timed_out = False
        with self._cond:
            while self._inflight.get(track_id, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                self._cond.wait(remaining)
        with self._stats_lock:
            self._stats.waits += 1
            self._stats.wait_seconds += time.monotonic() - started
            if timed_out:
                self._stats.wait_timeouts += 1
        return not timed_out

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait for the whole queue to clear. Used at end of run, before flush."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while any(self._inflight.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        # Anything still queued will never be read. Retire it so a caller
        # blocked in wait_for or drain wakes up now instead of at its deadline.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._cond:
            # Clear rather than decrement: the read the worker was *inside*
            # when we stopped is not in the queue, so draining alone leaves its
            # count stranded and every later wait_for burns its whole timeout.
            # If that straggler does finish it still adds its vote -- the voter
            # is thread-safe and a late vote is harmless -- but nothing waits.
            self._inflight.clear()
            self._cond.notify_all()

    def _retire(self, track_id: int) -> None:
        with self._cond:
            remaining = self._inflight.get(track_id, 0) - 1
            if remaining > 0:
                self._inflight[track_id] = remaining
            else:
                self._inflight.pop(track_id, None)
            self._cond.notify_all()

    def _loop(self) -> None:
        while self._running:
            try:
                track_id, crop, yolo_conf = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                text, confidence = self._reader.read(self._reader.preprocess(crop))
                if text:
                    self._voter.add(track_id, BibRead(text, confidence, yolo_conf))
                    with self._stats_lock:
                        self._stats.completed += 1
            except Exception:
                # A bad crop must not kill the worker: that would silently take
                # bib reading offline for the rest of the race.
                with self._stats_lock:
                    self._stats.errors += 1
            finally:
                self._retire(track_id)
