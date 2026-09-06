"""The frame loop.

Differences from the legacy loop that matter on race day:

* **No burst frame skipping.** The old cooldown dropped 30 frames -- a full
  second at 30 fps -- whenever 10 consecutive frames contained no *gated*
  person, then immediately re-armed. In backend.log it fired continuously,
  every ~310 ms, for hundreds of lines: racers could and did cross the line
  inside the blind window. Pacing here is uniform, driven by a target rate, and
  the achieved rate is reported.

* **No geometric gate on tracking.** Everyone detected is tracked. Geometry
  decides only who *finished*.

* **Emission is decoupled from timing.** A crossing is timestamped the moment it
  happens (interpolated between capture timestamps) but the finish event is held
  for a few frames so OCR can keep voting. Delaying the event costs nothing
  because the timestamp was already fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .capture import Frame
from .config import Config
from .detect import Detection, Detector
from .boundary import CourseBoundary
from .finish import Crossing, CrossingDetector, FinishLine
from .ocr import AsyncBibReader, BibRead, BibReader, BibVoter, crop_with_padding
from .sink import FinishEvent, make_event_id


@dataclass
class PipelineStats:
    """Health of a run. Every field is something an operator can act on."""

    frames_seen: int = 0
    frames_processed: int = 0
    frames_paced_out: int = 0
    people_detections: int = 0
    people_outside_boundary: int = 0
    bib_detections: int = 0
    ocr_reads: int = 0
    crossings: int = 0
    events_emitted: int = 0
    suppressed_first_seen_past: int = 0
    finishes_below_min_observations: int = 0
    handoffs: int = 0
    second_stage_bibs: int = 0
    two_stage_crops_skipped: int = 0
    two_stage_errors: int = 0
    unknown_bib_events: int = 0
    ocr_dropped: int = 0
    ocr_wait_timeouts: int = 0
    ocr_mean_wait_ms: float = 0.0
    first_capture_ts: float | None = None
    last_capture_ts: float | None = None

    @property
    def wall_span(self) -> float:
        if self.first_capture_ts is None or self.last_capture_ts is None:
            return 0.0
        return self.last_capture_ts - self.first_capture_ts

    @property
    def processed_fps(self) -> float:
        return self.frames_processed / self.wall_span if self.wall_span > 0 else 0.0


@dataclass
class FrameResult:
    """Everything the pipeline learned from one frame."""

    frame: Frame
    people: list[Detection] = field(default_factory=list)
    bibs: list[Detection] = field(default_factory=list)
    excluded_people: list[Detection] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)
    events: list[FinishEvent] = field(default_factory=list)


@dataclass
class _PendingFinish:
    crossing: Crossing
    frames_remaining: int


def associate_bib(person: Detection, bibs: Iterable[Detection]) -> Detection | None:
    """Pick the bib whose centre lies inside a person's box.

    When several qualify -- packs of runners overlap constantly -- the highest
    confidence one wins rather than whichever happened to be first in the list.
    """
    x1, y1, x2, y2 = person.xyxy
    candidates = [
        b for b in bibs if x1 <= b.center[0] <= x2 and y1 <= b.center[1] <= y2
    ]
    return max(candidates, key=lambda b: b.conf) if candidates else None


class Pipeline:
    """Owns per-run state and turns frames into finish events."""

    def __init__(
        self,
        config: Config,
        detector: Detector,
        frame_width: int,
        frame_height: int,
        run_id: str,
        bib_reader: BibReader | None = None,
        roster: set[str] | None = None,
        emit: Callable[[FinishEvent], None] | None = None,
    ):
        self.config = config
        self.detector = detector
        self.run_id = run_id
        self.line = FinishLine(config.finish_line, frame_width, frame_height)
        self.crossings = CrossingDetector(self.line, config.finish_line)
        self.boundary = CourseBoundary(config.course_boundary, frame_width, frame_height)
        self.voter = BibVoter(config.ocr, roster=roster)
        self.reader = bib_reader
        self.emit = emit or (lambda event: None)
        self.stats = PipelineStats()

        # OCR moves off the frame loop unless explicitly disabled. The thread
        # starts on first submit, so a Pipeline that never sees a bib -- most
        # unit tests -- never spawns one.
        self.async_ocr: AsyncBibReader | None = None
        if bib_reader is not None and config.ocr.enabled and config.ocr.async_reads:
            self.async_ocr = AsyncBibReader(
                bib_reader,
                self.voter,
                max_queue=config.ocr.async_queue_size,
                max_inflight_per_track=config.ocr.async_max_inflight_per_track,
            )

        self._pending: dict[int, _PendingFinish] = {}
        self._seen_last_frame: set[int] = set()
        self._observations: dict[int, int] = {}
        self._last_processed_ts: float | None = None
        self._next_due_ts: float | None = None

    def should_process(self, frame: Frame) -> bool:
        """Uniform pacing: never a burst, never a blind second.

        Scheduling runs off an accumulating deadline rather than "time since the
        last processed frame". The naive form loses a frame every few intervals
        to floating point (0.3 - 0.2 is fractionally less than 0.1), which
        silently runs the pipeline below its configured rate.

        Advances the schedule as a side effect when it returns True, so pacing
        state lives in exactly one place.
        """
        target = self.config.pipeline.target_fps
        if target <= 0:
            return True
        interval = 1.0 / target
        if self._next_due_ts is None:
            self._next_due_ts = frame.capture_ts + interval
            return True
        # Tolerance absorbs double-rounding without ever admitting a real burst.
        if frame.capture_ts < self._next_due_ts - interval * 1e-6:
            return False
        self._next_due_ts += interval
        if self._next_due_ts <= frame.capture_ts:
            # We fell behind (slow inference). Resync to now instead of firing a
            # catch-up burst -- bursts are what blinded the old pipeline.
            self._next_due_ts = frame.capture_ts + interval
        return True

    def process(self, frame: Frame) -> FrameResult:
        """Run one frame end to end."""
        self.stats.frames_processed += 1
        self._last_processed_ts = frame.capture_ts
        if self.stats.first_capture_ts is None:
            self.stats.first_capture_ts = frame.capture_ts
        self.stats.last_capture_ts = frame.capture_ts

        detections = self.detector.track(frame.image)
        people, bibs = self.detector.split(detections)
        self.stats.people_detections += len(people)
        self.stats.bib_detections += len(bibs)

        excluded_people: list[Detection] = []
        if self.boundary.enabled:
            # Bibs are never gated -- only whether a *person* counts as a
            # runner at all, matching the 2025 behavior exactly. A gated-out
            # person is invisible to OCR, crossing detection, and everything
            # downstream, same as they never existed for this frame -- but
            # unlike 2025, they're kept here for the overlay to draw, so a
            # miscalibrated boundary is visible instead of a silent drop.
            on_course = []
            for p in people:
                (on_course if self.boundary.contains_box(p.xyxy) else excluded_people).append(p)
            self.stats.people_outside_boundary += len(excluded_people)
            people = on_course

        if self.config.model.two_stage and people:
            # Second pass looks for bibs inside each runner's crop, where they
            # are hundreds of pixels wide rather than a dozen. Runs after the
            # boundary gate so we never pay for people who are off-course.
            extra = self.detector.bibs_in_people(frame.image, people)
            if extra:
                before = len(bibs)
                # Merge rather than concatenate: a bib visible at both scales is
                # found by both passes, and a duplicate box would let one racer
                # out-vote another during bib association.
                bibs = self.detector.merge(bibs, extra)
                added = len(bibs) - before
                self.stats.bib_detections += added
                self.stats.second_stage_bibs += added

        result = FrameResult(
            frame=frame, people=people, bibs=bibs, excluded_people=excluded_people
        )

        for person in people:
            track_id = person.track_id
            self._observations[track_id] = self._observations.get(track_id, 0) + 1
            self._read_bib(frame.image, person, bibs, track_id)
            crossing = self.crossings.update(
                track_id, person.xyxy, frame.capture_ts, frame.index
            )
            if crossing is not None:
                self.stats.crossings += 1
                result.crossings.append(crossing)
                predecessor = crossing.predecessor_track_id
                if predecessor is not None:
                    # A crossing recovered across a track break: the racer's
                    # history lives on the track that died. Inherit it, or the
                    # newborn -- a few frames old -- fails min_observations
                    # and has no bib votes, and the recovery was for nothing.
                    self.stats.handoffs += 1
                    self._observations[track_id] += self._observations.get(predecessor, 0)
                    self.voter.transfer(predecessor, track_id)
                minimum = self.config.finish_line.min_observations
                if minimum and self._observations[track_id] < minimum:
                    # A track the detector only glimpsed. Real racers are seen
                    # for tens of frames on their way in; fragments are not.
                    self.stats.finishes_below_min_observations += 1
                    continue
                self._pending[track_id] = _PendingFinish(
                    crossing=crossing,
                    frames_remaining=self.config.finish_line.confirm_frames,
                )

        seen_now = {p.track_id for p in people}
        result.events.extend(self._advance_pending(seen_now))
        self._seen_last_frame = seen_now
        self.stats.suppressed_first_seen_past = len(
            self.crossings.suppressed_first_seen_past
        )
        self._sync_ocr_stats()
        return result

    def _sync_ocr_stats(self) -> None:
        """Pull worker counters into PipelineStats.

        The worker cannot increment them itself: ``+=`` on an int is not atomic
        across threads, and these are read from the health line while the
        worker is running.
        """
        if self.async_ocr is None:
            return
        worker = self.async_ocr.stats
        self.stats.ocr_reads = worker.completed
        self.stats.ocr_dropped = worker.dropped_backlog + worker.skipped_inflight
        self.stats.ocr_wait_timeouts = worker.wait_timeouts
        self.stats.ocr_mean_wait_ms = worker.mean_wait_ms
        if self.detector is not None:
            self.stats.two_stage_crops_skipped = getattr(
                self.detector, "crops_skipped", 0
            )
            self.stats.two_stage_errors = getattr(self.detector, "two_stage_errors", 0)

    def _read_bib(
        self,
        image: np.ndarray,
        person: Detection,
        bibs: list[Detection],
        track_id: int,
    ) -> None:
        if self.reader is None or not self.config.ocr.enabled:
            return
        if self.voter.is_locked(track_id):
            return
        bib = associate_bib(person, bibs)
        if bib is None or bib.conf < self.config.ocr.min_bib_yolo_conf:
            return
        # OCR runs against the full-resolution frame, never the downscaled crop.
        crop = crop_with_padding(image, bib.xyxy, self.config.ocr.crop_padding)
        if crop.size == 0:
            return
        if self.async_ocr is not None:
            # .copy() matters: crop_with_padding returns a numpy *view*, which
            # keeps the entire 1080p frame alive for as long as it is queued.
            # A copy is a few KB; the view would pin ~6MB per pending read.
            self.async_ocr.submit(track_id, crop.copy(), bib.conf)
            return
        text, confidence = self.reader.read(self.reader.preprocess(crop))
        if text:
            self.stats.ocr_reads += 1
            self.voter.add(track_id, BibRead(text, confidence, bib.conf))

    def _advance_pending(self, seen_now: set[int]) -> list[FinishEvent]:
        """Emit finishes whose confirmation window elapsed or whose track ended."""
        ready: list[int] = []
        for track_id, pending in self._pending.items():
            pending.frames_remaining -= 1
            left_frame = track_id not in seen_now
            if pending.frames_remaining <= 0 or left_frame:
                ready.append(track_id)

        events = []
        for track_id in ready:
            pending = self._pending.pop(track_id)
            events.append(self._build_event(pending.crossing))
        for event in events:
            self.stats.events_emitted += 1
            if event.bib_number is None:
                self.stats.unknown_bib_events += 1
            self.emit(event)
        return events

    def _build_event(self, crossing: Crossing) -> FinishEvent:
        if self.async_ocr is not None:
            # This racer may still have crops queued. Waiting here is the one
            # place the frame loop blocks on OCR, and it happens once per
            # finisher rather than once per person per frame -- and normally
            # returns immediately, because confirm_frames has already given the
            # worker several frames of slack. The crossing timestamp was fixed
            # when it happened, so a late answer costs nothing but event
            # latency; resolving without waiting would throw away votes.
            self.async_ocr.wait_for(
                crossing.track_id, self.config.ocr.resolve_timeout
            )
        verdict = self.voter.resolve(crossing.track_id)
        return FinishEvent(
            track_observations=self._observations.get(crossing.track_id, 0),
            event_id=make_event_id(crossing.track_id, crossing.capture_ts, self.run_id),
            track_id=crossing.track_id,
            bib_number=verdict.text,
            capture_ts=crossing.capture_ts,
            frame_index=crossing.frame_index,
            ocr_votes=verdict.votes,
            ocr_score=verdict.score,
            bib_locked=verdict.locked,
            in_roster=verdict.in_roster,
            interpolated=crossing.interpolated,
        )

    def flush(self) -> list[FinishEvent]:
        """Emit every finish still waiting for confirmation.

        Called at the end of a run so a racer who crossed in the final frames is
        never lost to an unfinished confirmation window.

        Drains OCR first. These are precisely the finishers whose confirmation
        window never elapsed, so they are the ones most likely to still have
        reads in flight -- and unlike mid-race, there is no frame loop left to
        starve, so the wait is free.
        """
        if self.async_ocr is not None:
            self.async_ocr.drain(timeout=self.config.ocr.resolve_timeout * 8)
        events = [self._build_event(p.crossing) for p in self._pending.values()]
        self._pending.clear()
        for event in events:
            self.stats.events_emitted += 1
            if event.bib_number is None:
                self.stats.unknown_bib_events += 1
            self.emit(event)
        self._sync_ocr_stats()
        return events

    def close(self) -> None:
        """Release background workers. Safe to call more than once."""
        if self.async_ocr is not None:
            self.async_ocr.stop()

    def run(
        self,
        frames: Iterable[Frame],
        on_result: Callable[[FrameResult], None] | None = None,
    ) -> PipelineStats:
        """Consume a frame source until it is exhausted."""
        try:
            for frame in frames:
                self.stats.frames_seen += 1
                if not self.should_process(frame):
                    self.stats.frames_paced_out += 1
                    continue
                result = self.process(frame)
                if on_result is not None:
                    on_result(result)
            self.flush()
        finally:
            self.close()
        return self.stats
