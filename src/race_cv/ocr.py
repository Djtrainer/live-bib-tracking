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
    """Accumulates bib reads per track and resolves the most likely number."""

    def __init__(self, config: OcrConfig, roster: set[str] | None = None):
        self.config = config
        self.roster = roster or set()
        self._reads: dict[int, list[BibRead]] = {}
        self._locked: dict[int, BibRead] = {}

    def add(self, track_id: int, read: BibRead) -> None:
        """Record a read, locking the track if OCR is essentially certain."""
        if not self._is_plausible(read):
            return
        self._reads.setdefault(track_id, []).append(read)
        if track_id not in self._locked and read.ocr_conf >= self.config.lock_conf:
            self._locked[track_id] = read

    def _is_plausible(self, read: BibRead) -> bool:
        if not read.text or not read.text.isdigit():
            return False
        return self.config.min_len <= len(read.text) <= self.config.max_len

    def is_locked(self, track_id: int) -> bool:
        return track_id in self._locked

    def resolve(self, track_id: int) -> BibVerdict:
        """Resolve the winning bib for a track.

        Votes are summed per candidate, weighted by ``ocr_conf * yolo_conf``.
        When a roster is loaded, candidates that are real bib numbers are given
        precedence over ones that are not, regardless of raw score.
        """
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

        winner = max(scores, key=scores.get)
        return BibVerdict(
            text=winner,
            score=scores[winner],
            votes=len(reads),
            locked=False,
            in_roster=winner in self.roster,
        )

    def reads_for(self, track_id: int) -> list[BibRead]:
        return list(self._reads.get(track_id, []))

    def forget(self, track_id: int) -> None:
        self._reads.pop(track_id, None)
        self._locked.pop(track_id, None)
