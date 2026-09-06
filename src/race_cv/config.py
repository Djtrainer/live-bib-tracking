"""Configuration for the race CV pipeline.

Every threshold, geometry point and model parameter that used to be a hardcoded
constant or an unused CLI flag lives here and is loaded from YAML. Nothing in
this package reads a magic number from module scope.

Geometry is stored in **normalized** coordinates (fractions of frame width and
height) so a config survives a resolution change. Values outside ``[0, 1]`` are
legal -- the finish line is often anchored off-frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

Point = tuple[float, float]


@dataclass
class ModelConfig:
    """YOLO detection and tracking parameters.

    These were previously implicit: ``model.track()`` was called with no
    ``conf``, ``imgsz``, ``device`` or ``half``, while ``--conf`` and ``--fps``
    were parsed and logged but never reached the model.
    """

    path: str = "models/yolo11_white_bibs/weights/last.mlpackage"
    tracker: str = "config/custom_tracker.yaml"
    # An int for a square input, or [height, width] for a rectangular one.
    # Height-first, matching ultralytics' own convention, so the value here is
    # the value you would pass them directly. Rectangular is worth having: a
    # 1920x1080 frame letterboxed into a square 1280 model spends 43.8% of
    # every forward pass on grey padding. Measured raw forward pass on the
    # same weights -- 1280x1280 35.3ms, 1280x736 16.4ms, 960x736 14.5ms.
    imgsz: int | list[int] = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "cpu"
    half: bool = False
    person_class: int = 0
    bib_class: int = 1

    # Which Apple compute units CoreML may use. "DEFAULT" leaves ultralytics'
    # own choice alone; anything else overrides it.
    #
    # This is not a micro-optimisation. Ultralytics 8.4 hardcodes CPU_AND_NE
    # for detection models, commented as "~3x faster than CPU". Measured
    # end-to-end on this Mac with the deployed 1280 model, identical
    # detections in every case:
    #
    #     CPU_AND_NE (their default)  216.3ms
    #     CPU_ONLY                    152.7ms
    #     ALL (ANE+GPU+CPU)            59.5ms   <- 3.6x faster than default
    #
    # Their comment also warns that ALL can abort the process via an MPSGraph
    # compiler bug under coremltools 9.x. That does not reproduce here across
    # full smoke runs, but it is why this is a config value and not a constant:
    # if the pipeline ever dies at model load, set this to CPU_AND_NE.
    coreml_compute_units: str = "ALL"

    # Two-stage detection: find people on the full frame, then re-run the
    # detector on each person's crop to find their bib. A bib is a small object
    # -- median 46px wide in finish-line footage, which is 15px once a 1920px
    # frame is squeezed into a 640px input. Cropping to one runner and feeding
    # *that* to the same 640px input gives the bib several hundred pixels
    # instead, without paying for high resolution across the whole frame.
    #
    # Cost is one inference per person rather than per frame, so this is a win
    # when few runners are in shot and a loss in a dense pack; two_stage_max_crops
    # bounds the worst case.
    #
    # IMPORTANT: two_stage_imgsz only does anything if the model can actually
    # run at that size. A CoreML .mlpackage is exported at a *fixed* input --
    # verified on this repo's exports, none of which declare size flexibility --
    # so pointing the second stage at the 1280 export and asking for 640 gets
    # you 1280 anyway, at ~49ms per person instead of the ~12ms the setting
    # implies. That silent mismatch is the same class of bug as the old dead
    # --conf flag, so Detector now checks it at startup and says so.
    #
    # To get a genuinely cheap second stage, give it its own smaller export:
    # full-frame people at 1280, per-runner bib crops at 640. A crop is upscaled
    # to the model input regardless, so a 640 model loses nothing on a
    # 200x400px person and costs a quarter as much.
    two_stage: bool = False
    two_stage_model: str | None = None  # defaults to model.path
    two_stage_imgsz: int | list[int] = 640
    two_stage_padding: float = 0.15  # fraction of the person box, added around it
    two_stage_max_crops: int = 6     # bound the per-frame cost in a pack


@dataclass
class RoiConfig:
    """Region of interest fed to the detector.

    Disabled by default. The legacy pipeline cropped away the left ~28% of every
    frame, which made any racer on that side undetectable. If you enable this,
    calibrate it against real footage and confirm with the overlay.
    """

    enabled: bool = False
    polygon: list[Point] = field(default_factory=list)
    scale: float = 1.0


@dataclass
class CourseBoundaryConfig:
    """Restricts which detected people are tracked as runners at all.

    Disabled by default -- this is a per-venue decision, not a general
    default. A person's bounding box is kept only if at least one of its
    four corners falls between the left and right lines, interpolated at
    that corner's own y. That makes the region narrow or widen with distance
    the way a driveway does in perspective, rather than a fixed rectangle.

    This reproduces the 2025 setup's ``guide_line_left`` / ``guide_line_right``
    gate, which existed specifically to keep spectators and passersby outside
    the course from ever being counted as runners. The 2025 failure mode was
    never the gate's existence -- it was that the numbers were unlabeled
    constants in code with no way to confirm they still matched the camera,
    and no count of what they excluded. Both are fixed here: this is named,
    documented config with an overlay to confirm it visually, and
    PipelineStats.people_outside_boundary reports how many detections it
    dropped, every run.
    """

    enabled: bool = False
    left_p1: Point = (0.31, 1.0)
    left_p2: Point = (0.285, 0.49)
    right_p1: Point = (1.0, 0.78)
    right_p2: Point = (0.32, 0.49)


@dataclass
class FinishLineConfig:
    """The finish line, as a segment in normalized frame coordinates.

    ``side`` selects which half-plane counts as finished. "below" means a larger
    y value than the line at the same x, matching image coordinates where y
    grows downward.

    ``min_observations`` is how many times a track must have been seen before
    it is allowed to finish. 0 disables it. A real racer approaching the line
    is tracked for tens of frames; a fragment the tracker briefly invented is
    seen a handful of times and then vanishes. Measured on real footage with a
    sharper (1280px) model, one clip produced two finishes 0.44s apart -- one
    from a track observed 44 times (the racer) and one from a track observed 6
    times (a fragment of the same person). Raising detector resolution makes
    this worse, not better, because more sensitive detection fragments tracks
    more readily.
    """

    p1: Point = (0.0, 1.09)
    p2: Point = (1.0, 0.78)
    side: str = "below"
    confirm_frames: int = 8
    reference_point: str = "bottom_center"
    require_approach: bool = True
    min_observations: int = 0


@dataclass
class OcrConfig:
    """Bib reading and vote aggregation."""

    enabled: bool = True
    min_bib_yolo_conf: float = 0.70
    min_ocr_conf: float = 0.40
    lock_conf: float = 0.99
    min_len: int = 1
    max_len: int = 5
    crop_padding: int = 15
    target_height: int = 120

    # When a roster is loaded, a number that is NOT on it needs this many
    # agreeing reads before it can be the verdict. On-roster numbers are not
    # subject to it. Real bibs on real footage resolve with 3+ reads at
    # 0.99+; the case this guards is a bibless racer, seen for hundreds of
    # frames, collecting a single spurious "1" from a logo or a fold of
    # fabric and having their finish credited to whoever wears bib 1.
    # Two agreeing off-roster reads are kept as evidence the roster is
    # incomplete; one is noise. Set to 1 to restore the old behaviour.
    min_votes_off_roster: int = 2

    # Read bibs on a background thread instead of inline in the frame loop.
    # An EasyOCR read costs ~27ms and fires only when a bib is legible, i.e.
    # exactly at the finish line -- inline, that pushed the crossing frames
    # over the frame budget and the camera dropped them. See ocr.AsyncBibReader.
    async_reads: bool = True
    async_queue_size: int = 48
    async_max_inflight_per_track: int = 3
    # How long building a finish event may wait for that racer's outstanding
    # reads. Bounded so a wedged worker degrades to "resolve from the votes we
    # already have" rather than becoming a new stall. Normally ~0: the
    # confirm_frames window has already given the worker several frames of
    # slack before the event is built.
    resolve_timeout: float = 0.25


@dataclass
class PipelineConfig:
    """Frame pacing.

    ``target_fps`` of 0 means process every frame, which is what replay uses for
    determinism. Live capture drops frames *uniformly* to hold the target rate,
    never in the 30-frame bursts the old cooldown produced.
    """

    target_fps: float = 10.0
    log_every_n_frames: int = 300


@dataclass
class SinkConfig:
    """Result delivery."""

    api_url: str = "http://localhost:8001"
    event_log: str = "data/results/events.jsonl"
    retry_seconds: float = 2.0
    max_retry_seconds: float = 30.0
    timeout_seconds: float = 5.0


@dataclass
class StreamConfig:
    """Publishing annotated frames to the API server for browser viewing.

    Deliberately separate from result delivery: a dropped preview frame is
    nothing, a dropped finish event is a lost racer, so this uses drop-oldest
    semantics and short timeouts rather than sink.py's persist-and-retry.
    """

    enabled: bool = True
    target_fps: float = 8.0
    jpeg_quality: int = 80
    timeout_seconds: float = 2.0


@dataclass
class Config:
    """Top-level pipeline configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    course_boundary: CourseBoundaryConfig = field(default_factory=CourseBoundaryConfig)
    finish_line: FinishLineConfig = field(default_factory=FinishLineConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    sink: SinkConfig = field(default_factory=SinkConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Build a config from a YAML file, falling back to defaults.

        Unknown keys raise rather than being silently ignored -- a typo in a
        threshold name must not quietly leave the default in place.
        """
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        sections = {
            "model": ModelConfig,
            "roi": RoiConfig,
            "course_boundary": CourseBoundaryConfig,
            "finish_line": FinishLineConfig,
            "ocr": OcrConfig,
            "pipeline": PipelineConfig,
            "sink": SinkConfig,
            "stream": StreamConfig,
        }
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(
                f"Unknown config section(s): {sorted(unknown)}. "
                f"Expected any of {sorted(sections)}."
            )

        kwargs: dict[str, Any] = {}
        for name, section_cls in sections.items():
            values = raw.get(name) or {}
            if not isinstance(values, dict):
                raise ValueError(f"Config section '{name}' must be a mapping.")
            valid = {f for f in section_cls.__dataclass_fields__}
            bad = set(values) - valid
            if bad:
                raise ValueError(
                    f"Unknown key(s) in '{name}': {sorted(bad)}. "
                    f"Expected any of {sorted(valid)}."
                )
            kwargs[name] = section_cls(**values)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return out
