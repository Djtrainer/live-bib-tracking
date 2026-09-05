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
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "cpu"
    half: bool = False
    person_class: int = 0
    bib_class: int = 1


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
class FinishLineConfig:
    """The finish line, as a segment in normalized frame coordinates.

    ``side`` selects which half-plane counts as finished. "below" means a larger
    y value than the line at the same x, matching image coordinates where y
    grows downward.
    """

    p1: Point = (0.0, 1.09)
    p2: Point = (1.0, 0.78)
    side: str = "below"
    confirm_frames: int = 8
    reference_point: str = "bottom_center"
    require_approach: bool = True


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
