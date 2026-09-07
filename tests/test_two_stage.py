"""Tests for the two-stage bib pass.

These exist because two-stage shipped broken in a way nothing could see. A
CoreML export accepts exactly one input size and raises on anything else:

    RuntimeError: Image size 640 x 640 not in allowed set of image sizes

``two_stage_imgsz: 640`` against the deployed 1280 export therefore raised on
*every* crop, and a bare ``except Exception: continue`` swallowed all of it.
The feature reported ``second_stage_bibs: 0`` and looked like it was enabled
and simply not finding extra bibs. The two guards below -- resolve the size
at startup, and never swallow a failure -- are what make that visible.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import ModelConfig, RoiConfig
from race_cv.detect import (
    Detection, Detector, _fixed_input_size, normalize_imgsz, to_imgsz_arg,
)

REPO = Path(__file__).resolve().parents[1]
MODEL_1280 = REPO / "models/gpu_runs/yolo11n_1280/weights/best.mlpackage"
MODEL_640 = REPO / "models/yolo11_white_bibs/weights/last.mlpackage"

def _has_coremltools() -> bool:
    try:
        import coremltools  # noqa: F401
    except Exception:
        return False
    return True


# Reading an export's input size needs both the file and coremltools; CI has
# neither, a machine with the models but a lean env has one.
needs_models = pytest.mark.skipif(
    not (MODEL_1280.exists() and MODEL_640.exists() and _has_coremltools()),
    reason="CoreML exports or coremltools not present",
)


class TestFixedInputDetection:
    @needs_models
    def test_reads_the_exports_actual_size(self):
        assert _fixed_input_size(MODEL_1280) == (1280, 1280)
        assert _fixed_input_size(MODEL_640) == (640, 640)

    @pytest.mark.skipif(
        not ((REPO / "models/exports/rect_960x736.mlpackage").exists() and _has_coremltools()),
        reason="rectangular export or coremltools not present",
    )
    def test_reads_a_rectangular_export_as_width_height(self):
        assert _fixed_input_size(
            REPO / "models/exports/rect_960x736.mlpackage"
        ) == (960, 736)

    def test_non_coreml_is_treated_as_flexible(self, tmp_path):
        weights = tmp_path / "best.pt"
        weights.touch()
        assert _fixed_input_size(weights) is None

    def test_uninspectable_model_never_blocks_startup(self, tmp_path):
        broken = tmp_path / "broken.mlpackage"
        broken.mkdir()
        assert _fixed_input_size(broken) is None


class FakeModel:
    """A YOLO stand-in that enforces a fixed input size, as CoreML does."""

    def __init__(self, accepts: int):
        self.accepts = accepts
        self.calls = 0

    def predict(self, image, imgsz=None, **kwargs):
        self.calls += 1
        if imgsz != self.accepts:
            raise RuntimeError(
                f"Image size {imgsz} x {imgsz} not in allowed set of image sizes"
            )
        box = SimpleNamespace(
            xyxy=[SimpleNamespace(tolist=lambda: [10.0, 10.0, 30.0, 20.0])],
            conf=SimpleNamespace(item=lambda: 0.9),
            cls=SimpleNamespace(item=lambda: 1),
        )
        return [SimpleNamespace(boxes=[box])]


def detector_with(model, **overrides):
    """Build a Detector without touching ultralytics or the filesystem."""
    config = ModelConfig(imgsz=1280, two_stage=True, **overrides)
    detector = Detector.__new__(Detector)
    detector.config = config
    detector.roi = SimpleNamespace(
        crop=lambda img: img, to_full_frame=lambda box: box
    )
    detector.model = model
    detector.second_stage = model
    detector.crops_skipped = 0
    detector.two_stage_errors = 0
    detector.two_stage_last_error = None
    detector.warnings = []
    detector.imgsz = config.imgsz
    detector.two_stage_imgsz = config.two_stage_imgsz
    return detector


def person(x1=0.0, y1=0.0, x2=200.0, y2=400.0):
    return Detection(xyxy=(x1, y1, x2, y2), conf=0.9, cls=0, track_id=1)


def frame():
    return np.zeros((600, 400, 3), dtype=np.uint8)


class TestFailuresAreVisible:
    def test_a_rejected_crop_size_is_counted_not_swallowed(self):
        """The exact bug: every crop raised and nothing said so."""
        detector = detector_with(FakeModel(accepts=1280), two_stage_imgsz=640)
        found = detector.bibs_in_people(frame(), [person()])
        assert found == []
        assert detector.two_stage_errors == 1
        assert "not in allowed set" in detector.two_stage_last_error

    def test_a_matching_size_actually_finds_bibs(self):
        detector = detector_with(FakeModel(accepts=640), two_stage_imgsz=640)
        found = detector.bibs_in_people(frame(), [person()])
        assert len(found) == 1
        assert detector.two_stage_errors == 0


class TestCropBudget:
    def test_max_crops_bounds_the_per_frame_cost(self):
        model = FakeModel(accepts=640)
        detector = detector_with(model, two_stage_imgsz=640, two_stage_max_crops=3)
        detector.bibs_in_people(frame(), [person() for _ in range(10)])
        assert model.calls == 3
        assert detector.crops_skipped == 7

    def test_the_largest_runners_are_the_ones_kept(self):
        """Biggest box = nearest the camera = nearest the line.

        Taking the first N in detector order would make the choice arbitrary,
        and could spend the budget on someone in the far background while the
        racer about to finish goes unread.
        """
        model = FakeModel(accepts=640)
        detector = detector_with(model, two_stage_imgsz=640, two_stage_max_crops=1)
        far = person(0.0, 0.0, 20.0, 40.0)
        near = person(100.0, 100.0, 300.0, 500.0)
        found = detector.bibs_in_people(frame(), [far, near])
        assert model.calls == 1
        # Returned boxes are offset by the crop origin, so the origin identifies
        # which runner was chosen. near starts at x=100 less 15% padding of its
        # 200px width -> 70; far would have given an origin of 0.
        assert found[0].xyxy[0] == pytest.approx(70.0 + 10.0)

    def test_no_people_costs_nothing(self):
        model = FakeModel(accepts=640)
        detector = detector_with(model, two_stage_imgsz=640)
        assert detector.bibs_in_people(frame(), []) == []
        assert model.calls == 0


class TestImgszNormalization:
    """imgsz may be square or rectangular, and the ordering is a trap.

    ultralytics takes [height, width] -- height first -- which is the opposite
    of how every other size in this project is written. Getting it backwards
    silently swaps a 1280x736 model for a 736x1280 one, which CoreML then
    rejects on every frame.
    """

    def test_an_int_means_square(self):
        assert normalize_imgsz(1280) == (1280, 1280)

    def test_a_pair_is_height_first(self):
        # ultralytics imgsz=[736, 1280] produced a 1280-wide, 736-tall export.
        assert normalize_imgsz([736, 1280]) == (1280, 736)

    def test_tuples_work_too(self):
        assert normalize_imgsz((736, 960)) == (960, 736)

    def test_round_trips_back_to_ultralytics_form(self):
        assert to_imgsz_arg(normalize_imgsz([736, 1280])) == [736, 1280]
        assert to_imgsz_arg(normalize_imgsz(1280)) == 1280

    def test_a_square_pair_collapses_to_an_int(self):
        """ultralytics is happiest with a bare int for square inputs."""
        assert to_imgsz_arg(normalize_imgsz([864, 864])) == 864

    def test_nonsense_is_rejected_loudly(self):
        for bad in ("1280", [1280], [1, 2, 3], None, True):
            with pytest.raises(ValueError):
                normalize_imgsz(bad)
