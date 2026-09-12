"""The direct TensorRT path must put pixels and boxes exactly where ultralytics does.

No engine or GPU here: the geometry is checked against ultralytics' own
LetterBox and scale_boxes on random shapes, and the engine-metadata guard
is checked on fake files. The end-to-end equivalence (same boxes, same
tracks from the same engine) is a scratch script run on the Jetson and
recorded in JETSON_NOTES.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.detect import TrtRunner, letterbox_geometry, unletterbox_boxes

ultralytics = pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")


SHAPES = [
    ((756, 1383), (512, 928)),    # the racer crop into the race export
    ((756, 1383), (768, 1376)),   # near-native, tiny downscale, pad on top/bottom
    ((1080, 1920), (1088, 1920)), # full frame, pad only
    ((400, 200), (640, 640)),     # a person crop into the second stage: upscale
    ((37, 211), (640, 640)),      # a very wide sliver
]


class TestLetterboxGeometry:
    @pytest.mark.parametrize("src,dst", SHAPES)
    def test_pixels_land_where_ultralytics_puts_them(self, src, dst):
        from ultralytics.data.augment import LetterBox

        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(*src, 3), dtype=np.uint8)
        theirs = LetterBox(dst, auto=False, stride=32)(image=image)
        r, new_w, new_h, left, top = letterbox_geometry(src[0], src[1], dst[0], dst[1])
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) \
            if (new_h, new_w) != src else image
        assert theirs.shape == (dst[0], dst[1], 3)
        assert np.array_equal(theirs[top:top + new_h, left:left + new_w], resized)
        border = np.ones(dst, dtype=bool)
        border[top:top + new_h, left:left + new_w] = False
        assert (theirs[border] == 114).all()

    @pytest.mark.parametrize("src,dst", SHAPES)
    def test_boxes_come_back_where_scale_boxes_puts_them(self, src, dst):
        import torch
        from ultralytics.utils import ops

        rng = np.random.default_rng(1)
        n = 40
        x1 = rng.uniform(-10, dst[1], n); y1 = rng.uniform(-10, dst[0], n)
        boxes = np.stack([x1, y1, x1 + rng.uniform(1, 300, n), y1 + rng.uniform(1, 300, n)], 1)
        dets = np.concatenate([boxes, rng.uniform(0, 1, (n, 1)), np.zeros((n, 1))], 1).astype(np.float32)
        theirs = ops.scale_boxes(dst, torch.tensor(boxes, dtype=torch.float32), src).numpy()
        ours = unletterbox_boxes(dets.copy(), src[0], src[1], dst[0], dst[1])
        assert np.allclose(ours[:, :4], theirs, atol=1e-3)
        assert np.array_equal(ours[:, 4:], dets[:, 4:])  # conf and cls untouched


class TestEngineGuards:
    def test_an_engine_without_nms_is_refused_before_touching_the_gpu(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "tensorrt", None)  # importing it would fail loudly
        meta = json.dumps({"imgsz": [512, 928], "args": {"nms": False}}).encode()
        engine = tmp_path / "plain.engine"
        engine.write_bytes(len(meta).to_bytes(4, "little") + meta + b"\x00" * 32)
        with pytest.raises(ValueError, match="--nms"):
            TrtRunner(engine)

    def test_an_engine_without_metadata_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "tensorrt", None)
        engine = tmp_path / "bare.engine"
        engine.write_bytes(b"\xff\xff\xff\x7f" + b"\x00" * 32)
        with pytest.raises(ValueError, match="metadata"):
            TrtRunner(engine)
