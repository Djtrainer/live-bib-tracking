"""Tests for configuration loading.

The point of these is that a mistyped threshold must fail loudly. The old system
had knobs that were parsed, validated, logged and then silently ignored, which
made tuning sessions impossible to interpret.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import Config


def test_defaults_are_usable():
    config = Config()
    assert config.model.imgsz == 640
    assert config.pipeline.target_fps > 0
    assert config.ocr.min_len == 1, "single-digit bibs must be readable"
    assert config.roi.enabled is False, "ROI cropping must be opt-in"


def test_roundtrips_through_yaml(tmp_path):
    config = Config()
    config.model.conf = 0.4
    config.finish_line.p1 = (0.1, 0.2)
    path = config.save(tmp_path / "c.yaml")
    reloaded = Config.load(path)
    assert reloaded.model.conf == 0.4
    assert tuple(reloaded.finish_line.p1) == (0.1, 0.2)


def test_missing_path_falls_back_to_defaults():
    assert Config.load(None).model.imgsz == Config().model.imgsz


def test_unknown_section_is_rejected():
    with pytest.raises(ValueError, match="Unknown config section"):
        Config.from_dict({"modle": {}})


def test_unknown_key_is_rejected():
    """A typo must not silently leave the default in place."""
    with pytest.raises(ValueError, match="Unknown key"):
        Config.from_dict({"model": {"confidence": 0.3}})


def test_non_mapping_section_is_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        Config.from_dict({"model": [1, 2, 3]})


def test_shipped_config_is_valid():
    """config/race_cv.yaml must always load -- it is the race-day config."""
    repo_config = Path(__file__).resolve().parents[1] / "config" / "race_cv.yaml"
    config = Config.load(repo_config)
    assert config.finish_line.side in ("below", "above")
    assert config.ocr.min_len >= 1
    raw = yaml.safe_load(repo_config.read_text())
    assert "finish_line" in raw and "model" in raw
