"""Tests for the application configuration loader."""

import json

import pytest

from src.config import AppConfig, ConfigError, DEFAULT_CONFIG_PATH


class TestDefaults:
    def test_defaults_match_design_targets(self):
        cfg = AppConfig.defaults()
        assert cfg.stabilizer.window_frames == 7
        assert cfg.stabilizer.required_frames == 5
        assert cfg.stabilizer.stable_ms == 120
        assert cfg.pinch.drag_hold_ms == 400
        assert cfg.pinch.click_cooldown_ms == 350
        assert cfg.cursor.smoothing_alpha == pytest.approx(0.25)
        assert cfg.cursor.dead_zone == pytest.approx(4.0)

    def test_pinch_threshold_is_normalized_ratio(self):
        cfg = AppConfig.defaults()
        assert 0.0 < cfg.pinch.threshold < 1.0


class TestConfigFile:
    def test_default_config_file_loads_and_matches_defaults(self):
        cfg = AppConfig.load(DEFAULT_CONFIG_PATH)
        expected = AppConfig.defaults()
        assert cfg == expected

    def test_file_contains_full_explicit_configuration(self):
        data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        assert set(data) == {
            "camera",
            "stabilizer",
            "pinch",
            "cursor",
            "scroll",
            "volume",
            "hotkeys",
        }


class TestMerging:
    def test_partial_config_merges_over_defaults(self):
        cfg = AppConfig.from_dict({"pinch": {"drag_hold_ms": 1000}})
        assert cfg.pinch.drag_hold_ms == 1000
        assert cfg.pinch.threshold == pytest.approx(0.35)
        assert cfg.stabilizer.window_frames == 7

    def test_empty_config_yields_defaults(self):
        assert AppConfig.from_dict({}) == AppConfig.defaults()

    def test_nested_values_are_coerced(self):
        cfg = AppConfig.from_dict({"cursor": {"smoothing_alpha": 0.5}})
        assert isinstance(cfg.cursor.smoothing_alpha, float)
        assert cfg.cursor.smoothing_alpha == pytest.approx(0.5)


class TestErrors:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"bogus": 1},
            {"camera": {"bogus": 0}},
            {"pinch": {"threshold": 0.0}},
            {"pinch": {"threshold": 1.0}},
            {"pinch": {"drag_hold_ms": -1}},
            {"pinch": {"drag_hold_ms": "400"}},
            {"stabilizer": {"window_frames": 0}},
            {"stabilizer": {"required_frames": 0}},
            {"stabilizer": {"window_frames": 3, "required_frames": 5}},
            {"stabilizer": {"required_frames": "5"}},
            {"stabilizer": {"stable_ms": -10}},
            {"cursor": {"smoothing_alpha": 0.0}},
            {"cursor": {"smoothing_alpha": 1.5}},
            {"cursor": {"dead_zone": -1}},
            {"scroll": {"wheel_step": 0}},
            {"scroll": {"cooldown_ms": -1}},
            {"volume": {"step": 0.0}},
            {"volume": {"step": 2.0}},
            {"camera": {"index": -1}},
            {"camera": {"width": 0}},
            {"camera": {"index": True}},
            {"hotkeys": {"quit": ""}},
            {"hotkeys": {"toggle": True}},
            {"pinch": None},
            ["not", "an", "object"],
        ],
    )
    def test_invalid_configuration_raises(self, overrides):
        with pytest.raises(ConfigError):
            AppConfig.from_dict(overrides)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError):
            AppConfig.load(tmp_path / "does-not-exist.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ConfigError):
            AppConfig.load(path)
