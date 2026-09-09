"""Tests for live-loop helpers and overlay rendering."""

import numpy as np
import pytest

from src.config import CameraConfig
from src.main import FpsCounter, _key_matches, open_camera
from src.overlay import OverlayState, draw_overlay


class TestKeys:
    def test_escape_aliases(self):
        assert _key_matches(27, "esc")
        assert _key_matches(27, "escape")
        assert not _key_matches(32, "esc")

    def test_space_alias(self):
        assert _key_matches(32, "space")

    def test_single_character_key_is_case_insensitive(self):
        assert _key_matches(ord("q"), "Q")
        assert not _key_matches(ord("x"), "Q")

    def test_empty_key_does_not_match(self):
        assert not _key_matches(ord("a"), "")


class TestFpsCounter:
    def test_first_tick_has_no_rate(self):
        clock_values = iter((10.0,))
        counter = FpsCounter(clock=lambda: next(clock_values))
        assert counter.tick() == 0.0

    def test_fps_is_smoothed_from_frame_intervals(self):
        clock_values = iter((10.0, 10.1, 10.2))
        counter = FpsCounter(clock=lambda: next(clock_values))
        assert counter.tick() == 0.0
        assert counter.tick() == pytest.approx(10.0)
        assert counter.tick() == pytest.approx(10.0)

    def test_non_positive_interval_does_not_break_counter(self):
        clock_values = iter((10.0, 10.0, 9.0))
        counter = FpsCounter(clock=lambda: next(clock_values))
        counter.tick()
        assert counter.tick() == 0.0
        assert counter.tick() == 0.0


class TestOverlay:
    def test_overlay_modifies_frame_and_preserves_shape(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        result = draw_overlay(
            frame,
            OverlayState(
                gesture="POINT",
                fps=30.0,
                enabled=True,
                handedness="Right",
                pinch_ratio=0.42,
            ),
        )
        assert result is frame
        assert result.shape == (120, 160, 3)
        assert np.any(result != 0)

    def test_disabled_overlay_is_rendered(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        draw_overlay(frame, OverlayState(gesture="NO_HAND", enabled=False))
        assert np.any(frame != 0)
