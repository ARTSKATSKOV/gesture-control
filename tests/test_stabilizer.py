"""Tests for temporal stabilization and the pinch state machine."""

import pytest

from src.config import PinchConfig, StabilizerConfig
from src.stabilizer import (
    PinchDebounce,
    PinchEvent,
    PinchState,
    PinchStateMachine,
    TemporalStabilizer,
)

PINCH = "PINCH"
OPEN = "OPEN"
POINT = "POINT"


def frames_per_second(stabilizer, labels, step=0.05):
    results = []
    for index, label in enumerate(labels):
        results.append(stabilizer.update(label, now=index * step))
    return results


class TestPinchDebounce:
    def setup_method(self):
        self.debounce = PinchDebounce(frames=3)

    def test_single_frame_transient_is_ignored(self):
        assert self.debounce.update(True) is False
        assert self.debounce.update(False) is False
        assert self.debounce.state is False

    def test_confirms_after_required_frames(self):
        assert self.debounce.update(True) is False
        assert self.debounce.update(True) is False
        assert self.debounce.update(True) is True
        assert self.debounce.update(True) is True

    def test_release_also_requires_frames(self):
        for _ in range(3):
            self.debounce.update(True)
        assert self.debounce.state is True
        assert self.debounce.update(False) is True
        assert self.debounce.update(False) is True
        assert self.debounce.update(False) is False

    def test_reset_clears_state(self):
        for _ in range(3):
            self.debounce.update(True)
        self.debounce.reset()
        assert self.debounce.state is False
        assert self.debounce.update(True) is False

    def test_invalid_frames_count_raises(self):
        with pytest.raises(ValueError):
            PinchDebounce(frames=0)


class TestTemporalStabilizer:
    def test_confirms_after_required_frames_and_stable_time(self):
        stabilizer = TemporalStabilizer(
            StabilizerConfig(window_frames=5, required_frames=3, stable_ms=100)
        )
        results = frames_per_second(stabilizer, [PINCH, PINCH, PINCH, PINCH, PINCH])
        assert results == [None, None, PINCH, PINCH, PINCH]

    def test_requires_stable_time_not_just_frame_count(self):
        stabilizer = TemporalStabilizer(
            StabilizerConfig(window_frames=5, required_frames=3, stable_ms=150)
        )
        results = frames_per_second(stabilizer, [PINCH, PINCH, PINCH, PINCH])
        assert results == [None, None, None, PINCH]

    def test_requires_majority_of_window(self):
        config = StabilizerConfig(window_frames=5, required_frames=4, stable_ms=100)
        below = TemporalStabilizer(config)
        assert frames_per_second(below, [PINCH, PINCH, OPEN, PINCH]) == [None] * 4
        reached = TemporalStabilizer(config)
        results = frames_per_second(reached, [PINCH, PINCH, OPEN, PINCH, PINCH])
        assert results[-1] == PINCH

    def test_isolated_noise_frame_is_never_confirmed(self):
        stabilizer = TemporalStabilizer(
            StabilizerConfig(window_frames=5, required_frames=3, stable_ms=100)
        )
        results = frames_per_second(stabilizer, [PINCH, OPEN, OPEN, OPEN, OPEN])
        assert PINCH not in results

    def test_flip_resets_stability_anchor(self):
        stabilizer = TemporalStabilizer(
            StabilizerConfig(window_frames=5, required_frames=3, stable_ms=100)
        )
        labels = [PINCH] * 3 + [OPEN] * 5
        results = frames_per_second(stabilizer, labels)
        assert results == [None, None, PINCH, PINCH, PINCH, None, None, OPEN]

    def test_reset_clears_history(self):
        stabilizer = TemporalStabilizer(
            StabilizerConfig(window_frames=5, required_frames=3, stable_ms=100)
        )
        stabilizer.update(PINCH, now=0.0)
        stabilizer.reset()
        assert stabilizer.confirmed is None
        assert stabilizer.update(PINCH, now=0.05) is None

    def test_default_config_matches_spec(self):
        stabilizer = TemporalStabilizer(StabilizerConfig())
        results = frames_per_second(stabilizer, [PINCH] * 7, step=0.03)
        assert results[-1] == PINCH
        assert stabilizer.confirmed == PINCH


class TestPinchStateMachine:
    def setup_method(self):
        self.sm = PinchStateMachine(
            PinchConfig(threshold=0.35, drag_hold_ms=400, click_cooldown_ms=350)
        )

    def test_short_pinch_releases_as_single_click(self):
        assert self.sm.update(False, True, 0.00) == PinchEvent.NONE
        assert self.sm.state is PinchState.IDLE
        assert self.sm.update(True, True, 0.05) == PinchEvent.NONE
        assert self.sm.state is PinchState.PINCH_PENDING
        assert self.sm.update(False, True, 0.20) == PinchEvent.CLICK
        assert self.sm.state is PinchState.IDLE

    def test_hold_past_drag_hold_starts_drag(self):
        assert self.sm.update(True, True, 0.00) == PinchEvent.NONE
        assert self.sm.state is PinchState.PINCH_PENDING
        assert self.sm.update(True, True, 0.39) == PinchEvent.NONE
        assert self.sm.state is PinchState.PINCH_PENDING
        assert self.sm.update(True, True, 0.40) == PinchEvent.DRAG_START
        assert self.sm.state is PinchState.DRAGGING
        assert self.sm.update(True, True, 0.50) == PinchEvent.NONE
        assert self.sm.state is PinchState.DRAGGING

    def test_release_after_drag_ends_without_extra_click(self):
        self.sm.update(True, True, 0.00)
        self.sm.update(True, True, 0.40)
        event = self.sm.update(False, True, 0.55)
        assert event == PinchEvent.DRAG_END
        assert self.sm.state is PinchState.IDLE

    def test_release_frame_at_drag_hold_boundary_is_not_clicked(self):
        assert self.sm.update(True, True, 0.00) == PinchEvent.NONE
        assert self.sm.update(False, True, 0.40) == PinchEvent.NONE
        assert self.sm.state is PinchState.IDLE

    def test_pending_requires_hand_present(self):
        assert self.sm.update(True, False, 0.00) == PinchEvent.NONE
        assert self.sm.state is PinchState.IDLE

    def test_hand_loss_during_pending_cancels_silently(self):
        assert self.sm.update(True, True, 0.00) == PinchEvent.NONE
        assert self.sm.state is PinchState.PINCH_PENDING
        event = self.sm.update(True, False, 0.20)
        assert event == PinchEvent.NONE
        assert self.sm.state is PinchState.IDLE

    def test_hand_loss_during_drag_forces_release(self):
        self.sm.update(True, True, 0.00)
        self.sm.update(True, True, 0.40)
        event = self.sm.update(True, False, 0.50)
        assert event == PinchEvent.DRAG_END_FORCED
        assert self.sm.state is PinchState.IDLE

    def test_reset_during_drag_forces_release(self):
        self.sm.update(True, True, 0.00)
        self.sm.update(True, True, 0.40)
        event = self.sm.reset()
        assert event == PinchEvent.DRAG_END_FORCED
        assert self.sm.state is PinchState.IDLE
        assert self.sm.update(True, True, 0.60) == PinchEvent.NONE

    def test_reset_when_idle_is_noop(self):
        assert self.sm.reset() == PinchEvent.NONE
        assert self.sm.state is PinchState.IDLE

    def test_click_cooldown_suppresses_repeated_quick_clicks(self):
        self.sm.update(True, True, 0.05)
        assert self.sm.update(False, True, 0.20) == PinchEvent.CLICK
        self.sm.update(True, True, 0.30)
        event = self.sm.update(False, True, 0.40)
        assert event == PinchEvent.NONE
        assert self.sm.state is PinchState.IDLE
        self.sm.update(True, True, 0.55)
        assert self.sm.update(False, True, 0.70) == PinchEvent.CLICK

    def test_cooldown_does_not_block_drag(self):
        self.sm.update(True, True, 0.05)
        assert self.sm.update(False, True, 0.20) == PinchEvent.CLICK
        self.sm.update(True, True, 0.25)
        assert self.sm.state is PinchState.PINCH_PENDING
        assert self.sm.update(True, True, 0.70) == PinchEvent.DRAG_START
        assert self.sm.update(False, True, 0.75) == PinchEvent.DRAG_END
