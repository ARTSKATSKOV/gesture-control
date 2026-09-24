"""Tests for the tracking-quality gates that decide whether a frame may act.

These gates exist because of the first live acceptance run (2026-09-24): the
tracker extrapolates a hand that is leaving the picture and can report a face as
a hand, and both reached the Windows action layer unchanged.
"""

import pytest

from src.config import TrackingConfig
from src.hand_tracker import Hand, Landmark, LandmarkIndex as LI
from src.tracking_quality import (
    IMPLAUSIBLE_PALM,
    LOW_CONFIDENCE,
    NO_HAND,
    OK,
    PALM_OUT,
    POINTER_OUT,
    MotionGuard,
    assess_hand,
    pointer_of,
)
from test_hand_tracker import synthetic_hand

CONFIG = TrackingConfig()


def moved(dx=0.0, dy=0.0, score=1.0):
    hand = synthetic_hand(dx=dx, dy=dy)
    return Hand(landmarks=hand.landmarks, handedness=hand.handedness, score=score)


class TestAssessHand:
    def test_missing_hand_is_not_usable(self):
        verdict = assess_hand(None, CONFIG)
        assert not verdict.pointer_ok
        assert not verdict.full
        assert verdict.reason == NO_HAND

    def test_incomplete_landmark_set_is_not_usable(self):
        short = Hand(landmarks=moved().landmarks[:20], handedness="Right")
        assert assess_hand(short, CONFIG).reason == NO_HAND

    def test_a_well_tracked_hand_may_move_and_act(self):
        verdict = assess_hand(moved(), CONFIG)
        assert verdict.pointer_ok
        assert verdict.full
        assert verdict.reason == OK

    def test_low_handedness_score_is_rejected(self):
        """A face reported as a hand scores poorly; a real hand does not."""
        verdict = assess_hand(moved(score=0.4), CONFIG)
        assert not verdict.pointer_ok
        assert verdict.reason == LOW_CONFIDENCE

    def test_degenerate_palm_is_rejected(self):
        """A tracker that collapses every landmark reports an unusable hand."""
        collapsed = [Landmark(0.5, 0.5) for _ in range(21)]
        verdict = assess_hand(Hand(landmarks=tuple(collapsed), handedness="Right"), CONFIG)
        assert not verdict.pointer_ok
        assert verdict.reason == IMPLAUSIBLE_PALM

    def test_implausibly_large_hand_is_rejected(self):
        huge = synthetic_hand(scale=6.0)
        verdict = assess_hand(Hand(landmarks=huge.landmarks, handedness="Right"), CONFIG)
        assert not verdict.pointer_ok
        assert verdict.reason == IMPLAUSIBLE_PALM

    @pytest.mark.parametrize("dy", [1.0, 1.2])
    def test_extrapolated_hand_below_the_frame_cannot_move_at_all(self, dy):
        """This is the shape MediaPipe returns while the hand leaves the frame."""
        verdict = assess_hand(moved(dy=dy), CONFIG)
        assert not verdict.pointer_ok
        assert not verdict.full
        assert verdict.reason == POINTER_OUT

    @pytest.mark.parametrize("dx", [-0.45, -0.5])
    def test_hand_at_the_side_border_can_still_move_the_cursor(self, dx):
        """Only the palm left the frame; the fingertip is still observed."""
        hand = moved(dx=dx)
        assert hand.landmarks[LI.INDEX_FINGER_TIP].x >= 0.02
        verdict = assess_hand(hand, CONFIG)
        assert verdict.pointer_ok
        assert not verdict.full
        assert verdict.reason == PALM_OUT

    @pytest.mark.parametrize("dy", [0.5, 0.6])
    def test_hand_reaching_for_the_bottom_can_still_move_the_cursor(self, dy):
        """The posture that reaches the taskbar: palm low, fingertip in frame.

        Refusing these frames is what made the bottom of the screen unreachable
        in the 2026-09-24 live run.
        """
        verdict = assess_hand(moved(dy=dy), CONFIG)
        assert verdict.pointer_ok
        assert not verdict.full
        assert verdict.reason == PALM_OUT

    def test_hand_inside_the_frame_but_in_the_border_margin_is_not_full(self):
        """The wrist sits at 0.99: still inside the frame, already untrustworthy."""
        hand = moved(dy=0.49)
        assert hand.landmarks[LI.WRIST].y < 1.0
        verdict = assess_hand(hand, CONFIG)
        assert verdict.pointer_ok
        assert not verdict.full

    def test_a_folded_finger_clipping_the_edge_is_still_full(self):
        """The gate watches the anchors, not all twenty-one landmarks."""
        landmarks = list(moved().landmarks)
        landmarks[LI.PINKY_TIP] = Landmark(0.99, 0.99)
        landmarks[LI.MIDDLE_FINGER_TIP] = Landmark(0.0, 0.0)
        hand = Hand(landmarks=tuple(landmarks), handedness="Right", score=1.0)
        assert assess_hand(hand, CONFIG).full


class TestPointerOf:
    def test_no_hand_has_no_pointer(self):
        assert pointer_of(None) is None

    def test_pointer_is_the_index_fingertip(self):
        hand = moved()
        tip = hand.landmarks[LI.INDEX_FINGER_TIP]
        assert pointer_of(hand) == (tip.x, tip.y)


class TestMotionGuard:
    def test_first_frame_is_never_blurred(self):
        guard = MotionGuard(CONFIG, clock=lambda: 0.0)
        assert guard.update((0.5, 0.5), now=0.0) is False

    def test_slow_movement_is_not_blurred(self):
        guard = MotionGuard(CONFIG, clock=lambda: 0.0)
        guard.update((0.5, 0.5), now=0.0)
        assert guard.update((0.52, 0.5), now=0.1) is False

    def test_fast_movement_is_blurred(self):
        guard = MotionGuard(CONFIG, clock=lambda: 0.0)
        guard.update((0.5, 0.5), now=0.0)
        # 0.3 of the frame in 0.1 s is 3 frame widths/s, past the 2.5 limit.
        assert guard.update((0.5, 0.8), now=0.1) is True

    def test_slow_movement_after_a_fast_one_is_not_blurred(self):
        guard = MotionGuard(CONFIG, clock=lambda: 0.0)
        guard.update((0.5, 0.5), now=0.0)
        assert guard.update((0.5, 0.8), now=0.1) is True
        assert guard.update((0.5, 0.82), now=0.2) is False

    def test_losing_the_hand_does_not_make_the_next_frame_look_fast(self):
        """Re-acquiring a hand after a loss must not read as a throw."""
        guard = MotionGuard(CONFIG, clock=lambda: 0.0)
        guard.update((0.1, 0.1), now=0.0)
        assert guard.update(None, now=0.1) is False
        assert guard.update((0.9, 0.9), now=0.2) is False

    def test_a_long_gap_does_not_fake_a_throw(self):
        guard = MotionGuard(CONFIG, clock=lambda: 0.0)
        guard.update((0.1, 0.1), now=0.0)
        assert guard.update((0.9, 0.9), now=5.0) is False

    def test_repeated_position_is_never_blurred(self):
        guard = MotionGuard(CONFIG, clock=lambda: 0.0)
        guard.update((0.5, 0.5), now=0.0)
        for step in range(1, 5):
            assert guard.update((0.5, 0.5), now=step * 0.1) is False
