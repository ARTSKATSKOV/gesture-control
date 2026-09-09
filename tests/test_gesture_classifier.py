"""Tests for the geometric gesture classifier."""

import pytest

from src.config import PinchConfig
from src.gesture_classifier import (
    FINGER_JOINTS,
    Gesture,
    GestureClassifier,
    PinchDetector,
    pinch_ratio_of,
)
from src.hand_tracker import Hand, Landmark, LandmarkIndex as LI

PALM_SPAN = 0.14
MID_X = 0.5
MID_Y = 0.5


def land(x, y, z=0.0):
    return Landmark(x=x, y=y, z=z)


def make_hand(*points):
    """Build a Hand from 21 landmarks given as (x, y, z) triples or
    Landmark instances, padding the rest with WRIST-position duplicates."""
    landmarks = list(points)
    while len(landmarks) < 21:
        landmarks.append(land(WRIST_X, WRIST_Y))
    return Hand(landmarks=tuple(landmarks), handedness="Right", score=1.0)


WRIST_X, WRIST_Y = MID_X, MID_Y
THUMB_TIP = land(WRIST_X - 0.05, WRIST_Y - 0.02)


def _palm_up(y_offset=0.0):
    """Open hand pointing up: all four fingers above the palm."""
    return make_hand(
        land(WRIST_X, WRIST_Y),
        land(WRIST_X, WRIST_Y),
        land(WRIST_X, WRIST_Y),
        land(WRIST_X, WRIST_Y),
        THUMB_TIP,
        land(WRIST_X - 0.02, WRIST_Y - 0.05),
        land(WRIST_X - 0.02, WRIST_Y - 0.10),
        land(WRIST_X - 0.02, WRIST_Y - 0.15),
        land(WRIST_X - 0.02, WRIST_Y - 0.20),
        land(WRIST_X + 0.01, WRIST_Y + y_offset - 0.02),
        land(WRIST_X + 0.01, WRIST_Y + y_offset - 0.07),
        land(WRIST_X + 0.01, WRIST_Y + y_offset - 0.12),
        land(WRIST_X + 0.01, WRIST_Y + y_offset - 0.17),
        land(WRIST_X + 0.04, WRIST_Y - 0.02),
        land(WRIST_X + 0.04, WRIST_Y - 0.06),
        land(WRIST_X + 0.04, WRIST_Y - 0.10),
        land(WRIST_X + 0.04, WRIST_Y - 0.13),
        land(WRIST_X + 0.06, WRIST_Y - 0.01),
        land(WRIST_X + 0.06, WRIST_Y - 0.05),
        land(WRIST_X + 0.06, WRIST_Y - 0.08),
        land(WRIST_X + 0.06, WRIST_Y - 0.10),
    )


class TestPinchRatio:
    def test_zero_gap_gives_zero_ratio(self):
        closed = make_hand(
            land(WRIST_X, WRIST_Y),
            *[land(WRIST_X + 0.02, WRIST_Y) for _ in range(19)],
        )
        ratio = pinch_ratio_of(closed.landmarks)
        assert ratio == pytest.approx(0.0)

    def test_open_palm_ratio_is_scale_invariant(self):
        def ratio_for(scale):
            palm = _palm_up()
            scaled = Hand(
                landmarks=tuple(
                    land(WRIST_X + (p.x - WRIST_X) * scale,
                        WRIST_Y + (p.y - WRIST_Y) * scale)
                    for p in palm.landmarks
                ),
                handedness="Right",
                score=1.0,
            )
            return pinch_ratio_of(scaled.landmarks)

        assert ratio_for(1.0) == pytest.approx(ratio_for(2.0))
        assert ratio_for(1.0) > 1.0

    def test_no_palm_size_returns_inf(self):
        collapsed = make_hand(*[land(WRIST_X, WRIST_Y) for _ in range(21)])
        assert pinch_ratio_of(collapsed.landmarks) == float("inf")


class TestPinchDetector:
    def setup_method(self):
        self.detector = PinchDetector(threshold=0.35)

    def test_activates_below_enter_threshold(self):
        assert not self.detector.update(0.30)  # above on-threshold 0.28
        assert self.detector.update(0.25)
        assert self.detector.active

    def test_stays_active_across_off_threshold(self):
        assert self.detector.update(0.25)
        assert self.detector.update(0.38)  # still below off-threshold 0.42
        assert self.detector.active
        assert not self.detector.update(0.45)
        assert not self.detector.active

    def test_hysteresis_prevents_rapid_reentry(self):
        assert not self.detector.update(0.32)
        assert self.detector.update(0.25)
        assert not self.detector.update(0.45)
        assert not self.detector.update(0.36)  # too wide to re-enter (on <= 0.28)
        assert self.detector.update(0.25)
        assert self.detector.active

    def test_reset_clears_active(self):
        self.detector.update(0.25)
        self.detector.reset()
        assert not self.detector.active


class TestClassifier:
    def setup_method(self):
        self.classifier = GestureClassifier(PinchConfig(threshold=0.35))

    def classify(self, hand):
        return self.classifier.classify(hand)

    def test_no_hand(self):
        assert self.classify(None).gesture == Gesture.NO_HAND
        assert self.classify(None).pinch_active is False

    def test_open_palm_classifies_as_open_palm(self):
        result = self.classify(_palm_up())
        assert result.gesture == Gesture.OPEN_PALM
        assert result.pinch_active is False

    def test_closed_fist_classifies_as_fist(self):
        collapsed = make_hand(*[land(WRIST_X, WRIST_Y) for _ in range(21)])
        assert self.classify(collapsed).gesture == Gesture.FIST

    def test_point_with_other_fingers_folded_still_points(self):
        landmarks = list(_palm_up().landmarks)
        index_tip = land(WRIST_X - 0.02, WRIST_Y - 0.25)
        landmarks[LI.INDEX_FINGER_TIP] = index_tip
        for joint in (
            LI.MIDDLE_FINGER_MCP,
            LI.MIDDLE_FINGER_PIP,
            LI.MIDDLE_FINGER_TIP,
            LI.RING_FINGER_MCP,
            LI.RING_FINGER_PIP,
            LI.RING_FINGER_TIP,
            LI.PINKY_MCP,
            LI.PINKY_PIP,
            LI.PINKY_TIP,
        ):
            landmarks[joint] = land(WRIST_X + 0.02, WRIST_Y)
        result = self.classify(make_hand(*landmarks))
        assert result.gesture == Gesture.POINT
        assert result.pointer == (index_tip.x, index_tip.y)

    def test_two_fingers_up(self):
        palm = _palm_up()
        landmarks = list(palm.landmarks)
        for joint in (
            LI.RING_FINGER_MCP,
            LI.RING_FINGER_PIP,
            LI.RING_FINGER_TIP,
            LI.PINKY_MCP,
            LI.PINKY_PIP,
            LI.PINKY_TIP,
        ):
            landmarks[joint] = land(WRIST_X + 0.02, WRIST_Y - 0.01)
        assert self.classify(make_hand(*landmarks)).gesture == Gesture.TWO_UP

    def test_two_fingers_down(self):
        palm = _palm_up()
        landmarks = list(palm.landmarks)
        for finger_joints in (
            (LI.INDEX_FINGER_MCP, LI.INDEX_FINGER_PIP, LI.INDEX_FINGER_TIP),
            (LI.MIDDLE_FINGER_MCP, LI.MIDDLE_FINGER_PIP, LI.MIDDLE_FINGER_TIP),
        ):
            mcp, pip, tip = finger_joints
            landmarks[mcp] = land(WRIST_X + 0.02, WRIST_Y + 0.10)
            landmarks[pip] = land(WRIST_X + 0.02, WRIST_Y + 0.18)
            landmarks[tip] = land(WRIST_X + 0.02, WRIST_Y + 0.28)
        for joint in (
            LI.RING_FINGER_MCP,
            LI.RING_FINGER_PIP,
            LI.RING_FINGER_TIP,
            LI.PINKY_MCP,
            LI.PINKY_PIP,
            LI.PINKY_TIP,
        ):
            landmarks[joint] = land(WRIST_X + 0.02, WRIST_Y)
        assert self.classify(make_hand(*landmarks)).gesture == Gesture.TWO_DOWN

    def test_thumbs_up(self):
        landmarks = [land(WRIST_X + 0.02, WRIST_Y)] * 21
        landmarks[LI.THUMB_MCP] = land(WRIST_X + 0.02, WRIST_Y - 0.05)
        landmarks[LI.THUMB_IP] = land(WRIST_X + 0.02, WRIST_Y - 0.10)
        landmarks[LI.THUMB_TIP] = land(WRIST_X + 0.02, WRIST_Y - 0.20)
        result = self.classify(make_hand(*landmarks))
        assert result.gesture == Gesture.THUMBS_UP

    def test_thumbs_down(self):
        landmarks = [land(WRIST_X + 0.02, WRIST_Y)] * 21
        landmarks[LI.THUMB_MCP] = land(WRIST_X + 0.02, WRIST_Y + 0.05)
        landmarks[LI.THUMB_IP] = land(WRIST_X + 0.02, WRIST_Y + 0.10)
        landmarks[LI.THUMB_TIP] = land(WRIST_X + 0.02, WRIST_Y + 0.20)
        result = self.classify(make_hand(*landmarks))
        assert result.gesture == Gesture.THUMBS_DOWN

    def test_pinch_wins_over_posture(self):
        landmarks = list(_palm_up().landmarks)
        landmarks[LI.THUMB_TIP] = landmarks[LI.INDEX_FINGER_TIP]
        hand = make_hand(*landmarks)
        result = self.classify(hand)
        assert result.gesture == Gesture.PINCH
        assert result.pinch_active is True

    def test_fist_with_touching_tips_is_not_pinch(self):
        landmarks = [land(WRIST_X, WRIST_Y) for _ in range(21)]
        landmarks[LI.MIDDLE_FINGER_MCP] = land(WRIST_X, WRIST_Y + 0.10)
        for pip in (
            LI.INDEX_FINGER_PIP,
            LI.MIDDLE_FINGER_PIP,
            LI.RING_FINGER_PIP,
            LI.PINKY_PIP,
        ):
            landmarks[pip] = land(WRIST_X, WRIST_Y - 0.10)
        landmarks[LI.THUMB_TIP] = land(WRIST_X + 0.05, WRIST_Y)
        landmarks[LI.INDEX_FINGER_TIP] = land(WRIST_X + 0.05, WRIST_Y)
        result = self.classify(make_hand(*landmarks))
        assert result.gesture == Gesture.FIST
        assert result.pinch_active is False


class TestFingerEnumeration:
    def test_finger_joints_index_are_valid(self):
        for mcp, pip, tip in FINGER_JOINTS:
            assert mcp < pip < tip <= LI.PINKY_TIP
