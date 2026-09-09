"""Tests for hand landmark data model, selection, and drawing helpers."""

import numpy as np
import pytest

from src.hand_tracker import (
    HAND_CONNECTIONS,
    Hand,
    Landmark,
    LandmarkIndex,
    correct_handedness,
    distance,
    draw_hand,
    hand_size,
    landmarks_to_pixels,
    select_active_hand,
)

NUM_LANDMARKS = 21
WRIST = Landmark(x=0.5, y=0.5)
PALM = Landmark(x=0.5, y=0.4)  # middle-finger MCP, one tenth above wrist


def synthetic_hand(scale=1.0, handedness="Right", score=1.0, dx=0.0, dy=0.0):
    """Build a valid 21-point hand from a tiny keyed template."""
    template = {
        LandmarkIndex.WRIST: (0.50, 0.50),
        LandmarkIndex.THUMB_CMC: (0.46, 0.48),
        LandmarkIndex.THUMB_MCP: (0.42, 0.45),
        LandmarkIndex.THUMB_IP: (0.40, 0.40),
        LandmarkIndex.THUMB_TIP: (0.38, 0.35),
        LandmarkIndex.INDEX_FINGER_MCP: (0.52, 0.38),
        LandmarkIndex.INDEX_FINGER_PIP: (0.53, 0.30),
        LandmarkIndex.INDEX_FINGER_DIP: (0.54, 0.23),
        LandmarkIndex.INDEX_FINGER_TIP: (0.55, 0.16),
        LandmarkIndex.MIDDLE_FINGER_MCP: (0.50, 0.36),
        LandmarkIndex.MIDDLE_FINGER_PIP: (0.50, 0.27),
        LandmarkIndex.MIDDLE_FINGER_DIP: (0.50, 0.20),
        LandmarkIndex.MIDDLE_FINGER_TIP: (0.50, 0.12),
        LandmarkIndex.RING_FINGER_MCP: (0.48, 0.37),
        LandmarkIndex.RING_FINGER_PIP: (0.47, 0.29),
        LandmarkIndex.RING_FINGER_DIP: (0.46, 0.22),
        LandmarkIndex.RING_FINGER_TIP: (0.45, 0.16),
        LandmarkIndex.PINKY_MCP: (0.46, 0.40),
        LandmarkIndex.PINKY_PIP: (0.44, 0.34),
        LandmarkIndex.PINKY_DIP: (0.43, 0.29),
        LandmarkIndex.PINKY_TIP: (0.42, 0.24),
    }
    landmarks = []
    for index in range(NUM_LANDMARKS):
        x, y = template[index]
        landmarks.append(
            Landmark(
                x=0.5 + (x - 0.5) * scale + dx,
                y=0.5 + (y - 0.5) * scale + dy,
            )
        )
    return Hand(
        landmarks=tuple(landmarks), handedness=handedness, score=score
    )


class TestGeometry:
    def test_landmark_index_order_is_contiguous(self):
        assert [int(index) for index in LandmarkIndex] == list(range(NUM_LANDMARKS))

    def test_distance_uses_normalized_xy(self):
        assert distance(Landmark(0.0, 0.0), Landmark(0.3, 0.4)) == pytest.approx(0.5)

    def test_hand_size_is_hand_span(self):
        assert hand_size(synthetic_hand(scale=1.0).landmarks) == pytest.approx(0.14)
        assert hand_size(synthetic_hand(scale=2.0).landmarks) == pytest.approx(0.28)


class TestSelection:
    def test_empty_returns_none(self):
        assert select_active_hand([]) is None

    def test_single_hand_is_returned(self):
        hand = synthetic_hand()
        assert select_active_hand([hand]) is hand

    def test_largest_hand_wins(self):
        small = synthetic_hand(scale=1.0)
        large = synthetic_hand(scale=3.0)
        assert select_active_hand([small, large]) is large

    def test_tie_broken_by_confidence(self):
        low_conf = synthetic_hand(scale=1.0, score=0.6)
        high_conf = synthetic_hand(scale=1.0, score=0.9)
        assert select_active_hand([low_conf, high_conf]) is high_conf

    def test_supports_left_and_right_hands(self):
        left = synthetic_hand(handedness="Left")
        right = synthetic_hand(handedness="Right")
        assert {select_active_hand([left]).handedness, select_active_hand([right]).handedness} == {
            "Left",
            "Right",
        }


class TestHandednessCorrection:
    def test_selfie_mode_flips_labels(self):
        assert correct_handedness("Left", selfie_mode=True) == "Right"
        assert correct_handedness("Right", selfie_mode=True) == "Left"

    def test_no_flip_when_not_selfie(self):
        assert correct_handedness("Left", selfie_mode=False) == "Left"
        assert correct_handedness("Right", selfie_mode=False) == "Right"


class TestConversion:
    def test_mediapipe_hand_is_built_from_duck_typed_landmarks(self):
        raw_landmarks = [
            type("Fake", (), {"x": (i % 5) / 5.0, "y": (i % 7) / 7.0, "z": 0.0})()
            for i in range(NUM_LANDMARKS)
        ]
        hand = Hand.from_mediapipe(raw_landmarks, handedness="Left", score=0.9)
        assert len(hand.landmarks) == NUM_LANDMARKS
        assert hand.handedness == "Left"
        assert hand.score == pytest.approx(0.9)

    def test_landmarks_to_pixels_maps_normalized_to_frame(self):
        hand = synthetic_hand()
        pixels = landmarks_to_pixels(hand.landmarks, width=640, height=480)
        assert len(pixels) == NUM_LANDMARKS
        assert pixels[LandmarkIndex.WRIST] == (320, 240)

    def test_all_connections_reference_valid_indices(self):
        for start, end in HAND_CONNECTIONS:
            assert 0 <= start < NUM_LANDMARKS
            assert 0 <= end < NUM_LANDMARKS

    def test_draw_hand_modifies_frame_in_place(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = synthetic_hand()
        result = draw_hand(frame, hand)
        assert result is frame
        assert np.any(frame != 0)
