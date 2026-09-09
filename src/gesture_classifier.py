"""Per-frame geometric gesture classification.

The classifier is intentionally **stateless about time**: it maps one hand
into one raw classification. Decisions repeated over several frames are made
later by the temporal stabilizer and the pinch state machine, so this module
stays deterministic and unit-testable with synthetic landmark geometry.

Pinch is detected with a scale-invariant ratio (see the project spec):

    pinch_ratio = distance(THUMB_TIP, INDEX_FINGER_TIP) / hand_size

where ``hand_size = distance(WRIST, MIDDLE_FINGER_MCP)``. A small amount of
state is kept for pinch **hysteresis** (enter/exit at different ratios) to
avoid flicker exactly at the threshold.
"""

from dataclasses import dataclass
from enum import Enum

from .config import PinchConfig
from .hand_tracker import (
    Hand,
    Landmark,
    LandmarkIndex as LI,
    distance,
    hand_size,
)

# Finger landmark groups for the four fingers: (mcp, pip, tip).
FINGER_JOINTS = (
    (LI.INDEX_FINGER_MCP, LI.INDEX_FINGER_PIP, LI.INDEX_FINGER_TIP),
    (LI.MIDDLE_FINGER_MCP, LI.MIDDLE_FINGER_PIP, LI.MIDDLE_FINGER_TIP),
    (LI.RING_FINGER_MCP, LI.RING_FINGER_PIP, LI.RING_FINGER_TIP),
    (LI.PINKY_MCP, LI.PINKY_PIP, LI.PINKY_TIP),
)

# Hysteresis band around the configured pinch threshold. The hand must come
# well below the threshold to start a pinch and go well above it to release.
PINCH_ON_FACTOR = 0.8
PINCH_OFF_FACTOR = 1.2

# Minimum |dy| / |dx| ratio of the index finger vector before a two-finger
# pose is considered to be pointing clearly up or down (scroll direction).
TWO_FINGER_VERTICAL_SLOPE = 1.2

_INDEX_VECTOR_ORIGIN = LI.INDEX_FINGER_MCP
_INDEX_VECTOR_TIP = LI.INDEX_FINGER_TIP


class Gesture(str, Enum):
    NO_HAND = "NO_HAND"
    OPEN_PALM = "OPEN_PALM"
    POINT = "POINT"
    PINCH = "PINCH"
    FIST = "FIST"
    TWO_UP = "TWO_UP"            # V sign pointing up    -> scroll up
    TWO_DOWN = "TWO_DOWN"        # V sign pointing down  -> scroll down
    THUMBS_UP = "THUMBS_UP"      # volume up
    THUMBS_DOWN = "THUMBS_DOWN"  # volume down
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FrameGesture:
    gesture: Gesture
    pinch_active: bool
    pinch_ratio: float
    pointer: tuple[float, float] | None
    handedness: str = ""


def pinch_ratio_of(landmarks) -> float:
    """Distance between thumb and index tips normalized by palm size."""
    palm = hand_size(landmarks)
    if palm <= 0:
        return float("inf")
    gap = distance(landmarks[LI.THUMB_TIP], landmarks[LI.INDEX_FINGER_TIP])
    return gap / palm


def _is_extended(landmarks, mcp: int, pip: int, tip: int) -> bool:
    """A finger is extended when its tip is farther from the wrist than its
    pip joint. Orientation- and scale-invariant for a typical open hand."""
    wrist = landmarks[LI.WRIST]
    return distance(wrist, landmarks[tip]) > distance(wrist, landmarks[pip])


def _is_thumb_extended(landmarks) -> bool:
    """Same wrist-distance heuristic for the thumb."""
    return _is_extended(landmarks, LI.THUMB_MCP, LI.THUMB_IP, LI.THUMB_TIP)


def _two_finger_direction(landmarks) -> Gesture | None:
    """Direction of a two-finger (index+middle) gesture based on the index
    finger axis in image coordinates (y grows downward)."""
    origin = landmarks[_INDEX_VECTOR_ORIGIN]
    tip = landmarks[_INDEX_VECTOR_TIP]
    dx = tip.x - origin.x
    dy = tip.y - origin.y
    if abs(dy) > abs(dx) * TWO_FINGER_VERTICAL_SLOPE:
        return Gesture.TWO_UP if dy < 0 else Gesture.TWO_DOWN
    return None


def _thumb_pose(landmarks) -> Gesture | None:
    """Thumb-only poses (four fingers folded): thumbs-up / thumbs-down."""
    if not _is_thumb_extended(landmarks):
        return None
    tip = landmarks[LI.THUMB_TIP]
    ip = landmarks[LI.THUMB_IP]
    if tip.y < ip.y:
        return Gesture.THUMBS_UP
    if tip.y > ip.y:
        return Gesture.THUMBS_DOWN
    return None


def _classify_posture(landmarks) -> Gesture:
    index_ext = _is_extended(landmarks, *FINGER_JOINTS[0])
    middle_ext = _is_extended(landmarks, *FINGER_JOINTS[1])
    ring_ext = _is_extended(landmarks, *FINGER_JOINTS[2])
    pinky_ext = _is_extended(landmarks, *FINGER_JOINTS[3])
    extended = (index_ext, middle_ext, ring_ext, pinky_ext)
    count = sum(extended)

    if count == 4:
        return Gesture.OPEN_PALM
    if count == 1 and index_ext:
        return Gesture.POINT
    if count == 2 and index_ext and middle_ext:
        direction = _two_finger_direction(landmarks)
        if direction is not None:
            return direction
        return Gesture.UNKNOWN
    if count == 0:
        thumb_pose = _thumb_pose(landmarks)
        if thumb_pose is not None:
            return thumb_pose
        return Gesture.FIST
    return Gesture.UNKNOWN


class PinchDetector:
    """Hysteresis-smoothed pinch signal from the raw per-frame ratio."""

    def __init__(
        self,
        threshold: float,
        on_factor: float = PINCH_ON_FACTOR,
        off_factor: float = PINCH_OFF_FACTOR,
    ) -> None:
        self._on_ratio = threshold * on_factor
        self._off_ratio = min(1.0, threshold * off_factor)
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def update(self, ratio: float) -> bool:
        if not self._active and ratio <= self._on_ratio:
            self._active = True
        elif self._active and ratio >= self._off_ratio:
            self._active = False
        return self._active

    def reset(self) -> None:
        self._active = False


class GestureClassifier:
    def __init__(self, pinch: PinchConfig) -> None:
        self._pinch_detector = PinchDetector(threshold=pinch.threshold)

    def classify(self, hand: Hand | None) -> FrameGesture:
        if hand is None or len(hand.landmarks) < 21:
            self._pinch_detector.reset()
            return FrameGesture(
                gesture=Gesture.NO_HAND,
                pinch_active=False,
                pinch_ratio=0.0,
                pointer=None,
            )
        landmarks = hand.landmarks
        ratio = pinch_ratio_of(landmarks)
        pinch_active = self._pinch_detector.update(ratio)
        pointer = (
            landmarks[LI.INDEX_FINGER_TIP].x,
            landmarks[LI.INDEX_FINGER_TIP].y,
        )
        if pinch_active:
            gesture = Gesture.PINCH
        else:
            gesture = _classify_posture(landmarks)
        return FrameGesture(
            gesture=gesture,
            pinch_active=pinch_active,
            pinch_ratio=ratio,
            pointer=pointer,
            handedness=hand.handedness,
        )
