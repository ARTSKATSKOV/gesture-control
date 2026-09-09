"""MediaPipe hand tracking for a single active hand.

The MediaPipe 1.0.1 wheel removed the legacy ``mp.solutions`` API and no
longer bundles model weights, so this module wraps the Tasks API
(``HandLandmarker``) and loads a ``.task`` model file (see
``hand.model_path`` in ``config.json``; committed under ``models/``).

Output is deliberately decoupled from MediaPipe types: ``Hand`` is a plain
dataclass of ``Landmark`` (normalized x/y/z). Everything downstream
(classification, stabilization) works on plain geometry, keeps itself
testable without a camera, and survives MediaPipe API churn.

A camera may report several hands. ``select_active_hand`` keeps exactly one
active hand: the physically largest one (wrist-to-middle-MCP span). Both left
and right hands are supported and the handedness label is preserved.
"""

import enum
import time
from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .config import HandConfig


class LandmarkIndex(enum.IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (LandmarkIndex.WRIST, LandmarkIndex.THUMB_CMC),
    (LandmarkIndex.THUMB_CMC, LandmarkIndex.THUMB_MCP),
    (LandmarkIndex.THUMB_MCP, LandmarkIndex.THUMB_IP),
    (LandmarkIndex.THUMB_IP, LandmarkIndex.THUMB_TIP),
    (LandmarkIndex.WRIST, LandmarkIndex.INDEX_FINGER_MCP),
    (LandmarkIndex.INDEX_FINGER_MCP, LandmarkIndex.INDEX_FINGER_PIP),
    (LandmarkIndex.INDEX_FINGER_PIP, LandmarkIndex.INDEX_FINGER_DIP),
    (LandmarkIndex.INDEX_FINGER_DIP, LandmarkIndex.INDEX_FINGER_TIP),
    (LandmarkIndex.INDEX_FINGER_MCP, LandmarkIndex.MIDDLE_FINGER_MCP),
    (LandmarkIndex.MIDDLE_FINGER_MCP, LandmarkIndex.MIDDLE_FINGER_PIP),
    (LandmarkIndex.MIDDLE_FINGER_PIP, LandmarkIndex.MIDDLE_FINGER_DIP),
    (LandmarkIndex.MIDDLE_FINGER_DIP, LandmarkIndex.MIDDLE_FINGER_TIP),
    (LandmarkIndex.MIDDLE_FINGER_MCP, LandmarkIndex.RING_FINGER_MCP),
    (LandmarkIndex.RING_FINGER_MCP, LandmarkIndex.RING_FINGER_PIP),
    (LandmarkIndex.RING_FINGER_PIP, LandmarkIndex.RING_FINGER_DIP),
    (LandmarkIndex.RING_FINGER_DIP, LandmarkIndex.RING_FINGER_TIP),
    (LandmarkIndex.RING_FINGER_MCP, LandmarkIndex.PINKY_MCP),
    (LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP),
    (LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP),
    (LandmarkIndex.PINKY_DIP, LandmarkIndex.PINKY_TIP),
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0

    @classmethod
    def from_normalized(cls, landmark) -> "Landmark":
        return cls(x=float(landmark.x), y=float(landmark.y), z=float(landmark.z))


@dataclass(frozen=True)
class Hand:
    landmarks: tuple[Landmark, ...]
    handedness: str
    score: float = 1.0

    @classmethod
    def from_mediapipe(cls, landmarks: Iterable, handedness: str, score: float) -> "Hand":
        return cls(
            landmarks=tuple(Landmark.from_normalized(lm) for lm in landmarks),
            handedness=handedness,
            score=float(score),
        )


def distance(a: Landmark, b: Landmark) -> float:
    """Euclidean distance in normalized (x, y) space."""
    return dist((a.x, a.y), (b.x, b.y))


def hand_size(landmarks: Sequence[Landmark]) -> float:
    """Normalized palm span: wrist to middle-finger MCP."""
    return distance(landmarks[LandmarkIndex.WRIST], landmarks[LandmarkIndex.MIDDLE_FINGER_MCP])


def select_active_hand(hands: Sequence[Hand]) -> Optional[Hand]:
    """Pick the single active hand: the largest, then the most confident."""
    if not hands:
        return None
    return max(
        hands,
        key=lambda hand: (hand_size(hand.landmarks), hand.score),
    )


def correct_handedness(label: str, selfie_mode: bool) -> str:
    """Flip Left/Right when the frame is mirrored.

    MediaPipe labels handedness for a non-mirrored image. Selfie previews are
    conventionally mirrored, which swaps the perceived side.
    """
    if not selfie_mode or label not in ("Left", "Right"):
        return label
    return "Right" if label == "Left" else "Left"


class HandTracker:
    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        selfie_mode: bool = True,
        clock=time.monotonic,
    ) -> None:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:  # pragma: no cover - environment issue only
            raise RuntimeError("mediapipe is not installed") from exc

        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"hand landmark model not found: '{model_file}'. "
                "Download hand_landmarker.task and place it at hand.model_path."
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max(1, int(num_hands)),
            min_hand_detection_confidence=float(min_hand_detection_confidence),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._selfie_mode = bool(selfie_mode)
        self._clock = clock
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray) -> Optional[Hand]:
        """Track the active hand in one BGR frame (mirrored if selfie mode)."""
        if frame_bgr is None:
            return None
        import mediapipe as mp

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),
        )
        timestamp_ms = max(self._last_timestamp_ms + 1, round(self._clock() * 1000))
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands = [
            Hand.from_mediapipe(
                landmarks,
                handedness=correct_handedness(
                    categories[0].category_name, self._selfie_mode
                ),
                score=categories[0].score,
            )
            for landmarks, categories in zip(
                result.hand_landmarks, result.handedness
            )
        ]
        return select_active_hand(hands)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def landmarks_to_pixels(
    landmarks: Sequence[Landmark], width: int, height: int
) -> list[tuple[int, int]]:
    return [
        (int(round(lm.x * (width - 1))), int(round(lm.y * (height - 1))))
        for lm in landmarks
    ]


def draw_hand(
    frame: np.ndarray,
    hand: Hand,
    connection_color: tuple[int, int, int] = (0, 255, 0),
    landmark_color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw landmarks and connections onto a BGR frame (in place)."""
    height, width = frame.shape[:2]
    points = landmarks_to_pixels(hand.landmarks, width, height)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], connection_color, thickness)
    for point in points:
        cv2.circle(frame, point, thickness + 1, landmark_color, -1)
    return frame


def new_hand_tracker(config: HandConfig, model_root: Path | None = None) -> HandTracker:
    """Build a HandTracker from configuration, resolving a relative model path."""
    model_path = Path(config.model_path)
    if not model_path.is_absolute() and model_root is not None:
        model_path = model_root / model_path
    return HandTracker(
        model_path=model_path,
        num_hands=config.num_hands,
        min_hand_detection_confidence=config.min_hand_detection_confidence,
        min_hand_presence_confidence=config.min_hand_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
