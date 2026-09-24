"""Gates that decide what a tracked frame may be trusted to do.

Two separate problems are solved here, both observed on live runs:

* ``assess_hand`` judges a single tracker result. MediaPipe keeps extrapolating
  a hand that is leaving the frame and occasionally reports a face as a hand.
  Both reached the action layer unchanged: the extrapolated hand kept the mouse
  button pressed during a drag, and the collapsing geometry produced a click
  exactly when the hand disappeared.
* ``MotionGuard`` judges pointer motion between frames. A fast throw smears the
  landmarks until the thumb and index tips appear to touch, which fired a
  spurious pinch (the cursor froze and a click landed).

The verdict is deliberately split in two, because the two things the app does
have different requirements:

* moving the cursor needs the pointer (index fingertip) and nothing else — a
  hand that reaches for the taskbar has its palm partly out of frame, and
  refusing to move there is exactly what made the bottom of the screen
  unreachable;
* clicking, dragging, scrolling and volume need the whole palm, because pinch
  geometry is measured against the palm span.

Nothing here moves a cursor or presses a button: the module only returns
verdicts, so every rule stays testable without a camera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import dist
from typing import Callable, Optional

from .config import TrackingConfig
from .hand_tracker import Hand, LandmarkIndex as LI, hand_size

# The full set of landmarks the pinch/drag mapping depends on: the palm, which
# measures the hand, and the index tip, which is the pointer itself. A folded
# finger that clips the frame edge in an otherwise well-tracked hand is not a
# reason to stop.
ANCHOR_LANDMARKS = (
    LI.WRIST,
    LI.THUMB_CMC,
    LI.INDEX_FINGER_MCP,
    LI.MIDDLE_FINGER_MCP,
    LI.PINKY_MCP,
    LI.INDEX_FINGER_TIP,
)

OK = "ok"
NO_HAND = "no hand"
LOW_CONFIDENCE = "low confidence"
IMPLAUSIBLE_PALM = "implausible hand size"
POINTER_OUT = "pointer out of frame"
PALM_OUT = "palm out of frame, cursor only"


@dataclass(frozen=True)
class HandVerdict:
    """What this tracker result may be trusted to do.

    ``pointer_ok``: the pointer is observed, so the cursor may follow it.
    ``full``: every anchor is observed, so pinch/click/drag/scroll may act.
    """

    pointer_ok: bool
    full: bool
    reason: str


def _inside(value: float, margin: float) -> bool:
    return margin <= value <= 1.0 - margin


def assess_hand(hand: Optional[Hand], config: TrackingConfig) -> HandVerdict:
    """Decide what this tracker result may drive."""
    if hand is None or len(hand.landmarks) < len(LI):
        return HandVerdict(False, False, NO_HAND)
    if hand.score < config.min_hand_score:
        return HandVerdict(False, False, LOW_CONFIDENCE)
    palm = hand_size(hand.landmarks)
    if not config.min_palm_size <= palm <= config.max_palm_size:
        return HandVerdict(False, False, IMPLAUSIBLE_PALM)
    margin = config.frame_margin
    tip = hand.landmarks[LI.INDEX_FINGER_TIP]
    if not (_inside(tip.x, margin) and _inside(tip.y, margin)):
        return HandVerdict(False, False, POINTER_OUT)
    for index in ANCHOR_LANDMARKS:
        landmark = hand.landmarks[index]
        if not (_inside(landmark.x, margin) and _inside(landmark.y, margin)):
            return HandVerdict(True, False, PALM_OUT)
    return HandVerdict(True, True, OK)


def pointer_of(hand: Optional[Hand]) -> Optional[tuple[float, float]]:
    """The pointer the classifier would report (index fingertip), or ``None``."""
    if hand is None or len(hand.landmarks) < len(LI):
        return None
    tip = hand.landmarks[LI.INDEX_FINGER_TIP]
    return (float(tip.x), float(tip.y))


class MotionGuard:
    """Flag frames whose pointer moved faster than the camera could freeze.

    Above ``motion_blur_speed`` (frame widths per second) the landmark geometry
    is smeared enough that the pinch ratio is meaningless, so pinch decisions
    must not be taken on such a frame. Movement itself is not gated: the cursor
    still follows a throw, it just cannot click or scroll out of one.
    """

    MAX_GAP_SECONDS = 0.5

    def __init__(
        self,
        config: TrackingConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_speed = config.motion_blur_speed
        self._clock = clock
        self._last: Optional[tuple[float, float]] = None
        self._last_at: Optional[float] = None

    def update(
        self,
        pointer: Optional[tuple[float, float]],
        now: Optional[float] = None,
    ) -> bool:
        """Return True when this frame is too fast to trust for pinch decisions."""
        timestamp = self._clock() if now is None else now
        if pointer is None:
            self.reset()
            return False
        current = (float(pointer[0]), float(pointer[1]))
        previous, previous_at = self._last, self._last_at
        self._last, self._last_at = current, timestamp
        if previous is None or previous_at is None:
            return False
        elapsed = timestamp - previous_at
        if elapsed <= 0 or elapsed > self.MAX_GAP_SECONDS:
            return False
        return dist(current, previous) / elapsed > self._max_speed

    def reset(self) -> None:
        self._last = None
        self._last_at = None
