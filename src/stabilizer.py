"""Temporal stabilization of raw per-frame gesture classifications.

Raw classifier output is noisy and changes frame to frame, so it must not
drive Windows actions directly. Two collaborating pieces live here:

* ``TemporalStabilizer`` confirms a gesture once it appears in
  ``required_frames`` of the last ``window_frames`` frames and has stayed the
  dominant candidate for ``stable_ms``.
* ``PinchStateMachine`` owns the ``IDLE -> PINCH_PENDING -> DRAGGING``
  lifecycle and emits discrete click/drag events with configurable hold and
  click-cooldown timing. It always guarantees that an active drag is ended
  when the hand is lost, or when ``reset()`` is called during shutdown.
"""

import enum
import time
from collections import Counter, deque
from typing import Any, Callable, Optional

from .config import PinchConfig, StabilizerConfig


class PinchEvent(enum.Enum):
    NONE = 0
    CLICK = 1
    DRAG_START = 2
    DRAG_END = 3
    DRAG_END_FORCED = 4


class PinchState(enum.Enum):
    IDLE = "IDLE"
    PINCH_PENDING = "PINCH_PENDING"
    DRAGGING = "DRAGGING"


class TemporalStabilizer:
    def __init__(
        self,
        config: StabilizerConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_frames = config.window_frames
        self._required_frames = config.required_frames
        self._stable_seconds = config.stable_ms / 1000.0
        self._clock = clock
        self._history: deque[Any] = deque(maxlen=config.window_frames)
        self._dominant: Any = None
        self._dominant_since: Optional[float] = None
        self._confirmed: Any = None

    @property
    def confirmed(self) -> Any:
        return self._confirmed

    def update(self, raw_gesture: Any, now: Optional[float] = None) -> Any:
        timestamp = self._clock() if now is None else now
        self._history.append(raw_gesture)
        dominant = self._dominant_gesture()
        if dominant != self._dominant:
            self._dominant = dominant
            self._dominant_since = timestamp
        count = self._history.count(dominant)
        if (
            dominant is not None
            and count >= self._required_frames
            and self._dominant_since is not None
            and timestamp - self._dominant_since >= self._stable_seconds
        ):
            self._confirmed = dominant
        else:
            self._confirmed = None
        return self._confirmed

    def reset(self) -> None:
        self._history.clear()
        self._dominant = None
        self._dominant_since = None
        self._confirmed = None

    def _dominant_gesture(self) -> Any:
        if not self._history:
            return None
        counts = Counter(self._history)
        best = max(counts.values())
        candidates = [gesture for gesture, count in counts.items() if count == best]
        if len(candidates) == 1:
            return candidates[0]
        for gesture in reversed(self._history):
            if gesture in candidates:
                return gesture
        return None


class PinchDebounce:
    """Confirm pinch on/off only after it persists for N consecutive frames.

    The full ``TemporalStabilizer`` (majority-of-window + stable time) is too
    slow for a click, but a raw per-frame pinch signal lets a single transient
    frame produce a spurious click. This lightweight multi-frame gate sits
    between the classifier and the pinch state machine: it keeps click latency
    low while still requiring the signal to persist across several frames.
    """

    def __init__(self, frames: int = 3) -> None:
        if frames < 1:
            raise ValueError("frames must be >= 1")
        self._frames = frames
        self._count = 0
        self._state = False

    @property
    def state(self) -> bool:
        return self._state

    def update(self, active: bool) -> bool:
        if active == self._state:
            self._count = 0
        else:
            self._count += 1
            if self._count >= self._frames:
                self._state = active
                self._count = 0
        return self._state

    def reset(self) -> None:
        self._state = False
        self._count = 0


class PinchStateMachine:
    def __init__(
        self,
        config: PinchConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._drag_hold_seconds = config.drag_hold_ms / 1000.0
        self._click_cooldown_seconds = config.click_cooldown_ms / 1000.0
        self._clock = clock
        self._state = PinchState.IDLE
        self._state_since: Optional[float] = None
        self._last_click_at: float = float("-inf")

    @property
    def state(self) -> PinchState:
        return self._state

    def update(
        self,
        pinch_active: bool,
        hand_present: bool,
        now: Optional[float] = None,
    ) -> PinchEvent:
        timestamp = self._clock() if now is None else now
        state = self._state
        if state is PinchState.IDLE:
            if pinch_active and hand_present:
                self._enter(PinchState.PINCH_PENDING, timestamp)
            return PinchEvent.NONE

        if state is PinchState.PINCH_PENDING:
            return self._update_pending(pinch_active, hand_present, timestamp)
        return self._update_dragging(pinch_active, hand_present, timestamp)

    def reset(self) -> PinchEvent:
        event = PinchEvent.NONE
        if self._state is PinchState.DRAGGING:
            event = PinchEvent.DRAG_END_FORCED
        self._state = PinchState.IDLE
        self._state_since = None
        return event

    def _enter(self, state: PinchState, timestamp: float) -> None:
        self._state = state
        self._state_since = timestamp

    def _update_pending(
        self,
        pinch_active: bool,
        hand_present: bool,
        timestamp: float,
    ) -> PinchEvent:
        if not hand_present:
            self._enter(PinchState.IDLE, timestamp)
            return PinchEvent.NONE
        elapsed = timestamp - self._state_since
        if not pinch_active:
            if elapsed < self._drag_hold_seconds:
                if timestamp - self._last_click_at >= self._click_cooldown_seconds:
                    self._last_click_at = timestamp
                    self._enter(PinchState.IDLE, timestamp)
                    return PinchEvent.CLICK
            self._enter(PinchState.IDLE, timestamp)
            return PinchEvent.NONE
        if elapsed >= self._drag_hold_seconds:
            self._enter(PinchState.DRAGGING, timestamp)
            return PinchEvent.DRAG_START
        return PinchEvent.NONE

    def _update_dragging(
        self,
        pinch_active: bool,
        hand_present: bool,
        timestamp: float,
    ) -> PinchEvent:
        if not hand_present:
            self._enter(PinchState.IDLE, timestamp)
            return PinchEvent.DRAG_END_FORCED
        if not pinch_active:
            self._enter(PinchState.IDLE, timestamp)
            return PinchEvent.DRAG_END
        return PinchEvent.NONE
