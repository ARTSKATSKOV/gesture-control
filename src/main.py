"""Application entry point for the local gesture-control loop."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2

from .action_controller import ActionController
from .config import AppConfig
from .gesture_classifier import FrameGesture, Gesture, GestureClassifier
from .hand_tracker import HandTracker, new_hand_tracker
from .overlay import OverlayState, draw_overlay
from .stabilizer import (
    PinchDebounce,
    PinchEvent,
    PinchStateMachine,
    TemporalStabilizer,
)
from .tracking_quality import (
    MotionGuard,
    assess_hand,
    pointer_of,
)

WINDOW_NAME = "Gesture Control"


def control_frame(raw: FrameGesture, stable: Gesture | None, cursor_gesture: str) -> FrameGesture:
    """Keep fresh positions, but never put an old action on a new pose.

    Cursor movement can follow a current POINT immediately. Discrete actions
    still require temporal confirmation AND agreement with this frame.
    Pinch geometry always reaches the controller, even during transitions.
    """
    if raw.gesture is Gesture(cursor_gesture) or raw.gesture is stable:
        return raw
    return replace(raw, gesture=Gesture.UNKNOWN)


def open_camera(config) -> cv2.VideoCapture:
    backend_name = config.backend.strip().lower()
    if backend_name == "dshow":
        backends = (cv2.CAP_DSHOW,)
    elif backend_name == "msmf":
        backends = (cv2.CAP_MSMF,)
    else:
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF)

    for backend in backends:
        camera = cv2.VideoCapture(config.index, backend)
        if camera.isOpened():
            return camera
        camera.release()
    raise RuntimeError(
        f"cannot open camera index {config.index} with backend {backend_name}"
    )


def _key_matches(key: int, configured: str) -> bool:
    normalized = configured.strip().lower()
    if normalized in {"esc", "escape"}:
        return key == 27
    if normalized == "space":
        return key == 32
    return key == ord(normalized[0]) if normalized else False


def preview_visible() -> bool:
    """Whether the HighGUI preview window is currently on screen.

    ``WND_PROP_VISIBLE`` reports 1.0 while the window is shown and 0.0 once it
    has been dismissed; a window that never existed reports 0.0 as well, so
    callers must only consult this after the first ``imshow``. Some HighGUI
    backends raise instead of returning a value for a destroyed window.
    """
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


class FpsCounter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last: float | None = None
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def tick(self) -> float:
        now = self._clock()
        if self._last is not None:
            elapsed = now - self._last
            if elapsed > 0:
                instant = 1.0 / elapsed
                self._fps = instant if self._fps == 0 else self._fps * 0.9 + instant * 0.1
        self._last = now
        return self._fps


def run(config: AppConfig | None = None) -> None:
    config = config or AppConfig.load()
    project_root = Path(__file__).resolve().parent.parent
    camera = open_camera(config.camera)
    tracker: HandTracker | None = None
    controller: ActionController | None = None
    stabilizer = TemporalStabilizer(config.stabilizer)
    pinch_state = PinchStateMachine(config.pinch)
    pinch_debounce = PinchDebounce(config.pinch.confirm_frames)
    classifier = GestureClassifier(config.pinch)
    motion_guard = MotionGuard(config.tracking)
    fps = FpsCounter()
    enabled = True
    failed_reads = 0
    preview_shown = False

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
        tracker = new_hand_tracker(config.hand, model_root=project_root)
        controller = ActionController(
            cursor=config.cursor,
            mappings=config.mappings,
            scroll=config.scroll,
            volume=config.volume,
        )

        while True:
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                controller.process(None, pinch_state.reset())
                pinch_debounce.reset()
                stabilizer.reset()
                classifier.classify(None)
                failed_reads += 1
                key = cv2.waitKey(1) & 0xFF
                if _key_matches(key, config.hotkeys.quit):
                    break
                # Nothing is drawn on this path, so a window only counts as
                # dismissed once a preview has actually been shown.
                if preview_shown and not preview_visible():
                    break
                if failed_reads >= 30:
                    raise RuntimeError("Camera stopped delivering frames; reconnect it and restart.")
                time.sleep(0.01)
                continue
            failed_reads = 0
            frame = cv2.flip(frame, 1)
            track_frame = cv2.resize(
                frame,
                (config.hand.track_width, config.hand.track_height),
                interpolation=cv2.INTER_AREA,
            )
            hand = tracker.process(track_frame)
            # Only a plausible, fully observed hand may act on Windows. A hand
            # that MediaPipe extrapolated out of the frame used to keep the
            # mouse button pressed during a drag, and its collapsing geometry
            # produced a click exactly when the hand disappeared.
            verdict = assess_hand(hand, config.tracking)
            usable_hand = hand if verdict.usable else None
            blurred = motion_guard.update(pointer_of(usable_hand))
            raw = classifier.classify(usable_hand, motion_blurred=blurred)
            if usable_hand is None or not enabled:
                pinch_debounce.reset()
                stabilizer.reset()
            # A blurred frame must neither start nor end a pinch: skipping the
            # debounce and the state machine is what turns a fast throw from a
            # stray click into nothing at all.
            if blurred:
                stable_gesture = stabilizer.confirmed
                pinch_event = PinchEvent.NONE
            else:
                stable_gesture = stabilizer.update(raw.gesture)
                pinch_signal = pinch_debounce.update(raw.pinch_active) if enabled else False
                pinch_event = pinch_state.update(
                    pinch_active=pinch_signal,
                    hand_present=usable_hand is not None if enabled else False,
                )
            control = (
                control_frame(raw, stable_gesture, config.mappings.cursor)
                if enabled and usable_hand is not None
                else None
            )
            if blurred and control is not None:
                # A smeared frame may not freeze, click, scroll or drag on its
                # own; the one thing it must not break is an already pressed
                # button, so a running drag keeps following the hand.
                control = replace(
                    control,
                    pinch_active=controller.drag_active,
                    pinch_ratio=float("inf"),
                )
            controller.process(control, pinch_event)

            current_gesture = raw.gesture.value
            if controller.drag_active:
                interaction = "DRAG: move hand"
            elif blurred:
                interaction = "MOTION: events paused"
            elif enabled and usable_hand is not None and (
                raw.pinch_active or raw.pinch_ratio <= config.cursor.pinch_guard_ratio
            ):
                interaction = "PINCH: cursor locked"
            else:
                interaction = ""
            draw_overlay(
                frame,
                OverlayState(
                    gesture=current_gesture,
                    fps=fps.tick(),
                    enabled=enabled,
                    handedness=raw.handedness,
                    pinch_ratio=raw.pinch_ratio if usable_hand is not None else None,
                    interaction=interaction,
                    # Show why a frame was dropped instead of guessing later:
                    # Art can read the reason straight off the preview.
                    tracking=verdict.reason,
                    motion_blurred=blurred,
                ),
                hand,
                cursor=config.cursor,
            )
            cv2.imshow(WINDOW_NAME, frame)
            preview_shown = True
            key = cv2.waitKey(1) & 0xFF
            # waitKey pumps the GUI message loop, so this is the first point at
            # which a user close (X / Alt+F4) is observable. Leaving the loop
            # here is what keeps the next imshow from recreating the window.
            if not preview_visible() or _key_matches(key, config.hotkeys.quit):
                break
            if _key_matches(key, config.hotkeys.toggle):
                enabled = controller.toggle_enabled()
                stabilizer.reset()
                pinch_debounce.reset()
                event = pinch_state.reset()
                if event.value:
                    controller.process(None, event)
    finally:
        try:
            if controller is not None:
                controller.shutdown()
        finally:
            try:
                if tracker is not None:
                    tracker.close()
            finally:
                camera.release()
                cv2.destroyAllWindows()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
