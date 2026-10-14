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


def tracking_frame(frame, track_width: int, track_height: int):
    """Downscale a preview frame for the tracker without stretching it.

    Landmarks are normalized to this image and then drawn on the preview, so
    the two have to share an aspect ratio. Forcing ``track_width`` x
    ``track_height`` turns a 16:9 camera into 4:3, and both the cursor and the
    pinch distance come out wrong.
    """
    height, width = frame.shape[:2]
    scale = min(track_width / float(width), track_height / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if resized_width == width and resized_height == height:
        return frame
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )


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
    # True after a frame that was actually allowed to move the cursor. A fast
    # throw often stops looking like POINT; the cursor should keep following
    # that throw, and should not start following a scroll or a fist.
    was_following = False
    cursor_gesture = Gesture(config.mappings.cursor)

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
        # A deep driver queue shows an old frame, so the cursor trails the hand.
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                # The next delivered frame must not be compared with a pointer
                # from before the gap, or a reconnect looks like a throw.
                motion_guard.reset()
                was_following = False
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
            track_frame = tracking_frame(
                frame, config.hand.track_width, config.hand.track_height
            )
            hand = tracker.process(track_frame)
            # Two different questions, two different answers: moving the cursor
            # only needs the fingertip, while pinch and drag also need the palm
            # that measures the hand. Gating movement on the palm is what made
            # the bottom of the screen unreachable; letting an extrapolated hand
            # press buttons is what clicked when the hand disappeared.
            verdict = assess_hand(hand, config.tracking)
            movable_hand = hand if verdict.pointer_ok else None
            blurred = motion_guard.update(pointer_of(movable_hand))
            decides_pinch = enabled and verdict.full and not blurred
            raw = classifier.classify(movable_hand, decide_pinch=decides_pinch)
            if movable_hand is None or not enabled:
                pinch_debounce.reset()
                stabilizer.reset()
            if blurred:
                # A smeared frame must neither start nor end a pinch: skipping
                # the debounce and the state machine is what turns a fast throw
                # from a stray click into nothing at all. A half-counted
                # transition is dropped too, so the next settled frame cannot
                # finish a pinch the smear started.
                pinch_debounce.clear_pending()
                stable_gesture = stabilizer.confirmed
                pinch_event = PinchEvent.NONE
            elif not (enabled and verdict.full):
                # The palm is outside the frame, or the app is disabled: no pinch
                # may be decided, and a held mouse button has to be released.
                # Extrapolated landmarks must not be learned as a gesture, or a
                # hand leaving through the edge confirms a scroll on its way out.
                pinch_debounce.reset()
                stable_gesture = stabilizer.confirmed
                pinch_event = pinch_state.update(pinch_active=False, hand_present=False)
            else:
                stable_gesture = stabilizer.update(raw.gesture)
                pinch_signal = pinch_debounce.update(raw.pinch_active)
                pinch_event = pinch_state.update(pinch_active=pinch_signal, hand_present=True)
            control = (
                control_frame(raw, stable_gesture, config.mappings.cursor)
                if enabled and movable_hand is not None
                else None
            )
            if control is not None and not decides_pinch:
                # Such a frame may not click, scroll or drag by itself. The one
                # thing it must not break is a button that is already down, and
                # only a smeared frame may keep one: a palm that left the frame
                # means the hand is going away. A throw that started as pointing
                # keeps moving the cursor; any other pose becomes UNKNOWN so a
                # smeared or extrapolated TWO_UP cannot scroll.
                follow = control.gesture is cursor_gesture or (blurred and was_following)
                control = replace(
                    control,
                    gesture=cursor_gesture if follow else Gesture.UNKNOWN,
                    pinch_active=blurred and controller.drag_active,
                    pinch_ratio=float("inf"),
                )
            was_following = (
                control is not None
                and control.gesture is cursor_gesture
                and not control.pinch_active
            )
            controller.process(control, pinch_event)

            current_gesture = raw.gesture.value
            if controller.drag_active:
                interaction = "DRAG: move hand"
            elif blurred:
                interaction = "MOTION: events paused"
            elif enabled and movable_hand is not None and not verdict.full:
                interaction = "HAND LEAVING: cursor only"
            elif enabled and movable_hand is not None and (
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
                    pinch_ratio=raw.pinch_ratio if movable_hand is not None else None,
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
