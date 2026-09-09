"""Application entry point for the local gesture-control loop."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2

from .action_controller import ActionController
from .config import AppConfig
from .gesture_classifier import Gesture, GestureClassifier
from .hand_tracker import HandTracker, new_hand_tracker
from .overlay import OverlayState, draw_overlay
from .stabilizer import PinchStateMachine, TemporalStabilizer

WINDOW_NAME = "Gesture Control"
KEY_TOGGLE = ord(" ")
KEY_QUIT = 27


def _key_matches(key: int, configured: str) -> bool:
    normalized = configured.strip().lower()
    if normalized in {"esc", "escape"}:
        return key == 27
    if normalized == "space":
        return key == 32
    return key == ord(normalized[0]) if normalized else False


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
    camera = cv2.VideoCapture(config.camera.index)
    tracker: HandTracker | None = None
    controller: ActionController | None = None
    stabilizer = TemporalStabilizer(config.stabilizer)
    pinch_state = PinchStateMachine(config.pinch)
    classifier = GestureClassifier(config.pinch)
    fps = FpsCounter()
    enabled = True

    try:
        if not camera.isOpened():
            raise RuntimeError(f"cannot open camera index {config.camera.index}")
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
            if not ok:
                raise RuntimeError("failed to read frame from camera")
            frame = cv2.flip(frame, 1)
            hand = tracker.process(frame)
            raw = classifier.classify(hand)
            stable_gesture = stabilizer.update(raw.gesture)
            stable_frame = (
                replace(raw, gesture=stable_gesture)
                if stable_gesture is not None and enabled
                else None
            )
            stable_pinch = stable_gesture is Gesture.PINCH
            pinch_event = pinch_state.update(
                pinch_active=stable_pinch if enabled else False,
                hand_present=hand is not None if enabled else False,
            )
            controller.process(stable_frame, pinch_event)

            current_gesture = stable_gesture.value if stable_gesture is not None else raw.gesture.value
            draw_overlay(
                frame,
                OverlayState(
                    gesture=current_gesture,
                    fps=fps.tick(),
                    enabled=enabled,
                    handedness=raw.handedness,
                    pinch_ratio=raw.pinch_ratio if hand is not None else None,
                ),
                hand,
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if _key_matches(key, config.hotkeys.quit):
                break
            if _key_matches(key, config.hotkeys.toggle):
                enabled = controller.toggle_enabled()
                stabilizer.reset()
                event = pinch_state.reset()
                if event.value:
                    controller.process(None, event)
    except Exception:
        if controller is not None:
            controller.shutdown()
        raise
    finally:
        if controller is not None:
            controller.shutdown()
        if tracker is not None:
            tracker.close()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
