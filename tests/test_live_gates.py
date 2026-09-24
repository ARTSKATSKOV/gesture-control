"""Live-run regressions for the defects reported from the 2026-09-24 run.

Both tests drive the real classification, stabilization and action pipeline
through ``run()`` with scripted tracker output: no camera, no HighGUI window and
no cursor movement. They cover exactly what was reported by hand:

* a hand that leaves the frame during a drag used to keep the mouse button
  pressed (the window kept following the hand) and produced a click at the
  moment it disappeared;
* a fast throw used to fire a spurious pinch, which froze the cursor and landed
  a click.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import AppConfig
from src.hand_tracker import Hand, LandmarkIndex as LI
from src.main import run
from src.stabilizer import PinchStateMachine
from src.tracking_quality import MotionGuard
from test_action_controller import FakeMouse, controller
from test_hand_tracker import synthetic_hand

IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)
ESC = 27
NO_KEY = 255


def hand(dx=0.0, dy=0.0, pinch=False):
    """A pointing hand (other fingers folded) offset in the frame."""
    landmarks = list(synthetic_hand(dx=dx, dy=dy).landmarks)
    for tip in (LI.MIDDLE_FINGER_TIP, LI.RING_FINGER_TIP, LI.PINKY_TIP):
        landmarks[tip] = landmarks[LI.WRIST]
    if pinch:
        landmarks[LI.THUMB_TIP] = landmarks[LI.INDEX_FINGER_TIP]
    return Hand(landmarks=tuple(landmarks), handedness="Right", score=1.0)


@dataclass
class Run:
    mouse: FakeMouse
    overlay: MagicMock

    def actions(self) -> list[str]:
        return [call[0] for call in self.mouse.calls]

    def overlay_states(self) -> list:
        return [call.args[1] for call in self.overlay.call_args_list]


def drive(script, mouse=None):
    """Run the real loop over ``script`` (one tracker result per frame)."""
    assert script, "the script must contain at least one frame"
    config = AppConfig.defaults()
    timestamp = [0.0]
    mouse = mouse or FakeMouse()
    action_controller = controller(mouse, clock=lambda: timestamp[0])
    camera = MagicMock()
    tracker = MagicMock()
    hands = deque(script)

    def read():
        timestamp[0] += 0.1
        return (True, IMAGE)

    camera.read.side_effect = read
    tracker.process.side_effect = lambda _frame: hands.popleft() if hands else None
    # One waitKey per processed frame, and the loop leaves on the last one, so
    # exactly len(script) frames are rendered.
    keys = [NO_KEY] * (len(script) - 1) + [ESC]
    with patch("src.main.open_camera", return_value=camera), \
         patch("src.main.new_hand_tracker", return_value=tracker), \
         patch("src.main.ActionController", return_value=action_controller), \
         patch("src.main.PinchStateMachine",
               return_value=PinchStateMachine(config.pinch,
                                              clock=lambda: timestamp[0])), \
         patch("src.main.MotionGuard",
               return_value=MotionGuard(config.tracking,
                                        clock=lambda: timestamp[0])), \
         patch("src.main.draw_overlay") as overlay, \
         patch("src.main.cv2.imshow"), \
         patch("src.main.cv2.destroyAllWindows"), \
         patch("src.main.cv2.getWindowProperty", return_value=1.0), \
         patch("src.main.cv2.waitKey", side_effect=keys):
        run(config)
    assert not hands, "the loop must consume the whole script"
    assert action_controller.drag_active is False
    return Run(mouse=mouse, overlay=overlay)


def test_losing_the_hand_during_a_drag_releases_the_button_without_clicking():
    """The reported defect: the window kept following the hand off screen."""
    held_pinch = [hand(dy=0.02 * step, pinch=True) for step in range(1, 9)]
    left_the_frame = [hand(dy=1.0, pinch=True)] * 2
    gone = [None] * 2
    run_result = drive([hand(), hand()] + held_pinch + left_the_frame + gone)
    actions = run_result.actions()
    assert "down" in actions, "a held pinch must start a drag"
    assert actions.index("up") > actions.index("down")
    assert actions.count("up") == 1, "the button must be released exactly once"
    assert "click" not in actions, "losing the hand must not click"


def test_a_fast_throw_cannot_click_but_still_moves_the_cursor():
    """The reported defect: a sharp throw fired a false pinch and a click.

    Every frame after the first moves the fingertip 0.28 of the frame in 0.1 s
    (2.8 frame widths per second, past the 2.5 blur limit) while the pinch-shaped
    hand stays inside the picture. Without the gate the second frame would
    confirm a pinch and the release at the end would land a click.
    """
    thrown = [
        hand(pinch=True),
        hand(dx=0.28, pinch=True),
        hand(pinch=True),
        hand(dx=0.28, pinch=True),
    ]
    run_result = drive(thrown + [hand(dy=0.05)] * 3)
    actions = run_result.actions()
    assert "click" not in actions
    assert "down" not in actions
    assert "up" not in actions
    assert "move" in actions, "the cursor must still follow the throw"


def test_a_slow_pinch_still_clicks():
    """The gate must not cost a normal, deliberate click."""
    deliberate = [hand(dy=0.02 * step, pinch=True) for step in range(1, 4)]
    release = [hand(dy=0.06)] * 3
    run_result = drive(deliberate + release)
    actions = run_result.actions()
    assert "click" in actions
    assert "down" not in actions


def test_overlay_shows_why_a_frame_was_dropped():
    """Art reads the drop reason straight off the preview during a live run."""
    run_result = drive([hand(), hand(dy=1.0), hand()])
    states = run_result.overlay_states()
    assert [state.tracking for state in states] == ["ok", "hand out of frame", "ok"]
    assert [state.motion_blurred for state in states] == [False, False, False]


def test_overlay_marks_a_blurred_frame():
    run_result = drive([hand(pinch=True), hand(dx=0.28, pinch=True), hand(pinch=True)])
    states = run_result.overlay_states()
    assert [state.motion_blurred for state in states] == [False, True, True]
    assert "MOTION" not in states[0].interaction
