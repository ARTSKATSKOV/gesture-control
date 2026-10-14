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
from src.hand_tracker import Hand, Landmark, LandmarkIndex as LI
from src.main import run
from src.stabilizer import PinchStateMachine, TemporalStabilizer
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
         patch("src.main.TemporalStabilizer",
               side_effect=lambda cfg, clock=None: TemporalStabilizer(
                   cfg, clock=lambda: timestamp[0]
               )), \
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


def two_down_hand(dy=0.0):
    """Index and middle point down; the other fingers stay folded at the wrist."""
    points = [Landmark(0.50, 0.40 + dy) for _ in range(21)]

    def put(index, x, y):
        points[index] = Landmark(x, y + dy)

    put(LI.INDEX_FINGER_MCP, 0.48, 0.48)
    put(LI.INDEX_FINGER_PIP, 0.48, 0.56)
    put(LI.INDEX_FINGER_DIP, 0.48, 0.62)
    put(LI.INDEX_FINGER_TIP, 0.48, 0.70)
    put(LI.MIDDLE_FINGER_MCP, 0.53, 0.48)
    put(LI.MIDDLE_FINGER_PIP, 0.53, 0.56)
    put(LI.MIDDLE_FINGER_DIP, 0.53, 0.62)
    put(LI.MIDDLE_FINGER_TIP, 0.53, 0.70)
    put(LI.THUMB_CMC, 0.46, 0.42)
    put(LI.THUMB_MCP, 0.42, 0.40)
    put(LI.THUMB_IP, 0.40, 0.36)
    put(LI.THUMB_TIP, 0.38, 0.32)
    return Hand(landmarks=tuple(points), handedness="Right", score=1.0)


def _scroll_count(run_result) -> int:
    return run_result.actions().count("scroll")


def test_a_palm_leaving_the_frame_cannot_scroll():
    """An extrapolated two-finger pose used to keep wheeling the page."""
    settled = [two_down_hand()] * 8
    leaving = [two_down_hand(dy=-0.50)] * 4
    parked = [hand()] * 4
    while_leaving = drive(settled + leaving)
    while_pointing = drive(settled + parked)
    assert _scroll_count(while_pointing) > 0, "an in-frame two-down pose must scroll"
    assert _scroll_count(while_leaving) == _scroll_count(while_pointing)
    assert "palm out" in while_leaving.overlay_states()[-1].tracking
    assert "click" not in while_leaving.actions()


def test_palm_out_frames_do_not_arm_a_scroll_for_the_next_good_frame():
    """Frames with the wrist already gone must not fill the gesture window."""
    armed = [two_down_hand(dy=-0.50)] * 8
    # The jump back into frame is blurred; the frames after it are stationary.
    returned = [two_down_hand()] * 4
    run_result = drive(armed + returned)
    assert _scroll_count(run_result) == 0
    assert any("palm out" in state.tracking for state in run_result.overlay_states())


def test_a_blurred_throw_keeps_moving_when_the_pose_is_no_longer_point():
    """A fast open palm used to freeze the cursor because it is not POINT."""
    thrown_open_palm = synthetic_hand(dx=0.28)
    run_result = drive([hand(), thrown_open_palm])
    moves = [call for call in run_result.mouse.calls if call[0] == "move"]
    assert len(moves) >= 2
    assert moves[-1][1] > moves[0][1] + 100
    assert "click" not in run_result.actions()
    assert run_result.overlay_states()[-1].motion_blurred is True


def test_a_smeared_frame_does_not_finish_a_half_confirmed_pinch():
    """One pinched frame, a throw, then one more pinch must not click."""
    frames = [
        hand(pinch=True),
        hand(dx=0.28, pinch=True),
        hand(dx=0.28, dy=0.02, pinch=True),
        hand(dx=0.28, dy=0.02),
    ]
    run_result = drive(frames)
    assert "click" not in run_result.actions()
    assert "down" not in run_result.actions()


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
    assert [state.tracking for state in states] == ["ok", "pointer out of frame", "ok"]
    assert [state.motion_blurred for state in states] == [False, False, False]


def test_a_hand_reaching_the_bottom_keeps_moving_but_cannot_click():
    """The second live report: the bottom stayed unreachable.

    Reaching the taskbar lowers the palm out of the frame while the fingertip
    stays visible. That frame used to be thrown away entirely, which froze the
    cursor right where the screen edge is; it must move the cursor and still
    never click.
    """
    reaching = [hand(pinch=True), hand(dy=0.5, pinch=True), hand(dy=0.5, pinch=True)]
    run_result = drive(reaching + [hand(dy=0.5)])
    actions = run_result.actions()
    assert "move" in actions, "a low palm must not freeze the cursor"
    assert "click" not in actions
    assert "down" not in actions
    assert run_result.overlay_states()[-1].tracking == "palm out of frame, cursor only"


def test_a_drag_ends_when_the_palm_leaves_the_frame():
    """A button must never stay down because the palm dipped out of the frame."""
    held = [hand(dy=0.02 * step, pinch=True) for step in range(1, 9)]
    leaving = [hand(dy=0.5, pinch=True)] * 3
    run_result = drive([hand(), hand()] + held + leaving)
    actions = run_result.actions()
    assert "down" in actions, "a held pinch must start a drag"
    assert actions.count("up") == 1
    assert "click" not in actions


def test_overlay_marks_a_blurred_frame():
    run_result = drive([hand(pinch=True), hand(dx=0.28, pinch=True), hand(pinch=True)])
    states = run_result.overlay_states()
    assert [state.motion_blurred for state in states] == [False, True, True]
    assert "MOTION" not in states[0].interaction
