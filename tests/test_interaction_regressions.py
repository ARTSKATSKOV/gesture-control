"""User-facing gesture transitions and full-screen reachability."""

from dataclasses import replace

import pytest

from src.action_controller import CursorSmoother
from src.config import AppConfig, ConfigError, CursorConfig
from src.gesture_classifier import Gesture
from src.main import control_frame
from src.stabilizer import PinchEvent
from test_action_controller import FakeMouse, controller, frame
from test_gesture_classifier import _palm_up, land, make_hand, WRIST_X, WRIST_Y
from src.gesture_classifier import GestureClassifier
from src.hand_tracker import LandmarkIndex as LI
from src.config import PinchConfig


def test_pinch_approach_and_one_frame_dropout_cannot_move_cursor():
    mouse = FakeMouse()
    c = controller(mouse)
    c.process(frame(Gesture.POINT), now=0)
    initial = mouse.calls.copy()
    c.process(replace(frame(Gesture.POINT, (0.5, 0.6)), pinch_ratio=0.5), now=0.05)
    c.process(frame(Gesture.PINCH, (0.5, 0.7)), now=0.1)
    c.process(frame(Gesture.POINT, (0.5, 0.8)), now=0.15)
    assert mouse.calls == initial
    c.process(frame(Gesture.POINT, (0.5, 0.6)), now=0.4)
    assert len(mouse.calls) == len(initial) + 1


def test_click_release_does_not_move_or_scroll_in_same_frame():
    mouse = FakeMouse()
    c = controller(mouse)
    c.process(frame(Gesture.POINT, (0.2, 0.3)), PinchEvent.CLICK, now=0)
    c.process(frame(Gesture.TWO_UP), now=0.1)
    assert mouse.calls == [("click",)]


def test_stale_point_is_not_applied_to_new_fist_or_pinch_pose():
    mouse = FakeMouse()
    c = controller(mouse)
    for pose in (Gesture.FIST, Gesture.PINCH, Gesture.TWO_DOWN):
        c.process(control_frame(frame(pose), Gesture.POINT, "POINT"), now=0)
    assert mouse.calls == []


def test_point_uses_fresh_geometry_without_waiting_for_majority_vote():
    assert control_frame(frame(Gesture.POINT), None, "POINT").gesture is Gesture.POINT


def test_scroll_requires_current_pose_to_match_confirmed_pose():
    mouse = FakeMouse()
    c = controller(mouse)
    c.process(control_frame(frame(Gesture.OPEN_PALM), Gesture.TWO_UP, "POINT"))
    assert mouse.calls == []


def test_linear_overscan_keeps_center_and_reaches_all_edges_early():
    smoother = CursorSmoother(CursorConfig(smoothing_alpha=1, dead_zone=0), (1280, 960))
    assert smoother.update((0.5, 0.5)) == (640, 480)
    assert smoother.update((1 / 6, 1 / 6)) == (0, 0)
    assert smoother.update((5 / 6, 5 / 6)) == (1279, 959)
    assert smoother.update((0.75, 0.25)) == (1120, 120)


def test_drag_uses_knuckle_and_ignores_curling_tip():
    mouse = FakeMouse()
    c = controller(mouse)
    c.process(frame(Gesture.POINT))
    start = replace(frame(Gesture.PINCH), drag_pointer=(0.5, 0.5))
    c.process(start, PinchEvent.DRAG_START)
    c.process(replace(start, pointer=(0.5, 0.9)))
    assert mouse.calls == [("move", 500, 400), ("down",)]
    c.process(replace(start, drag_pointer=(0.5, 0.6)))
    assert mouse.calls[-1] == ("move", 500, 480)


def test_drag_freezes_during_raw_release_and_stays_on_screen():
    mouse = FakeMouse()
    c = controller(mouse)
    c.process(frame(Gesture.PINCH), PinchEvent.DRAG_START)
    c.process(frame(Gesture.PINCH, (3, 3)))
    assert mouse.calls[-1] == ("move", 999, 799)
    initial = mouse.calls.copy()
    c.process(frame(Gesture.POINT, (0.1, 0.1)))
    assert mouse.calls == initial
    c.process(None, PinchEvent.DRAG_END_FORCED)
    assert mouse.calls[-1] == ("up",)


def test_pinching_does_not_release_when_index_becomes_folded():
    classifier = GestureClassifier(PinchConfig())
    open_landmarks = list(_palm_up().landmarks)
    open_landmarks[LI.THUMB_TIP] = open_landmarks[LI.INDEX_FINGER_TIP]
    assert classifier.classify(make_hand(*open_landmarks)).pinch_active
    folded = [land(WRIST_X, WRIST_Y) for _ in range(21)]
    folded[LI.MIDDLE_FINGER_MCP] = land(WRIST_X, WRIST_Y + 0.1)
    for joint in (LI.INDEX_FINGER_PIP, LI.MIDDLE_FINGER_PIP,
                  LI.RING_FINGER_PIP, LI.PINKY_PIP):
        folded[joint] = land(WRIST_X, WRIST_Y - 0.1)
    hand = make_hand(*folded)
    assert classifier.classify(hand).pinch_active
    classifier.classify(None)
    assert not classifier.classify(hand).pinch_active


def test_camera_failure_releases_drag_and_preserves_full_size_preview():
    from unittest.mock import MagicMock, patch
    import numpy as np
    from src.main import run
    from src.stabilizer import PinchStateMachine

    timestamp = [0.0]
    mouse = FakeMouse()
    c = controller(mouse, clock=lambda: timestamp[0])
    camera = MagicMock()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    reads = [(True, image)] * 10 + [(False, None)]

    def read():
        timestamp[0] += 0.1
        return reads.pop(0)

    camera.read.side_effect = read
    tracker = MagicMock()
    tracker.process.return_value = _palm_up()
    classifier = MagicMock()
    classifier.classify.return_value = frame(Gesture.PINCH)
    state = PinchStateMachine(PinchConfig(), clock=lambda: timestamp[0])
    with patch("src.main.open_camera", return_value=camera), \
         patch("src.main.new_hand_tracker", return_value=tracker), \
         patch("src.main.GestureClassifier", return_value=classifier), \
         patch("src.main.PinchStateMachine", return_value=state), \
         patch("src.main.ActionController", return_value=c), \
         patch("src.main.draw_overlay") as overlay, \
         patch("src.main.cv2.imshow"), \
         patch("src.main.cv2.destroyAllWindows"), \
         patch("src.main.cv2.waitKey", side_effect=[255] * 10 + [27]):
        run(AppConfig.defaults())
    assert mouse.calls == [("down",), ("up",)]
    assert not c.drag_active
    assert all(call.args[0].shape == (192, 256, 3) for call in tracker.process.call_args_list)
    assert all(call.args[0].shape == (480, 640, 3) for call in overlay.call_args_list)
    camera.release.assert_called_once()
    tracker.close.assert_called_once()


@pytest.mark.parametrize("overrides", [
    {"cursor": {"overscan": 0.9}},
    {"cursor": {"overscan": 3.1}},
    {"cursor": {"pinch_guard_ratio": 0}},
    {"cursor": {"pinch_release_ms": -1}},
    {"cursor": {"overscan": float("nan")}},
    {"cursor": {"dead_zone": float("inf")}},
])
def test_invalid_interaction_settings_are_rejected(overrides):
    with pytest.raises(ConfigError):
        AppConfig.from_dict(overrides)
