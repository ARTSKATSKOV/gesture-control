"""Output timing checks without touching the real mouse or camera."""

from unittest.mock import patch

import pytest

from src.cursor_motion import CursorMotion
from src.action_controller import PyAutoGuiBackend
from src.config import CursorConfig


def motion(alpha=0.3):
    calls = []
    clock = [0.0]
    output = CursorMotion(lambda x, y: calls.append((x, y)), lambda: (100, 100),
                          alpha=alpha, dead_zone=0, clock=lambda: clock[0])
    return output, calls, clock


def test_output_has_intermediate_steps_between_15_fps_camera_frames():
    output, calls, clock = motion()
    for frame in range(6):
        output.target(100 + 150 * (frame + 1), 100)
        for tick in range(8):
            clock[0] += 1 / 120
            output.tick()
    assert len(calls) > 35  # Six camera frames produced many smaller movements.
    assert all(a[0] < b[0] for a, b in zip(calls, calls[1:]))


def test_freeze_cancels_pending_motion_without_a_final_jump():
    output, calls, clock = motion()
    output.target(800, 700)
    clock[0] = 0.01
    output.tick()
    before = calls.copy()
    output.freeze()
    for _ in range(20):
        clock[0] += 0.01
        output.tick()
    assert calls == before


def test_stalled_camera_expires_target():
    output, calls, clock = motion()
    output.target(800, 700)
    clock[0] = 0.21
    output.tick()
    assert calls == []


def test_filter_response_is_independent_of_output_tick_rate():
    def position_at_100ms(rate):
        output, calls, clock = motion()
        output.target(900, 700)
        for step in range(1, round(rate / 10) + 1):
            clock[0] = step / rate
            output.tick()
        return calls[-1]
    assert position_at_100ms(60) == position_at_100ms(120)


def test_latest_target_replaces_old_target():
    output, calls, clock = motion(alpha=1)
    output.target(900, 900)
    output.target(200, 200)
    clock[0] = 0.01
    output.tick()
    assert calls == [(200, 200)]


def test_real_backend_opts_out_of_pyautogui_100ms_pause():
    with patch("src.action_controller.CursorMotion.start"), \
         patch("src.action_controller.pyautogui.position", return_value=(100, 100)), \
         patch("src.action_controller.pyautogui.moveTo") as move, \
         patch("src.action_controller.pyautogui.click") as click, \
         patch("src.action_controller.pyautogui.mouseDown") as down, \
         patch("src.action_controller.pyautogui.mouseUp") as up, \
         patch("src.action_controller.pyautogui.scroll") as scroll:
        backend = PyAutoGuiBackend(CursorConfig(smoothing_alpha=1))
        try:
            backend.move_to(200, 300)
            backend.motion.tick()
            backend.click()
            backend.mouse_down()
            backend.mouse_up()
            backend.scroll(3)
            move.assert_called_once_with(200, 300, duration=0, _pause=False)
            click.assert_called_once_with(_pause=False)
            down.assert_called_once_with(button="left", _pause=False)
            up.assert_called_once_with(button="left", _pause=False)
            scroll.assert_called_once_with(3, _pause=False)
        finally:
            backend.close()


def test_worker_reports_output_failure_to_controller():
    output, calls, clock = motion()
    output.target(200, 300)
    clock[0] = 0.01
    with patch.object(output, "_move", side_effect=RuntimeError("output failed")), \
         patch.object(output._stop, "wait", return_value=False):
        output._run()
    with pytest.raises(RuntimeError, match="output failed"):
        output.target(200, 300)
    assert output._target is None


def test_generated_corner_does_not_trigger_emergency_stop():
    with patch("src.action_controller.pyautogui.FAILSAFE_POINTS", [(0, 0), (999, 799)]), \
         patch("src.action_controller.pyautogui.size", return_value=(1000, 800)), \
         patch("src.action_controller.pyautogui.moveTo") as move:
        PyAutoGuiBackend._send_position(999, 799)
        move.assert_called_once_with(998, 799, duration=0, _pause=False)
