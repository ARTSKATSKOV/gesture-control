"""Tests for OS action orchestration using fake backends."""

from unittest.mock import patch

from src.action_controller import (
    ActionController,
    CursorSmoother,
    PycawVolumeBackend,
)
from src.config import CursorConfig, GestureMappingsConfig, ScrollConfig, VolumeConfig
from src.gesture_classifier import FrameGesture, Gesture
from src.stabilizer import PinchEvent


class FakeMouse:
    def __init__(self):
        self.calls = []
        self.size = (1000, 800)

    def move_to(self, x, y):
        self.calls.append(("move", x, y))

    def click(self):
        self.calls.append(("click",))

    def mouse_down(self):
        self.calls.append(("down",))

    def mouse_up(self):
        self.calls.append(("up",))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def screen_size(self):
        return self.size


class FakeVolume:
    def __init__(self, level=0.5):
        self.level = level
        self.calls = []

    def get_level(self):
        return self.level

    def set_level(self, level):
        self.level = level
        self.calls.append(level)


class TestPycawLazyBackend:
    def test_construction_does_not_touch_audio_endpoint(self):
        with patch("pycaw.pycaw.AudioUtilities") as fake:
            backend = PycawVolumeBackend()
            fake.GetSpeakers.assert_not_called()
            assert backend._endpoint is None

    def test_endpoint_acquired_on_first_use(self):
        endpoint = type(
            "Endpoint",
            (),
            {"GetMasterVolumeLevelScalar": lambda self: 0.5},
        )()
        with patch("pycaw.pycaw.AudioUtilities") as fake:
            fake.GetSpeakers.return_value.EndpointVolume = endpoint
            backend = PycawVolumeBackend()
            assert backend.get_level() == 0.5
            fake.GetSpeakers.assert_called_once()


def frame(gesture, pointer=(0.5, 0.5)):
    return FrameGesture(
        gesture=gesture,
        pinch_active=gesture is Gesture.PINCH,
        pinch_ratio=0.1 if gesture is Gesture.PINCH else 1.0,
        pointer=pointer,
    )


def controller(mouse=None, volume=None, **kwargs):
    return ActionController(
        cursor=CursorConfig(
            smoothing_alpha=1.0,
            dead_zone=0.0,
            overscan=1.0,
            drag_sensitivity=1.0,
        ),
        mappings=GestureMappingsConfig(),
        scroll=ScrollConfig(**kwargs.pop("scroll", {})),
        volume=VolumeConfig(**kwargs.pop("volume_config", {})),
        mouse=mouse or FakeMouse(),
        volume_backend=volume or FakeVolume(),
        screen_size=(1000, 800),
        **kwargs,
    )


class TestCursorSmoother:
    def test_ema_and_dead_zone(self):
        smoother = CursorSmoother(
            CursorConfig(
                smoothing_alpha=0.5,
                dead_zone=10,
                overscan=1.0,
            ),
            (100, 100),
        )
        assert smoother.update((0.0, 0.0)) == (0, 0)
        assert smoother.update((1.0, 1.0)) == (50, 50)
        assert smoother.update((0.49, 0.49)) == (50, 50)

    def test_normalized_coordinates_are_clamped(self):
        smoother = CursorSmoother(
            CursorConfig(
                smoothing_alpha=1.0,
                overscan=1.0,
            ),
            (100, 100),
        )
        assert smoother.update((-1.0, 2.0)) == (0, 99)

    def test_reset_discards_previous_position(self):
        smoother = CursorSmoother(
            CursorConfig(
                smoothing_alpha=1.0,
                overscan=1.0,
            ),
            (100, 100),
        )
        smoother.update((0.2, 0.2))
        smoother.reset()
        assert smoother.update((0.8, 0.8)) == (80, 80)

    def test_overscan_reaches_screen_edges_early(self):
        smoother = CursorSmoother(
            CursorConfig(smoothing_alpha=1.0, dead_zone=0.0, overscan=1.25),
            (100, 100),
        )
        assert smoother.update((0.1, 0.1)) == (0, 0)
        assert smoother.update((0.9, 0.9)) == (99, 99)


class TestMouseActions:
    def setup_method(self):
        self.mouse = FakeMouse()
        self.volume = FakeVolume()
        self.controller = controller(self.mouse, self.volume)

    def test_pointer_moves_from_normalized_point(self):
        position = self.controller.move_pointer((0.25, 0.5))
        assert position == (250, 400)
        assert self.mouse.calls == [("move", 250, 400)]

    def test_raw_pinch_does_not_move_point_cursor(self):
        pinch_frame = FrameGesture(
            gesture=Gesture.POINT,
            pinch_active=True,
            pinch_ratio=0.1,
            pointer=(0.25, 0.5),
        )
        self.controller.process(pinch_frame)
        assert self.mouse.calls == []

    def test_click_is_disabled_when_controller_disabled(self):
        self.controller.set_enabled(False)
        assert self.controller.click() is False
        assert self.mouse.calls == []

    def test_short_pinch_event_clicks_without_moving_cursor(self):
        self.controller.process(frame(Gesture.PINCH), PinchEvent.CLICK)
        assert self.mouse.calls == [("click",)]

    def test_pinch_hold_before_drag_does_not_move_cursor(self):
        self.controller.process(frame(Gesture.PINCH, pointer=(0.2, 0.3)))
        self.controller.process(frame(Gesture.PINCH, pointer=(0.8, 0.9)))
        assert self.mouse.calls == []

    def test_drag_moves_relative_to_anchor(self):
        self.controller.process(
            frame(Gesture.PINCH, pointer=(0.5, 0.5)), PinchEvent.DRAG_START
        )
        assert self.controller.drag_active
        self.controller.process(frame(Gesture.PINCH, pointer=(0.6, 0.5)))
        assert self.mouse.calls[-1] == ("move", 600, 400)
        self.controller.process(frame(Gesture.PINCH, pointer=(0.6, 0.7)))
        assert self.mouse.calls[-1] == ("move", 600, 560)
        self.controller.process(None, PinchEvent.DRAG_END)
        assert self.mouse.calls[-1] == ("up",)
        assert not self.controller.drag_active

    def test_drag_does_not_jump_to_finger_tip(self):
        self.controller.process(
            frame(Gesture.POINT, pointer=(0.5, 0.5))
        )
        assert self.mouse.calls[-1] == ("move", 500, 400)
        self.controller.process(
            frame(Gesture.PINCH, pointer=(0.5, 0.5)), PinchEvent.DRAG_START
        )
        self.controller.process(frame(Gesture.PINCH, pointer=(0.4, 0.9)))
        assert self.mouse.calls == [
            ("move", 500, 400),
            ("down",),
            ("move", 400, 720),
        ]

    def test_drag_lifecycle_has_exactly_one_down_and_up(self):
        self.controller.process(None, PinchEvent.DRAG_START)
        assert [call[0] for call in self.mouse.calls] == ["down"]
        self.controller.process(None, PinchEvent.DRAG_END)
        assert [call[0] for call in self.mouse.calls] == ["down", "up"]
        assert not self.controller.drag_active

    def test_forced_drag_end_releases_mouse(self):
        self.controller.process(None, PinchEvent.DRAG_START)
        self.controller.process(None, PinchEvent.DRAG_END_FORCED)
        assert self.mouse.calls == [("down",), ("up",)]

    def test_shutdown_is_idempotent_and_safe(self):
        self.controller.process(None, PinchEvent.DRAG_START)
        self.controller.shutdown()
        self.controller.shutdown()
        assert self.mouse.calls.count(("up",)) == 1
        assert not self.controller.drag_active

    def test_toggle_resets_smoothing_and_releases_drag(self):
        self.controller.process(None, PinchEvent.DRAG_START)
        assert self.controller.toggle_enabled() is False
        assert self.mouse.calls[-1] == ("up",)
        assert self.controller.toggle_enabled() is True
        self.controller.move_pointer((0.8, 0.8))
        assert self.mouse.calls[-1] == ("move", 800, 640)


class TestCooldownActions:
    def test_scroll_cooldown(self):
        mouse = FakeMouse()
        c = controller(mouse, clock=lambda: 0.0)
        assert c.scroll(1, now=0.0) is True
        assert c.scroll(1, now=0.1) is False
        assert c.scroll(-1, now=0.3) is True
        assert mouse.calls == [("scroll", 1), ("scroll", -1)]

    def test_volume_cooldown_and_clamp(self):
        mouse = FakeMouse()
        volume = FakeVolume(level=0.98)
        c = controller(mouse, volume, clock=lambda: 0.0)
        assert c.adjust_volume(0.04, now=0.0) is True
        assert volume.calls == [1.0]
        assert c.adjust_volume(0.04, now=0.1) is False
        assert c.adjust_volume(-0.04, now=0.3) is True
        assert volume.calls == [1.0, 0.96]

    def test_process_mapping_dispatches_scroll_and_volume(self):
        mouse = FakeMouse()
        volume = FakeVolume()
        c = controller(mouse, volume)
        assert c.process(frame(Gesture.TWO_UP), now=0.0) is None
        assert c.process(frame(Gesture.THUMBS_DOWN), now=0.0) is None
        assert mouse.calls == [("scroll", 300)]
        assert volume.calls == [0.46]

    def test_disabled_controller_blocks_scroll_and_volume(self):
        mouse = FakeMouse()
        volume = FakeVolume()
        c = controller(mouse, volume)
        c.set_enabled(False)
        assert c.scroll(1, now=0.0) is False
        assert c.adjust_volume(0.1, now=0.0) is False
        assert mouse.calls == []
        assert volume.calls == []
