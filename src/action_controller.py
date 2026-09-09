"""Windows side effects for cursor, mouse, scroll, and volume actions."""

from __future__ import annotations

import time
from math import hypot
from pathlib import Path
from typing import Callable, Protocol

import pyautogui

from .config import CursorConfig, GestureMappingsConfig, ScrollConfig, VolumeConfig
from .gesture_classifier import FrameGesture, Gesture
from .stabilizer import PinchEvent


class MouseBackend(Protocol):
    def move_to(self, x: int, y: int) -> None: ...
    def click(self) -> None: ...
    def mouse_down(self) -> None: ...
    def mouse_up(self) -> None: ...
    def scroll(self, amount: int) -> None: ...
    def screen_size(self) -> tuple[int, int]: ...


class VolumeBackend(Protocol):
    def get_level(self) -> float: ...
    def set_level(self, level: float) -> None: ...


class PyAutoGuiBackend:
    def move_to(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y, duration=0)

    def click(self) -> None:
        pyautogui.click()

    def mouse_down(self) -> None:
        pyautogui.mouseDown(button="left")

    def mouse_up(self) -> None:
        pyautogui.mouseUp(button="left")

    def scroll(self, amount: int) -> None:
        pyautogui.scroll(amount)

    def screen_size(self) -> tuple[int, int]:
        size = pyautogui.size()
        return size.width, size.height


class PycawVolumeBackend:
    def __init__(self) -> None:
        from pycaw.pycaw import AudioUtilities

        self._endpoint = AudioUtilities.GetSpeakers().EndpointVolume

    def get_level(self) -> float:
        return float(self._endpoint.GetMasterVolumeLevelScalar())

    def set_level(self, level: float) -> None:
        self._endpoint.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level)), None)


class CursorSmoother:
    def __init__(
        self,
        config: CursorConfig,
        screen_size: tuple[int, int],
    ) -> None:
        self._alpha = config.smoothing_alpha
        self._dead_zone = config.dead_zone
        self._width, self._height = screen_size
        self._position: tuple[float, float] | None = None

    @property
    def position(self) -> tuple[int, int] | None:
        if self._position is None:
            return None
        return round(self._position[0]), round(self._position[1])

    def update(self, normalized: tuple[float, float]) -> tuple[int, int]:
        x = max(0.0, min(1.0, float(normalized[0]))) * (self._width - 1)
        y = max(0.0, min(1.0, float(normalized[1]))) * (self._height - 1)
        target = (x, y)
        if self._position is None:
            self._position = target
        elif hypot(target[0] - self._position[0], target[1] - self._position[1]) >= self._dead_zone:
            self._position = (
                self._alpha * target[0] + (1.0 - self._alpha) * self._position[0],
                self._alpha * target[1] + (1.0 - self._alpha) * self._position[1],
            )
        return self.position

    def reset(self) -> None:
        self._position = None


class ActionController:
    def __init__(
        self,
        cursor: CursorConfig,
        mappings: GestureMappingsConfig,
        scroll: ScrollConfig,
        volume: VolumeConfig,
        mouse: MouseBackend | None = None,
        volume_backend: VolumeBackend | None = None,
        screen_size: tuple[int, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mouse = mouse or PyAutoGuiBackend()
        self._volume = volume_backend or PycawVolumeBackend()
        size = screen_size or self._mouse.screen_size()
        self._smoother = CursorSmoother(cursor, size)
        self._mappings = mappings
        self._scroll = scroll
        self._volume_config = volume
        self._clock = clock
        self._last_scroll_at = float("-inf")
        self._last_volume_at = float("-inf")
        self._enabled = True
        self._drag_active = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def drag_active(self) -> bool:
        return self._drag_active

    def set_enabled(self, enabled: bool) -> None:
        if not enabled and self._drag_active:
            self.release_drag()
        self._enabled = enabled
        if not enabled:
            self._smoother.reset()

    def toggle_enabled(self) -> bool:
        self.set_enabled(not self._enabled)
        return self._enabled

    def move_pointer(self, pointer: tuple[float, float] | None) -> tuple[int, int] | None:
        if not self._enabled or pointer is None:
            return None
        position = self._smoother.update(pointer)
        self._mouse.move_to(*position)
        return position

    def click(self) -> bool:
        if not self._enabled:
            return False
        self._mouse.click()
        return True

    def start_drag(self) -> bool:
        if not self._enabled or self._drag_active:
            return False
        self._mouse.mouse_down()
        self._drag_active = True
        return True

    def release_drag(self) -> bool:
        if not self._drag_active:
            return False
        self._mouse.mouse_up()
        self._drag_active = False
        return True

    def scroll(self, amount: int, now: float | None = None) -> bool:
        if not self._enabled or not self._scroll.enabled:
            return False
        timestamp = self._clock() if now is None else now
        if timestamp - self._last_scroll_at < self._scroll.cooldown_ms / 1000.0:
            return False
        self._mouse.scroll(amount)
        self._last_scroll_at = timestamp
        return True

    def adjust_volume(self, delta: float, now: float | None = None) -> bool:
        if not self._enabled or not self._volume_config.enabled:
            return False
        timestamp = self._clock() if now is None else now
        if timestamp - self._last_volume_at < self._volume_config.cooldown_ms / 1000.0:
            return False
        level = self._volume.get_level()
        self._volume.set_level(max(0.0, min(1.0, level + delta)))
        self._last_volume_at = timestamp
        return True

    def process(
        self,
        frame: FrameGesture | None,
        pinch_event: PinchEvent = PinchEvent.NONE,
        now: float | None = None,
    ) -> None:
        if pinch_event is PinchEvent.CLICK:
            self.click()
        elif pinch_event is PinchEvent.DRAG_START:
            self.start_drag()
        elif pinch_event in (PinchEvent.DRAG_END, PinchEvent.DRAG_END_FORCED):
            self.release_drag()

        if frame is None:
            return
        cursor_gesture = Gesture(self._mappings.cursor)
        pinch_gesture = Gesture(self._mappings.pinch)
        if frame.gesture in (cursor_gesture, pinch_gesture) or self._drag_active:
            self.move_pointer(frame.pointer)

        gesture = frame.gesture
        if gesture is Gesture(self._mappings.scroll_up):
            self.scroll(self._scroll.wheel_step, now)
        elif gesture is Gesture(self._mappings.scroll_down):
            self.scroll(-self._scroll.wheel_step, now)
        elif gesture is Gesture(self._mappings.volume_up):
            self.adjust_volume(self._volume_config.step, now)
        elif gesture is Gesture(self._mappings.volume_down):
            self.adjust_volume(-self._volume_config.step, now)

    def shutdown(self) -> None:
        self.release_drag()
        self._smoother.reset()
