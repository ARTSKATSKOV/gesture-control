"""Windows side effects for cursor, mouse, scroll, and volume actions."""

from __future__ import annotations

import time
from dataclasses import replace
from math import hypot
from typing import Callable, Protocol

import pyautogui

from .config import CursorConfig, GestureMappingsConfig, ScrollConfig, VolumeConfig
from .gesture_classifier import FrameGesture, Gesture
from .cursor_motion import CursorMotion
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
    def __init__(self, cursor: CursorConfig):
        self.motion = CursorMotion(
            self._send_position,
            pyautogui.position,
            cursor.smoothing_alpha,
            cursor.dead_zone,
        )

    @staticmethod
    def _send_position(x, y):
        # Generated motion must not park on PyAutoGUI's emergency-stop corners.
        # A physical mouse moved into a corner still activates its fail-safe.
        if (x, y) in pyautogui.FAILSAFE_POINTS:
            width, _ = pyautogui.size()
            x = 1 if x == 0 else width - 2
        pyautogui.moveTo(x, y, duration=0, _pause=False)

    def move_to(self, x: int, y: int) -> None:
        self.motion.start()
        self.motion.target(x, y)

    def freeze(self):
        self.motion.freeze()

    def position(self):
        return tuple(pyautogui.position())

    def check(self):
        self.motion.check()

    def close(self):
        self.motion.close()

    def click(self) -> None:
        self.freeze()
        pyautogui.click(_pause=False)

    def mouse_down(self) -> None:
        self.freeze()
        pyautogui.mouseDown(button="left", _pause=False)

    def mouse_up(self) -> None:
        self.freeze()
        try:
            pyautogui.mouseUp(button="left", _pause=False)
        except pyautogui.FailSafeException:
            # Releasing a held button must still work after the corner fail-safe.
            previous = pyautogui.FAILSAFE
            try:
                pyautogui.FAILSAFE = False
                pyautogui.mouseUp(button="left", _pause=False)
            finally:
                pyautogui.FAILSAFE = previous

    def scroll(self, amount: int) -> None:
        pyautogui.scroll(amount, _pause=False)

    def screen_size(self) -> tuple[int, int]:
        size = pyautogui.size()
        return size.width, size.height


class PycawVolumeBackend:
    """pycaw volume with lazy endpoint acquisition.

    The endpoint is resolved on first use so that starting the app does not
    fail (or touch system audio) on machines where no endpoint is available.
    """

    def __init__(self) -> None:
        self._endpoint = None

    def _ensure_endpoint(self):
        if self._endpoint is None:
            from pycaw.pycaw import AudioUtilities

            self._endpoint = AudioUtilities.GetSpeakers().EndpointVolume
        return self._endpoint

    def get_level(self) -> float:
        return float(self._ensure_endpoint().GetMasterVolumeLevelScalar())

    def set_level(self, level: float) -> None:
        self._ensure_endpoint().SetMasterVolumeLevelScalar(
            max(0.0, min(1.0, level)), None
        )


def normalized_to_screen(
    normalized: tuple[float, float],
    width: int,
    height: int,
    sensitivity: float,
    edge_margin: float,
    vertical_sensitivity: float | None = None,
    vertical_bottom: float = 1.0,
) -> tuple[float, float]:
    """Map normalized camera coordinates to screen-space float coordinates.

    The tracking area shrinks to the ``edge_margin`` band, then expands around
    the centre by the horizontal and vertical sensitivity values (clamped at
    screen edges). A separate helper keeps ordinary pointer control and the
    drag anchor math consistent.
    """
    x = max(0.0, min(1.0, float(normalized[0])))
    y = max(0.0, min(1.0, float(normalized[1])))
    raw_y = y
    if edge_margin > 0.0:
        x = (x - edge_margin) / (1.0 - 2.0 * edge_margin)
        y = (y - edge_margin) / (1.0 - 2.0 * edge_margin)
    sensitivity_y = sensitivity if vertical_sensitivity is None else vertical_sensitivity
    x = max(0.0, min(1.0, 0.5 + (x - 0.5) * sensitivity))
    y = max(0.0, min(1.0, 0.5 + (y - 0.5) * sensitivity_y))
    if raw_y >= 0.5:
        y = max(y, min(1.0, 0.5 + 0.5 * (raw_y - 0.5) / (vertical_bottom - 0.5)))
    return x * (width - 1), y * (height - 1)


class CursorSmoother:
    def __init__(
        self,
        config: CursorConfig,
        screen_size: tuple[int, int],
    ) -> None:
        self._alpha = config.smoothing_alpha
        self._dead_zone = config.dead_zone
        self._width, self._height = screen_size
        self._sensitivity = config.sensitivity
        self._vertical_sensitivity = config.vertical_sensitivity
        self._edge_margin = config.edge_margin
        self._vertical_bottom = config.vertical_bottom
        self._position: tuple[float, float] | None = None

    @property
    def position(self) -> tuple[int, int] | None:
        if self._position is None:
            return None
        return round(self._position[0]), round(self._position[1])

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._width, self._height

    def update(self, normalized: tuple[float, float]) -> tuple[int, int]:
        target = normalized_to_screen(
            normalized,
            self._width,
            self._height,
            self._sensitivity,
            self._edge_margin,
            self._vertical_sensitivity,
            self._vertical_bottom,
        )
        if self._position is None:
            self._position = target
        elif (
            hypot(target[0] - self._position[0], target[1] - self._position[1])
            >= self._dead_zone
        ):
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
        self._mouse = mouse or PyAutoGuiBackend(cursor)
        self._volume = volume_backend or PycawVolumeBackend()
        size = screen_size or self._mouse.screen_size()
        # The live backend smooths at 120 Hz; avoid a second camera-rate filter.
        filter_config = replace(cursor, smoothing_alpha=1.0, dead_zone=0.0) if mouse is None else cursor
        self._smoother = CursorSmoother(filter_config, size)
        self._width, self._height = size
        self._alpha = filter_config.smoothing_alpha
        self._dead_zone = cursor.dead_zone
        self._drag_gain_x = cursor.sensitivity
        self._drag_gain_y = cursor.vertical_sensitivity
        self._mappings = mappings
        self._scroll = scroll
        self._volume_config = volume
        self._clock = clock
        self._last_scroll_at = float("-inf")
        self._last_volume_at = float("-inf")
        self._enabled = True
        self._drag_active = False
        self._drag_anchor_screen: tuple[int, int] | None = None
        self._drag_anchor_norm: tuple[float, float] | None = None
        self._drag_smoothed: tuple[float, float] | None = None
        self._drag_last_sent: tuple[int, int] | None = None
        self._pinch_guard_ratio = cursor.pinch_guard_ratio
        self._pinch_release_seconds = cursor.pinch_release_ms / 1000
        self._cursor_blocked_until = float("-inf")

    def freeze_pointer(self):
        freeze = getattr(self._mouse, "freeze", None)
        if freeze is not None:
            freeze()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def drag_active(self) -> bool:
        return self._drag_active

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.freeze_pointer()
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

    def _begin_drag(self, frame: FrameGesture | None) -> bool:
        if not self._enabled or self._drag_active:
            return False
        self._mouse.mouse_down()
        self._drag_active = True
        anchor_norm = (frame.drag_pointer or frame.pointer) if frame is not None else None
        anchor_screen = self._smoother.position
        read_position = getattr(self._mouse, "position", None)
        if read_position is not None:
            anchor_screen = read_position()
        if anchor_screen is None and anchor_norm is not None:
            mapped = normalized_to_screen(
                anchor_norm,
                self._width,
                self._height,
                self._drag_gain_x,
                0.0,
                self._drag_gain_y,
            )
            anchor_screen = (round(mapped[0]), round(mapped[1]))
        self._drag_anchor_screen = anchor_screen
        self._drag_anchor_norm = anchor_norm
        self._drag_smoothed = None
        self._drag_last_sent = anchor_screen
        return True

    def _drag_move(self, pointer_norm: tuple[float, float] | None) -> tuple[int, int] | None:
        if not self._enabled or not self._drag_active or pointer_norm is None:
            return None
        anchor_screen = self._drag_anchor_screen
        anchor_norm = self._drag_anchor_norm
        if anchor_screen is None or anchor_norm is None:
            return None
        target = (
            anchor_screen[0]
            + (pointer_norm[0] - anchor_norm[0])
            * (self._width - 1)
            * self._drag_gain_x,
            anchor_screen[1]
            + (pointer_norm[1] - anchor_norm[1])
            * (self._height - 1)
            * self._drag_gain_y,
        )
        target = (max(0, min(self._width - 1, target[0])),
                  max(0, min(self._height - 1, target[1])))
        if self._drag_smoothed is None:
            self._drag_smoothed = (float(target[0]), float(target[1]))
        else:
            prev_x, prev_y = self._drag_smoothed
            self._drag_smoothed = (
                self._alpha * target[0] + (1.0 - self._alpha) * prev_x,
                self._alpha * target[1] + (1.0 - self._alpha) * prev_y,
            )
        pos = (round(self._drag_smoothed[0]), round(self._drag_smoothed[1]))
        last_sent = self._drag_last_sent
        if last_sent is not None:
            if pos == last_sent:
                return last_sent
            if self._dead_zone > 0 and (
                hypot(pos[0] - last_sent[0], pos[1] - last_sent[1]) < self._dead_zone
            ):
                return last_sent
        self._drag_last_sent = pos
        self._mouse.move_to(*pos)
        return pos

    def release_drag(self) -> bool:
        if not self._drag_active:
            return False
        self._mouse.mouse_up()
        self._drag_active = False
        self._drag_anchor_screen = None
        self._drag_anchor_norm = None
        self._drag_smoothed = None
        self._drag_last_sent = None
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
        check = getattr(self._mouse, "check", None)
        if check is not None:
            check()
        timestamp = self._clock() if now is None else now
        if not self._enabled:
            self.freeze_pointer()
            return
        near_pinch = frame is not None and frame.pointer is not None and (
            frame.pinch_active or frame.pinch_ratio <= self._pinch_guard_ratio
        )
        if near_pinch or pinch_event is not PinchEvent.NONE:
            self._cursor_blocked_until = timestamp + self._pinch_release_seconds
        if not self._drag_active and (near_pinch or timestamp < self._cursor_blocked_until):
            self.freeze_pointer()
        if pinch_event is PinchEvent.CLICK:
            self.click()
        elif pinch_event is PinchEvent.DRAG_START:
            self._begin_drag(frame)
        elif pinch_event in (PinchEvent.DRAG_END, PinchEvent.DRAG_END_FORCED):
            self.release_drag()

        if frame is None:
            self.freeze_pointer()
            return

        # Cursor follows the index finger only in the POINT pose, or while an
        # active drag moves relative to its anchor. While a pinch is held
        # (before drag), the cursor stays put so clicking does not jitter.
        cursor_gesture = Gesture(self._mappings.cursor)
        if self._drag_active:
            # Never follow a fingertip reopening while release debounce runs.
            if frame.pinch_active:
                self._drag_move(frame.drag_pointer or frame.pointer)
            else:
                self.freeze_pointer()
            return
        elif (frame.gesture is cursor_gesture and not near_pinch
              and timestamp >= self._cursor_blocked_until):
            self.move_pointer(frame.pointer)
        else:
            self.freeze_pointer()

        if near_pinch or timestamp < self._cursor_blocked_until:
            return

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
        self.freeze_pointer()
        try:
            self.release_drag()
        finally:
            close = getattr(self._mouse, "close", None)
            if close is not None:
                close()
            self._smoother.reset()
