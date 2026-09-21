"""Closing the preview window is a real, intended exit — not a loop trap.

The loop is driven by ``waitKey``; once the user closes the window with X or
Alt+F4 that call reports no key at all, so the window state itself has to end
the loop. These tests use doubles only: no camera, no real HighGUI window and
no cursor movement.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import AppConfig
from src.main import run

ESC = 27
NO_KEY = 255
FRAME_LIMIT = 4


def window_values(*values: float):
    """Report ``values`` in order, then keep repeating the last one."""
    queue = deque(values)

    def value(*_args, **_kwargs) -> float:
        return queue.popleft() if len(queue) > 1 else queue[0]

    return value


@dataclass
class Doubles:
    camera: MagicMock
    tracker: MagicMock
    controller: MagicMock
    imshow: MagicMock
    destroy: MagicMock
    window_property: MagicMock
    reads: list = field(default_factory=list)

    def assert_clean_shutdown(self) -> None:
        self.controller.shutdown.assert_called_once()
        self.tracker.close.assert_called_once()
        self.camera.release.assert_called_once()
        self.destroy.assert_called_once()


@contextmanager
def running_loop(*, reads, window, keys=NO_KEY):
    """Run-patch every device ``run()`` touches.

    ``reads`` is a list of ``(ok, frame)`` results; running off its end raises
    instead of spinning forever, so a loop that fails to exit fails the test
    rather than hanging it.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    doubles = Doubles(
        camera=MagicMock(),
        tracker=MagicMock(),
        controller=MagicMock(),
        imshow=None,
        destroy=None,
        window_property=None,
    )
    doubles.tracker.process.return_value = None
    doubles.controller.drag_active = False

    def read():
        doubles.reads.append(1)
        assert len(doubles.reads) <= len(reads), "run() kept looping instead of exiting"
        return reads[len(doubles.reads) - 1]

    doubles.camera.read.side_effect = read
    with patch("src.main.open_camera", return_value=doubles.camera), \
         patch("src.main.new_hand_tracker", return_value=doubles.tracker), \
         patch("src.main.ActionController", return_value=doubles.controller), \
         patch("src.main.draw_overlay"), \
         patch("src.main.cv2.imshow") as imshow, \
         patch("src.main.cv2.destroyAllWindows") as destroy, \
         patch("src.main.cv2.getWindowProperty", side_effect=window) as window_property, \
         patch("src.main.cv2.waitKey", return_value=keys):
        doubles.imshow = imshow
        doubles.destroy = destroy
        doubles.window_property = window_property
        yield doubles


def frames(count: int) -> list[tuple[bool, np.ndarray]]:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return [(True, image) for _ in range(count)]


def test_run_returns_when_the_preview_window_is_closed():
    with running_loop(reads=frames(FRAME_LIMIT), window=window_values(0.0)) as doubles:
        run(AppConfig.defaults())
    assert doubles.reads == [1], "the loop must leave after the window is gone"
    assert doubles.imshow.call_count == 1, "the closed window must not be recreated"
    doubles.assert_clean_shutdown()


def test_run_still_exits_on_escape_and_still_watches_the_window():
    with running_loop(reads=frames(FRAME_LIMIT), window=window_values(1.0), keys=ESC) as doubles:
        run(AppConfig.defaults())
    assert doubles.reads == [1]
    assert doubles.imshow.call_count == 1
    assert doubles.window_property.called, "Esc must exit through the window-aware loop"
    doubles.controller.toggle_enabled.assert_not_called()
    doubles.assert_clean_shutdown()


def test_closing_the_window_while_the_camera_fails_stops_the_loop():
    reads = frames(1) + [(False, None)] * FRAME_LIMIT
    with running_loop(reads=reads, window=window_values(1.0, 0.0)) as doubles:
        run(AppConfig.defaults())
    assert doubles.reads == [1, 1], "the closed window must end the retry path too"
    assert doubles.imshow.call_count == 1
    doubles.assert_clean_shutdown()


def test_hidden_window_before_the_first_frame_is_not_treated_as_closed():
    with running_loop(reads=[(False, None)] * 30, window=window_values(0.0)) as doubles:
        with pytest.raises(RuntimeError, match="Camera stopped delivering frames"):
            run(AppConfig.defaults())
    assert doubles.imshow.call_count == 0
    doubles.camera.release.assert_called_once()
    doubles.tracker.close.assert_called_once()
