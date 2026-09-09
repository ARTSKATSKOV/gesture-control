"""Camera backend selection tests."""

from unittest.mock import patch

import pytest

from src.config import CameraConfig
from src.main import open_camera


class FakeCapture:
    def __init__(self, index, backend, opened):
        self.index = index
        self.backend = backend
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def release(self):
        self.released = True


class TestOpenCamera:
    def test_dshow_is_selected_for_windows_default(self):
        capture = FakeCapture(0, 700, True)
        with patch("src.main.cv2.VideoCapture", return_value=capture) as factory:
            result = open_camera(CameraConfig(backend="dshow"))
        assert result is capture
        factory.assert_called_once_with(0, 700)

    def test_auto_falls_back_to_msmf(self):
        first = FakeCapture(0, 700, False)
        second = FakeCapture(0, 800, True)
        with patch("src.main.cv2.CAP_DSHOW", 700), patch(
            "src.main.cv2.CAP_MSMF", 800
        ), patch("src.main.cv2.VideoCapture", side_effect=[first, second]):
            result = open_camera(CameraConfig(backend="auto"))
        assert result is second
        assert first.released

    def test_raises_when_backend_cannot_open(self):
        capture = FakeCapture(0, 700, False)
        with patch("src.main.cv2.VideoCapture", return_value=capture):
            with pytest.raises(RuntimeError, match="cannot open camera"):
                open_camera(CameraConfig(backend="dshow"))
        assert capture.released
