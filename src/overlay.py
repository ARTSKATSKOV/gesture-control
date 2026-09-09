"""OpenCV overlay helpers for the live gesture-control window."""

from dataclasses import dataclass

import cv2
import numpy as np

from .hand_tracker import Hand, draw_hand
from .config import CursorConfig


@dataclass(frozen=True)
class OverlayState:
    gesture: str = "NO_HAND"
    fps: float = 0.0
    enabled: bool = True
    handedness: str = ""
    pinch_ratio: float | None = None
    interaction: str = ""


def draw_overlay(frame: np.ndarray, state: OverlayState, hand: Hand | None = None,
                 cursor: CursorConfig | None = None) -> np.ndarray:
    """Draw hand landmarks and runtime status onto a BGR frame in place."""
    if hand is not None:
        draw_hand(frame, hand)
    if cursor is not None:
        height, width = frame.shape[:2]
        half_x = (0.5 - cursor.edge_margin) / cursor.sensitivity
        half_y = (0.5 - cursor.edge_margin) / cursor.vertical_sensitivity
        left = round(max(0, 0.5 - half_x) * (width - 1))
        right = round(min(1, 0.5 + half_x) * (width - 1))
        top = round(max(0, 0.5 - half_y) * (height - 1))
        bottom = round(min(cursor.vertical_bottom, 0.5 + half_y) * (height - 1))
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 220, 0), 1)
        cv2.drawMarker(frame, (round((width - 1) / 2), round((height - 1) / 2)),
                       (255, 220, 0), cv2.MARKER_CROSS, 12, 1)

    status = "Enabled" if state.enabled else "Disabled"
    status_color = (0, 200, 0) if state.enabled else (0, 0, 255)
    lines = [
        (f"Gesture: {state.gesture}", (255, 255, 255)),
        (f"Tracking FPS: {state.fps:.1f}", (255, 255, 255)),
        (f"Status: {status}", status_color),
    ]
    if state.handedness:
        lines.append((f"Hand: {state.handedness}", (255, 255, 255)))
    if state.pinch_ratio is not None:
        lines.append((f"Pinch ratio: {state.pinch_ratio:.2f}", (255, 255, 255)))
    if state.interaction:
        lines.append((state.interaction, (0, 220, 255)))

    x, y = 12, 28
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 30
    return frame
