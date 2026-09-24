# Gesture Control

Camera-based hand gestures for controlling the Windows cursor: move, pinch-click, drag, scroll, and adjust volume.

## Status

The control pipeline is implemented and covered by automated tests (216 passing on 2026-09-24, synthetic — no camera). The first live acceptance run with a webcam and the real Windows cursor happened on 2026-09-24; its findings are closed and its defects fixed in [the live checklist](docs/acceptance/CHECKLIST-live.md), which also lists what the next run has to re-verify. Two things came out of that run: cursor reach is now configured per axis (`cursor.overscan` horizontally, `cursor.overscan_y` vertically, because the palm hangs below the pointing fingertip) and a new `tracking` section gates which frames may act at all (hand plausibility, frame edges, motion blur).

## Run on Windows

Requires Python 3.14 and a webcam. The MediaPipe hand-landmarker model is included in `models/hand_landmarker.task`.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.main
```

Run the automated checks with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The preview window has to be focused for its Space and Esc shortcuts. The physical mouse corner remains the PyAutoGUI fail-safe. Current limitations and the live test procedure are described in [UX_REVIEW.md](UX_REVIEW.md) and the [checklist](docs/acceptance/CHECKLIST-live.md).
