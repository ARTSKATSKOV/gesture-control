# Gesture Control

Camera-based hand gestures for controlling the Windows cursor: move, pinch-click, drag, scroll, and adjust volume.

## Status

The control pipeline is implemented and covered by automated tests (173 passing on 2026-09-24). A documented live acceptance run with a webcam and the actual Windows cursor is still pending. See [the live checklist](docs/acceptance/CHECKLIST-live.md) for what remains to be verified before calling the application ready for everyday use.

## Run on Windows

Requires Python 3.14 and a webcam. The MediaPipe hand-landmarker model is included in `models/hand_landmarker.task`.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/main.py
```

Run the automated checks with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The preview window has to be focused for its Space and Esc shortcuts. The physical mouse corner remains the PyAutoGUI fail-safe. Current limitations and the live test procedure are described in [UX_REVIEW.md](UX_REVIEW.md) and the [checklist](docs/acceptance/CHECKLIST-live.md).
