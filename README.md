# Gesture Control (185)

Camera-based hand-gesture control of the Windows cursor — move, pinch-click, drag, scroll.

- **Repo:** `C:\Users\awesa\jarvis\Projects\gesture-control` (canonical)
- **Stack:** Python — OpenCV, MediaPipe, PyAutoGUI, Pillow, NumPy; pytest
- **Tests:** 169 passing (`\.venv/Scripts/python.exe -m pytest`)
- **Run:** `\.venv/Scripts/python.exe src/main.py`

## Engineering Memory

Architecture, decisions, progress and risks live in the Obsidian vault:

`C:\Users\awesa\jarvis\Obsidian Vault\Raw\Engineering Memory\Projects\185 - Gesture Controlled Interface\`

## Status

Core pipeline implemented and covered by tests. **Not yet validated against real hardware** — no live webcam frame, MediaPipe run or PyAutoGUI side effect has happened. See `UX_REVIEW.md` and `docs/acceptance/CHECKLIST-live.md` for the live acceptance run.
