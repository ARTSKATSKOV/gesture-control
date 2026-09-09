# Gesture control: UX code review — 2026-09-10

Reviewed the camera loop, hand tracking, geometric classification, temporal
stabilization, mouse/audio backends, mapping, configuration, overlay, and tests.
Findings below are based on local source inspection and reproducible synthetic
tests. Actual hand recognition, camera FPS, and subjective aiming still need a
live session with the user. No camera footage was captured or uploaded.

## High-impact findings fixed

| Priority | Trigger and user-visible problem | Evidence and correction |
| --- | --- | --- |
| P1 | Moving the pointer stalls the tracking loop. | The installed PyAutoGUI library sets `PAUSE = 0.1`; its decorated calls sleep after each operation by default. The mouse backend now passes `_pause=False` to each action, preserving fail-safe checks. |
| P1 | Cursor looks like low-FPS motion even when the camera is acceptable. | Previously only one OS movement was emitted per inference frame, with another camera-rate smoothing stage. `CursorMotion` now interpolates toward the latest target using elapsed time on a separate worker. Its requested cadence is 120 Hz; scheduling and hardware determine actual cadence. The live controller bypasses the previous second smoothing stage. |
| P1 | Pinching moves the pointer downward. | The old loop could replace a new pose with the previous stabilized `POINT`; raw pinch flicker and folding of the last extended finger reopened movement. Current geometry now wins over stale labels, approaching pinch freezes output, established pinch survives index folding, and normal pointing stays blocked for 180 ms after the last pinch evidence/event. Freezing cancels pending interpolated movement under the same lock as output. |
| P1 | Pinch release or dragging follows a bending fingertip. | Drag uses the index MCP (knuckle), anchored to the actual system cursor for the live backend. A raw release freezes drag movement while release confirmation completes. The relative target is clamped to the screen. |
| P1 | Bottom of screen requires moving the hand out of view. | Former bottom threshold was approximately 72% of camera height. The lower half now reaches the bottom at 60%, while the center remains 50% and horizontal gain remains 1.4. The overlay shows the usable rectangle and center. Downward gain is deliberately higher than upward gain. |
| P1 | Camera read failure can leave a button held and ignore Esc. | Failed reads now cancel pinch state, release drag, clear gesture history, and process Esc. Repeated failures terminate with an error. Nested cleanup releases camera/window resources even if controller or tracker cleanup fails. The motion worker expires stale movement targets after 200 ms. |
| P2 | A scroll gesture jumps many pages. | `wheel_step=100` was sent directly as 100 wheel clicks, not 100 pixels. Default is now 3 wheel clicks. |
| P2 | Stale scroll/volume action fires after pose changes, including during pinch. | Non-cursor actions require agreement between the current and confirmed gesture. Pinch and its release guard suppress these actions. |
| P2 | Normal screen-corner targeting activates the emergency stop. | Generated movement avoids the exact fail-safe corner by one horizontal pixel. Moving the physical mouse into a corner still triggers PyAutoGUI's fail-safe. Button release is allowed during fail-safe cleanup. |
| P2 | Non-finite JSON numbers can propagate into coordinates. | Configuration loading rejects NaN and infinity for numeric float fields. |

## Verification

- Regression tests exercise pinch approach, a transient POINT dropout, click
  release, fingertip folding, stale action labels, knuckle-based dragging,
  screen clamping, output freeze/expiry, output failure propagation, and bottom
  reachability with a preserved center.
- A mocked full camera loop checks drag release on capture failure and verifies
  that inference uses 256x192 frames while preview remains 640x480.
- Deterministic output tests feed 15 camera frames per second and verify multiple
  intermediate movements, equivalent filter response at 60/120 output ticks per
  second, and replacement of old targets instead of a growing queue.
- A short real-thread scheduling experiment on this Windows host, using a fake
  mouse backend, produced 52 movements in 0.805 seconds (about 64.6 updates/s)
  from a simulated 15 FPS input. This measures worker scheduling, not camera,
  MediaPipe performance, or observed physical-cursor smoothness.
- Full pytest, bytecode compilation, and whitespace checks run before handoff.

## Remaining findings and practical limits

1. **P2 — Hotkeys depend on preview focus.** `cv2.waitKey` is not a global Windows
   hotkey listener. After clicking another application, Space/Esc may require
   returning focus to the preview. Global hotkeys need a separate keyboard
   listener and a deliberate policy for Space while typing. The physical-mouse
   corner fail-safe remains available when an output action is attempted.
2. **P2 — Blocking camera drivers.** A failed `read()` is handled, but a driver
   call that never returns still blocks the main loop. The worker expires cursor
   motion, but cannot guarantee releasing an already-held drag button in that
   situation. Isolated capture with a watchdog is a further architectural step.
3. **P2 — Pinch timing still depends partly on FPS.** Click/drag timing uses
   elapsed time, but pinch confirmation remains three consecutive frames. A
   very short tap can be missed at low FPS. A held pinch intentionally becomes
   drag after the configured 400 ms following confirmation; the overlay now
   explicitly shows DRAG. This behavior is distinct from unintentional POINT
   motion during a pinch.
4. **P2 — Hardware and personal calibration.** The new 60% bottom boundary should
   allow more of the palm to stay visible, but it is not personalized. High
   downward gain also magnifies tracking noise. Test with the user's normal
   posture and camera angle; adjust `cursor.vertical_bottom` if necessary.
   Finger extension remains a wrist-distance heuristic and can misclassify
   strongly rotated/occluded hands. Smaller input frames may reduce precision;
   reducing resolution further is not justified without measurements.
5. **P2 — Single screen and concurrent launches.** Cursor bounds come from
   PyAutoGUI's screen size; this is not a calibrated multi-monitor desktop.
   There is no single-instance lock, so launching a second instance can create
   conflicting camera/mouse control. Stop the previous process before restarting.
6. **P2 — Audio endpoint failure.** Lazy audio initialization avoids startup
   failures, but an unavailable endpoint during a volume gesture can still
   terminate the application. Handling this as a disabled volume feature with a
   visible error would improve resilience.

## Live acceptance check

Restart the application. Move the index tip from the preview center to the
rectangle's bottom edge and check the taskbar is reachable with the whole hand
still visible. Hold the hand steady, pinch/release several times, then deliberately
hold long enough for DRAG and move the hand. Confirm that finger curling alone
does not move the cursor. Finally test hand loss and disabling during a drag.
The displayed Tracking FPS describes the camera/control loop, not output-worker Hz.

Higgsfield's existing connection was checked successfully with a read-only
account query. It is not used for live cursor tracking or this code review.
