"""Time-based cursor output, independent of camera/inference frame rate."""

from __future__ import annotations

import math
import threading
import time


class CursorMotion:
    """Interpolate toward the latest target; never queue old camera frames.

    All output and freezing share a lock so a pinch cannot race a queued move.
    A stale target expires after 200 ms, even if camera capture gets stuck.
    """

    def __init__(self, move, position, alpha=0.3, dead_zone=3.0, clock=time.monotonic):
        self._move = move
        self._read_position = position
        self._clock = clock
        self._tau = 0.0 if alpha == 1 else -1 / (60 * math.log1p(-alpha))
        self._dead_zone = dead_zone
        self._speed_distance = 120.0
        self._max_speedup = 8.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._target = None
        self._position = None
        self._sent = None
        self._updated = 0.0
        self._last_tick = None
        self._error = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="cursor-output", daemon=True)
        self._thread.start()

    def check(self):
        if self._error is not None:
            raise self._error

    def target(self, x, y):
        with self._lock:
            self.check()
            if self._target is None:
                self._position = tuple(self._read_position())
                self._sent = tuple(round(v) for v in self._position)
                self._last_tick = self._clock()
            target = (x, y)
            if self._target is None or math.dist(target, self._target) >= self._dead_zone:
                self._target = target
            self._updated = self._clock()

    def freeze(self):
        with self._lock:
            self._target = None
            self._last_tick = None

    def tick(self, now=None):
        """One deterministic output step, also used by the worker and tests."""
        with self._lock:
            now = self._clock() if now is None else now
            if self._target is None:
                return
            if now - self._updated >= 0.2:
                self.freeze()
                return
            dt = max(0.0, now - self._last_tick)
            self._last_tick = now
            distance = math.dist(self._position, self._target)
            speedup = min(
                self._max_speedup,
                1.0 + distance / self._speed_distance,
            )
            effective_tau = self._tau / speedup if self._tau else 0.0
            gain = 1.0 if effective_tau == 0 else -math.expm1(-dt / effective_tau)
            self._position = tuple(
                p + gain * (t - p) for p, t in zip(self._position, self._target)
            )
            if math.dist(self._position, self._target) < 0.5:
                self._position = self._target
            output = tuple(round(v) for v in self._position)
            if output != self._sent:
                self._move(*output)
                self._sent = output

    def _run(self):
        try:
            while not self._stop.wait(1 / 120):
                self.tick()
        except Exception as exc:
            self._error = exc
            self.freeze()

    def close(self):
        self._stop.set()
        self.freeze()
        if self._thread is not None:
            self._thread.join(timeout=1)
