"""Optional fullscreen output window on a chosen monitor.

Architecture:
- The viewer runs in a separate process via subprocess.Popen
  (NOT multiprocessing). multiprocessing's spawn-mode child re-imports
  __main__ to set itself up, which would re-run all of main.py — DPG init,
  model loading, audio — effectively spawning a second instance of the app.
- The viewer is `fullscreen_viewer.py`, a standalone entry-point script.
- Frames cross the process boundary via SharedMemory (one ~450KB block at
  384², rewritten in-place every render tick). The viewer polls at ~60fps
  and always reads the latest contents — no event IPC, no deadlock risk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from multiprocessing.shared_memory import SharedMemory

import numpy as np

_VIEWER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fullscreen_viewer.py"
)


class FullscreenOutput:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._shm: SharedMemory | None = None
        self._src_size: tuple[int, int] | None = None
        self._monitor: int | None = None

    @staticmethod
    def list_monitors() -> list[tuple[int, tuple[int, int]]]:
        """Returns [(index, (w, h))]. Runs pygame in a one-shot subprocess so
        a flaky display setup can't crash the parent. Returns [] on failure."""
        code = (
            "import pygame, json\n"
            "pygame.display.init()\n"
            "print(json.dumps(pygame.display.get_desktop_sizes()))\n"
        )
        try:
            res = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=10,
            )
            if res.returncode != 0:
                return []
            sizes = json.loads(res.stdout.strip().splitlines()[-1])
            return [(i, (int(w), int(h))) for i, (w, h) in enumerate(sizes)]
        except Exception:
            return []

    def is_open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def monitor(self) -> int | None:
        return self._monitor if self.is_open() else None

    def open(self, monitor_index: int, src_size: tuple[int, int]) -> None:
        """src_size is (W, H) of the AI render."""
        if self.is_open():
            self.close()

        sw, sh = src_size
        nbytes = int(sw) * int(sh) * 3
        self._shm = SharedMemory(create=True, size=nbytes)
        # Zero the buffer so the viewer doesn't briefly show whatever garbage
        # the kernel handed us when the page was mapped in.
        np.ndarray((sh, sw, 3), dtype=np.uint8, buffer=self._shm.buf)[:] = 0

        self._proc = subprocess.Popen(
            [sys.executable, _VIEWER_SCRIPT, str(monitor_index), str(sw), str(sh), self._shm.name],
        )
        self._src_size = (sw, sh)
        self._monitor = monitor_index

    def update(self, rgb_uint8: np.ndarray) -> None:
        """rgb_uint8: (H, W, 3) uint8. Called from the main loop every
        rendered frame. Cheap — one ~450KB memcpy. Detects a dead viewer
        (ESC pressed, crash, monitor unplug) and cleans up so is_open()
        reports honestly on the next call."""
        if self._proc is None or self._shm is None or self._src_size is None:
            return
        if self._proc.poll() is not None:
            self.close()
            return
        sw, sh = self._src_size
        if rgb_uint8.shape != (sh, sw, 3):
            return  # unexpected size; drop frame rather than corrupt SHM
        view = np.ndarray((sh, sw, 3), dtype=np.uint8, buffer=self._shm.buf)
        np.copyto(view, rgb_uint8)

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=1.0)
            except Exception:
                pass
            self._proc = None
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
        self._src_size = None
        self._monitor = None
