"""Standalone fullscreen viewer process.

Launched by FullscreenOutput.open() via subprocess.Popen — NOT via the
multiprocessing module, because that would re-import main.py at child
startup and trigger the entire DPG/CUDA/audio stack to spin up a second
time. This file is the entry point, so it stays self-contained.

Args (positional): monitor_index src_w src_h shm_name

The shared memory block holds a (H, W, 3) uint8 RGB frame the parent
rewrites every render. We poll at ~60fps and always render the latest
contents — no event/signal IPC, no risk of deadlock if the parent dies.
"""

import signal
import sys

import numpy as np
import pygame
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory


def main() -> int:
    if len(sys.argv) != 5:
        print(f"usage: {sys.argv[0]} <monitor_index> <src_w> <src_h> <shm_name>", file=sys.stderr)
        return 2

    monitor_index = int(sys.argv[1])
    sw = int(sys.argv[2])
    sh = int(sys.argv[3])
    shm_name = sys.argv[4]

    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    if not sizes:
        print("[fullscreen-viewer] no displays found", file=sys.stderr)
        return 1
    monitor_index = max(0, min(monitor_index, len(sizes) - 1))
    mon_size = sizes[monitor_index]

    try:
        surface = pygame.display.set_mode(
            mon_size, pygame.FULLSCREEN, display=monitor_index
        )
    except Exception as e:
        print(f"[fullscreen-viewer] set_mode failed: {e}", file=sys.stderr)
        return 1

    pygame.display.set_caption("113.RecursiveAI Output")
    pygame.mouse.set_visible(False)

    # Aspect-preserving fit. Square AI render → letterboxed onto wider monitor.
    dw, dh = mon_size
    scale = min(dw / sw, dh / sh)
    out_w = int(sw * scale)
    out_h = int(sh * scale)
    x = (dw - out_w) // 2
    y = (dh - out_h) // 2

    shm = SharedMemory(name=shm_name)
    # Tell THIS process's resource_tracker to ignore the SHM. The parent
    # created it and owns its lifecycle (unlink); without this, the child's
    # tracker also registers it and either (a) leaks at shutdown or (b)
    # warns "No such file or directory" when it tries to clean up after the
    # parent already unlinked. Python bpo-38119.
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    # SIGTERM (sent by parent's subprocess.terminate()) should run the
    # finally block so pygame and SHM close cleanly. Without a handler,
    # Python kills the interpreter outright and leaves resources dangling.
    def _on_sigterm(_sig, _frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        frame_view = np.ndarray((sh, sw, 3), dtype=np.uint8, buffer=shm.buf)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            # Snapshot so a parent write mid-blit doesn't tear the on-screen
            # frame. The copy is ~450KB at 384² — sub-millisecond.
            snapshot = frame_view.copy()
            src = pygame.image.frombuffer(snapshot.tobytes(), (sw, sh), "RGB")
            if (sw, sh) != (out_w, out_h):
                src = pygame.transform.scale(src, (out_w, out_h))
            surface.fill((0, 0, 0))
            surface.blit(src, (x, y))
            pygame.display.flip()
            clock.tick(60)
    finally:
        try:
            pygame.mouse.set_visible(True)
            pygame.display.quit()
        except Exception:
            pass
        try:
            shm.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
