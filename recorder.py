import os
import signal
import subprocess
import threading
import time


class Recorder:
    """
    Records the AI canvas to MP4 via a single ffmpeg process that ingests
    rawvideo frames from stdin and PulseAudio output simultaneously.

    Sharing one ffmpeg instance means both streams use the same clock, and
    `-use_wallclock_as_timestamps` on the rawvideo input stamps each frame
    with its actual arrival time. ffmpeg then resamples to a clean CFR
    output at the target fps — so audio and video stay locked even when the
    main render loop's framerate jitters.
    """

    def __init__(self, width: int, height: int, fps: int = 30, output_dir: str = "recordings"):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir

        self._proc: subprocess.Popen | None = None
        self._final_path: str | None = None
        self._start_time: float | None = None
        self._is_recording = False
        self._last_message: str = ""
        self._lock = threading.Lock()
        self._active_size: tuple[int, int] = (width, height)

    @property
    def active_size(self) -> tuple[int, int]:
        """Frame dimensions ffmpeg is currently expecting. Used by callers
        to validate they're sending correctly-sized frames after a toggle."""
        return self._active_size

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def last_message(self) -> str:
        return self._last_message

    def elapsed(self) -> float:
        if self._start_time is None or not self._is_recording:
            return 0.0
        return time.time() - self._start_time

    @staticmethod
    def default_sink_monitor() -> str | None:
        """PulseAudio source name that captures whatever's playing on the
        default speaker output (i.e. system audio, not the mic)."""
        try:
            sink = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            return f"{sink}.monitor" if sink else None
        except Exception:
            return None

    def start(
        self,
        include_audio: bool = True,
        audio_source: str = "system",
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """audio_source: 'system' (loopback of speaker output) or 'mic'
        (default input). Falls back to video-only if pulse lookup fails.

        width/height override the init defaults for THIS recording. Lets the
        recorder follow the upscaler's output size when GPU upscale is on,
        without re-instantiating the Recorder. The chosen size sticks for the
        whole take — ffmpeg's rawvideo input can't change resolution mid-stream."""
        if self._is_recording:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._final_path = os.path.join(self.output_dir, f"{timestamp}.mp4")

        active_w = int(width) if width else self.width
        active_h = int(height) if height else self.height
        self._active_size = (active_w, active_h)

        pulse_source: str | None = None
        if include_audio:
            if audio_source == "system":
                pulse_source = self.default_sink_monitor()
            else:
                pulse_source = "default"

        cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]

        if pulse_source:
            cmd += [
                "-thread_queue_size", "1024",
                "-f", "pulse",
                "-i", pulse_source,
            ]

        # Wallclock timestamps + thread_queue let the rawvideo input survive
        # bursty producers without dropping or mis-stamping frames.
        cmd += [
            "-thread_queue_size", "1024",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{active_w}x{active_h}",
            "-framerate", str(self.fps),
            "-use_wallclock_as_timestamps", "1",
            "-i", "-",
        ]

        if pulse_source:
            cmd += ["-map", "0:a", "-map", "1:v"]

        # libx264 instead of NVENC: NVENC fights the SD/TensorRT pipeline for
        # the GPU and intermittently fails with "no capable device". At this
        # resolution x264 veryfast is far under realtime on any CPU and gives
        # better quality per bitrate than NVENC anyway.
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr",
            "-r", str(self.fps),
        ]

        if pulse_source:
            cmd += [
                "-c:a", "aac",
                "-b:a", "192k",
            ]

        cmd += [self._final_path]

        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            self._last_message = "ffmpeg not found"
            return

        self._is_recording = True
        self._start_time = time.time()
        self._last_message = "recording" if pulse_source else "recording (no audio)"

    def write_frame(self, rgb_uint8) -> None:
        if not self._is_recording or self._proc is None:
            return
        try:
            self._proc.stdin.write(rgb_uint8.tobytes())
        except (BrokenPipeError, ValueError, OSError):
            self.stop()

    def stop(self) -> None:
        with self._lock:
            if not self._is_recording:
                return
            self._is_recording = False

        # Closing stdin signals video EOF, but the pulse input is unbounded —
        # ffmpeg won't finalize the MP4 (moov atom) until told to wrap up.
        # SIGINT triggers ffmpeg's graceful-shutdown path: flush buffers,
        # write trailer, exit 0.
        try:
            if self._proc:
                if self._proc.stdin:
                    try:
                        self._proc.stdin.close()
                    except Exception:
                        pass
                try:
                    self._proc.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=15)
        except Exception:
            try:
                if self._proc:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            except Exception:
                pass

        proc, self._proc = self._proc, None
        # ffmpeg exits 255 after handling SIGINT — that's our graceful path,
        # not a failure. Treat any case where the output file looks valid as
        # success; only surface errors when the file is missing/empty.
        file_ok = (
            self._final_path is not None
            and os.path.exists(self._final_path)
            and os.path.getsize(self._final_path) > 1024
        )
        if file_ok:
            self._last_message = f"saved: {os.path.basename(self._final_path)}"
        else:
            err = b""
            try:
                if proc and proc.stderr:
                    err = proc.stderr.read() or b""
            except Exception:
                pass
            rc = proc.returncode if proc else "?"
            self._last_message = f"ffmpeg exit {rc}: {err.decode(errors='ignore')[:200]}"
