import os
import subprocess
import threading
import time


class Recorder:
    """
    Records the AI canvas to MP4 via two parallel ffmpeg processes:
      - video: raw RGB frames piped from the main loop, encoded with NVENC
      - audio: pulled directly from PulseAudio so we get full-rate capture
        without competing with PyAudio's in-process stream
    On stop, the two files are remuxed into a single MP4 in a background
    thread so the UI doesn't block.
    """

    def __init__(self, width: int, height: int, fps: int = 24, output_dir: str = "recordings"):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir

        self._video_proc: subprocess.Popen | None = None
        self._audio_proc: subprocess.Popen | None = None
        self._video_path: str | None = None
        self._audio_path: str | None = None
        self._final_path: str | None = None
        self._start_time: float | None = None
        self._is_recording = False
        self._last_message: str = ""
        self._lock = threading.Lock()

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

    def start(self, include_audio: bool = True, audio_source: str = "system") -> None:
        """audio_source: 'system' (loopback of speaker output) or 'mic'
        (default input). Falls back to 'default' if monitor lookup fails."""
        if self._is_recording:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if audio_source == "system":
            pulse_source = self.default_sink_monitor() or "default"
        else:
            pulse_source = "default"

        if include_audio:
            self._video_path = os.path.join(self.output_dir, f"{timestamp}_v.mp4")
            self._audio_path = os.path.join(self.output_dir, f"{timestamp}_a.wav")
            self._final_path = os.path.join(self.output_dir, f"{timestamp}.mp4")
        else:
            self._final_path = os.path.join(self.output_dir, f"{timestamp}.mp4")
            self._video_path = self._final_path
            self._audio_path = None

        video_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "ll",
            "-rc", "constqp",
            "-qp", "22",
            "-pix_fmt", "yuv420p",
            self._video_path,
        ]
        try:
            self._video_proc = subprocess.Popen(
                video_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            self._last_message = "ffmpeg not found"
            return

        if include_audio:
            audio_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "pulse",
                "-i", pulse_source,
                "-ac", "2",
                "-ar", "44100",
                "-c:a", "pcm_s16le",
                self._audio_path,
            ]
            try:
                self._audio_proc = subprocess.Popen(
                    audio_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
            except FileNotFoundError:
                self._audio_proc = None
                self._audio_path = None

        self._is_recording = True
        self._start_time = time.time()
        self._last_message = "recording"

    def write_frame(self, rgb_uint8) -> None:
        if not self._is_recording or self._video_proc is None:
            return
        try:
            self._video_proc.stdin.write(rgb_uint8.tobytes())
        except (BrokenPipeError, ValueError, OSError):
            # Encoder died. Stop cleanly so the user gets the partial file.
            self.stop()

    def stop(self) -> None:
        with self._lock:
            if not self._is_recording:
                return
            self._is_recording = False

        # Close video encoder.
        try:
            if self._video_proc and self._video_proc.stdin:
                self._video_proc.stdin.close()
            if self._video_proc:
                self._video_proc.wait(timeout=10)
        except Exception:
            try:
                self._video_proc.kill()
            except Exception:
                pass
        self._video_proc = None

        # Stop audio capture.
        if self._audio_proc:
            try:
                self._audio_proc.terminate()
                self._audio_proc.wait(timeout=5)
            except Exception:
                try:
                    self._audio_proc.kill()
                except Exception:
                    pass
            self._audio_proc = None

        # Mux video + audio if both were captured. Do it off-thread so the UI
        # doesn't freeze.
        if (
            self._audio_path
            and os.path.exists(self._audio_path)
            and os.path.getsize(self._audio_path) > 1024
            and self._video_path != self._final_path
        ):
            threading.Thread(target=self._mux_and_cleanup, daemon=True).start()
        elif self._video_path and self._video_path != self._final_path:
            # Audio was requested but failed — keep the video file under the final name.
            try:
                os.replace(self._video_path, self._final_path)
            except OSError:
                pass
            self._last_message = f"saved (no audio): {os.path.basename(self._final_path)}"
        else:
            self._last_message = f"saved: {os.path.basename(self._final_path)}" if self._final_path else "saved"

    def _mux_and_cleanup(self) -> None:
        try:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", self._video_path,
                "-i", self._audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                self._final_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                try:
                    os.remove(self._video_path)
                    os.remove(self._audio_path)
                except OSError:
                    pass
                self._last_message = f"saved: {os.path.basename(self._final_path)}"
            else:
                err = result.stderr.decode(errors="ignore")[:200]
                self._last_message = f"mux failed: {err}"
        except Exception as e:
            self._last_message = f"mux error: {e}"
