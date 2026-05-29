import threading

import pyaudio
import numpy as np


class AudioAnalyzer:
    def __init__(self, chunk=1024, rate=44100, device_index: int | None = None):
        self.chunk = chunk
        self.rate = rate
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self._device_index = device_index
        # Held when (re)opening the stream so a device switch can't race
        # against an in-flight read in get_bands().
        self._device_lock = threading.Lock()
        self._open_stream(device_index)

        # Envelope follower: fast attack so transients punch through,
        # slow release (controlled by smoothing_factor) keeps them visible.
        self.smoothed_bands = np.array([0.0, 0.0, 0.0])
        self.smoothing_factor = 0.5
        self.attack_factor = 0.15

        # Hann window — without it, spectral leakage bleeds energy between bands.
        self.window = np.hanning(chunk).astype(np.float32)

        # Musical EQ split (Hz), mapped to FFT bins for this sample rate.
        freqs = np.fft.rfftfreq(chunk, d=1.0 / rate)
        self.bass_mask = (freqs >= 20.0)   & (freqs < 250.0)
        self.mid_mask  = (freqs >= 250.0)  & (freqs < 2000.0)
        self.high_mask = (freqs >= 2000.0) & (freqs < 8000.0)

        # Per-band running peak for auto-gain. Peaks decay slowly so the
        # visualization doesn't over-react during quiet passages.
        self.peaks = np.array([0.05, 0.05, 0.05])
        self.peak_decay = 0.995  # ~5s half-life at 30fps
        self.peak_floor = 0.02   # ignore signals below room noise

    @classmethod
    def list_input_devices(cls) -> list[tuple[int, str]]:
        """Enumerate all input-capable devices. Returns [(index, label)].
        Each PyAudio init re-queries the OS so newly-plugged devices appear
        on the next refresh. PulseAudio/PipeWire 'monitor' sources show up
        here too — that's how to capture system audio on Linux."""
        out: list[tuple[int, str]] = []
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if info.get("maxInputChannels", 0) > 0:
                    name = str(info.get("name", f"device {i}"))
                    out.append((i, name))
        finally:
            pa.terminate()
        return out

    def _open_stream(self, device_index: int | None) -> None:
        """Close any existing stream and open a new one on the chosen device.
        Caller is responsible for holding _device_lock. None resolves to the
        OS default input device, but we record the resolved index so the UI
        combo can reflect the real selection rather than 'default'."""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if device_index is None:
            try:
                device_index = int(self.pa.get_default_input_device_info().get("index"))
            except Exception:
                device_index = None
        try:
            kwargs = dict(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
            if device_index is not None:
                kwargs["input_device_index"] = device_index
            self.stream = self.pa.open(**kwargs)
            self._device_index = device_index
        except Exception as e:
            print(f"Warning: Could not open audio stream (device={device_index}): {e}")
            self.stream = None

    def set_device(self, device_index: int | None) -> None:
        """Switch the input device. Safe to call from the UI thread — the
        next get_bands() call will read from the new stream."""
        with self._device_lock:
            if device_index == self._device_index and self.stream is not None:
                return
            self._open_stream(device_index)

    @property
    def current_device(self) -> int | None:
        return self._device_index

    def get_bands(self):
        with self._device_lock:
            if self.stream is None:
                return [0.1, 0.1, 0.1]
            try:
                raw = np.frombuffer(
                    self.stream.read(self.chunk, exception_on_overflow=False),
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
            except Exception:
                return [0.0, 0.0, 0.0]

        # Heavy work outside the lock — device switches don't need to block
        # FFT and envelope updates.
        power = np.abs(np.fft.rfft(raw * self.window)) ** 2

        bass = np.sqrt(np.mean(power[self.bass_mask]) + 1e-12)
        mids = np.sqrt(np.mean(power[self.mid_mask])  + 1e-12)
        high = np.sqrt(np.mean(power[self.high_mask]) + 1e-12)
        band_energy = np.array([bass, mids, high])

        self.peaks = np.maximum(band_energy, self.peaks * self.peak_decay)
        denom = np.maximum(self.peaks, self.peak_floor)
        normalized = np.minimum(band_energy / denom, 1.5)

        rising = normalized > self.smoothed_bands
        mix = np.where(rising, self.attack_factor, self.smoothing_factor)
        self.smoothed_bands = self.smoothed_bands * mix + normalized * (1.0 - mix)

        return self.smoothed_bands.tolist()

    def close(self):
        with self._device_lock:
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
        self.pa.terminate()
