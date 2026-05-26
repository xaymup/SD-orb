import pyaudio
import numpy as np

class AudioAnalyzer:
    def __init__(self, chunk=1024, rate=44100):
        self.chunk = chunk
        self.rate = rate
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.try_open_stream()

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

    def try_open_stream(self):
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            print(f"Warning: Could not open audio stream: {e}")
            self.stream = None

    def get_bands(self):
        if self.stream is None:
            return [0.1, 0.1, 0.1]
        try:
            raw = np.frombuffer(
                self.stream.read(self.chunk, exception_on_overflow=False),
                dtype=np.int16
            ).astype(np.float32) / 32768.0

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
        except Exception:
            return [0.0, 0.0, 0.0]

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
