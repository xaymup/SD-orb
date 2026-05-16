import pyaudio
import numpy as np

class AudioAnalyzer:
    def __init__(self, chunk=1024, rate=44100):
        self.chunk = chunk
        self.rate = rate
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.try_open_stream()

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
            data = np.frombuffer(
                self.stream.read(self.chunk, exception_on_overflow=False),
                dtype=np.int16
            )
            fft = np.abs(np.fft.rfft(data))
            # Normalize and clamp bands
            bass  = min(np.mean(fft[:8])   / 12000, 1.5)
            mids  = min(np.mean(fft[8:60]) / 6000,  1.5)
            highs = min(np.mean(fft[60:])  / 3000,  1.5)
            return [bass, mids, highs]
        except Exception:
            return [0.0, 0.0, 0.0]

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
