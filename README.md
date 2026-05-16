# SD-orb

**SD-orb** is a high-performance, real-time AI VJ orchestrator. It combines the power of Stable Diffusion (via NVIDIA TensorRT) with audio-reactive feedback loops to create immersive, recursive visuals that respond to live music.

![Sample Output](test_lcm.png)

## 🚀 Key Features

- **Real-time AI Generation**: Powered by Stable Diffusion 1.5, LCM (Latent Consistency Models), and NVIDIA TensorRT 10.x for ultra-low latency inference.
- **Audio Reactivity**: Integrated `AudioAnalyzer` captures system/mic audio and maps FFT frequency bands (Bass, Mids, Highs) to visual parameters.
- **Recursive Feedback Engine**: A custom `Visualizer` implementing warp, zoom, rotation, and decay effects that feed the previous AI frame back into the next generation.
- **Interactive UI**: Built with `DearPyGui` for real-time control over prompt playlists, AI strength, temporal smoothing, and feedback geometry.
- **Optimized Pipeline**: Uses `TAESD` (Tiny Autoencoder for Stable Diffusion) for instantaneous decoding of latents.

## 💻 Hardware Requirements

- **GPU**: NVIDIA RTX 30-series or 40-series (8GB+ VRAM recommended).
- **Driver**: NVIDIA Driver 535+
- **CUDA**: 12.x
- **TensorRT**: 10.x

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/SD-orb.git
cd SD-orb
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
Place your Stable Diffusion 1.5 checkpoints in the `models/` directory.
Recommended: [Realistic Vision V6.0 B1](https://civitai.com/models/4201/realistic-vision-v60-b1)

### 5. Build TensorRT Engine
Building the engine is hardware-specific and can take 10-20 minutes.
```bash
python builder.py
```

## 🎮 Usage

Run the main application:
```bash
python main.py
```

- **Prompt Playlist**: Add, edit, and shuffle prompts in real-time.
- **AI Strength**: Controls how much the AI modifies the input feedback loop.
- **Temporal Smooth**: Blends the current frame with the previous one for more fluid transitions.
- **Feedback Engine**: Adjust Zoom, Rotation, and Audio Sensitivity.

## 📊 Performance Metrics

Benchmarks conducted on NVIDIA RTX 4090 / CUDA 12.4:

| Component | Backend | Latency (ms) | FPS |
|--- |--- |--- |--- |
| **UNet Inference** | PyTorch (FP16) | ~45ms | ~22 |
| **UNet Inference** | TensorRT 10 | **~8ms** | **~120** |
| **End-to-End** | Full Pipeline | ~12ms | ~80 |

*Note: Performance may vary based on GPU and input resolution (default 512x512).*

## 📂 Project Structure

- `main.py`: Entry point and UI management.
- `pipeline.py`: AI inference logic (TensorRT + LCM).
- `visualizer.py`: Feedback transformation engine.
- `audio_analyzer.py`: Real-time audio processing.
- `builder.py`: TensorRT engine compiler.
- `models/`: (Ignored) Storage for .safetensors.
- `engines/`: (Ignored) Compiled TensorRT engines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) for acceleration patterns.
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers).
- [DearPyGui](https://github.com/hoffstadt/DearPyGui).
