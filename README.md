# SD-orb

**SD-orb** is a high-performance, real-time AI VJ orchestrator. It combines the power of Stable Diffusion (via NVIDIA TensorRT) with audio-reactive feedback loops to create immersive, recursive visuals that respond to live music.

![Sample Output](test_lcm.png)

## 🚀 Key Features

- **Real-time AI Generation**: Stable Diffusion 1.5 + LCM (Latent Consistency Model) compiled to NVIDIA TensorRT 10.x for one-step inference per frame.
- **Accurate Audio Reactivity**: `AudioAnalyzer` with a Hann-windowed FFT, musical EQ bands (Bass 20–250 Hz / Mids 250–2000 Hz / Highs 2–8 kHz), per-band auto-gain, and an asymmetric envelope follower (fast attack, slow release) so transients punch through.
- **GPU Feedback Engine**: `Visualizer` does its polar warp on the GPU with `F.grid_sample` — zoom, rotation and decay all happen on resident CUDA tensors, no CPU round-trip.
- **Image-Space Recursion**: The AI step blends in image space (`output = lerp(warped_input, ai_image, strength)`), so geometric warps survive the denoiser instead of getting redrawn back to centre.
- **🎚️ Hot-Swappable Model Bank**: Multiple SD 1.5 engines (RealisticVision, DreamShaper, Deliberate, MeinaMix, …) can be compiled and selected at runtime via a UI dropdown. Engine swap takes ~1–2 s with atomic restore-on-failure, so a bad load can't crash the loop.
- **🌀 Deep Dream**: Local Ollama LLM generates an evolving surreal narrative — each new prompt continues the previous one as a single dream. Picker for any installed model, temperature slider, keyword steering (Influences), history reset.
- **🎥 Recording**: One-click capture of the canvas to MP4. Video uses NVENC (zero CUDA contention with SD); audio is captured in parallel from PulseAudio's system-output monitor (or mic, configurable) at full rate and muxed at stop. Files land in `recordings/`.
- **🎛️ MIDI Learn**: Bind any hardware knob, fader or pad to any UI control. Mappings persist in `midi_mappings.json` across restarts. CC for sliders, Note for buttons/toggles.
- **Interactive UI**: `DearPyGui`, with a scalable display so the rendered canvas can be shown at 2× or more without re-rendering at higher resolution.

## 💻 Hardware Requirements

- **GPU**: NVIDIA RTX 30-series or 40-series (8 GB+ VRAM). NVENC required for the in-app recorder.
- **Driver**: NVIDIA Driver 535+
- **CUDA**: 12.x  •  **TensorRT**: 10.x
- **Audio**: PulseAudio or PipeWire (for the recording feature). PyAudio for the visualizer's analyzer works on ALSA.
- **(Optional) Ollama**: Required only for Deep Dream mode. Any installed instruction-tuned model works; `llama3.1:8b` is a fast default.
- **(Optional) MIDI Controller**: Any class-compliant USB MIDI device — picked up via mido/rtmidi.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/xaymup/SD-orb.git
cd SD-orb
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
Place SD 1.5 checkpoints in `models/`. Recommended starter:
[Realistic Vision V6.0 B1](https://civitai.com/models/4201/realistic-vision-v60-b1)

For more variety, see the [Model Bank](#-model-bank) section below.

### 5. Build TensorRT Engine
Engine is shape-baked at the resolution in `main.py` (default `384x384`). Takes 10–20 min the first time.
```bash
python builder.py <model_name>
```
- `<model_name>` looks up `models/<model_name>.safetensors` and writes `engines/<model_name>/unet.engine`.
- Run with no argument to build the legacy default (`realisticVisionV60B1_v51HyperVAE`).

### 6. (Optional) Start Ollama for Deep Dream
```bash
ollama serve &
ollama pull llama3.1:8b
```

## 🎮 Usage

```bash
python main.py
```

The viewport title shows live FPS, audio bands, the actual zoom/rot being applied, and warp/AI diff diagnostics:
```
25.9 fps | bass 0.74 mid 0.62 hi 0.47 | zoom 0.877 rot +0.224 | warp_diff 0.16 ai_diff 0.08
```

### Controls

| Section | Control | Notes |
|---|---|---|
| **SD Model** | Dropdown | Hot-swap between any compiled engine in `engines/`. Status line confirms active model. |
| Prompt Playlist | Add / edit / shuffle, Auto Switch on interval | Manual or auto cycling |
| AI | **Strength** | 0 = pure warp passthrough · 0.3 default · 1 = AI fully redraws each frame |
| AI | **Temporal Smooth** | Image-space blend with previous output frame |
| AI | **Bypass AI** | Skip the diffusion step entirely — useful for tuning the warp in isolation |
| Feedback Engine | **Zoom Base / React** | Constant zoom + bass-driven zoom-in pulse |
| Feedback Engine | **Rotate Base / React** | Constant rotation + mids-driven rotation |
| Feedback Engine | **Audio Smooth** | Release time of the envelope follower |
| Recording | **Start / Stop, Audio source** | NVENC video + PulseAudio capture → MP4 in `recordings/`. Pick `system` (loopback) or `mic`. |
| MIDI Learn | **Port, Control, Listen** | Pick port + target control, hit Listen, touch your MIDI device |
| Deep Dream | **Enable** | Switches the auto-prompt cycle to Ollama-generated dreams |
| Deep Dream | **Model / Temp / Reset** | Pick model, push temperature up for weirder dreams, reset history |
| Deep Dream | **Influences** | Comma-separated keywords steer the dream softly toward themes |

### Recommended Starting Settings

For a clearly perceptible "dive into the image" effect with audio reactivity:
- **AI Strength**: 0.3
- **Temporal Smooth**: 0.4
- **Zoom React**: 1.5–2.0
- **Rotate React**: 1.0–1.5

## 🎚️ Model Bank

The runtime hot-swaps between any engines in `engines/`. All engines share the canonical SD 1.5 VAE (`stabilityai/sd-vae-ft-mse`), the same text encoder, and the same LCM scheduler — only the UNet weights differ.

To add a model:

1. Drop the `.safetensors` in `models/` with a clean name (e.g. `models/dreamshaper8.safetensors`).
2. Build the engine:
   ```bash
   python builder.py dreamshaper8
   ```
3. Restart `main.py` — it appears in the **SD Model** dropdown.

Engines live at `engines/<name>/unet.engine`. The flat layout (`engines/<name>.engine`) also works for one-off engines.

### Aesthetic guide

| Model | Vibe | Best for |
|---|---|---|
| **RealisticVision V6** | Cinematic photoreal portraits | "cinematic portrait" / realistic scenes |
| **DreamShaper 8** | Versatile painterly-photoreal | Fantasy, sci-fi, abstract — all-rounder |
| **Deliberate v11** | Strongly painted / illustrated | Best for recursive feedback (brushstrokes trail beautifully on bass kicks) |
| **MeinaMix V8** | Anime, saturated, high-contrast | Anime / "trippy" sets, holds up under aggressive warp |

Many other SD 1.5 finetunes work the same way (AbsoluteReality, EpicRealism, CyberRealistic, Counterfeit, etc.).

## 📊 Performance Notes

On an RTX 4080 at 384×384:
- ~22–26 fps end-to-end (including full-VAE encode + decode and image-space blending)
- Recording active: ~0 fps drop (NVENC uses the dedicated video engine)
- Deep Dream prompt generation: ~1–3 s per dream, runs in a background thread; never blocks the render loop
- Engine swap: ~1–2 s (releases old engine atomically, falls back if the new one fails to load)

512² or higher requires an engine rebuild and will roughly halve the framerate.

## 📂 Project Structure

```
main.py            Entry point, UI, main render loop
pipeline.py        SD1.5 + LCM TRT pipeline (image-space strength/temporal blend,
                   atomic engine swap, canonical SD 1.5 VAE)
visualizer.py      Polar warp + decay, all on GPU via F.grid_sample
audio_analyzer.py  FFT bands with auto-gain + asymmetric envelope
dreamer.py         Background Ollama client for evolving dream prompts + keyword steering
recorder.py        NVENC video + PulseAudio capture, async mux on stop
midi.py            mido-based MIDI input with learn mode and JSON-persisted mappings
builder.py         TensorRT engine compiler (gitignored) — accepts <model_name> arg
models/            (gitignored) .safetensors checkpoints
engines/           (gitignored) Compiled TensorRT engines (per-model subdirs)
recordings/        (gitignored) Output MP4 files
midi_mappings.json (gitignored) Persisted MIDI bindings
```

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) for acceleration patterns and the TRT compile path.
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers).
- [Ollama](https://ollama.com/) for the local LLM runtime that powers Deep Dream.
- [DearPyGui](https://github.com/hoffstadt/DearPyGui).
- [mido + python-rtmidi](https://mido.readthedocs.io/) for MIDI input.
