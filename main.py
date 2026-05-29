import json
import os
import time
import torch
import numpy as np
import dearpygui.dearpygui as dpg

import settings as settings_io
from audio_analyzer import AudioAnalyzer
from visualizer import Visualizer
from pipeline import AIPipeline
from dreamer import Dreamer
from recorder import Recorder
from midi import MidiController
from fullscreen import FullscreenOutput
from upscaler import Upscaler

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 384, 384  # was 512 — requires builder.py engine rebuild


def discover_engines() -> list[tuple[str, str]]:
    """Find compiled UNet engines. Supports two layouts:
    - flat:   engines/<name>.engine
    - subdir: engines/<name>/unet.engine
    Returns [(display_name, path)] sorted alphabetically.
    """
    out = []
    if os.path.isdir("engines"):
        for entry in os.listdir("engines"):
            full = os.path.join("engines", entry)
            if entry.endswith(".engine") and os.path.isfile(full):
                out.append((entry[:-len(".engine")], full))
            elif os.path.isdir(full) and entry != "onnx":
                sub = os.path.join(full, "unet.engine")
                if os.path.isfile(sub):
                    out.append((entry, sub))
    return sorted(out)


def _checkpoint_for(model_name: str) -> str | None:
    for ext in (".safetensors", ".ckpt"):
        p = f"models/{model_name}{ext}"
        if os.path.exists(p):
            return p
    return None


def resolve_engine(engine_name: str) -> tuple[str | None, list[tuple[str, float]], str]:
    """Return (checkpoint_path, loras, style_hint) for an engine. New-style
    engines have a manifest.json pinning the exact base + LoRA recipe plus a
    style hint for the dreamer; legacy engines fall back to engine-name-
    equals-checkpoint-name with no LoRAs and no style."""
    manifest_path = f"engines/{engine_name}/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        ckpt = _checkpoint_for(m["base"])
        loras = [
            (f"loras/{l['name']}.safetensors", float(l["scale"]))
            for l in m.get("loras", [])
        ]
        return ckpt, loras, m.get("style", "")
    return _checkpoint_for(engine_name), [], ""


# Pick whatever engine is available at startup. Hardcoding a single path
# breaks the moment that engine gets renamed or removed.
_available_engines = discover_engines()
if not _available_engines:
    raise SystemExit("No engines found under engines/. Run `python builder.py <name>` first.")
STARTUP_ENGINE_NAME, ENGINE_PATH = _available_engines[0]
MODEL_PATH, STARTUP_LORAS, STARTUP_STYLE = resolve_engine(STARTUP_ENGINE_NAME)
if MODEL_PATH is None:
    raise SystemExit(
        f"Startup engine '{STARTUP_ENGINE_NAME}' has no resolvable checkpoint in models/."
    )
print(f"Startup engine: {STARTUP_ENGINE_NAME} ({ENGINE_PATH})")
if STARTUP_LORAS:
    print(f"  with LoRAs: {', '.join(f'{p}@{s}' for p, s in STARTUP_LORAS)}")
if STARTUP_STYLE:
    print(f"  dreamer style hint: {STARTUP_STYLE}")

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
state = {
    "prompts": [
        "cinematic portrait, realistic lighting, 8k, ultra detailed, bokeh",
        "neon cyberpunk city, rain, reflections, futuristic atmosphere",
        "dreamy ethereal forest, mystical lights, soft focus, fantasy",
        "abstract liquid metal, iridescent colors, 3d render, sleek"
    ],
    "current_p_idx": 0,
    "current_p": "",
    "auto_switch": True,
    "switch_interval": 30.0,
    "last_switch": time.time(),
    "shuffle": True,
    "embeds": None,
    "deep_dream": False,
    "neg_p": "nudity, nsfw, naked, nude, sexual, breasts, genitals, suggestive, erotic, lingerie, underwear, porn, explicit",
    "neg_embeds": None,
}
state["current_p"] = state["prompts"][0]

# Pre-UI restore: prompt list + current selection have to be applied before
# refresh_prompt_list() runs, so widgets render with the saved playlist.
# Other knobs are restored post-UI via dpg.set_value (see further below).
_SAVED = settings_io.load()
if isinstance(_SAVED.get("prompts"), list) and _SAVED["prompts"]:
    state["prompts"] = [str(p) for p in _SAVED["prompts"]]
    idx = _SAVED.get("current_p_idx", 0)
    state["current_p_idx"] = max(0, min(int(idx), len(state["prompts"]) - 1))
    state["current_p"] = _SAVED.get("current_p") or state["prompts"][state["current_p_idx"]]

# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------
print("Initializing 113.Milkdrop AI Orchestrator...")
audio = AudioAnalyzer()
viz = Visualizer(WIDTH, HEIGHT)
ai = AIPipeline(MODEL_PATH, ENGINE_PATH, width=WIDTH, height=HEIGHT, loras=STARTUP_LORAS)
dreamer = Dreamer()
dreamer.set_style(STARTUP_STYLE)
recorder = Recorder(WIDTH, HEIGHT, fps=30)
midi = MidiController()
fullscreen = FullscreenOutput()
upscaler = Upscaler()


def _output_size() -> tuple[int, int]:
    """Frame size for downstream consumers (fullscreen viewer, recorder).
    Returns the upscaled size only if the upscaler is both enabled AND
    loaded — a disabled or unloaded upscaler must not change the size."""
    if upscaler.enabled and upscaler.is_loaded():
        return WIDTH * upscaler.scale, HEIGHT * upscaler.scale
    return WIDTH, HEIGHT


# Rec.709 luminance weights, broadcast-shaped for (1, 3, H, W).
_LUMA_W = torch.tensor([0.2126, 0.7152, 0.0722], device="cuda", dtype=torch.float16).view(1, 3, 1, 1)


@torch.no_grad()
def adjust_color(t: torch.Tensor, brightness: float, contrast: float,
                 saturation: float, gamma: float) -> torch.Tensor:
    """Image-space color decoration applied to the DISPLAY tensor only,
    never to the recursive ai_tensor. Keeping it out of the feedback loop
    means a brightness boost lifts the perceived output without compounding
    each frame (which would saturate to white in seconds). Defaults are
    identity — return the input when all sliders are at neutral."""
    out = t
    if brightness != 0.0:
        out = out + brightness
    if contrast != 1.0:
        out = (out - 0.5) * contrast + 0.5
    if saturation != 1.0:
        # Per-pixel luminance, broadcast back across RGB.
        lum = (out * _LUMA_W).sum(dim=1, keepdim=True)
        out = lum + (out - lum) * saturation
    out = out.clamp(0.0, 1.0)
    if gamma != 1.0:
        out = out.pow(1.0 / gamma)
    return out

# (tag, label, type, min, max). Buttons use ("trigger" routing in apply_midi).
MIDI_CONTROLS = [
    ("strength_sl",     "AI Strength",      "slider",   0.0,  1.0),
    ("delta_sl",        "Temporal Smooth",  "slider",   0.0,  0.98),
    ("audio_smooth_sl", "Audio Smooth",     "slider",   0.0,  0.99),
    ("zoom_base_sl",    "Zoom Base",        "slider",   0.8,  1.1),
    ("zoom_sens_sl",    "Zoom React",       "slider",   0.0,  2.0),
    ("rot_base_sl",     "Rotate Base",      "slider",  -0.1,  0.1),
    ("rot_sens_sl",     "Rotate React",     "slider",   0.0,  2.0),
    ("kaleido_base_sl", "Kaleido Base",     "slider",   0.0,  1.0),
    ("kaleido_sens_sl", "Kaleido React",    "slider",   0.0,  2.0),
    ("interval_sl",     "Switch Interval",  "slider",   5.0,  300.0),
    ("dream_temp_sl",   "Dream Temp",       "slider",   0.3,  1.6),
    ("auto_sw_cb",      "Auto Switch",      "checkbox", 0,    1),
    ("shuffle_cb",      "Shuffle",          "checkbox", 0,    1),
    ("bypass_ai_cb",    "Bypass AI",        "checkbox", 0,    1),
    ("deep_dream_cb",   "Deep Dream",       "checkbox", 0,    1),
    ("record_audio_cb", "Record Audio",     "checkbox", 0,    1),
    ("next_prompt",     "→ Next Prompt",    "button",   0,    1),
    ("record_btn",      "→ Toggle Recording","button",  0,    1),
    ("dream_reset",     "→ Reset Dream",    "button",   0,    1),
    ("random_engine",   "→ Random Engine",  "button",   0,    1),
    ("brightness_sl",   "Brightness",       "slider",  -0.3,  0.3),
    ("contrast_sl",     "Contrast",         "slider",   0.5,  2.0),
    ("saturation_sl",   "Saturation",       "slider",   0.0,  2.0),
    ("gamma_sl",        "Gamma",            "slider",   0.5,  2.0),
]

# Registry populated by UI setup: maps a MIDI button tag to a zero-arg
# callable. Lets apply_midi_updates trigger actions that are defined inside
# the dpg.window setup block (closures over UI-local state) without forcing
# them to module level.
midi_actions: dict[str, "callable"] = {}

DREAM_MODELS = [
    "llama3.1:8b",
    "qwen3.5:9b",
    "qwen3:14b",
    "deepseek-r1:14b",
]

def update_embeds(prompt):
    state["embeds"] = ai.get_embeds(prompt)
    state["current_p"] = prompt

def update_neg_embeds(neg_prompt):
    """Re-encode the negative prompt against the active text encoder. Called
    on edit and after an SD engine swap (the new text encoder produces
    different embeddings, so a cached neg_embeds from before the swap would
    pull the conditioning in the wrong direction)."""
    state["neg_p"] = neg_prompt
    state["neg_embeds"] = ai.get_embeds(neg_prompt) if neg_prompt.strip() else None

def next_prompt():
    # Deep Dream takes over the prompt cycle when enabled and we have a dream ready.
    if state["deep_dream"]:
        dream = dreamer.latest_dream
        if dream:
            update_embeds(dream)
            dpg.set_value("prompt_in", dream)
            if dpg.does_item_exist("dream_text"):
                dpg.set_value("dream_text", dream)
            dreamer.request_dream()  # queue the next one for the following switch
            state["last_switch"] = time.time()
            return
        # No dream ready yet — kick one off and fall through to manual this round.
        dreamer.request_dream()

    if not state["prompts"]:
        return
    if state["shuffle"]:
        state["current_p_idx"] = np.random.randint(0, len(state["prompts"]))
    else:
        state["current_p_idx"] = (state["current_p_idx"] + 1) % len(state["prompts"])

    p = state["prompts"][state["current_p_idx"]]
    update_embeds(p)
    dpg.set_value("prompt_in", p)
    state["last_switch"] = time.time()

def edit_prompt(idx, new_val):
    state["prompts"][idx] = new_val
    if state["current_p_idx"] == idx:
        update_embeds(new_val)
        dpg.set_value("prompt_in", new_val)

def select_prompt(idx):
    state["current_p_idx"] = idx
    p = state["prompts"][idx]
    update_embeds(p)
    dpg.set_value("prompt_in", p)
    state["last_switch"] = time.time()

def refresh_prompt_list():
    if not dpg.does_item_exist("prompt_list_group"): return
    dpg.delete_item("prompt_list_group", children_only=True)
    for i, p in enumerate(state["prompts"]):
        with dpg.group(horizontal=True, parent="prompt_list_group"):
            dpg.add_button(label=">", callback=lambda s, a, u: select_prompt(u), user_data=i)
            dpg.add_input_text(default_value=p, width=230, callback=lambda s, a, u: edit_prompt(u, a), user_data=i)
            dpg.add_button(label="X", callback=lambda s, a, u: remove_prompt(u), user_data=i)

def add_prompt(s, a):
    new_p = dpg.get_value("new_prompt_in")
    if new_p:
        state["prompts"].append(new_p)
        dpg.set_value("new_prompt_in", "")
        refresh_prompt_list()

def remove_prompt(idx):
    if len(state["prompts"]) > 1:
        state["prompts"].pop(idx)
        refresh_prompt_list()

update_embeds(state["current_p"])


def apply_midi_updates():
    pending = midi.consume_pending()
    for tag, value in pending.items():
        # Buttons / triggers
        if value == "trigger":
            handler = midi_actions.get(tag)
            if handler:
                handler()
                continue
            if tag == "next_prompt":
                next_prompt()
            elif tag == "record_btn":
                if recorder.is_recording:
                    recorder.stop()
                    if dpg.does_item_exist("record_btn"):
                        dpg.set_item_label("record_btn", "● Start Recording")
                else:
                    w, h = _output_size()
                    recorder.start(
                        include_audio=dpg.get_value("record_audio_cb"),
                        audio_source=dpg.get_value("audio_source_combo"),
                        width=w, height=h,
                    )
                    if dpg.does_item_exist("record_btn"):
                        dpg.set_item_label("record_btn", "■ Stop Recording")
            elif tag == "dream_reset":
                dreamer.reset()
            continue

        # Toggle (note-on on a checkbox)
        if value == "toggle":
            if dpg.does_item_exist(tag):
                cur = dpg.get_value(tag)
                value = not cur
            else:
                continue

        # Apply to widget if it exists
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)

        # Mirror into state / dreamer for fields driven by callbacks rather
        # than read-each-frame.
        if tag == "auto_sw_cb":
            state["auto_switch"] = bool(value)
        elif tag == "shuffle_cb":
            state["shuffle"] = bool(value)
        elif tag == "deep_dream_cb":
            state["deep_dream"] = bool(value)
            if value and dreamer.latest_dream is None:
                dreamer.request_dream()
        elif tag == "interval_sl":
            state["switch_interval"] = float(value)
        elif tag == "dream_temp_sl":
            dreamer.set_temperature(float(value))

# ---------------------------------------------------------------------------
# UI SETUP
# ---------------------------------------------------------------------------
dpg.create_context()
with dpg.texture_registry():
    dpg.add_dynamic_texture(width=WIDTH, height=HEIGHT, default_value=[0]*WIDTH*HEIGHT*4, tag="ai_tex")

DISPLAY_SCALE = 2  # show the 384² render at 768² in the UI
DISPLAY_W = WIDTH  * DISPLAY_SCALE
DISPLAY_H = HEIGHT * DISPLAY_SCALE

with dpg.window(label="113.RECURSIVE AI — Modular RV6", tag="main_window", width=350 + DISPLAY_W + 40, height=DISPLAY_H + 60):
    # Top prompt strip — spans the full viewport so the prompt has the room
    # it deserves. Negative prompt + strength sit beside each other below.
    dpg.add_input_text(
        tag="prompt_in", default_value=state["current_p"],
        callback=lambda s, a: update_embeds(a),
        multiline=True, width=-1, height=70,
    )
    with dpg.group(horizontal=True):
        dpg.add_input_text(
            label="", tag="neg_prompt_in",
            default_value=state["neg_p"],
            callback=lambda s, a: update_neg_embeds(a),
            multiline=True, width=-220, height=50,
        )
        with dpg.group():
            dpg.add_text("Negative prompt + strength", color=(150, 150, 150))
            dpg.add_slider_float(
                label="", tag="neg_strength_sl",
                min_value=0.0, max_value=1.5, default_value=0.5,
                width=180,
            )
    dpg.add_separator()

    with dpg.table(header_row=False):
        dpg.add_table_column(width_fixed=True, init_width_or_weight=330)
        dpg.add_table_column()
        with dpg.table_row():
            with dpg.group():
                # =========================================================
                # ALWAYS-VISIBLE: engine, master mix knobs. Everything else
                # lives in a tab bar below to keep the panel short.
                # =========================================================

                # SD Model picker — hot-swaps the TRT engine in place.
                _engines = discover_engines()
                _engine_names = [n for n, _ in _engines] or ["(none)"]
                _engine_paths = {n: p for n, p in _engines}
                _current_engine_name = STARTUP_ENGINE_NAME if STARTUP_ENGINE_NAME in _engine_paths else _engine_names[0]

                def _format_engine_details(name: str) -> str:
                    """Multi-line summary of an engine's LoRA stack + style for the
                    detail widget under the dropdown. Pulled live from manifest.json
                    so edits to a manifest show up after a swap, no restart needed."""
                    _, loras, style = resolve_engine(name)
                    lines = []
                    if loras:
                        lines.append("LoRAs: " + ", ".join(f"{os.path.basename(p).replace('.safetensors','')}@{s}" for p, s in loras))
                    else:
                        lines.append("LoRAs: (none)")
                    if style:
                        lines.append("Style: " + style)
                    return "\n".join(lines)

                def _swap_sd_model(s, a):
                    path = _engine_paths.get(a)
                    if not path:
                        return
                    checkpoint, loras, style_hint = resolve_engine(a)
                    if checkpoint is None:
                        dpg.set_value("sd_model_status", f"no checkpoint for {a}")
                        return
                    dpg.set_value("sd_model_status", f"loading {a}…")
                    try:
                        ai.swap_engine(path, checkpoint, loras)
                        # Re-encode prompt with the model's own text encoder.
                        if state["current_p"]:
                            update_embeds(state["current_p"])
                        # Same text-encoder swap invalidates the cached
                        # negative embedding.
                        update_neg_embeds(state["neg_p"])
                        # Pivot the dreamer aesthetic to match the new engine.
                        dreamer.set_style(style_hint)
                        dpg.set_value("sd_model_status", f"active: {a}")
                        dpg.set_value("sd_model_details", _format_engine_details(a))
                    except Exception as e:
                        dpg.set_value("sd_model_status", f"swap failed: {e}")

                def _random_engine_swap():
                    """Pick a different engine at random. Excludes the current one
                    so the action always changes something — otherwise on a one-
                    engine setup the MIDI button would silently no-op."""
                    cur = dpg.get_value("sd_model_combo")
                    candidates = [n for n in _engine_names if n != cur and n != "(none)"]
                    if not candidates:
                        return
                    pick = candidates[np.random.randint(0, len(candidates))]
                    dpg.set_value("sd_model_combo", pick)
                    _swap_sd_model(None, pick)

                # Expose to MIDI: maps "random_engine" trigger → this callable.
                midi_actions["random_engine"] = _random_engine_swap

                dpg.add_combo(
                    items=_engine_names, default_value=_current_engine_name,
                    label="SD Model", tag="sd_model_combo",
                    callback=_swap_sd_model, width=-80,
                )
                dpg.add_button(label="🎲 Random Engine", tag="random_engine",
                               callback=lambda: _random_engine_swap(), width=-1)
                dpg.add_text(f"active: {_current_engine_name}", tag="sd_model_status", color=(160, 200, 160))
                dpg.add_text(
                    _format_engine_details(_current_engine_name),
                    tag="sd_model_details", color=(160, 160, 200), wrap=320,
                )
                dpg.add_separator()

                dpg.add_slider_float(label="AI Strength",     tag="strength_sl", min_value=0.0, max_value=1.0,  default_value=0.3)
                dpg.add_slider_float(label="Temporal Smooth", tag="delta_sl",    min_value=0.0, max_value=0.98, default_value=0.4)
                dpg.add_checkbox(label="Bypass AI (raw warp feedback)", tag="bypass_ai_cb", default_value=False)
                dpg.add_separator()

                with dpg.tab_bar():
                    # ---------------------------------------------- Playlist
                    with dpg.tab(label="Playlist"):
                        dpg.add_checkbox(label="Auto Switch", tag="auto_sw_cb", default_value=state["auto_switch"], callback=lambda s, a: state.update({"auto_switch": a}))
                        dpg.add_slider_float(label="Interval (s)", tag="interval_sl", min_value=5.0, max_value=300.0, default_value=state["switch_interval"], callback=lambda s, a: state.update({"switch_interval": a}))
                        dpg.add_checkbox(label="Shuffle", tag="shuffle_cb", default_value=state["shuffle"], callback=lambda s, a: state.update({"shuffle": a}))
                        dpg.add_button(label="Next Prompt", callback=next_prompt, width=-1)
                        dpg.add_separator()
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="new_prompt_in", hint="Add new prompt...", width=230)
                            dpg.add_button(label="+", callback=add_prompt)
                        with dpg.group(tag="prompt_list_group"):
                            pass
                        refresh_prompt_list()

                    # ---------------------------------------------- Look
                    with dpg.tab(label="Look"):
                        dpg.add_text("Output Color", color=(200, 200, 200))
                        dpg.add_slider_float(label="Brightness", tag="brightness_sl", min_value=-0.3, max_value=0.3, default_value=0.0)
                        dpg.add_slider_float(label="Contrast",   tag="contrast_sl",   min_value=0.5,  max_value=2.0, default_value=1.0)
                        dpg.add_slider_float(label="Saturation", tag="saturation_sl", min_value=0.0,  max_value=2.0, default_value=1.0)
                        dpg.add_slider_float(label="Gamma",      tag="gamma_sl",      min_value=0.5,  max_value=2.0, default_value=1.0)
                        def _reset_color():
                            dpg.set_value("brightness_sl", 0.0)
                            dpg.set_value("contrast_sl",   1.0)
                            dpg.set_value("saturation_sl", 1.0)
                            dpg.set_value("gamma_sl",      1.0)
                        dpg.add_button(label="Reset Color", callback=lambda: _reset_color(), width=-1)

                        dpg.add_separator()
                        dpg.add_text("Feedback Engine", color=(200, 200, 200))
                        dpg.add_slider_float(label="Audio Smooth",   tag="audio_smooth_sl", min_value=0.0, max_value=0.99, default_value=0.7)
                        dpg.add_slider_float(label="Zoom Base",      tag="zoom_base_sl",  min_value=0.8, max_value=1.1,  default_value=0.98)
                        dpg.add_slider_float(label="Zoom React",    tag="zoom_sens_sl",  min_value=0.0, max_value=2.0,  default_value=1.0)
                        dpg.add_slider_float(label="Rotate Base",    tag="rot_base_sl",   min_value=-0.1, max_value=0.1, default_value=0.01)
                        dpg.add_slider_float(label="Rotate React",  tag="rot_sens_sl",   min_value=0.0, max_value=2.0,  default_value=1.0)
                        dpg.add_combo(
                            items=["Off", "Mirror-fold", "Anisotropic stretch", "Stretch-in-folds"],
                            default_value="Off",
                            label="Kaleido Mode", tag="kaleido_mode_combo",
                        )
                        dpg.add_combo(
                            items=["Bass", "Mids", "Highs"],
                            default_value="Bass",
                            label="Kaleido Band", tag="kaleido_band_combo",
                        )
                        dpg.add_slider_float(label="Kaleido Base",   tag="kaleido_base_sl", min_value=0.0, max_value=1.0, default_value=0.0)
                        dpg.add_slider_float(label="Kaleido React",  tag="kaleido_sens_sl", min_value=0.0, max_value=2.0, default_value=1.0)

                    # ---------------------------------------------- Dream
                    with dpg.tab(label="Dream"):
                        def _toggle_dream(s, a):
                            state["deep_dream"] = a
                            if a and dreamer.latest_dream is None:
                                dreamer.request_dream()
                        dpg.add_checkbox(label="Enable Deep Dream", tag="deep_dream_cb", default_value=False, callback=_toggle_dream)
                        dpg.add_checkbox(
                            label="Run llama on CPU (smoother FPS)",
                            tag="dream_cpu_cb", default_value=True,
                            callback=lambda s, a: dreamer.set_cpu_only(a),
                        )
                        dpg.add_combo(items=DREAM_MODELS, default_value=DREAM_MODELS[0], tag="dream_model_cb", callback=lambda s, a: dreamer.set_model(a), width=-1)
                        dpg.add_slider_float(label="Dream Temp", tag="dream_temp_sl", min_value=0.3, max_value=1.6, default_value=1.1, callback=lambda s, a: dreamer.set_temperature(a))
                        dpg.add_input_text(
                            label="Influences", tag="dream_keywords_in",
                            hint="e.g. underwater, neon, glass",
                            callback=lambda s, a: dreamer.set_keywords(a),
                            width=-80,
                        )
                        dpg.add_text("Current dream:")
                        dpg.add_text("(idle)", tag="dream_text", wrap=320)
                        dpg.add_text("", tag="dream_status", color=(200, 160, 80))
                        dpg.add_button(label="Reset Dream", callback=lambda: dreamer.reset(), width=-1)

                    # ---------------------------------------------- Devices
                    with dpg.tab(label="Devices"):
                        # ----- Audio Input -----
                        dpg.add_text("Audio Input", color=(200, 200, 200))
                        def _refresh_audio_devices():
                            devices = AudioAnalyzer.list_input_devices()
                            items = [f"{idx}: {name}" for idx, name in devices] or ["(none)"]
                            dpg.configure_item("audio_in_combo", items=items)
                            cur = audio.current_device
                            match = next(
                                (s for s in items if cur is not None and s.startswith(f"{cur}:")),
                                items[0],
                            )
                            dpg.set_value("audio_in_combo", match)

                        def _select_audio_device(s, a):
                            if not a or a == "(none)":
                                return
                            idx_str, _, _ = a.partition(":")
                            try:
                                audio.set_device(int(idx_str))
                            except ValueError:
                                pass

                        dpg.add_combo(
                            items=["(none)"], default_value="(none)", tag="audio_in_combo",
                            label="Device", callback=_select_audio_device, width=-80,
                        )
                        dpg.add_button(label="Refresh Devices", callback=lambda: _refresh_audio_devices(), width=-1)
                        dpg.add_text(
                            "Tip: on Linux, pick a 'monitor' source to react to system audio.",
                            color=(160, 160, 160), wrap=320,
                        )
                        _refresh_audio_devices()

                        dpg.add_separator()
                        # ----- Fullscreen Output -----
                        dpg.add_text("Fullscreen Output", color=(200, 200, 200))
                        def _refresh_monitors():
                            try:
                                mons = FullscreenOutput.list_monitors()
                            except Exception as e:
                                dpg.set_value("fs_status", f"pygame error: {e}")
                                dpg.configure_item("fs_monitor_combo", items=["(none)"])
                                return
                            items = [f"{i}: {w}x{h}" for i, (w, h) in mons] or ["(none)"]
                            dpg.configure_item("fs_monitor_combo", items=items)
                            if mons:
                                cur = dpg.get_value("fs_monitor_combo")
                                if not cur or cur not in items:
                                    dpg.set_value("fs_monitor_combo", items[-1])

                        def _open_fullscreen():
                            sel = dpg.get_value("fs_monitor_combo")
                            if not sel or sel == "(none)":
                                dpg.set_value("fs_status", "no monitor selected")
                                return
                            try:
                                idx = int(sel.split(":", 1)[0])
                            except ValueError:
                                return
                            try:
                                fullscreen.open(idx, _output_size())
                                dpg.set_value("fs_status", f"open on monitor {idx} (ESC to close)")
                                dpg.set_item_label("fs_toggle_btn", "Close Fullscreen")
                            except Exception as e:
                                dpg.set_value("fs_status", f"open failed: {e}")

                        def _close_fullscreen():
                            fullscreen.close()
                            dpg.set_value("fs_status", "closed")
                            dpg.set_item_label("fs_toggle_btn", "Open Fullscreen")

                        def _toggle_fullscreen():
                            if fullscreen.is_open():
                                _close_fullscreen()
                            else:
                                _open_fullscreen()

                        dpg.add_combo(
                            items=["(none)"], default_value="(none)", tag="fs_monitor_combo",
                            label="Monitor", width=-80,
                        )
                        dpg.add_button(label="Refresh Monitors", callback=lambda: _refresh_monitors(), width=-1)
                        dpg.add_button(label="Open Fullscreen", tag="fs_toggle_btn", callback=lambda: _toggle_fullscreen(), width=-1)
                        dpg.add_text("closed", tag="fs_status", color=(160, 200, 160))
                        _refresh_monitors()

                        dpg.add_separator()
                        # ----- GPU Upscaler -----
                        dpg.add_text("GPU Upscaler (fullscreen + recording)", color=(200, 200, 200))

                        def _toggle_upscaler():
                            want = dpg.get_value("upscale_cb")
                            if recorder.is_recording:
                                dpg.set_value("upscale_cb", upscaler.enabled)
                                dpg.set_value("upscale_status", "stop recording first")
                                return
                            if want and not upscaler.is_loaded():
                                dpg.set_value("upscale_status", "loading model...")
                                try:
                                    upscaler.load()
                                except Exception as e:
                                    dpg.set_value("upscale_cb", False)
                                    dpg.set_value("upscale_status", f"load failed: {e}")
                                    return
                            upscaler.enabled = bool(want)
                            if fullscreen.is_open():
                                mon = fullscreen.monitor
                                fullscreen.close()
                                if mon is not None:
                                    try:
                                        fullscreen.open(mon, _output_size())
                                    except Exception as e:
                                        dpg.set_value("fs_status", f"reopen failed: {e}")
                            w, h = _output_size()
                            dpg.set_value(
                                "upscale_status",
                                f"on (output {w}x{h})" if upscaler.enabled else "off",
                            )

                        def _select_upscaler_profile(s, a):
                            if recorder.is_recording or fullscreen.is_open():
                                dpg.set_value("upscale_profile_combo", upscaler.profile)
                                dpg.set_value("upscale_status", "close fullscreen / stop recording first")
                                return
                            try:
                                upscaler.set_profile(a)
                                w, h = _output_size()
                                dpg.set_value(
                                    "upscale_status",
                                    f"profile {a}; on (output {w}x{h})" if upscaler.enabled else f"profile {a}; off",
                                )
                            except Exception as e:
                                dpg.set_value("upscale_status", f"profile change failed: {e}")

                        dpg.add_combo(
                            items=Upscaler.list_profiles(), default_value=upscaler.profile,
                            label="Model", tag="upscale_profile_combo",
                            callback=_select_upscaler_profile, width=-80,
                        )
                        dpg.add_checkbox(
                            label="Enable GPU Upscale", tag="upscale_cb",
                            default_value=False, callback=lambda: _toggle_upscaler(),
                        )
                        dpg.add_text("off", tag="upscale_status", color=(160, 160, 160), wrap=320)

                        dpg.add_separator()
                        # ----- Recording -----
                        dpg.add_text("Recording", color=(200, 200, 200))
                        def _toggle_record(s, a, u):
                            if recorder.is_recording:
                                recorder.stop()
                                dpg.set_item_label("record_btn", "Start Recording")
                            else:
                                w, h = _output_size()
                                recorder.start(
                                    include_audio=dpg.get_value("record_audio_cb"),
                                    audio_source=dpg.get_value("audio_source_combo"),
                                    width=w, height=h,
                                )
                                dpg.set_item_label("record_btn", "Stop Recording")
                        dpg.add_checkbox(label="Include audio", tag="record_audio_cb", default_value=True)
                        dpg.add_combo(items=["system", "mic"], default_value="system",
                                      label="Audio source", tag="audio_source_combo")
                        dpg.add_button(label="Start Recording", tag="record_btn", callback=_toggle_record, width=-1)
                        dpg.add_text("", tag="record_status", color=(220, 120, 120))

                        dpg.add_separator()
                        # ----- MIDI Learn -----
                        dpg.add_text("MIDI Learn", color=(200, 200, 200))
                        def _refresh_ports():
                            ports = MidiController.list_inputs()
                            if not ports:
                                ports = ["(none)"]
                            dpg.configure_item("midi_port_combo", items=ports)
                            if midi.port_name and midi.port_name in ports:
                                dpg.set_value("midi_port_combo", midi.port_name)
                            else:
                                dpg.set_value("midi_port_combo", ports[0])

                        def _connect_port(s, a):
                            if a and a != "(none)":
                                midi.open(a)
                                dpg.set_value("midi_status", f"connected: {a}")
                            else:
                                midi.close()
                                dpg.set_value("midi_status", "disconnected")

                        def _start_learn():
                            sel = dpg.get_value("midi_target_combo")
                            for tag, label, ctype, vmin, vmax in MIDI_CONTROLS:
                                if label == sel:
                                    midi.start_learn(tag, ctype, vmin, vmax)
                                    return

                        def _cancel_learn():
                            midi.cancel_learn()

                        def _clear_mapping():
                            sel = dpg.get_value("midi_target_combo")
                            for tag, label, *_ in MIDI_CONTROLS:
                                if label == sel:
                                    midi.remove_mapping(tag)
                                    return

                        dpg.add_combo(items=[], tag="midi_port_combo", label="Port", callback=_connect_port, width=-80)
                        dpg.add_button(label="Refresh Ports", callback=lambda: _refresh_ports(), width=-1)
                        dpg.add_text("disconnected", tag="midi_status", color=(180, 180, 180))
                        dpg.add_separator()
                        dpg.add_combo(
                            items=[label for _, label, *_ in MIDI_CONTROLS],
                            tag="midi_target_combo",
                            label="Control",
                            default_value=MIDI_CONTROLS[0][1],
                            width=-80,
                        )
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Listen", callback=lambda: _start_learn())
                            dpg.add_button(label="Cancel", callback=lambda: _cancel_learn())
                            dpg.add_button(label="Clear", callback=lambda: _clear_mapping())
                        dpg.add_text("", tag="midi_event", color=(200, 200, 120), wrap=320)

                        _refresh_ports()
                        initial_ports = [p for p in MidiController.list_inputs() if "Through" not in p]
                        if initial_ports:
                            midi.open(initial_ports[0])
                            dpg.set_value("midi_port_combo", initial_ports[0])
                            dpg.set_value("midi_status", f"connected: {initial_ports[0]}")
            with dpg.group():
                dpg.add_image("ai_tex", width=DISPLAY_W, height=DISPLAY_H)

dpg.create_viewport(title="113.Recursive AI VJ — RV6", width=350 + DISPLAY_W + 80, height=DISPLAY_H + 100)
dpg.setup_dearpygui()
dpg.show_viewport()
# Pin the main window to the viewport so DPG doesn't render its own
# title bar + frame inside the OS window (the "window in a window" look).
dpg.set_primary_window("main_window", True)

# ---------------------------------------------------------------------------
# RESTORE WIDGET STATE FROM settings.json
# ---------------------------------------------------------------------------
# Tags here are restored via dpg.set_value, which does NOT fire callbacks —
# anything that's callback-driven (state dict mirrors, dreamer setters, the
# SD engine swap, audio device select) is re-synced explicitly below.
_PERSISTED_WIDGETS = (
    "strength_sl", "delta_sl", "audio_smooth_sl",
    "zoom_base_sl", "zoom_sens_sl", "rot_base_sl", "rot_sens_sl",
    "kaleido_mode_combo", "kaleido_band_combo",
    "kaleido_base_sl", "kaleido_sens_sl",
    "interval_sl", "auto_sw_cb", "shuffle_cb", "bypass_ai_cb",
    "brightness_sl", "contrast_sl", "saturation_sl", "gamma_sl",
    "deep_dream_cb", "dream_cpu_cb", "dream_model_cb", "dream_temp_sl",
    "dream_keywords_in", "record_audio_cb", "audio_source_combo",
    "upscale_profile_combo",
    "neg_prompt_in", "neg_strength_sl",
)
for _tag in _PERSISTED_WIDGETS:
    if _tag in _SAVED and dpg.does_item_exist(_tag):
        try:
            dpg.set_value(_tag, _SAVED[_tag])
        except Exception as e:
            print(f"[settings] restore {_tag}: {e}")

state["auto_switch"] = dpg.get_value("auto_sw_cb")
state["shuffle"] = dpg.get_value("shuffle_cb")
state["deep_dream"] = dpg.get_value("deep_dream_cb")
state["switch_interval"] = dpg.get_value("interval_sl")
dreamer.set_temperature(dpg.get_value("dream_temp_sl"))
dreamer.set_keywords(dpg.get_value("dream_keywords_in"))
dreamer.set_model(dpg.get_value("dream_model_cb"))
dreamer.set_cpu_only(dpg.get_value("dream_cpu_cb"))
# Mirror the side effect of _toggle_dream: setting the checkbox via
# dpg.set_value above does NOT fire its callback, so without this kick
# the dreamer never gets a request and dream_text sits at "(idle)".
if state["deep_dream"] and dreamer.latest_dream is None:
    dreamer.request_dream()

# SD engine: only swap if the saved engine differs from the startup pick and
# still exists on disk. Engines can be deleted between sessions.
_saved_engine = _SAVED.get("sd_model_combo")
if _saved_engine and _saved_engine in _engine_paths and _saved_engine != STARTUP_ENGINE_NAME:
    dpg.set_value("sd_model_combo", _saved_engine)
    _swap_sd_model(None, _saved_engine)
dpg.set_value("prompt_in", state["current_p"])
update_embeds(state["current_p"])
if dpg.does_item_exist("neg_prompt_in"):
    state["neg_p"] = dpg.get_value("neg_prompt_in")
update_neg_embeds(state["neg_p"])

# Combos with dynamic item lists: only set the value if it's still in the
# current items, else fall back to whatever the refresh helper picked.
_saved_audio = _SAVED.get("audio_in_combo")
if _saved_audio and dpg.does_item_exist("audio_in_combo"):
    if _saved_audio in dpg.get_item_configuration("audio_in_combo").get("items", []):
        dpg.set_value("audio_in_combo", _saved_audio)
        _select_audio_device(None, _saved_audio)

_saved_mon = _SAVED.get("fs_monitor_combo")
if _saved_mon and dpg.does_item_exist("fs_monitor_combo"):
    if _saved_mon in dpg.get_item_configuration("fs_monitor_combo").get("items", []):
        dpg.set_value("fs_monitor_combo", _saved_mon)

# Upscaler: restore the on/off intent but actually load the model lazily,
# matching _toggle_upscaler's behavior. A load failure shouldn't block startup.
if _SAVED.get("upscale_cb"):
    try:
        upscaler.load()
        upscaler.enabled = True
        dpg.set_value("upscale_cb", True)
    except Exception as e:
        dpg.set_value("upscale_cb", False)
        print(f"[settings] upscaler load failed: {e}")


def _collect_settings() -> dict:
    out = {tag: dpg.get_value(tag) for tag in _PERSISTED_WIDGETS if dpg.does_item_exist(tag)}
    out["prompts"] = list(state["prompts"])
    out["current_p_idx"] = state["current_p_idx"]
    out["current_p"] = state["current_p"]
    for tag in ("sd_model_combo", "audio_in_combo", "fs_monitor_combo", "upscale_cb"):
        if dpg.does_item_exist(tag):
            out[tag] = dpg.get_value(tag)
    return out


_last_save = time.time()

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
# Recursive state stays on GPU between frames — only crosses to CPU for the
# UI texture upload at the bottom of the loop.
ai_tensor = torch.rand(1, 3, HEIGHT, WIDTH, device="cuda", dtype=torch.float16)
ai_rgba = np.ones((HEIGHT, WIDTH, 4), dtype=np.float32)

print("VJ SYSTEM ONLINE (RECURSIVE MODE)!")
fps_frames = 0
fps_t0 = time.time()
while dpg.is_dearpygui_running():
    if state["auto_switch"] and (time.time() - state["last_switch"] > state["switch_interval"]):
        next_prompt()

    apply_midi_updates()

    audio.smoothing_factor = dpg.get_value("audio_smooth_sl")
    bands = audio.get_bands()

    viz_config = {
        'zoom_base':     dpg.get_value("zoom_base_sl"),
        'zoom_sens':     dpg.get_value("zoom_sens_sl"),
        'rot_base':      dpg.get_value("rot_base_sl"),
        'rot_sens':      dpg.get_value("rot_sens_sl"),
        'kaleido_mode':  dpg.get_value("kaleido_mode_combo"),
        'kaleido_band':  dpg.get_value("kaleido_band_combo"),
        'kaleido_base':  dpg.get_value("kaleido_base_sl"),
        'kaleido_sens':  dpg.get_value("kaleido_sens_sl"),
    }
    img_tensor = viz.apply_feedback(ai_tensor, bands, viz_config)
    warp_diff = (img_tensor - ai_tensor).abs().mean().item()

    if dpg.get_value("bypass_ai_cb"):
        ai_tensor = img_tensor
        ai_diff = 0.0
    else:
        strength = dpg.get_value("strength_sl")
        delta    = dpg.get_value("delta_sl")
        neg_strength = dpg.get_value("neg_strength_sl")
        ai_tensor = ai.step(
            img_tensor, state["embeds"], strength, delta,
            neg_embeds=state["neg_embeds"], neg_strength=neg_strength,
        )
        ai_diff = (ai_tensor - img_tensor).abs().mean().item()

    # Apply color decoration to the DISPLAY path only — ai_tensor stays raw
    # so the recursive feedback loop doesn't compound a brightness boost
    # (which would saturate to white in a few frames).
    display_tensor = adjust_color(
        ai_tensor,
        brightness=dpg.get_value("brightness_sl"),
        contrast=dpg.get_value("contrast_sl"),
        saturation=dpg.get_value("saturation_sl"),
        gamma=dpg.get_value("gamma_sl"),
    )

    # Single GPU→CPU copy per frame for the UI texture (always at native res).
    ai_rgba[:, :, :3] = display_tensor[0].permute(1, 2, 0).float().cpu().numpy()
    dpg.set_value("ai_tex", ai_rgba.ravel())

    # Downstream consumers (recorder, fullscreen) want the upscaled frame
    # when the upscaler is on, the color-adjusted native frame otherwise.
    # The DPG preview always stays at native res — no point spending GPU on
    # a higher-res preview that just gets squashed back into the UI thumbnail.
    needs_output_frame = recorder.is_recording or fullscreen.is_open()
    output_rgb = None
    if needs_output_frame:
        if upscaler.enabled and upscaler.is_loaded():
            # Upscale the color-adjusted frame so the projected output reflects
            # the slider state, not the raw (often dim) AI output.
            upscaled = upscaler.upscale(display_tensor)
            output_rgb = (
                upscaled[0].permute(1, 2, 0).float().cpu().numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)
        else:
            output_rgb = (ai_rgba[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)

    if recorder.is_recording and output_rgb is not None:
        recorder.write_frame(output_rgb)

    if fullscreen.is_open() and output_rgb is not None:
        fullscreen.update(output_rgb)
    
    dpg.render_dearpygui_frame()

    # Cheap (<1 ms JSON dump of a small dict) but no need to do it every
    # frame — once every 10 s caps disk wear and still bounds crash loss.
    if time.time() - _last_save > 10.0:
        settings_io.save(_collect_settings())
        _last_save = time.time()

    fps_frames += 1
    if fps_frames >= 30:
        now = time.time()
        fps = fps_frames / max(now - fps_t0, 1e-6)
        b, m, h = bands
        zoom_live = dpg.get_value("zoom_base_sl") - (b * 0.25 * dpg.get_value("zoom_sens_sl"))
        rot_live  = dpg.get_value("rot_base_sl")  + (m * 0.15 * dpg.get_value("rot_sens_sl"))
        line = (
            f"{fps:.1f} fps | bass {b:.2f} mid {m:.2f} hi {h:.2f} "
            f"| zoom {zoom_live:.3f} rot {rot_live:+.3f} "
            f"| warp_diff {warp_diff:.4f} ai_diff {ai_diff:.4f}"
        )
        dpg.set_viewport_title(line)
        print(line, flush=True)
        fps_frames = 0
        fps_t0 = now

        if dpg.does_item_exist("dream_status"):
            if dreamer.error:
                dpg.set_value("dream_status", f"⚠ {dreamer.error}")
            elif dreamer.is_dreaming:
                dpg.set_value("dream_status", "dreaming…")
            else:
                dpg.set_value("dream_status", "")

        # Show the latest dream as soon as it's produced, not when it's
        # eventually consumed by next_prompt — otherwise with auto-switch
        # off the panel sits at "(idle)" even though a dream is ready.
        if dpg.does_item_exist("dream_text"):
            latest = dreamer.latest_dream
            if latest and dpg.get_value("dream_text") != latest:
                dpg.set_value("dream_text", latest)

        if dpg.does_item_exist("record_status"):
            if recorder.is_recording:
                dpg.set_value("record_status", f"REC ● {int(recorder.elapsed())}s")
            else:
                dpg.set_value("record_status", recorder.last_message)

        if dpg.does_item_exist("fs_toggle_btn"):
            expected_label = "Close Fullscreen" if fullscreen.is_open() else "Open Fullscreen"
            if dpg.get_item_label("fs_toggle_btn") != expected_label:
                dpg.set_item_label("fs_toggle_btn", expected_label)
                dpg.set_value(
                    "fs_status",
                    "open" if fullscreen.is_open() else "closed (ESC pressed)",
                )

        if dpg.does_item_exist("midi_event"):
            if midi.is_learning():
                dpg.set_value("midi_event", f"learning {midi.learn_target_tag()} — touch a MIDI control")
            else:
                dpg.set_value("midi_event", midi.last_event)

settings_io.save(_collect_settings())
dpg.destroy_context()
audio.close()
dreamer.shutdown()
midi.close()
fullscreen.close()
if recorder.is_recording:
    recorder.stop()
