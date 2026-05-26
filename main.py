import os
import time
import torch
import numpy as np
import dearpygui.dearpygui as dpg

from audio_analyzer import AudioAnalyzer
from visualizer import Visualizer
from pipeline import AIPipeline
from dreamer import Dreamer
from recorder import Recorder
from midi import MidiController

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 384, 384  # was 512 — requires builder.py engine rebuild
MODEL_PATH = "models/realisticVisionV60B1_v51HyperVAE.safetensors"
ENGINE_PATH = "engines/unet.engine"


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
}
state["current_p"] = state["prompts"][0]

# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------
print("Initializing 113.Milkdrop AI Orchestrator...")
audio = AudioAnalyzer()
viz = Visualizer(WIDTH, HEIGHT)
ai = AIPipeline(MODEL_PATH, ENGINE_PATH, width=WIDTH, height=HEIGHT)
dreamer = Dreamer()
recorder = Recorder(WIDTH, HEIGHT, fps=24)
midi = MidiController()

# (tag, label, type, min, max). Buttons use ("trigger" routing in apply_midi).
MIDI_CONTROLS = [
    ("strength_sl",     "AI Strength",      "slider",   0.0,  1.0),
    ("delta_sl",        "Temporal Smooth",  "slider",   0.0,  0.98),
    ("audio_smooth_sl", "Audio Smooth",     "slider",   0.0,  0.99),
    ("zoom_base_sl",    "Zoom Base",        "slider",   0.8,  1.1),
    ("zoom_sens_sl",    "Zoom React",       "slider",   0.0,  2.0),
    ("rot_base_sl",     "Rotate Base",      "slider",  -0.1,  0.1),
    ("rot_sens_sl",     "Rotate React",     "slider",   0.0,  2.0),
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
]

DREAM_MODELS = [
    "llama3.1:8b",
    "mistral-nemo:12b",
    "qwen3.5:9b",
    "qwen3:14b",
    "gemma4:latest",
    "deepseek-r1:14b",
    "gpt-oss:20b",
]

def update_embeds(prompt):
    state["embeds"] = ai.get_embeds(prompt)
    state["current_p"] = prompt

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
            if tag == "next_prompt":
                next_prompt()
            elif tag == "record_btn":
                if recorder.is_recording:
                    recorder.stop()
                    if dpg.does_item_exist("record_btn"):
                        dpg.set_item_label("record_btn", "● Start Recording")
                else:
                    recorder.start(
                        include_audio=dpg.get_value("record_audio_cb"),
                        audio_source=dpg.get_value("audio_source_combo"),
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

with dpg.window(label="113.RECURSIVE AI — Modular RV6", width=350 + DISPLAY_W + 40, height=DISPLAY_H + 60):
    with dpg.table(header_row=False):
        dpg.add_table_column(width_fixed=True, init_width_or_weight=350)
        dpg.add_table_column()
        with dpg.table_row():
            with dpg.group():
                dpg.add_input_text(tag="prompt_in", default_value=state["current_p"], callback=lambda s, a: update_embeds(a))
                
                with dpg.collapsing_header(label="Prompt Playlist", default_open=True):
                    dpg.add_checkbox(label="Auto Switch", tag="auto_sw_cb", default_value=state["auto_switch"], callback=lambda s, a: state.update({"auto_switch": a}))
                    dpg.add_slider_float(label="Interval (s)", tag="interval_sl", min_value=5.0, max_value=300.0, default_value=state["switch_interval"], callback=lambda s, a: state.update({"switch_interval": a}))
                    dpg.add_checkbox(label="Shuffle", tag="shuffle_cb", default_value=state["shuffle"], callback=lambda s, a: state.update({"shuffle": a}))
                    dpg.add_button(label="Next Prompt", callback=next_prompt, width=-1)
                    
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="new_prompt_in", hint="Add new prompt...", width=250)
                        dpg.add_button(label="+", callback=add_prompt)
                    
                    with dpg.group(tag="prompt_list_group"):
                        pass 
                
                refresh_prompt_list()

                # SD Model picker — hot-swaps the TRT engine in place.
                _engines = discover_engines()
                _engine_names = [n for n, _ in _engines] or ["(none)"]
                _engine_paths = {n: p for n, p in _engines}
                _current_engine_name = next(
                    (n for n, p in _engines if os.path.abspath(p) == os.path.abspath(ENGINE_PATH)),
                    _engine_names[0],
                )

                def _swap_sd_model(s, a):
                    path = _engine_paths.get(a)
                    if not path:
                        return
                    dpg.set_value("sd_model_status", f"loading {a}…")
                    try:
                        ai.swap_engine(path)
                        dpg.set_value("sd_model_status", f"active: {a}")
                    except Exception as e:
                        dpg.set_value("sd_model_status", f"swap failed: {e}")

                dpg.add_combo(
                    items=_engine_names, default_value=_current_engine_name,
                    label="SD Model", tag="sd_model_combo",
                    callback=_swap_sd_model, width=-80,
                )
                dpg.add_text(f"active: {_current_engine_name}", tag="sd_model_status", color=(160, 200, 160))
                dpg.add_separator()

                dpg.add_slider_float(label="AI Strength",     tag="strength_sl", min_value=0.0, max_value=1.0,  default_value=0.3)
                dpg.add_slider_float(label="Temporal Smooth", tag="delta_sl",    min_value=0.0, max_value=0.98, default_value=0.4)
                dpg.add_checkbox(label="Bypass AI (raw warp feedback)", tag="bypass_ai_cb", default_value=False)

                with dpg.collapsing_header(label="Feedback Engine", default_open=True):
                    dpg.add_slider_float(label="Audio Smooth",   tag="audio_smooth_sl", min_value=0.0, max_value=0.99, default_value=0.7)
                    dpg.add_slider_float(label="Zoom Base",      tag="zoom_base_sl",  min_value=0.8, max_value=1.1,  default_value=0.98)
                    dpg.add_slider_float(label="Zoom React",    tag="zoom_sens_sl",  min_value=0.0, max_value=2.0,  default_value=1.0)
                    dpg.add_slider_float(label="Rotate Base",    tag="rot_base_sl",   min_value=-0.1, max_value=0.1, default_value=0.01)
                    dpg.add_slider_float(label="Rotate React",  tag="rot_sens_sl",   min_value=0.0, max_value=2.0,  default_value=1.0)

                with dpg.collapsing_header(label="Recording", default_open=False):
                    def _toggle_record(s, a, u):
                        if recorder.is_recording:
                            recorder.stop()
                            dpg.set_item_label("record_btn", "● Start Recording")
                        else:
                            recorder.start(
                                include_audio=dpg.get_value("record_audio_cb"),
                                audio_source=dpg.get_value("audio_source_combo"),
                            )
                            dpg.set_item_label("record_btn", "■ Stop Recording")
                    dpg.add_checkbox(label="Include audio", tag="record_audio_cb", default_value=True)
                    dpg.add_combo(items=["system", "mic"], default_value="system",
                                  label="Audio source", tag="audio_source_combo")
                    dpg.add_button(label="● Start Recording", tag="record_btn", callback=_toggle_record, width=-1)
                    dpg.add_text("", tag="record_status", color=(220, 120, 120))

                with dpg.collapsing_header(label="MIDI Learn", default_open=False):
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

                    # Initialise port list and auto-connect to the first non-Through input.
                    _refresh_ports()
                    initial_ports = [p for p in MidiController.list_inputs() if "Through" not in p]
                    if initial_ports:
                        midi.open(initial_ports[0])
                        dpg.set_value("midi_port_combo", initial_ports[0])
                        dpg.set_value("midi_status", f"connected: {initial_ports[0]}")

                with dpg.collapsing_header(label="Deep Dream", default_open=False):
                    def _toggle_dream(s, a):
                        state["deep_dream"] = a
                        if a and dreamer.latest_dream is None:
                            dreamer.request_dream()
                    dpg.add_checkbox(label="Enable Deep Dream", tag="deep_dream_cb", default_value=False, callback=_toggle_dream)
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
            with dpg.group():
                dpg.add_image("ai_tex", width=DISPLAY_W, height=DISPLAY_H)

dpg.create_viewport(title="113.Recursive AI VJ — RV6", width=350 + DISPLAY_W + 80, height=DISPLAY_H + 100)
dpg.setup_dearpygui()
dpg.show_viewport()

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
        'zoom_base': dpg.get_value("zoom_base_sl"),
        'zoom_sens': dpg.get_value("zoom_sens_sl"),
        'rot_base':  dpg.get_value("rot_base_sl"),
        'rot_sens':  dpg.get_value("rot_sens_sl"),
    }
    img_tensor = viz.apply_feedback(ai_tensor, bands, viz_config)
    warp_diff = (img_tensor - ai_tensor).abs().mean().item()

    if dpg.get_value("bypass_ai_cb"):
        ai_tensor = img_tensor
        ai_diff = 0.0
    else:
        strength = dpg.get_value("strength_sl")
        delta    = dpg.get_value("delta_sl")
        ai_tensor = ai.step(img_tensor, state["embeds"], strength, delta)
        ai_diff = (ai_tensor - img_tensor).abs().mean().item()

    # Single GPU→CPU copy per frame, just for the UI texture.
    ai_rgba[:, :, :3] = ai_tensor[0].permute(1, 2, 0).float().cpu().numpy()
    dpg.set_value("ai_tex", ai_rgba.ravel())

    if recorder.is_recording:
        # rgb24 frame for ffmpeg stdin. Reuse the already-on-CPU RGB slice.
        recorder.write_frame((ai_rgba[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8))
    
    dpg.render_dearpygui_frame()

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

        if dpg.does_item_exist("record_status"):
            if recorder.is_recording:
                dpg.set_value("record_status", f"REC ● {int(recorder.elapsed())}s")
            else:
                dpg.set_value("record_status", recorder.last_message)

        if dpg.does_item_exist("midi_event"):
            if midi.is_learning():
                dpg.set_value("midi_event", f"learning {midi.learn_target_tag()} — touch a MIDI control")
            else:
                dpg.set_value("midi_event", midi.last_event)

dpg.destroy_context()
audio.close()
dreamer.shutdown()
midi.close()
if recorder.is_recording:
    recorder.stop()
