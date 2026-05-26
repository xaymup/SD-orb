import time
import torch
import numpy as np
import dearpygui.dearpygui as dpg

from audio_analyzer import AudioAnalyzer
from visualizer import Visualizer
from pipeline import AIPipeline
from dreamer import Dreamer
from recorder import Recorder

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 384, 384  # was 512 — requires builder.py engine rebuild
MODEL_PATH = "models/realisticVisionV60B1_v51HyperVAE.safetensors"
ENGINE_PATH = "engines/unet.engine"

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
                            recorder.start(include_audio=dpg.get_value("record_audio_cb"))
                            dpg.set_item_label("record_btn", "■ Stop Recording")
                    dpg.add_checkbox(label="Include audio (PulseAudio default)", tag="record_audio_cb", default_value=True)
                    dpg.add_button(label="● Start Recording", tag="record_btn", callback=_toggle_record, width=-1)
                    dpg.add_text("", tag="record_status", color=(220, 120, 120))

                with dpg.collapsing_header(label="Deep Dream", default_open=False):
                    def _toggle_dream(s, a):
                        state["deep_dream"] = a
                        if a and dreamer.latest_dream is None:
                            dreamer.request_dream()
                    dpg.add_checkbox(label="Enable Deep Dream", tag="deep_dream_cb", default_value=False, callback=_toggle_dream)
                    dpg.add_combo(items=DREAM_MODELS, default_value=DREAM_MODELS[0], tag="dream_model_cb", callback=lambda s, a: dreamer.set_model(a), width=-1)
                    dpg.add_slider_float(label="Dream Temp", tag="dream_temp_sl", min_value=0.3, max_value=1.6, default_value=1.1, callback=lambda s, a: dreamer.set_temperature(a))
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

dpg.destroy_context()
audio.close()
dreamer.shutdown()
if recorder.is_recording:
    recorder.stop()
