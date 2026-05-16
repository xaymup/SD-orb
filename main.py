import os
import time
import torch
import numpy as np
import pygame
import dearpygui.dearpygui as dpg

from audio_analyzer import AudioAnalyzer
from visualizer import Visualizer
from pipeline import AIPipeline

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 512, 512
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
}
state["current_p"] = state["prompts"][0]

# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()

print("Initializing 113.Milkdrop AI Orchestrator...")
audio = AudioAnalyzer()
viz = Visualizer(WIDTH, HEIGHT)
ai = AIPipeline(MODEL_PATH, ENGINE_PATH)

def update_embeds(prompt):
    state["embeds"] = ai.get_embeds(prompt)
    state["current_p"] = prompt

def next_prompt():
    if not state["prompts"]: return
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

with dpg.window(label="113.RECURSIVE AI — Modular RV6", width=900, height=650):
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

                dpg.add_slider_float(label="AI Strength",     tag="strength_sl", min_value=0.0, max_value=1.0,  default_value=0.6)
                dpg.add_slider_float(label="Temporal Smooth", tag="delta_sl",    min_value=0.0, max_value=0.98, default_value=0.85)

                with dpg.collapsing_header(label="Feedback Engine", default_open=True):
                    dpg.add_slider_float(label="Zoom Base",      tag="zoom_base_sl",  min_value=0.8, max_value=1.1,  default_value=0.98)
                    dpg.add_slider_float(label="Zoom React",    tag="zoom_sens_sl",  min_value=0.0, max_value=2.0,  default_value=1.0)
                    dpg.add_slider_float(label="Rotate Base",    tag="rot_base_sl",   min_value=-0.1, max_value=0.1, default_value=0.01)
                    dpg.add_slider_float(label="Rotate React",  tag="rot_sens_sl",   min_value=0.0, max_value=2.0,  default_value=1.0)
            with dpg.group():
                dpg.add_image("ai_tex")

dpg.create_viewport(title="113.Recursive AI VJ — RV6", width=950, height=700)
dpg.setup_dearpygui()
dpg.show_viewport()

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
ai_arr = np.random.rand(HEIGHT, WIDTH, 3).astype(np.float32)
ai_rgba = np.ones((HEIGHT, WIDTH, 4), dtype=np.float32)

print("VJ SYSTEM ONLINE (RECURSIVE MODE)!")
while dpg.is_dearpygui_running():
    # Prompt Auto-Switch Logic
    if state["auto_switch"] and (time.time() - state["last_switch"] > state["switch_interval"]):
        next_prompt()

    # 1. Audio
    bands = audio.get_bands()
    
    # 2. Visualization (Recursive Feedback)
    viz_config = {
        'zoom_base': dpg.get_value("zoom_base_sl"),
        'zoom_sens': dpg.get_value("zoom_sens_sl"),
        'rot_base':  dpg.get_value("rot_base_sl"),
        'rot_sens':  dpg.get_value("rot_sens_sl")
    }
    # Apply feedback to the PREVIOUS AI output
    img_array = viz.apply_feedback(ai_arr, bands, viz_config)

    # 3. AI Inference
    strength = dpg.get_value("strength_sl")
    delta    = dpg.get_value("delta_sl")
    
    # Pre-process image for AI (convert to tensor, normalize)
    input_image = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) * 2.0 - 1.0
    
    # Get NEW AI output
    ai_arr = ai.step(input_image, state["embeds"], strength, delta)

    # 4. Update UI Texture
    ai_rgba[:, :, :3] = ai_arr
    dpg.set_value("ai_tex", ai_rgba.ravel())
    
    dpg.render_dearpygui_frame()

dpg.destroy_context()
audio.close()
pygame.quit()
