"""Engine variants to build via build_all.py.

Each recipe produces:
    engines/<name>/unet.engine   (TensorRT UNet)
    engines/<name>/manifest.json (base + loras + style for the runtime)

Add or remove entries here; build_all.py will skip any engine that's
already on disk so you can iterate without rebuilding everything.

Fields:
    name   — directory name under engines/. Also what shows in the SD Model
             dropdown. Convention: <base>+<lora>+<lora>... but you can use
             anything. The build script passes this as --name to builder.py.
    base   — model_name in models/ (without extension). models/<base>.safetensors
    loras  — list of (lora_name, scale). lora_name is the file in loras/
             without the .safetensors extension.
    style  — free-text aesthetic hint stored in manifest.json. The dreamer
             injects this into its system prompt when this engine is active.
"""

RECIPES: list[dict] = [
    # ===== Plain engines (already built; here for completeness so recipes.py
    # is a single source of truth). build_all.py will skip these. =====
    {"name": "realisticvision", "base": "realisticvision", "loras": [],
     "style": "photorealistic, cinematic framing, naturalistic lighting"},
    {"name": "dreamshaper8", "base": "dreamshaper8", "loras": [],
     "style": "painterly surreal, vivid fantasy, dramatic lighting"},
    {"name": "meinamix", "base": "meinamix", "loras": [],
     "style": "anime illustration, vibrant cel shading, soft pastel palette"},

    # ===== Utility-boosted realism (subtle detail/contrast lift) =====
    {"name": "realisticvision+detail", "base": "realisticvision",
     "loras": [("more_details", 0.5), ("epinoise", 0.5)],
     "style": "photorealistic, punchy detail, cinematic contrast"},

    # ===== Painterly fantasy =====
    {"name": "dreamshaper8+watercolor", "base": "dreamshaper8",
     "loras": [("watercolor", 0.8)],
     "style": "loose watercolor wash, painterly fantasy, soft edges"},

    {"name": "dreamshaper8+ghibli", "base": "dreamshaper8",
     "loras": [("ghibli", 0.8)],
     "style": "cozy ghibli fantasy, soft pastels, warm light"},

    # ===== Anime variations =====
    {"name": "meinamix+pixel", "base": "meinamix",
     "loras": [("pixelart_8bit", 0.9)],
     "style": "anime pixel art, lo-fi sprites, retro game aesthetic"},

    {"name": "meinamix+sketch", "base": "meinamix",
     "loras": [("pencil_sketch", 0.8)],
     "style": "anime pencil sketch, hand-drawn linework"},

    {"name": "meinamix+charcoal", "base": "meinamix",
     "loras": [("charcoal", 0.7)],
     "style": "anime in charcoal, rough graphic strokes, high contrast"},
]
