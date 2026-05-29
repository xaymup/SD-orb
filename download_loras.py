"""Download the curated SD 1.5 LoRA pack from Civitai.

Usage:
    python download_loras.py                       # download everything missing
    python download_loras.py --only ghibli,watercolor
    python download_loras.py --skip charcoal,pixelart_8bit
    CIVITAI_TOKEN=<token> python download_loras.py # if any model needs auth

Each LoRA is saved as loras/<short_name>.safetensors so build commands stay
short, e.g. `python builder.py dreamshaper8 --lora ghibli --style "cozy fantasy"`.

The script fetches model metadata via Civitai's public API, picks the latest
version flagged baseModel='SD 1.5', and prefers the .safetensors file.
"""

import argparse
import os
import sys
import time

import requests

LORA_DIR = "loras"

# Curated SD 1.5 LoRA pack: (civitai_model_id, save_as, default_scale, note).
# Model IDs were verified via web search; if Civitai ever migrates URLs the
# numeric IDs stay stable.
PACK: list[tuple[int, str, float, str]] = [
    # === Utility boosters (subtle quality lift; bake into many engines at low scale) ===
    (82098,  "more_details",      0.5, "Lykon's Add More Details — sharper edges, more texture"),
    (13941,  "epinoise",          0.5, "epi_noiseoffset — better contrast and darker exposure"),
    (58390,  "detail_tweaker",    0.5, "CyberAIchemist's Detail Tweaker — granular detail control"),

    # === Style LoRAs ===
    (6526,   "ghibli",            0.8, "Studio Ghibli style (offset version)"),
    (64560,  "watercolor",        0.8, "WATERCOLOR by Clumsy_Trainer (SD 1.5)"),
    (155490, "pencil_sketch",     0.9, "Pencil sketch with real stroke textures"),
    (300324, "charcoal",          0.9, "Dark Charcoal Style — SD1.5 + SDXL"),
    (977315, "cyberpunk_space",   0.8, "Cyberpunk Style by malibu79617 — SD 1.5 (older diffusers format)"),
    (159705, "pixelart_8bit",     0.9, "Pixel Art Style"),
]


def fetch_model_info(model_id: int, token: str | None) -> dict:
    url = f"https://civitai.com/api/v1/models/{model_id}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_sd15_version(model_info: dict) -> tuple[int, str, str] | None:
    """Find the most recent SD 1.5 version. Returns (version_id, file_name,
    download_url). Returns None if the model has no SD 1.5 version (some
    are SDXL-only despite the search hit)."""
    for v in model_info.get("modelVersions", []):
        base = str(v.get("baseModel", "")).lower()
        if "sd 1" not in base and base not in ("sd1.5", "sd 1.5"):
            continue
        files = v.get("files", [])
        safe = [f for f in files if str(f.get("name", "")).endswith(".safetensors")]
        chosen = safe[0] if safe else (files[0] if files else None)
        if not chosen:
            continue
        return v.get("id"), chosen.get("name", "<unknown>"), chosen.get("downloadUrl")
    return None


def download(url: str, out_path: str, token: str | None) -> None:
    """Stream to .part then atomic rename so an interrupted download doesn't
    leave a half-file the next run would treat as already present."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".part"
        downloaded = 0
        last_print = 0.0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print > 0.5:
                    if total > 0:
                        pct = downloaded * 100.0 / total
                        sys.stdout.write(
                            f"\r  {downloaded/1e6:6.1f} / {total/1e6:6.1f} MB ({pct:5.1f}%)"
                        )
                    else:
                        sys.stdout.write(f"\r  {downloaded/1e6:6.1f} MB")
                    sys.stdout.flush()
                    last_print = now
        os.rename(tmp_path, out_path)
        sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download curated SD 1.5 LoRA pack from Civitai.")
    parser.add_argument(
        "--token", default=os.environ.get("CIVITAI_TOKEN"),
        help="Civitai API token (or set CIVITAI_TOKEN). Required for some auth-gated LoRAs.",
    )
    parser.add_argument("--only", default="", help="Comma-separated short names to download (others skipped)")
    parser.add_argument("--skip", default="", help="Comma-separated short names to skip")
    parser.add_argument("--list", action="store_true", help="List the pack and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'name':<20} {'scale':<6} {'civitai':<8} note")
        print("-" * 80)
        for mid, name, scale, note in PACK:
            print(f"{name:<20} {scale:<6} {mid:<8} {note}")
        return 0

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    os.makedirs(LORA_DIR, exist_ok=True)

    succeeded: list[str] = []
    already: list[str] = []
    failed: list[tuple[str, str]] = []

    for model_id, name, scale, note in PACK:
        if only and name not in only:
            continue
        if name in skip:
            continue
        out_path = os.path.join(LORA_DIR, f"{name}.safetensors")
        if os.path.exists(out_path):
            print(f"✓ {name} (already present)")
            already.append(name)
            continue
        print(f"\n→ {name} (civitai {model_id}, scale {scale}): {note}")
        try:
            info = fetch_model_info(model_id, args.token)
            pick = pick_sd15_version(info)
            if pick is None:
                raise RuntimeError("no SD 1.5 version on this model page")
            version_id, fname, dl_url = pick
            print(f"  version {version_id} ({fname})")
            download(dl_url, out_path, args.token)
            print(f"  ✓ saved to {out_path}")
            succeeded.append(name)
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
            failed.append((name, str(e)))

    print(f"\n{'='*60}")
    print(f"  downloaded: {len(succeeded)}")
    print(f"  already had: {len(already)}")
    print(f"  failed:     {len(failed)}")
    if failed:
        print("\nFailures:")
        for n, r in failed:
            print(f"  {n:<20} {r}")
        print(
            "\nHint: 401/403 errors mean the model requires a Civitai login. "
            "Get a free API token at https://civitai.com/user/account "
            "(API Keys → Add API key), then:"
            "\n  CIVITAI_TOKEN=<token> python download_loras.py"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
