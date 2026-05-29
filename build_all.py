"""Batch-build every engine in recipes.py that isn't already on disk.

Usage:
    python build_all.py                       # build everything missing
    python build_all.py --only realisticvision+detail,dreamshaper8+ghibli
    python build_all.py --skip inkpunk+sketch,inkpunk+charcoal
    python build_all.py --list                # list recipes + status, no build
    python build_all.py --rebuild realisticvision+detail   # wipe & rebuild

Each build takes ~10-20 min on a 4090-class GPU and ~1.5 GB of disk. Plan
on a few hours if you're building the full pack from scratch.
"""

import argparse
import os
import shutil
import subprocess
import sys

from recipes import RECIPES


def engine_exists(name: str) -> bool:
    return os.path.isfile(f"engines/{name}/unet.engine")


def wipe_engine(name: str) -> None:
    p = f"engines/{name}"
    if os.path.isdir(p):
        shutil.rmtree(p)


def build_one(recipe: dict) -> bool:
    name = recipe["name"]
    cmd = [sys.executable, "builder.py", recipe["base"], "--name", name]
    for lora_name, scale in recipe.get("loras", []):
        cmd += ["--lora", f"{lora_name}:{scale}"]
    if recipe.get("style"):
        cmd += ["--style", recipe["style"]]
    print(f"\n{'='*70}\n>>> Building {name}\n>>> {' '.join(cmd)}\n{'='*70}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Build failed for {name}: exit {e.returncode}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-build engines from recipes.py.")
    parser.add_argument("--only", default="", help="Comma-separated recipe names to build (others skipped)")
    parser.add_argument("--skip", default="", help="Comma-separated recipe names to skip")
    parser.add_argument("--rebuild", default="", help="Comma-separated recipe names to wipe and rebuild")
    parser.add_argument("--list", action="store_true", help="List recipes + status and exit")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    rebuild = {s.strip() for s in args.rebuild.split(",") if s.strip()}

    if args.list:
        print(f"{'status':<10} {'name':<32} {'base':<18} loras / style")
        print("-" * 100)
        for r in RECIPES:
            status = "BUILT" if engine_exists(r["name"]) else "missing"
            loras = ", ".join(f"{n}@{s}" for n, s in r.get("loras", [])) or "—"
            print(f"{status:<10} {r['name']:<32} {r['base']:<18} {loras}")
            if r.get("style"):
                print(f"{'':<62} style: {r['style']}")
        return 0

    # Wipe anything requested first so the existence check below treats them
    # as missing and builds them.
    for name in rebuild:
        wipe_engine(name)
        print(f"wiped engines/{name}")

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for r in RECIPES:
        name = r["name"]
        if only and name not in only:
            continue
        if name in skip:
            continue
        if engine_exists(name):
            print(f"✓ {name} (already built)")
            skipped.append(name)
            continue
        if build_one(r):
            succeeded.append(name)
        else:
            failed.append(name)

    print(f"\n{'='*70}")
    print(f"  built:   {len(succeeded)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed:  {len(failed)}")
    if failed:
        print("  failed names: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
