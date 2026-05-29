import json
import os

PATH = "settings.json"


def load() -> dict:
    if not os.path.exists(PATH):
        return {}
    try:
        with open(PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"[settings] load error: {e}")
        return {}


def save(data: dict) -> None:
    try:
        with open(PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[settings] save error: {e}")
