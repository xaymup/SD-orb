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
    # Atomic write: dump to a sibling tmp file, then os.replace (atomic on
    # POSIX). A crash mid-dump leaves the previous settings.json untouched
    # rather than truncated/corrupt.
    tmp = PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, PATH)
    except Exception as e:
        print(f"[settings] save error: {e}")
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
