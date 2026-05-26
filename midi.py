import json
import os
import threading

import mido


class MidiController:
    """
    Listens to a MIDI input port in a background thread (mido callback) and
    exposes:
      - learn mode: bind the next incoming CC or Note to a UI tag
      - mappings persisted to JSON so they survive restart
      - a thread-safe "pending updates" dict that the main loop drains each
        frame and applies to DearPyGui widgets
    """

    def __init__(self, mappings_path: str = "midi_mappings.json"):
        self.mappings_path = mappings_path
        # key "cc:<channel>:<control>" or "note:<channel>:<note>"
        # value {"tag": str, "type": "slider"|"checkbox"|"button", "min": float, "max": float}
        self.mappings: dict[str, dict] = {}
        self._load()

        self._learn_target: tuple | None = None  # (tag, type, min, max)
        self._last_event: str = ""

        self._pending: dict[str, object] = {}
        self._lock = threading.Lock()

        self._port: mido.ports.BaseInput | None = None
        self._port_name: str | None = None

    # ---- port management ----

    @staticmethod
    def list_inputs() -> list[str]:
        try:
            return mido.get_input_names()
        except Exception:
            return []

    def open(self, name: str) -> bool:
        self.close()
        try:
            self._port = mido.open_input(name, callback=self._on_message)
            self._port_name = name
            return True
        except Exception as e:
            self._last_event = f"open failed: {e}"
            self._port = None
            self._port_name = None
            return False

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        self._port_name = None

    @property
    def port_name(self) -> str | None:
        return self._port_name

    @property
    def last_event(self) -> str:
        return self._last_event

    # ---- learn mode ----

    def start_learn(self, tag: str, ctype: str, vmin: float, vmax: float) -> None:
        with self._lock:
            self._learn_target = (tag, ctype, vmin, vmax)
            self._last_event = f"learning {tag} — touch a MIDI control…"

    def cancel_learn(self) -> None:
        with self._lock:
            self._learn_target = None
            self._last_event = "learn cancelled"

    def is_learning(self) -> bool:
        with self._lock:
            return self._learn_target is not None

    def learn_target_tag(self) -> str | None:
        with self._lock:
            return self._learn_target[0] if self._learn_target else None

    # ---- mappings ----

    def remove_mapping(self, tag: str) -> None:
        with self._lock:
            self.mappings = {k: v for k, v in self.mappings.items() if v["tag"] != tag}
            self._save()
            self._last_event = f"cleared mapping for {tag}"

    def mappings_for(self, tag: str) -> list[str]:
        return [k for k, v in self.mappings.items() if v["tag"] == tag]

    # ---- message handling ----

    def _on_message(self, msg: mido.Message) -> None:
        key = None
        ctype_hint = None
        norm_value = None  # 0..1 for sliders/checkboxes
        trigger = False    # for buttons

        if msg.type == "control_change":
            key = f"cc:{msg.channel}:{msg.control}"
            ctype_hint = "slider"
            norm_value = msg.value / 127.0
            trigger = msg.value >= 64
        elif msg.type == "note_on" and msg.velocity > 0:
            key = f"note:{msg.channel}:{msg.note}"
            ctype_hint = "button"
            norm_value = msg.velocity / 127.0
            trigger = True
        else:
            return

        with self._lock:
            # Learn mode: bind this incoming message to the pending target.
            if self._learn_target is not None:
                tag, ctype, vmin, vmax = self._learn_target
                self.mappings[key] = {"tag": tag, "type": ctype, "min": vmin, "max": vmax}
                self._learn_target = None
                self._save()
                self._last_event = f"bound {key} → {tag}"
                return

            mapping = self.mappings.get(key)
            self._last_event = f"{key} = {getattr(msg, 'value', getattr(msg, 'velocity', 0))}"

            if mapping is None:
                return

            tag = mapping["tag"]
            ctype = mapping["type"]
            vmin = mapping.get("min", 0.0)
            vmax = mapping.get("max", 1.0)

            if ctype == "slider":
                if norm_value is None:
                    return
                self._pending[tag] = vmin + norm_value * (vmax - vmin)
            elif ctype == "checkbox":
                # CC ≥ 64 → on, < 64 → off. Notes always toggle.
                if msg.type == "control_change":
                    self._pending[tag] = trigger
                else:
                    self._pending[tag] = "toggle"
            elif ctype == "button":
                if trigger:
                    self._pending[tag] = "trigger"

    def consume_pending(self) -> dict[str, object]:
        with self._lock:
            out = self._pending
            self._pending = {}
            return out

    # ---- persistence ----

    def _load(self) -> None:
        if not os.path.exists(self.mappings_path):
            return
        try:
            with open(self.mappings_path) as f:
                self.mappings = json.load(f)
        except Exception as e:
            print(f"[MIDI] load error: {e}")
            self.mappings = {}

    def _save(self) -> None:
        try:
            with open(self.mappings_path, "w") as f:
                json.dump(self.mappings, f, indent=2)
        except Exception as e:
            print(f"[MIDI] save error: {e}")
