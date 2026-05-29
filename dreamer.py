import re
import threading
import time

import requests

# Strip trailing parenthetical meta the LLM leaks despite being told not to:
# "(8 words)", "(word count: 9)", "(Word Count - 7)", "(words: 6)". End-
# anchored so legitimate inline parens are safe.
_META_RE = re.compile(
    r"\s*\(\s*(?:"
    r"\d+\s*words?\.?"            # 8 words
    r"|word\s*count[:\-=\s]*\d+"   # word count: 5, Word Count - 6
    r"|words?[:\-=\s]+\d+"         # words: 8
    r")\s*\)\s*$",
    re.IGNORECASE,
)

BASE_SYSTEM_PROMPT = (
    "You are a dreaming mind generating Stable Diffusion 1.5 prompts for "
    "real-time LCM inference. Output ONE prompt per turn: comma-separated "
    "tags, 10 WORDS OR FEWER total. No labels, quotes, numbering, "
    "narration, or full sentences.\n"
    "\n"
    "Aesthetic: raw, chaotic, hallucinatory scenes built from SENSORY "
    "specifics — TEXTURE (wet asphalt, cracked vinyl, oily film, ash "
    "crust, rust bloom, frayed silk, melted plastic), LIGHTING "
    "(flickering tube, blown flash, sodium glare, dying CRT, undercaught "
    "backlight, dawn sodium), and DEGRADATION (chromatic bleed, film "
    "scratch, VHS tracking, halftone rot, JPEG smear, photocopy grain). "
    "One concrete subject anchors the scene; the rest is grit.\n"
    "\n"
    "FORBIDDEN — never use: 8k, 4k, ultra detailed, hyperrealistic, "
    "masterpiece, best quality, highly detailed, intricate, sharp focus, "
    "cinematic, beautiful, stunning, professional, breathtaking, "
    "atmospheric, epic. These are empty noise and degrade LCM output.\n"
    "\n"
    "Each turn EVOLVES the dream — sometimes a small swap (texture, "
    "lighting), often a bold leap (new subject, new environment, "
    "unexpected pairing). Welcome surprise. Drag the imagery into "
    "stranger territory: hybrid creatures, impossible architecture, "
    "ruined sacred objects, scenes from invented decades. The thread "
    "between turns is mood and sensory grammar, NOT literal similarity. "
    "Do not loop, do not repeat tags from recent turns, do not settle "
    "into a comfortable subject — keep mutating.\n"
    "\n"
    "Output ONLY the comma-separated tags. No preamble, no commentary, "
    "no parentheses, no word counts, no notes — just the tags. Examples "
    "of valid replies:\n"
    "  wet asphalt subway, flickering tubes, chromatic bleed\n"
    "  rusted chrome bust, VHS tracking, dying CRT glow\n"
    "  torn silk dress, sodium glare, film scratch, ash haze"
)


def build_system_prompt(style_hint: str | None, keywords: str | None = None) -> str:
    """Inject the active SD style register + the user's influence list into
    the system prompt. Both are constraints, not hints — they go in the
    system message so the LLM weighs them above turn-by-turn nudges.

    Putting keywords here (instead of as a "lean toward" suffix on the user
    message) is the difference between the LLM treating them as a soft
    suggestion vs. an active rule. The "bold leaps" instruction in the
    base prompt makes the soft path almost invisible — the LLM picks its
    own examples over the user's."""
    parts = [BASE_SYSTEM_PROMPT]
    if style_hint:
        parts.append(
            f"\n\nActive visual register: {style_hint}. Lean the sensory grit "
            "toward this register — pick textures, lighting, and degradation "
            "that suit it. Still 10 words or fewer."
        )
    if keywords:
        parts.append(
            f"\n\nACTIVE INFLUENCES (use these every turn): {keywords}. Every "
            "dream MUST integrate at least one of these — as a texture, "
            "subject, environment, or framing device. Translate them into "
            "the sensory grammar above (don't list them verbatim as tags). "
            "If a turn drifts off these influences, pull it back next turn."
        )
    return "".join(parts)


class Dreamer:
    """
    Background worker that queries Ollama for surreal SD prompts and keeps a
    rolling conversation so successive dreams flow into each other. The main
    loop never blocks on the HTTP call — request_dream() returns immediately,
    latest_dream is whatever the worker has produced so far.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        history_turns: int = 8,
        temperature: float = 1.1,
        cpu_only: bool = True,
    ):
        self._model = model
        self._host = host
        self._history_turns = history_turns
        self._temperature = temperature
        self._cpu_only = cpu_only
        self._keywords: str = ""
        self._style_hint: str = ""

        self._history: list[dict] = []
        self._latest: str | None = None
        self._error: str | None = None
        self._is_dreaming = False

        self._lock = threading.Lock()
        self._request = threading.Event()
        self._stop = threading.Event()

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    @property
    def latest_dream(self) -> str | None:
        with self._lock:
            return self._latest

    @property
    def is_dreaming(self) -> bool:
        with self._lock:
            return self._is_dreaming

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def request_dream(self) -> None:
        self._request.set()

    def set_model(self, model: str) -> None:
        with self._lock:
            self._model = model

    def set_temperature(self, t: float) -> None:
        with self._lock:
            self._temperature = t

    def set_cpu_only(self, value: bool) -> None:
        with self._lock:
            self._cpu_only = bool(value)

    def set_keywords(self, keywords: str) -> None:
        with self._lock:
            self._keywords = keywords.strip()

    def set_style(self, style_hint: str) -> None:
        """Update the visual register the dreamer biases toward. Called when
        the SD model changes so dreams shift aesthetically alongside the
        renderer. Clears history because the prior turns were written under a
        different aesthetic and seeing them would anchor the LLM back."""
        hint = (style_hint or "").strip()
        with self._lock:
            if hint == self._style_hint:
                return
            self._style_hint = hint
            self._history.clear()

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._latest = None

    def shutdown(self) -> None:
        self._stop.set()
        self._request.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._request.wait()
            self._request.clear()
            if self._stop.is_set():
                return

            with self._lock:
                self._is_dreaming = True
                model = self._model
                temp = self._temperature
                cpu_only = self._cpu_only
                keywords = self._keywords
                style_hint = self._style_hint
                # Keywords are baked into the system prompt now — much more
                # weight than a soft "lean toward" suffix on the user msg.
                messages = [{"role": "system", "content": build_system_prompt(style_hint, keywords)}]
                # Keep only the most recent turns to bound context size.
                tail = self._history[-(self._history_turns * 2):]
                messages.extend(tail)
                user_msg = (
                    "Continue the dream — next scene. Remember the active influences."
                    if tail
                    else "Begin the dream — first scene. Remember the active influences."
                )
                messages.append({"role": "user", "content": user_msg})

            # num_gpu=0 forces Ollama to load llama on CPU. The render loop
            # shares the GPU with SD inference; full-GPU llama generation
            # contends for SMs and visibly drops FPS during the ~1-3 s the
            # response is being produced. CPU generation takes a few seconds
            # but the dream is queued one switch interval ahead, so the extra
            # latency is invisible.
            options = {"temperature": temp}
            if cpu_only:
                options["num_gpu"] = 0

            try:
                response = requests.post(
                    f"{self._host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": options,
                    },
                    timeout=120 if cpu_only else 60,
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                # Strip wrapping quotes/backticks/markdown the model sometimes adds.
                content = content.strip('"\'` \n\t').replace("\n", " ").strip()
                # Strip parenthetical word-count meta the model leaks even
                # after being told not to. Belt-and-suspenders with the
                # system prompt — the prompt usually wins but not always.
                content = _META_RE.sub("", content).strip().rstrip(",").strip()
                if not content:
                    raise RuntimeError("empty response from Ollama")

                with self._lock:
                    self._history.append({"role": "user", "content": user_msg})
                    self._history.append({"role": "assistant", "content": content})
                    # Bound history.
                    excess = len(self._history) - self._history_turns * 2
                    if excess > 0:
                        self._history = self._history[excess:]
                    self._latest = content
                    self._error = None
            except Exception as e:
                with self._lock:
                    self._error = f"{type(e).__name__}: {e}"
                print(f"[Dreamer] {self._error}", flush=True)
                # Brief backoff so a down Ollama doesn't get hammered.
                time.sleep(1.0)
            finally:
                with self._lock:
                    self._is_dreaming = False
