import threading
import time

import requests

SYSTEM_PROMPT = (
    "You are the dreaming mind of a sleeping VJ artist. Each turn, respond "
    "with exactly one Stable Diffusion prompt describing the next vivid scene "
    "in the dream. Visual description only: subject, environment, mood, "
    "lighting, art style. Under 200 characters. No quotes, no narration, no "
    "numbering — just the prompt. Each scene should drift from the previous "
    "one: the same setting twisted, an object transformed, the light shifting. "
    "Don't reset the dream."
)


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
    ):
        self._model = model
        self._host = host
        self._history_turns = history_turns
        self._temperature = temperature

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
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                # Keep only the most recent turns to bound context size.
                tail = self._history[-(self._history_turns * 2):]
                messages.extend(tail)
                user_msg = (
                    "Continue the dream — describe the next scene."
                    if tail
                    else "Begin the dream — describe the first scene."
                )
                messages.append({"role": "user", "content": user_msg})

            try:
                response = requests.post(
                    f"{self._host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temp},
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                # Strip wrapping quotes/backticks/markdown the model sometimes adds.
                content = content.strip('"\'` \n\t').replace("\n", " ").strip()
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
