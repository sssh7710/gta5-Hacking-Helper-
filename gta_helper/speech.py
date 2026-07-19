from __future__ import annotations

import queue
import threading


class SpeechService:
    def __init__(self, enabled: bool, rate: int) -> None:
        self.enabled = enabled
        self.rate = rate
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, daemon=True, name="gta-helper-speech")
            self._thread.start()

    def say(self, text: str) -> None:
        if self.enabled:
            self._queue.put(text)

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception:
            return
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
