import os
import subprocess
import time
from typing import Optional


class PersistentChatterboxAnnouncer:
    def __init__(
        self,
        enabled: bool,
        tts_python: str = ".venv_tts/bin/python",
        model: str = "turbo",
        audio_prompt: Optional[str] = None,
        cooldown_s: float = 0.4,
    ):
        self.enabled = enabled
        self.tts_python = tts_python
        self.model = model
        self.audio_prompt = audio_prompt
        self.cooldown_s = cooldown_s
        self._proc: Optional[subprocess.Popen] = None
        self._last_spoken = 0.0
        self._warned_error = False

    def start(self):
        if not self.enabled:
            return
        if self._proc is not None and self._proc.poll() is None:
            return
        if not os.path.exists(self.tts_python):
            if not self._warned_error:
                print(f"[TTS] Python not found at {self.tts_python}. Disabling TTS.")
                self._warned_error = True
            self.enabled = False
            return

        cmd = [self.tts_python, "-m", "app.tts_worker", "--model", self.model]
        if self.audio_prompt:
            cmd.extend(["--audio-prompt", self.audio_prompt])

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=None,
                text=True,
            )
        except Exception as exc:
            if not self._warned_error:
                print(f"[TTS] Failed to start worker: {exc}. Disabling TTS.")
                self._warned_error = True
            self.enabled = False
            return

    def start_nonblocking(self):
        self.start()

    def stop(self):
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.write("__quit__\n")
                self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        self._proc = None

    def on_rep(self, exercise: str, rep_count: int):
        if not self.enabled:
            return
        if self._proc is None or self._proc.poll() is not None:
            self.start()
            if self._proc is None or self._proc.poll() is not None:
                return

        now = time.monotonic()
        if (now - self._last_spoken) < self.cooldown_s:
            return

        phrase = f"{exercise.capitalize()} {rep_count}"
        try:
            if self._proc.stdin:
                self._proc.stdin.write(phrase + "\n")
                self._proc.stdin.flush()
            self._last_spoken = now
        except Exception as exc:
            if not self._warned_error:
                print(f"[TTS] Failed to send text: {exc}. Disabling TTS.")
                self._warned_error = True
            self.enabled = False
