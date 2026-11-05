"""Speech-to-text and text-to-speech adapters.

Interfaces: STTAdapter.transcribe(wav) -> text, TTSAdapter.synthesize(text) -> wav. Audio is
always a 16-bit PCM mono WAV file in bytes.

Selected by environment variable:
    HELPDESK_STT = fake (default) | faster-whisper
    HELPDESK_TTS = fake (default) | piper

Real adapters load their models lazily, on first use or warm_up().
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .audio import embedded_text, silent_wav, wav_to_float32_mono

DEFAULT_CANNED_TRANSCRIPT = "Sorry, it's still doing the same thing."


class STTAdapter(ABC):
    name = "stt"

    @abstractmethod
    async def transcribe(self, wav: bytes) -> str:
        """Return the text spoken in a WAV file."""

    async def warm_up(self) -> None:
        """Load models ahead of the first call (no-op by default)."""


class TTSAdapter(ABC):
    name = "tts"

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Return a 16-bit PCM mono WAV file of `text` being spoken."""

    async def warm_up(self) -> None:
        """Load models ahead of the first call (no-op by default)."""


# --------------------------------------------------------------------------- fakes


class FakeSTT(STTAdapter):
    """Returns canned text. If the audio came from FakeTTS it returns the text embedded in it,
    which lets end-to-end tests run real conversations through the voice pipeline."""

    name = "fake"

    def __init__(self, canned: list[str] | None = None, default: str = DEFAULT_CANNED_TRANSCRIPT):
        self._canned = list(canned or [])
        self.default = default

    async def transcribe(self, wav: bytes) -> str:
        text = embedded_text(wav)
        if text is not None:
            return text
        if self._canned:
            return self._canned.pop(0)
        return self.default


class FakeTTS(TTSAdapter):
    """Returns silent audio whose length roughly matches the text (and carries the text)."""

    name = "fake"

    def __init__(self, sample_rate: int = 16_000, seconds_per_char: float = 0.06, max_seconds: float = 12.0):
        self.sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char
        self.max_seconds = max_seconds

    async def synthesize(self, text: str) -> bytes:
        seconds = min(self.max_seconds, max(0.2, len(text) * self.seconds_per_char))
        return silent_wav(seconds, self.sample_rate, text=text)


# --------------------------------------------------------------------------- real


class FasterWhisperSTT(STTAdapter):
    """faster-whisper (CTranslate2). The model downloads from huggingface.co on first use."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str | None = "en",
        download_root: str | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.download_root = download_root
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "faster-whisper is not installed. Run: pip install -e '.[voice]'"
                    ) from exc
                try:
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=self.download_root,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load the Whisper '{self.model_size}' model ({type(exc).__name__}: {exc}). "
                        "It downloads from huggingface.co on first use; see SETUP_LOCAL.md."
                    ) from exc
            return self._model

    def _transcribe_sync(self, wav: bytes) -> str:
        audio = wav_to_float32_mono(wav, 16_000)
        if audio.size == 0:
            return ""
        segments, _info = self._load().transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def transcribe(self, wav: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_sync, wav)

    async def warm_up(self) -> None:
        await asyncio.to_thread(self._load)


class PiperTTS(TTSAdapter):
    """Piper neural TTS. Needs a voice file (.onnx plus its .onnx.json) on disk."""

    name = "piper"

    def __init__(self, voice_path: str | Path, length_scale: float | None = None, use_cuda: bool = False):
        self.voice_path = Path(voice_path).expanduser()
        self.length_scale = length_scale
        self.use_cuda = use_cuda
        self._voice: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._voice is None:
                try:
                    from piper import PiperVoice
                except ImportError as exc:
                    raise RuntimeError("piper-tts is not installed. Run: pip install -e '.[voice]'") from exc
                config = Path(f"{self.voice_path}.json")
                if not self.voice_path.exists() or not config.exists():
                    raise RuntimeError(
                        f"Piper voice not found: need {self.voice_path} and {config}. "
                        "Download a voice (see SETUP_LOCAL.md) and set HELPDESK_PIPER_VOICE."
                    )
                self._voice = PiperVoice.load(str(self.voice_path), use_cuda=self.use_cuda)
            return self._voice

    def _synthesize_sync(self, text: str) -> bytes:
        voice = self._load()
        syn_config = None
        if self.length_scale is not None:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(length_scale=self.length_scale)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        return buffer.getvalue()

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    async def warm_up(self) -> None:
        await asyncio.to_thread(self._load)


# --------------------------------------------------------------------------- selection


def get_stt(kind: str | None = None) -> STTAdapter:
    kind = (kind or os.environ.get("HELPDESK_STT", "fake")).strip().lower()
    if kind == "fake":
        return FakeSTT()
    if kind in ("faster-whisper", "faster_whisper", "whisper"):
        return FasterWhisperSTT(
            model_size=os.environ.get("HELPDESK_WHISPER_MODEL", "small"),
            device=os.environ.get("HELPDESK_WHISPER_DEVICE", "cpu"),
            compute_type=os.environ.get("HELPDESK_WHISPER_COMPUTE", "int8"),
            beam_size=int(os.environ.get("HELPDESK_WHISPER_BEAM", "1")),
            download_root=os.environ.get("HELPDESK_WHISPER_DIR") or None,
        )
    raise ValueError(f"unknown HELPDESK_STT {kind!r} (use 'fake' or 'faster-whisper')")


def get_tts(kind: str | None = None, role: str = "agent") -> TTSAdapter:
    """role='caller' uses HELPDESK_PIPER_CALLER_VOICE if set, so the no-mic test can give the
    simulated caller a different voice from the agent."""
    kind = (kind or os.environ.get("HELPDESK_TTS", "fake")).strip().lower()
    if kind == "fake":
        return FakeTTS()
    if kind == "piper":
        voice = os.environ.get("HELPDESK_PIPER_CALLER_VOICE") if role == "caller" else None
        voice = voice or os.environ.get("HELPDESK_PIPER_VOICE")
        if not voice:
            raise ValueError("HELPDESK_TTS=piper needs HELPDESK_PIPER_VOICE=/path/to/voice.onnx")
        scale = os.environ.get("HELPDESK_PIPER_LENGTH_SCALE")
        return PiperTTS(voice, length_scale=float(scale) if scale else None)
    raise ValueError(f"unknown HELPDESK_TTS {kind!r} (use 'fake' or 'piper')")
