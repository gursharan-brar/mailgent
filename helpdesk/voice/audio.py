"""WAV helpers (16-bit PCM). Pure stdlib except wav_to_float32_mono, which uses numpy."""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass

# Custom RIFF chunk the fake TTS uses to carry the source text inside otherwise silent audio,
# so the fake STT can "hear" it. Standard WAV readers skip unknown chunks.
TEXT_CHUNK = b"htxt"


def _chunk(chunk_id: bytes, data: bytes) -> bytes:
    pad = b"\x00" if len(data) % 2 else b""
    return chunk_id + struct.pack("<I", len(data)) + data + pad


def pcm16_wav(
    pcm: bytes, sample_rate: int, channels: int = 1, extra_chunks: dict[bytes, bytes] | None = None
) -> bytes:
    """Build a 16-bit PCM WAV file. Extra chunks go after the data chunk."""
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16)
    body = b"WAVE" + _chunk(b"fmt ", fmt) + _chunk(b"data", pcm)
    for chunk_id, data in (extra_chunks or {}).items():
        body += _chunk(chunk_id, data)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def read_chunks(wav: bytes) -> dict[bytes, bytes]:
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    chunks: dict[bytes, bytes] = {}
    pos = 12
    while pos + 8 <= len(wav):
        chunk_id = wav[pos : pos + 4]
        size = struct.unpack("<I", wav[pos + 4 : pos + 8])[0]
        chunks[chunk_id] = wav[pos + 8 : pos + 8 + size]
        pos += 8 + size + (size & 1)
    return chunks


def silent_wav(duration_s: float, sample_rate: int = 16_000, text: str | None = None) -> bytes:
    frames = max(0, int(duration_s * sample_rate))
    extra = {TEXT_CHUNK: text.encode("utf-8")} if text is not None else None
    return pcm16_wav(b"\x00\x00" * frames, sample_rate, extra_chunks=extra)


def embedded_text(wav: bytes) -> str | None:
    try:
        data = read_chunks(wav).get(TEXT_CHUNK)
    except ValueError:
        return None
    return data.decode("utf-8") if data is not None else None


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def wav_info(wav: bytes) -> WavInfo:
    with wave.open(io.BytesIO(wav), "rb") as wf:
        return WavInfo(wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes())


def wav_to_float32_mono(wav: bytes, target_rate: int = 16_000):
    """Decode a 16-bit PCM WAV to a mono float32 numpy array at target_rate (what Whisper wants)."""
    import numpy as np

    with wave.open(io.BytesIO(wav), "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM audio, got {8 * width}-bit")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != target_rate and audio.size:
        n_out = int(round(audio.size * target_rate / rate))
        old_x = np.arange(audio.size, dtype=np.float64) / rate
        new_x = np.arange(n_out, dtype=np.float64) / target_rate
        audio = np.interp(new_x, old_x, audio).astype(np.float32)
    return audio
