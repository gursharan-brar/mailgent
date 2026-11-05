"""Audio helpers, sentence splitting, and STT/TTS adapters."""

import asyncio
import io
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from helpdesk.voice.adapters import (
    DEFAULT_CANNED_TRANSCRIPT,
    FakeSTT,
    FakeTTS,
    FasterWhisperSTT,
    PiperTTS,
    STTAdapter,
    TTSAdapter,
    get_stt,
    get_tts,
)
from helpdesk.voice.audio import embedded_text, pcm16_wav, silent_wav, wav_info, wav_to_float32_mono
from helpdesk.voice.sentences import SentenceSplitter

FIXTURE_VOICE = Path(__file__).parent / "fixtures" / "tiny_piper_voice.onnx"


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ audio


def test_silent_wav_is_valid_and_carries_text():
    wav = silent_wav(0.5, 16000, text="hello there")
    info = wav_info(wav)
    assert (info.sample_rate, info.channels, info.sample_width, info.frames) == (16000, 1, 2, 8000)
    assert embedded_text(wav) == "hello there"
    with wave.open(io.BytesIO(wav)) as wf:  # the stdlib reader skips our extra chunk
        assert wf.readframes(10) == b"\x00" * 20


def test_plain_wav_has_no_embedded_text():
    assert embedded_text(pcm16_wav(b"\x00\x00" * 10, 8000)) is None
    assert embedded_text(b"not a wav") is None


def test_wav_to_float32_resamples_and_mixes_down():
    tone = (np.sin(np.linspace(0, 200 * np.pi, 22050)) * 20000).astype("<i2")
    mono = wav_to_float32_mono(pcm16_wav(tone.tobytes(), 22050), 16000)
    assert mono.dtype == np.float32 and abs(len(mono) - 16000) <= 1
    assert 0.5 < np.abs(mono).max() <= 1.0
    stereo = np.stack([tone, tone], axis=1).astype("<i2")
    mixed = wav_to_float32_mono(pcm16_wav(stereo.tobytes(), 22050, channels=2), 22050)
    assert len(mixed) == len(tone)


# ------------------------------------------------------------------ sentences


def test_splitter_emits_sentences_as_they_complete():
    s = SentenceSplitter()
    out = []
    for chunk in ["Sorry to hear ", "that! Can you", " tell me what the screen says? I'll", " wait."]:
        out += s.feed(chunk)
    assert out == ["Sorry to hear that!", "Can you tell me what the screen says?"]
    assert s.flush() == ["I'll wait."]
    assert s.flush() == []


def test_splitter_keeps_abbreviations_and_decimals():
    s = SentenceSplitter()
    out = s.feed("Open Settings, e.g. from the Start menu. Version 2.5 is fine. ")
    assert out == ["Open Settings, e.g. from the Start menu.", "Version 2.5 is fine."]


def test_splitter_cuts_long_runs_and_newlines():
    s = SentenceSplitter(max_chars=40)
    out = s.feed("first line\nthis is a long clause without any stop, and it keeps going on and on")
    assert out[0] == "first line"
    assert all(len(x) <= 40 for x in out)


# ------------------------------------------------------------------ adapters


def test_interfaces():
    assert issubclass(FakeSTT, STTAdapter) and issubclass(FasterWhisperSTT, STTAdapter)
    assert issubclass(FakeTTS, TTSAdapter) and issubclass(PiperTTS, TTSAdapter)


def test_fake_tts_to_fake_stt_round_trip():
    wav = run(FakeTTS().synthesize("My printer stopped working."))
    assert wav_info(wav).duration_s > 0.5
    assert np.abs(wav_to_float32_mono(wav)).max() == 0  # silent
    assert run(FakeSTT().transcribe(wav)) == "My printer stopped working."


def test_fake_stt_canned_then_default():
    stt = FakeSTT(canned=["one", "two"])
    plain = pcm16_wav(b"\x00\x00" * 100, 16000)
    assert [run(stt.transcribe(plain)) for _ in range(3)] == ["one", "two", DEFAULT_CANNED_TRANSCRIPT]


def test_selection_by_env(monkeypatch):
    monkeypatch.delenv("HELPDESK_STT", raising=False)
    monkeypatch.delenv("HELPDESK_TTS", raising=False)
    assert isinstance(get_stt(), FakeSTT) and isinstance(get_tts(), FakeTTS)
    monkeypatch.setenv("HELPDESK_STT", "faster-whisper")
    monkeypatch.setenv("HELPDESK_WHISPER_MODEL", "tiny")
    stt = get_stt()
    assert isinstance(stt, FasterWhisperSTT) and stt.model_size == "tiny" and stt._model is None  # lazy
    monkeypatch.setenv("HELPDESK_TTS", "piper")
    with pytest.raises(ValueError, match="HELPDESK_PIPER_VOICE"):
        get_tts()
    monkeypatch.setenv("HELPDESK_PIPER_VOICE", "/voices/agent.onnx")
    monkeypatch.setenv("HELPDESK_PIPER_CALLER_VOICE", "/voices/caller.onnx")
    assert get_tts(role="agent").voice_path == Path("/voices/agent.onnx")
    assert get_tts(role="caller").voice_path == Path("/voices/caller.onnx")
    with pytest.raises(ValueError):
        get_stt("nope")


def test_faster_whisper_adapter_glue(monkeypatch):
    calls = {}

    class FakeWhisperModel:
        def __init__(self, size, device, compute_type, download_root):
            calls["init"] = (size, device, compute_type)

        def transcribe(self, audio, language, beam_size, condition_on_previous_text):
            calls["audio"] = (audio.dtype, len(audio), language, beam_size)
            seg = types.SimpleNamespace
            return iter([seg(text=" My printer "), seg(text=" isn't working. ")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    stt = FasterWhisperSTT(model_size="small")
    wav = pcm16_wav((np.ones(22050) * 1000).astype("<i2").tobytes(), 22050)
    assert run(stt.transcribe(wav)) == "My printer isn't working."
    assert calls["init"] == ("small", "cpu", "int8")
    assert calls["audio"][0] == np.float32 and abs(calls["audio"][1] - 16000) <= 1


def test_faster_whisper_load_failure_is_explained(monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise OSError("403 Forbidden")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Broken))
    with pytest.raises(RuntimeError, match="SETUP_LOCAL.md"):
        run(FasterWhisperSTT().warm_up())


def test_piper_missing_voice_is_explained(tmp_path):
    pytest.importorskip("piper")
    with pytest.raises(RuntimeError, match="HELPDESK_PIPER_VOICE"):
        run(PiperTTS(tmp_path / "missing.onnx").synthesize("hi"))


def test_piper_adapter_with_real_library_and_stand_in_voice():
    """Runs the real piper-tts code (espeak phonemizer, onnxruntime, WAV writer) with a tiny
    stand-in model from tests/fixtures. Checks the adapter, not voice quality."""
    pytest.importorskip("piper")
    tts = PiperTTS(FIXTURE_VOICE, length_scale=1.0)
    short = run(tts.synthesize("Hi."))
    longer = run(tts.synthesize("Thanks for calling the help desk. What seems to be the problem today?"))
    info = wav_info(longer)
    assert (info.sample_rate, info.channels, info.sample_width) == (16000, 1, 2)
    assert info.frames > wav_info(short).frames > 0


@pytest.mark.skipif(
    __import__("os").environ.get("HELPDESK_REAL_VOICE_TESTS") != "1",
    reason="set HELPDESK_REAL_VOICE_TESTS=1 with real models installed (see SETUP_LOCAL.md)",
)
def test_real_models_round_trip():
    tts, stt = get_tts("piper"), get_stt("faster-whisper")
    text = run(stt.transcribe(run(tts.synthesize("My printer is not working this morning."))))
    assert "printer" in text.lower()
