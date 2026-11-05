"""One voice turn: caller audio -> STT -> streamed agent reply -> TTS sentence by sentence.

TTS runs concurrently with the LLM stream: each sentence is synthesized as soon as it is
complete, so the first audio is ready long before the full reply is written.

Latency per turn (milliseconds):
    stt_ms              speech-to-text time for the caller's audio
    llm_first_token_ms  from sending the agent request to the first text token
    tts_first_audio_ms  synthesis time of the first sentence
    first_audio_ms      end of caller audio -> first reply audio ready (what the caller waits)
    total_ms            end of caller audio -> last reply audio ready
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..agent import AgentReply, next_agent_message
from ..config import AGENT_MODEL, RESULTS_DIR
from ..llm import TokenUsage
from ..models import Turn
from .adapters import STTAdapter, TTSAdapter
from .sentences import SentenceSplitter

AudioSink = Callable[[int, str, bytes], Awaitable[None]]
TextSink = Callable[[str], Any]


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


async def _call(sink: TextSink | None, value: str) -> None:
    if sink is None:
        return
    result = sink(value)
    if inspect.isawaitable(result):
        await result


@dataclass
class TurnLatency:
    stt_ms: float | None
    llm_first_token_ms: float | None
    tts_first_audio_ms: float | None
    first_audio_ms: float | None
    total_ms: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass
class VoiceTurnResult:
    caller_text: str
    reply: AgentReply
    latency: TurnLatency
    audio: list[tuple[str, bytes]] = field(default_factory=list)  # (sentence, wav)


async def run_voice_turn(
    conversation: list[Turn],
    *,
    stt: STTAdapter,
    tts: TTSAdapter,
    client: Any,
    audio: bytes | None = None,
    caller_text: str | None = None,
    model: str = AGENT_MODEL,
    usage: TokenUsage | None = None,
    on_caller_text: TextSink | None = None,
    on_agent_text: TextSink | None = None,
    on_audio: AudioSink | None = None,
) -> VoiceTurnResult:
    """Run one turn. Pass the caller's `audio` (WAV) or, to skip STT, `caller_text`.
    Appends the caller turn and the agent turn to `conversation`."""
    if audio is None and caller_text is None:
        raise ValueError("need audio or caller_text")
    t0 = time.perf_counter()
    stt_ms = None
    if audio is not None:
        t_stt = time.perf_counter()
        caller_text = await stt.transcribe(audio)
        stt_ms = _ms(t_stt)
    caller_text = (caller_text or "").strip() or "(inaudible)"
    await _call(on_caller_text, caller_text)
    conversation.append(Turn(role="caller", text=caller_text))

    splitter = SentenceSplitter()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    spoken: list[tuple[str, bytes]] = []
    first: dict[str, float] = {}

    async def tts_worker() -> None:
        seq = 0
        while (sentence := await queue.get()) is not None:
            t_tts = time.perf_counter()
            wav = await tts.synthesize(sentence)
            if not first:
                first["tts_first_audio_ms"] = _ms(t_tts)
                first["first_audio_ms"] = _ms(t0)
            spoken.append((sentence, wav))
            if on_audio is not None:
                await on_audio(seq, sentence, wav)
            seq += 1

    async def on_text(delta: str) -> None:
        for sentence in splitter.feed(delta):
            queue.put_nowait(sentence)
        await _call(on_agent_text, delta)

    worker = asyncio.create_task(tts_worker())
    try:
        reply = await next_agent_message(conversation, client=client, model=model, usage=usage, on_text=on_text)
    except BaseException:
        worker.cancel()
        raise
    for sentence in splitter.flush():
        queue.put_nowait(sentence)
    queue.put_nowait(None)
    await worker

    conversation.append(reply.as_turn())
    latency = TurnLatency(
        stt_ms=stt_ms,
        llm_first_token_ms=round(reply.first_token_ms, 1) if reply.first_token_ms is not None else None,
        tts_first_audio_ms=first.get("tts_first_audio_ms"),
        first_audio_ms=first.get("first_audio_ms"),
        total_ms=_ms(t0),
    )
    return VoiceTurnResult(caller_text=caller_text, reply=reply, latency=latency, audio=spoken)


class LatencyLog:
    """Appends one JSON line per turn to results/latency.jsonl (or HELPDESK_LATENCY_LOG)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("HELPDESK_LATENCY_LOG") or RESULTS_DIR / "latency.jsonl")

    def write(self, *, source: str, call_id: str, turn: int, latency: TurnLatency, **extra: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "source": source,
            "call_id": call_id,
            "turn": turn,
            **latency.to_dict(),
            **extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record
