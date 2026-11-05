"""Voice call server: FastAPI + one WebSocket endpoint, plus the static call page.

    python -m helpdesk.voice.server              # http://127.0.0.1:8000
    HELPDESK_STT=faster-whisper HELPDESK_TTS=piper HELPDESK_PIPER_VOICE=models/en_US-lessac-medium.onnx \
        python -m helpdesk.voice.server

WebSocket protocol on /ws:

client -> server
    binary                                   one caller turn as a 16-bit PCM WAV
    {"type": "caller_text", "text": "..."}   a typed caller turn (skips STT)
    {"type": "hangup"}

server -> client (JSON text frames, plus binary WAV frames)
    {"type": "ready", "call_id", "stt", "tts", "agent_model"}
    {"type": "caller_text", "text"}          what STT heard
    {"type": "agent_delta", "text"}          streamed reply text
    {"type": "audio", "seq", "text"}         followed immediately by one binary WAV frame
    {"type": "turn_done", "turn", "text", "latency"}
    {"type": "ticket", "ticket_id", "ticket"} then {"type": "call_ended"}
    {"type": "error", "message"}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..agent import count_agent_replies
from ..config import AGENT_MODEL, HARNESS_MAX_TURNS, STATIC_DIR, resolve_model
from ..llm import MissingAPIKeyError, TokenUsage, make_client
from ..models import Turn
from .adapters import STTAdapter, TTSAdapter, get_stt, get_tts
from .pipeline import LatencyLog, run_voice_turn

GREETING = "Thanks for calling the IT help desk, this is Alex. What's going on?"


def create_app(
    *,
    client: Any = None,
    stt: STTAdapter | None = None,
    tts: TTSAdapter | None = None,
    latency_log: LatencyLog | None = None,
    model: str | None = None,
    warm_up: bool = False,
) -> FastAPI:
    stt = stt or get_stt()
    tts = tts or get_tts(role="agent")
    log = latency_log or LatencyLog()
    agent_model = resolve_model(model or os.environ.get("HELPDESK_AGENT_MODEL", AGENT_MODEL))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up:
            # Load speech models before the first call so turn 1 isn't slow.
            await asyncio.gather(stt.warm_up(), tts.warm_up())
        yield

    app = FastAPI(title="helpdesk-voice", lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "call.html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "stt": stt.name,
            "tts": tts.name,
            "agent_model": agent_model,
            "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()) or client is not None,
        }

    @app.websocket("/ws")
    async def call(ws: WebSocket) -> None:
        await ws.accept()
        send_lock = asyncio.Lock()

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(payload))

        async def send_audio(seq: int, text: str, wav: bytes) -> None:
            async with send_lock:  # keep each audio header and its WAV frame together
                await ws.send_text(json.dumps({"type": "audio", "seq": seq, "text": text}))
                await ws.send_bytes(wav)

        try:
            llm = client or make_client()
        except MissingAPIKeyError as exc:
            await send_json({"type": "error", "message": str(exc)})
            await ws.close()
            return

        call_id = uuid.uuid4().hex[:8]
        conversation: list[Turn] = []
        usage = TokenUsage(model=agent_model)
        ended = False
        turn = 0
        await send_json({"type": "ready", "call_id": call_id, "stt": stt.name, "tts": tts.name, "agent_model": agent_model})
        await send_audio(0, GREETING, await tts.synthesize(GREETING))
        await send_json({"type": "turn_done", "turn": 0, "text": GREETING, "latency": None})

        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                audio, typed = message.get("bytes"), None
                if audio is None:
                    try:
                        data = json.loads(message.get("text") or "{}")
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "hangup":
                        break
                    if data.get("type") != "caller_text" or not str(data.get("text", "")).strip():
                        continue
                    typed = str(data["text"])
                if ended:
                    await send_json({"type": "error", "message": "The call has ended."})
                    continue
                turn += 1
                try:
                    result = await run_voice_turn(
                        conversation,
                        stt=stt,
                        tts=tts,
                        client=llm,
                        audio=audio,
                        caller_text=typed,
                        model=agent_model,
                        usage=usage,
                        on_caller_text=lambda t: send_json({"type": "caller_text", "text": t}),
                        on_agent_text=lambda d: send_json({"type": "agent_delta", "text": d}),
                        on_audio=send_audio,
                    )
                except Exception as exc:
                    await send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                    await send_json({"type": "turn_done", "turn": turn, "text": "", "latency": None})
                    continue
                log.write(
                    source="server",
                    call_id=call_id,
                    turn=turn,
                    latency=result.latency,
                    stt=stt.name,
                    tts=tts.name,
                    agent_model=agent_model,
                    caller_audio=audio is not None,
                )
                await send_json({"type": "turn_done", "turn": turn, "text": result.reply.text, "latency": result.latency.to_dict()})
                if result.reply.ticket is not None:
                    await send_json({"type": "ticket", "ticket_id": result.reply.ticket_id, "ticket": result.reply.ticket.model_dump(mode="json")})
                    await send_json({"type": "call_ended"})
                    ended = True
                elif count_agent_replies(conversation) >= HARNESS_MAX_TURNS:
                    await send_json({"type": "error", "message": f"Stopping after {HARNESS_MAX_TURNS} agent replies."})
                    await send_json({"type": "call_ended"})
                    ended = True
        except WebSocketDisconnect:
            pass

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m helpdesk.voice.server", description="Run the voice call server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None, help="agent model alias or id (default haiku)")
    args = parser.parse_args(argv)
    app = create_app(model=args.model, warm_up=True)
    print(f"Open http://{args.host}:{args.port}/ in Chrome, Edge or Firefox and allow the microphone.")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
