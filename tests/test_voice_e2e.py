"""Voice pipeline, WebSocket server, and the no-mic voice test, all with fake adapters."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from helpdesk import voice_test
from helpdesk.models import Turn
from helpdesk.voice.adapters import FakeSTT, FakeTTS
from helpdesk.voice.audio import embedded_text
from helpdesk.voice.pipeline import LatencyLog, run_voice_turn
from helpdesk.voice.server import GREETING, create_app
from tests.fake_api import GOOD_TICKET, FakeAPI, text, ticket_call

LATENCY_KEYS = {"stt_ms", "llm_first_token_ms", "tts_first_audio_ms", "first_audio_ms", "total_ms"}
FIXTURE_VOICE = Path(__file__).parent / "fixtures" / "tiny_piper_voice.onnx"


# ------------------------------------------------------------------ pipeline


def test_voice_turn_streams_sentence_by_sentence():
    api = FakeAPI(agent=[text("Sorry about that. Is the little Caps Lock light on? Take a look.")])
    convo: list[Turn] = []
    sent = []

    async def on_audio(seq, sentence, wav):
        sent.append((seq, sentence, embedded_text(wav)))

    caller_wav = asyncio.run(FakeTTS().synthesize("It says my password is wrong."))
    result = asyncio.run(
        run_voice_turn(convo, stt=FakeSTT(), tts=FakeTTS(), client=api.client(), audio=caller_wav, on_audio=on_audio)
    )
    assert result.caller_text == "It says my password is wrong."
    assert [s for _, s, _ in sent] == ["Sorry about that.", "Is the little Caps Lock light on?", "Take a look."]
    assert [seq for seq, _, _ in sent] == [0, 1, 2]
    assert all(s == t for _, s, t in sent)  # each WAV carries its own sentence
    lat = result.latency.to_dict()
    assert set(lat) == LATENCY_KEYS and all(v is not None and v >= 0 for v in lat.values())
    assert lat["first_audio_ms"] <= lat["total_ms"]
    assert [t.role for t in convo] == ["caller", "agent"]


def test_voice_turn_with_typed_text_skips_stt():
    api = FakeAPI(agent=[text("Okay.")])
    result = asyncio.run(run_voice_turn([], stt=FakeSTT(), tts=FakeTTS(), client=api.client(), caller_text="hello"))
    assert result.latency.stt_ms is None and result.caller_text == "hello"


def test_latency_log_appends_jsonl(tmp_path):
    api = FakeAPI(agent=[text("Okay.")])
    result = asyncio.run(run_voice_turn([], stt=FakeSTT(), tts=FakeTTS(), client=api.client(), caller_text="hi"))
    log = LatencyLog(tmp_path / "latency.jsonl")
    log.write(source="test", call_id="abc", turn=1, latency=result.latency)
    log.write(source="test", call_id="abc", turn=2, latency=result.latency)
    lines = [json.loads(l) for l in (tmp_path / "latency.jsonl").read_text().splitlines()]
    assert [l["turn"] for l in lines] == [1, 2] and LATENCY_KEYS <= set(lines[0])


# ------------------------------------------------------------------ server


def collect_until(ws, stop_type):
    """Read frames until a JSON frame of stop_type; pair each audio header with its WAV."""
    events, audio = [], []
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            audio.append(frame["bytes"])
            continue
        msg = json.loads(frame["text"])
        events.append(msg)
        if msg["type"] == stop_type:
            return events, audio


@pytest.fixture
def server(tmp_path):
    api = FakeAPI(
        agent=[
            text("Got it. What does the message on the screen say exactly?"),
            ticket_call(GOOD_TICKET),
            text("You're all set, have a good one."),
        ]
    )
    app = create_app(client=api.client(), stt=FakeSTT(), tts=FakeTTS(), latency_log=LatencyLog(tmp_path / "latency.jsonl"))
    with TestClient(app) as client:
        yield client, api, tmp_path / "latency.jsonl"


def test_static_page_and_health(server):
    client, _, _ = server
    page = client.get("/")
    assert page.status_code == 200 and "getUserMedia" in page.text and "/ws" in page.text
    health = client.get("/health").json()
    assert health["stt"] == "fake" and health["tts"] == "fake" and health["agent_model"] == "claude-haiku-5-5"


def test_websocket_call_audio_and_typed_turns(server):
    client, api, latency_file = server
    with client.websocket_connect("/ws") as ws:
        events, audio = collect_until(ws, "turn_done")
        assert events[0]["type"] == "ready"
        assert embedded_text(audio[0]) == GREETING

        caller_wav = asyncio.run(FakeTTS().synthesize("My computer won't let me in."))
        ws.send_bytes(caller_wav)
        events, audio = collect_until(ws, "turn_done")
        types = [e["type"] for e in events]
        assert events[0] == {"type": "caller_text", "text": "My computer won't let me in."}
        assert "agent_delta" in types and types.count("audio") == len(audio) == 2
        headers = [e for e in events if e["type"] == "audio"]
        assert [embedded_text(w) for w in audio] == [h["text"] for h in headers]
        done = events[-1]
        assert done["text"].startswith("Got it.") and set(done["latency"]) == LATENCY_KEYS

        ws.send_text(json.dumps({"type": "caller_text", "text": "Oh, Caps Lock was on. I'm in!"}))
        events, _ = collect_until(ws, "call_ended")
        ticket = next(e for e in events if e["type"] == "ticket")
        assert ticket["ticket"]["status"] == "resolved" and ticket["ticket_id"].startswith("HD-")

    lines = [json.loads(l) for l in latency_file.read_text().splitlines()]
    assert [l["turn"] for l in lines] == [1, 2]
    assert lines[0]["source"] == "server" and lines[0]["caller_audio"] is True and lines[1]["stt_ms"] is None
    # The agent's history was replayed append-only across turns.
    first, second = api.requests_for("agent")[0], api.requests_for("agent")[1]
    assert second["messages"][: len(first["messages"])] == first["messages"]


def test_websocket_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(stt=FakeSTT(), tts=FakeTTS(), latency_log=LatencyLog(tmp_path / "l.jsonl"))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error" and "ANTHROPIC_API_KEY" in msg["message"]


# ------------------------------------------------------------------ no-mic voice test


def scripted_agent(body):
    last = body["messages"][-1]
    if last["role"] == "user" and isinstance(last["content"], list) and last["content"][0].get("type") == "tool_result":
        return text("Great, you're all set. Bye!")
    if sum(1 for m in body["messages"] if m["role"] == "assistant") < 2:
        return text("Okay. What does the screen say? Read it to me.")
    return ticket_call(GOOD_TICKET)


def test_voice_test_cli_with_fakes(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HELPDESK_STT", raising=False)
    monkeypatch.delenv("HELPDESK_TTS", raising=False)
    api = FakeAPI(agent=scripted_agent, caller=["It says the password is incorrect.", "Oh! It worked."])
    latency = tmp_path / "latency.jsonl"
    code = voice_test.main(
        ["--scenario", "hd-001", "--out-dir", str(tmp_path / "calls"), "--latency-log", str(latency), "--grade"],
        client=api.client(),
    )
    assert code == 0
    call_dir = next((tmp_path / "calls").iterdir())
    record = json.loads((call_dir / "call.json").read_text())
    assert record["status"] == "ticket_submitted" and record["ticket"]["status"] == "resolved"
    assert record["stt"] == "fake" and record["tts"] == "fake"
    assert len(record["turns"]) == 3
    first = record["turns"][0]
    assert first["caller_said"].startswith("Hi, um, yeah") and first["caller_heard_as"] == first["caller_said"]
    assert first["agent_sentences"] == ["Okay.", "What does the screen say?", "Read it to me."]
    for t in record["turns"]:
        assert (call_dir / t["caller_audio"]).exists()
        assert all((call_dir / f).exists() for f in t["agent_audio"])
    assert record["grade"]["deterministic"]["passed"]
    assert set(record["latency_mean"]) == LATENCY_KEYS
    lines = latency.read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["source"] == "voice_test"
    assert "Mean latency (ms)" in capsys.readouterr().out


def test_voice_test_with_piper_selected_by_env(tmp_path, monkeypatch):
    """HELPDESK_TTS=piper runs the real piper-tts library (stand-in voice). FakeSTT can't read
    real audio, so the agent hears the canned transcript, as it would with any fake STT."""
    pytest.importorskip("piper")
    monkeypatch.setenv("HELPDESK_TTS", "piper")
    monkeypatch.setenv("HELPDESK_PIPER_VOICE", str(FIXTURE_VOICE))
    monkeypatch.delenv("HELPDESK_STT", raising=False)
    api = FakeAPI(agent=scripted_agent)
    voice_test.main(
        ["--scenario", "hd-003", "--max-turns", "2", "--out-dir", str(tmp_path), "--latency-log", str(tmp_path / "l.jsonl")],
        client=api.client(),
    )
    record = json.loads(next(tmp_path.glob("*/call.json")).read_text())
    assert record["tts"] == "piper"
    assert record["turns"][0]["caller_heard_as"] == "Sorry, it's still doing the same thing."


def test_voice_test_rejects_stub(capsys):
    assert voice_test.main(["--scenario", "hd-011"], client=FakeAPI().client()) == 2


def test_latency_report(tmp_path, capsys):
    from helpdesk import latency_report

    path = tmp_path / "latency.jsonl"
    rows = [
        {"source": "voice_test", "stt": "fake", "tts": "fake", "agent_model": "m", "stt_ms": v, "llm_first_token_ms": 2 * v,
         "tts_first_audio_ms": None, "first_audio_ms": 3 * v, "total_ms": 4 * v}
        for v in (10, 20, 30)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert latency_report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "| stt_ms | 20 | 20 | 28 |" in out and "| tts_first_audio_ms | – | – | – |" in out
