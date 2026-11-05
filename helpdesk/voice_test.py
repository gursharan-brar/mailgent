"""No-mic end-to-end voice test.

    python -m helpdesk.voice_test --scenario hd-001

The simulated caller's lines are spoken with the TTS adapter, transcribed with the STT adapter,
and answered by the agent, whose reply is streamed and spoken sentence by sentence: the same
pipeline as the browser call, with no audio hardware. The agent hears only what STT produced.

Writes results/calls/<timestamp>-<scenario>/call.json plus every WAV file, and appends one line
per turn to results/latency.jsonl. Uses the adapters chosen by HELPDESK_STT / HELPDESK_TTS
(default: fakes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .caller import next_caller_message
from .config import CALLER_MODEL, DEFAULT_RUN_TOKEN_LIMIT, GRADER_MODEL, HARNESS_MAX_TURNS, RESULTS_DIR, resolve_model
from .grader import grade_call
from .llm import MissingAPIKeyError, TokenUsage, make_client
from .models import Scenario, Turn
from .scenarios import get_scenario
from .voice.adapters import STTAdapter, TTSAdapter, get_stt, get_tts
from .voice.audio import wav_info
from .voice.pipeline import LatencyLog, run_voice_turn

LATENCY_FIELDS = ("stt_ms", "llm_first_token_ms", "tts_first_audio_ms", "first_audio_ms", "total_ms")


async def run_voice_call(
    scenario: Scenario,
    *,
    client: Any,
    stt: STTAdapter,
    agent_tts: TTSAdapter,
    caller_tts: TTSAdapter,
    model: str,
    out_dir: Path,
    latency_log: LatencyLog,
    max_turns: int = HARNESS_MAX_TURNS,
    grade: bool = False,
    log=print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    call_id = uuid.uuid4().hex[:8]
    agent_usage, caller_usage, grader_usage = TokenUsage(model=model), TokenUsage(model=CALLER_MODEL), TokenUsage(model=GRADER_MODEL)
    heard: list[Turn] = []        # the agent's view: caller turns as STT heard them
    intended: list[Turn] = []     # the caller's view: what the caller actually said
    turns: list[dict[str, Any]] = []
    caller_line = scenario.opening_line
    ticket = ticket_id = None
    status = "turn_limit"

    for n in range(1, max_turns + 1):
        if agent_usage.total + caller_usage.total >= DEFAULT_RUN_TOKEN_LIMIT:
            status = "token_limit"
            break
        caller_wav = await caller_tts.synthesize(caller_line)
        caller_file = out_dir / f"turn{n:02d}_caller.wav"
        caller_file.write_bytes(caller_wav)
        agent_files: list[str] = []

        async def save_audio(seq: int, sentence: str, wav: bytes, n: int = n) -> None:
            path = out_dir / f"turn{n:02d}_agent_{seq:02d}.wav"
            path.write_bytes(wav)
            agent_files.append(path.name)

        result = await run_voice_turn(
            heard, stt=stt, tts=agent_tts, client=client, audio=caller_wav,
            model=model, usage=agent_usage, on_audio=save_audio,
        )
        intended += [Turn(role="caller", text=caller_line), Turn(role="agent", text=result.reply.text)]
        latency = result.latency.to_dict()
        latency_log.write(
            source="voice_test", call_id=call_id, turn=n, latency=result.latency,
            scenario_id=scenario.id, stt=stt.name, tts=agent_tts.name, agent_model=model,
            caller_audio_s=round(wav_info(caller_wav).duration_s, 2),
        )
        turns.append({
            "turn": n,
            "caller_said": caller_line,
            "caller_heard_as": result.caller_text,
            "agent_said": result.reply.text,
            "agent_sentences": [s for s, _ in result.audio],
            "caller_audio": caller_file.name,
            "agent_audio": agent_files,
            "latency": latency,
        })
        log(f"turn {n}: caller: {caller_line}")
        if result.caller_text != caller_line:
            log(f"         heard as: {result.caller_text}")
        log(f"         agent: {result.reply.text}")
        log("         latency: " + ", ".join(f"{k}={latency[k]}" for k in LATENCY_FIELDS))
        if result.reply.ticket is not None:
            ticket, ticket_id, status = result.reply.ticket, result.reply.ticket_id, "ticket_submitted"
            break
        if n == max_turns:
            break
        caller_line = await next_caller_message(scenario, intended, client=client, usage=caller_usage)

    record: dict[str, Any] = {
        "call_id": call_id,
        "scenario_id": scenario.id,
        "agent_model": model,
        "stt": stt.name,
        "tts": agent_tts.name,
        "status": status,
        "turns": turns,
        "ticket_id": ticket_id,
        "ticket": ticket.model_dump(mode="json") if ticket else None,
        "tokens": {"agent": agent_usage.to_dict(), "caller": caller_usage.to_dict()},
        "latency_mean": {
            k: round(mean(v), 1) if (v := [t["latency"][k] for t in turns if t["latency"][k] is not None]) else None
            for k in LATENCY_FIELDS
        },
    }
    if grade:
        graded = await grade_call(scenario, heard, ticket, client=client, usage=grader_usage)
        record["grade"] = graded.model_dump(mode="json")
        record["tokens"]["grader"] = grader_usage.to_dict()
    (out_dir / "call.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m helpdesk.voice_test", description="No-mic end-to-end voice test.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--model", default="haiku", help="agent model alias or id (default haiku)")
    parser.add_argument("--max-turns", type=int, default=HARNESS_MAX_TURNS)
    parser.add_argument("--grade", action="store_true", help="also grade the call with the grader")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "calls")
    parser.add_argument("--latency-log", type=Path, default=None, help="default results/latency.jsonl")
    args = parser.parse_args(argv)

    scenario = get_scenario(args.scenario)
    if not isinstance(scenario, Scenario):
        print(f"error: {scenario.id} is a stub scenario; pick one of hd-001..hd-010", file=sys.stderr)
        return 2
    try:
        client = client or make_client()
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stt, agent_tts, caller_tts = get_stt(), get_tts(role="agent"), get_tts(role="caller")
    out_dir = args.out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{scenario.id}"
    print(f"Voice test {scenario.id}: STT={stt.name}, TTS={agent_tts.name}, agent={resolve_model(args.model)}")
    record = asyncio.run(
        run_voice_call(
            scenario, client=client, stt=stt, agent_tts=agent_tts, caller_tts=caller_tts,
            model=resolve_model(args.model), out_dir=out_dir, latency_log=LatencyLog(args.latency_log),
            max_turns=args.max_turns, grade=args.grade,
        )
    )
    print(f"\nStatus: {record['status']}; ticket: {(record['ticket'] or {}).get('status', 'none')}")
    if "grade" in record:
        g = record["grade"]
        print(f"Grade: score={g.get('score')} passed={g.get('passed')}")
    print("Mean latency (ms): " + ", ".join(f"{k}={v}" for k, v in record["latency_mean"].items()))
    print(f"Saved {out_dir / 'call.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
