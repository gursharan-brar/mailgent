"""Interactive text chat with the agent. You type the caller's side.

    python -m helpdesk.chat --scenario hd-001

The scenario brief (persona, hidden root cause, red herrings) is printed for you, the person
playing the caller; the agent never sees it. Type /quit to hang up. Transcripts are saved to
results/chats/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

from .agent import count_agent_replies, next_agent_message
from .config import AGENT_TURN_LIMIT, HARNESS_MAX_TURNS, RESULTS_DIR, resolve_model
from .llm import MissingAPIKeyError, RefusalError, TokenUsage, make_client
from .models import Scenario, Turn
from .scenarios import AnyScenario, get_scenario


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m helpdesk.chat", description=__doc__.splitlines()[0])
    p.add_argument("--scenario", required=True, help="scenario id, e.g. hd-001")
    p.add_argument("--model", default="haiku", help="agent model alias or id (default: haiku)")
    p.add_argument(
        "--no-brief", action="store_true", help="show only the persona, not the hidden details"
    )
    p.add_argument("--save-dir", type=Path, default=RESULTS_DIR / "chats")
    return p


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=96, initial_indent=indent, subsequent_indent=indent)


def print_brief(scenario: AnyScenario, out: TextIO, full: bool = True) -> None:
    print(f"=== Scenario {scenario.id} (tier {scenario.tier}) ===", file=out)
    if not isinstance(scenario, Scenario):
        print("This is a stub scenario. Problem:", file=out)
        print(_wrap(scenario.problem), file=out)
        print("Make up the details as you go.\n", file=out)
        return
    p = scenario.caller_persona
    print(f"You are {p.name} (tech skill: {p.tech_skill}; mood: {p.mood}).", file=out)
    print(f"Device: {scenario.device}", file=out)
    print("Opening line (press Enter at the first prompt to use it):", file=out)
    print(_wrap(scenario.opening_line), file=out)
    if full:
        print("Hidden root cause (don't say this; answer consistently with it):", file=out)
        print(_wrap(scenario.hidden_root_cause), file=out)
        print("What fixes it:", file=out)
        for i, step in enumerate(scenario.correct_fix_steps, 1):
            print(_wrap(f"{i}. {step}"), file=out)
        print("Red herrings (follow along if the agent goes there):", file=out)
        for item in scenario.red_herrings:
            print(_wrap(f"- {item}"), file=out)
        print(f"Expected outcome: {'escalated' if scenario.must_escalate else 'resolved'}", file=out)
    print("Type /quit to hang up.\n", file=out)


async def run_chat(
    scenario: AnyScenario,
    *,
    client: Any,
    model: str,
    input_fn: Callable[[str], str],
    out: TextIO,
    show_brief: bool = True,
) -> dict[str, Any]:
    print_brief(scenario, out, full=show_brief)
    conversation: list[Turn] = []
    usage = TokenUsage(model=model)
    ticket: dict[str, Any] | None = None
    ticket_id: str | None = None
    ended = "hangup"

    while True:
        try:
            line = await asyncio.to_thread(input_fn, "caller> ")
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            break
        line = line.strip()
        if line in ("/quit", "/exit", "/q"):
            break
        if not line:
            if not conversation and isinstance(scenario, Scenario):
                line = scenario.opening_line
                print(f"caller> {line}", file=out)
            else:
                continue
        conversation.append(Turn(role="caller", text=line))
        reply_no = count_agent_replies(conversation) + 1
        print(f"agent [{reply_no}/{AGENT_TURN_LIMIT}]> ", end="", file=out, flush=True)
        try:
            reply = await next_agent_message(
                conversation,
                client=client,
                model=model,
                usage=usage,
                on_text=lambda t: print(t, end="", file=out, flush=True),
            )
        except RefusalError as exc:
            print(f"\n[the model declined: {exc}]", file=out)
            ended = "refusal"
            break
        print(file=out)
        conversation.append(reply.as_turn())
        for problem in reply.ticket_errors:
            print(f"  [ticket rejected and sent back to the agent: {problem}]", file=out)
        if reply.ticket:
            ticket = reply.ticket.model_dump(mode="json")
            ticket_id = reply.ticket_id
            print(f"\n=== Ticket {ticket_id} ===", file=out)
            print(json.dumps(ticket, indent=2), file=out)
            ended = "ticket"
            break
        if count_agent_replies(conversation) >= HARNESS_MAX_TURNS:
            print(f"[stopping: {HARNESS_MAX_TURNS}-reply cap reached without a ticket]", file=out)
            ended = "turn_limit"
            break

    if ticket is None and ended != "refusal":
        print("[call ended without a ticket]", file=out)
    print(
        f"[tokens: {usage.input_tokens} in, {usage.output_tokens} out, "
        f"{usage.cache_read_input_tokens} cache read, {usage.cache_creation_input_tokens} cache write]",
        file=out,
    )
    return {
        "scenario_id": scenario.id,
        "agent_model": model,
        "ended": ended,
        "transcript": [t.public() for t in conversation],
        "ticket_id": ticket_id,
        "ticket": ticket,
        "tokens": usage.to_dict(),
    }


def main(
    argv: list[str] | None = None,
    *,
    client: Any = None,
    input_fn: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenario = get_scenario(args.scenario)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        client = client or make_client()
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    record = asyncio.run(
        run_chat(
            scenario,
            client=client,
            model=resolve_model(args.model),
            input_fn=input_fn,
            out=out,
            show_brief=not args.no_brief,
        )
    )
    if record["transcript"]:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = args.save_dir / f"{stamp}-{scenario.id}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[transcript saved to {path}]", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
