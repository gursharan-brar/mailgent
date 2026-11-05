"""The help desk agent.

`next_agent_message(conversation)` takes the call so far (a list of Turn, ending with the
caller) and returns the agent's next message. The agent can file the ticket with the
`submit_ticket` tool; when it does, the reply carries the validated Ticket and the call is over.

History is replayed append-only: each agent Turn keeps the exact API messages it produced
(including Haiku 5.5's thinking blocks), and the per-reply system note is derived from the
turn's position, so earlier messages are byte-identical on every request.
"""

from __future__ import annotations

import asyncio
import inspect
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from pydantic import ValidationError

from .config import (
    AGENT_MAX_TOKENS,
    AGENT_MODEL,
    AGENT_TURN_LIMIT,
    MID_CONVERSATION_SYSTEM_MODELS,
    env_effort,
    prompt_cache_enabled,
)
from .llm import TokenUsage, block_to_param, check_refusal, make_client, text_of
from .models import Ticket, Turn, ticket_tool_schema
from .prompts import load_prompt

SUBMIT_TICKET_TOOL: dict[str, Any] = {
    "name": "submit_ticket",
    "description": (
        "File the help desk ticket for this call. Call it exactly once, at the end of the call: "
        "when the caller has confirmed the fix works (status 'resolved'), when you escalate to an "
        "admin (status 'escalated'), or when the call ends without either (status 'unresolved'). "
        "After it succeeds, give the caller a short closing message."
    ),
    # Strict tool use: the API guarantees the input matches the schema. We still validate it
    # with the Ticket model, which also enforces the cross-field rules.
    "strict": True,
    "input_schema": ticket_tool_schema(),
}

# One reply may take a few API calls: text and/or a submit_ticket call, then (after the tool
# result) the closing line. Three rounds also allow one corrected resubmission.
MAX_TOOL_ROUNDS = 3

TextCallback = Callable[[str], Any] | Callable[[str], Awaitable[Any]]


@dataclass
class AgentReply:
    text: str
    ticket: Ticket | None = None
    ticket_id: str | None = None
    ticket_errors: list[str] = field(default_factory=list)
    api_messages: list[dict[str, Any]] = field(default_factory=list)
    first_token_ms: float | None = None
    stop_reason: str | None = None

    def as_turn(self) -> Turn:
        return Turn(role="agent", text=self.text, api_messages=self.api_messages)


def reply_note(reply_no: int, limit: int = AGENT_TURN_LIMIT) -> str:
    """The help desk software's note after each caller message."""
    if reply_no < limit:
        return f"Help desk system: this is your reply {reply_no} of at most {limit}."
    if reply_no == limit:
        return (
            f"Help desk system: this is reply {limit} of {limit}, your last. Call submit_ticket "
            "now: status resolved only if the caller already confirmed the fix works, "
            "otherwise escalated."
        )
    return (
        f"Help desk system: you are past the {limit}-reply limit. Call submit_ticket now "
        "with status escalated."
    )


def _note_via_system(model: str) -> bool:
    return model in MID_CONVERSATION_SYSTEM_MODELS


def build_agent_messages(
    conversation: Sequence[Turn], *, model: str = AGENT_MODEL, turn_limit: int = AGENT_TURN_LIMIT
) -> list[dict[str, Any]]:
    """Convert the call into API messages.

    Caller turns become user messages, each followed by the reply-count note (a
    mid-conversation system message on models that support it, otherwise a separate text
    block). Consecutive caller turns are merged. Agent turns replay their stored API
    messages; agent turns before the first caller turn (e.g. a spoken greeting) are skipped.
    """
    messages: list[dict[str, Any]] = []
    pending: list[str] = []
    agent_replies = 0

    def flush_caller() -> None:
        text = "\n".join(t for t in pending if t.strip()) or "(the caller said nothing)"
        note = reply_note(agent_replies + 1, turn_limit)
        if _note_via_system(model):
            messages.append({"role": "user", "content": text})
            messages.append({"role": "system", "content": note})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "text", "text": f"<help_desk_note>{note}</help_desk_note>"},
                    ],
                }
            )
        pending.clear()

    for turn in conversation:
        if turn.role == "caller":
            pending.append(turn.text)
            continue
        if pending:
            flush_caller()
        elif not messages:
            continue  # greeting before the caller spoke: not part of the API conversation
        if turn.api_messages:
            messages.extend(turn.api_messages)
        else:
            messages.append({"role": "assistant", "content": turn.text or "..."})
        agent_replies += 1
    if pending:
        flush_caller()
    return messages


def count_agent_replies(conversation: Sequence[Turn]) -> int:
    return sum(1 for t in conversation if t.role == "agent")


def new_ticket_id() -> str:
    return f"HD-{secrets.randbelow(90000) + 10000}"


async def _emit(callback: TextCallback | None, text: str) -> None:
    if callback is None or not text:
        return
    result = callback(text)
    if inspect.isawaitable(result):
        await result


async def next_agent_message(
    conversation: Sequence[Turn],
    *,
    client: Any = None,
    model: str = AGENT_MODEL,
    on_text: TextCallback | None = None,
    usage: TokenUsage | None = None,
    system_prompt: str | None = None,
    effort: str | None = None,
    turn_limit: int = AGENT_TURN_LIMIT,
) -> AgentReply:
    """Return the agent's next message for `conversation` (which must end with the caller).

    on_text, if given, receives the reply text as it streams (used by the voice layer).
    usage, if given, accumulates token counts.
    """
    if not conversation or conversation[-1].role != "caller":
        raise ValueError("conversation must end with a caller turn")
    client = client or make_client()
    usage = usage if usage is not None else TokenUsage(model=model)
    usage.model = usage.model or model
    system = system_prompt or load_prompt("agent.md")
    effort = effort or env_effort("HELPDESK_AGENT_EFFORT")

    history = build_agent_messages(conversation, model=model, turn_limit=turn_limit)
    new_messages: list[dict[str, Any]] = []
    texts: list[str] = []
    reply = AgentReply(text="")
    started = time.perf_counter()
    emitted_any = False

    for _ in range(MAX_TOOL_ROUNDS):
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": AGENT_MAX_TOKENS,
            "system": system,
            "tools": [SUBMIT_TICKET_TOOL],
            "messages": history + new_messages,
        }
        if effort:
            params["output_config"] = {"effort": effort}
        if prompt_cache_enabled():
            params["cache_control"] = {"type": "ephemeral"}

        async with client.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "text":
                    if emitted_any:
                        await _emit(on_text, " ")
                elif event.type == "text" and event.text:
                    if reply.first_token_ms is None:
                        reply.first_token_ms = (time.perf_counter() - started) * 1000
                    emitted_any = True
                    await _emit(on_text, event.text)
            message = await stream.get_final_message()

        usage.add(message.usage)
        reply.stop_reason = message.stop_reason
        check_refusal(message, "agent")
        new_messages.append(
            {"role": "assistant", "content": [block_to_param(b) for b in message.content]}
        )
        text = text_of(message)
        if text:
            texts.append(text)

        tool_uses = [b for b in message.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            break
        truncated = message.stop_reason == "max_tokens"
        results = [_handle_tool_use(b, reply, truncated) for b in tool_uses]
        new_messages.append({"role": "user", "content": results})

    reply.text = " ".join(texts).strip()
    reply.api_messages = new_messages
    return reply


def _handle_tool_use(block: Any, reply: AgentReply, truncated: bool) -> dict[str, Any]:
    def error(message: str) -> dict[str, Any]:
        return {"type": "tool_result", "tool_use_id": block.id, "content": message, "is_error": True}

    if str(block.name).lower() != "submit_ticket":
        return error(f"Unknown tool {block.name!r}. The only tool is submit_ticket.")
    if reply.ticket is not None:
        return error(
            f"Ticket {reply.ticket_id} was already submitted. Don't submit again; "
            "just give the caller your closing message."
        )
    if truncated:
        reply.ticket_errors.append("tool call was cut off by max_tokens")
        return error("Your submit_ticket call was cut off. Call submit_ticket again with every field.")
    try:
        ticket = Ticket.model_validate(block.input)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'ticket'}: {err['msg']}" for err in exc.errors()
        )
        reply.ticket_errors.append(problems)
        return error(f"Ticket rejected: {problems}. Fix those fields and call submit_ticket again.")
    reply.ticket = ticket
    reply.ticket_id = new_ticket_id()
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": (
            f"Ticket {reply.ticket_id} created with status '{ticket.status.value}'. "
            "Now give the caller a short closing message."
        ),
    }


def next_agent_message_sync(conversation: Sequence[Turn], **kwargs: Any) -> AgentReply:
    """Blocking wrapper around next_agent_message."""
    return asyncio.run(next_agent_message(conversation, **kwargs))
