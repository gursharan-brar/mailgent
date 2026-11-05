"""Simulated caller: an LLM plays the person described by a scenario.

Each reply is a single-turn request: the caller's system prompt (persona, the secret root
cause, how the world reacts) plus the transcript so far, asking for the caller's next line.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .config import CALLER_MAX_TOKENS, CALLER_MODEL, env_effort, prompt_cache_enabled
from .llm import TokenUsage, check_refusal, make_client, text_of
from .models import Scenario, Turn
from .prompts import load_prompt, render


def caller_system_prompt(scenario: Scenario) -> str:
    persona = scenario.caller_persona
    return render(
        load_prompt("caller.md"),
        {
            "name": persona.name,
            "tech_skill": persona.tech_skill,
            "mood": persona.mood,
            "device": scenario.device,
            "opening_line": scenario.opening_line,
            "hidden_root_cause": scenario.hidden_root_cause,
            "correct_fix_steps": "\n".join(
                f"{i}. {step}" for i, step in enumerate(scenario.correct_fix_steps, 1)
            ),
            "red_herrings": "\n".join(f"- {r}" for r in scenario.red_herrings) or "- (none)",
        },
    )


def format_transcript(conversation: Sequence[Turn], caller_label: str = "YOU") -> str:
    return "\n".join(
        f"{caller_label if t.role == 'caller' else 'HELP DESK'}: {t.text.strip() or '(silence)'}"
        for t in conversation
    )


def caller_user_message(scenario: Scenario, conversation: Sequence[Turn]) -> str:
    name = scenario.caller_persona.name
    return (
        f"Here is the call so far. YOU are {name}.\n\n"
        f"<transcript>\n{format_transcript(conversation)}\n</transcript>\n\n"
        f"Write {name}'s next reply to the help desk's last message: only the spoken words."
    )


def clean_caller_text(text: str, name: str) -> str:
    """Strip labels and wrapping quotes the model sometimes adds."""
    text = text.strip()
    first = re.escape(name.split()[0]) if name.strip() else "CALLER"
    label = re.compile(rf"^(YOU|CALLER|{re.escape(name)}|{first})\s*:\s*", re.I)
    text = label.sub("", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text)


async def next_caller_message(
    scenario: Scenario,
    conversation: Sequence[Turn],
    *,
    client: Any = None,
    model: str = CALLER_MODEL,
    usage: TokenUsage | None = None,
    effort: str | None = None,
) -> str:
    """Return the simulated caller's reply to the agent's latest message."""
    if not conversation or conversation[-1].role != "agent":
        raise ValueError("conversation must end with an agent turn")
    client = client or make_client()
    usage = usage if usage is not None else TokenUsage(model=model)
    usage.model = usage.model or model
    effort = effort or env_effort("HELPDESK_CALLER_EFFORT")

    system: Any = caller_system_prompt(scenario)
    if prompt_cache_enabled():
        # The system prompt is fixed for the whole call; cache it.
        system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": CALLER_MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": caller_user_message(scenario, conversation)}],
    }
    if effort:
        params["output_config"] = {"effort": effort}

    message = await client.messages.create(**params)
    usage.add(message.usage)
    check_refusal(message, "caller")
    return clean_caller_text(text_of(message), scenario.caller_persona.name) or "Sorry, can you say that again?"
