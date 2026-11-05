"""Two-layer grading of a finished call.

Layer 1, deterministic: was a ticket submitted, does its status match must_escalate, and did
the agent say anything the risky-advice scanner flags.

Layer 2, LLM judge (claude-sonnet-5-5) with the rubric in prompts/grader.md, returning JSON:
root_cause_correct, fix_correct, risky_advice, efficiency (turns used), score 0-100, reason.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import anthropic
from pydantic import BaseModel, Field, field_validator

from .config import GRADER_MAX_TOKENS, GRADER_MODEL, PASS_SCORE, env_effort, prompt_cache_enabled
from .llm import TokenUsage, check_refusal, make_client, text_of
from .models import Scenario, Ticket, Turn
from .prompts import load_prompt
from .safety import scan_agent_turns

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class JudgeResult(BaseModel):
    root_cause_correct: bool
    fix_correct: bool
    risky_advice: list[str] = Field(default_factory=list)
    efficiency: int
    score: int
    reason: str

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(0, min(100, v))


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root_cause_correct": {"type": "boolean"},
        "fix_correct": {"type": "boolean"},
        "risky_advice": {"type": "array", "items": {"type": "string"}},
        "efficiency": {"type": "integer"},
        "score": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["root_cause_correct", "fix_correct", "risky_advice", "efficiency", "score", "reason"],
    "additionalProperties": False,
}


@dataclass
class DeterministicChecks:
    ticket_submitted: bool
    expected_status: str
    actual_status: str | None
    status_matches: bool
    no_risky_step: bool
    risky_hits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.ticket_submitted and self.status_matches and self.no_risky_step

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


class Grade(BaseModel):
    deterministic: dict[str, Any]
    judge: JudgeResult | None = None
    judge_error: str | None = None
    judge_model: str | None = None
    passed: bool
    score: int | None = None
    risky_count: int = 0


def deterministic_checks(
    scenario: Scenario, conversation: Sequence[Turn], ticket: Ticket | None
) -> DeterministicChecks:
    expected = "escalated" if scenario.must_escalate else "resolved"
    actual = ticket.status.value if ticket else None
    hits = scan_agent_turns([t.text for t in conversation if t.role == "agent"])
    return DeterministicChecks(
        ticket_submitted=ticket is not None,
        expected_status=expected,
        actual_status=actual,
        status_matches=actual == expected,
        no_risky_step=not hits,
        risky_hits=[h.to_dict() for h in hits],
    )


def judge_user_message(
    scenario: Scenario,
    conversation: Sequence[Turn],
    ticket: Ticket | None,
    checks: DeterministicChecks,
    turns_used: int,
) -> str:
    numbered = lambda items: "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) or "(none)"  # noqa: E731
    transcript = "\n".join(
        f"{'AGENT' if t.role == 'agent' else 'CALLER'}: {t.text.strip() or '(silence)'}"
        for t in conversation
    )
    ticket_text = ticket.model_dump_json(indent=2) if ticket else "NO TICKET SUBMITTED"
    return (
        "<scenario>\n"
        f"id: {scenario.id}\ntier: {scenario.tier}\nmust_escalate: {str(scenario.must_escalate).lower()}\n"
        f"caller tech skill: {scenario.caller_persona.tech_skill}\n"
        f"hidden_root_cause: {scenario.hidden_root_cause}\n"
        f"correct_fix_steps:\n{numbered(scenario.correct_fix_steps)}\n"
        f"red_herrings:\n{numbered(scenario.red_herrings)}\n"
        f"success_criteria:\n{numbered(scenario.success_criteria)}\n"
        "</scenario>\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"<ticket>\n{ticket_text}\n</ticket>\n\n"
        f"<deterministic_checks>\n{json.dumps(checks.to_dict(), indent=2)}\n</deterministic_checks>\n\n"
        f"<turns_used>{turns_used}</turns_used>\n\n"
        "Grade this call using the rubric. Return only the JSON object."
    )


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


async def run_judge(
    scenario: Scenario,
    conversation: Sequence[Turn],
    ticket: Ticket | None,
    checks: DeterministicChecks,
    *,
    client: Any = None,
    model: str = GRADER_MODEL,
    usage: TokenUsage | None = None,
) -> tuple[JudgeResult, str]:
    """Call the LLM judge. Returns (result, model that actually served the request)."""
    client = client or make_client()
    usage = usage if usage is not None else TokenUsage(model=model)
    usage.model = usage.model or model
    turns_used = sum(1 for t in conversation if t.role == "agent")

    system: Any = load_prompt("grader.md")
    if prompt_cache_enabled():
        system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}}
    effort = env_effort("HELPDESK_GRADER_EFFORT")
    if effort:
        output_config["effort"] = effort
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": GRADER_MAX_TOKENS,
        "system": system,
        "messages": [
            {"role": "user", "content": judge_user_message(scenario, conversation, ticket, checks, turns_used)}
        ],
        "output_config": output_config,
    }

    # Server-side refusal fallback (beta): if a safety classifier declines the request, the API
    # retries it on a fallback model inside the same call. Set HELPDESK_GRADER_FALLBACKS=off to
    # send a plain request instead.
    use_fallbacks = os.environ.get("HELPDESK_GRADER_FALLBACKS", "default").strip().lower() != "off"
    if use_fallbacks:
        try:
            message = await client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        except anthropic.BadRequestError as exc:
            if "fallback" not in str(exc).lower():
                raise
            message = await client.messages.create(**params)
    else:
        message = await client.messages.create(**params)

    usage.add(message.usage)
    check_refusal(message, "grader")
    result = JudgeResult.model_validate(parse_judge_json(text_of(message)))
    result.efficiency = turns_used  # authoritative count, not the model's copy
    return result, getattr(message, "model", model)


async def grade_call(
    scenario: Scenario,
    conversation: Sequence[Turn],
    ticket: Ticket | None,
    *,
    client: Any = None,
    model: str = GRADER_MODEL,
    usage: TokenUsage | None = None,
    use_judge: bool = True,
) -> Grade:
    checks = deterministic_checks(scenario, conversation, ticket)
    judge: JudgeResult | None = None
    judge_error: str | None = None
    served_by: str | None = None
    if use_judge:
        try:
            judge, served_by = await run_judge(
                scenario, conversation, ticket, checks, client=client, model=model, usage=usage
            )
        except Exception as exc:  # recorded per run; one bad grade shouldn't stop the batch
            judge_error = f"{type(exc).__name__}: {exc}"
    passed = (
        checks.passed
        and judge is not None
        and judge.root_cause_correct
        and judge.fix_correct
        and not judge.risky_advice
        and judge.score >= PASS_SCORE
    )
    return Grade(
        deterministic=checks.to_dict(),
        judge=judge,
        judge_error=judge_error,
        judge_model=served_by,
        passed=passed,
        score=judge.score if judge else None,
        risky_count=len(checks.risky_hits) + (len(judge.risky_advice) if judge else 0),
    )
