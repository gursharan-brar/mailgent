"""Anthropic client creation, token accounting, and small response helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import anthropic


class MissingAPIKeyError(RuntimeError):
    pass


class RefusalError(RuntimeError):
    """The model declined the request (stop_reason == 'refusal')."""

    def __init__(self, who: str, category: str | None, explanation: str | None):
        self.who, self.category, self.explanation = who, category, explanation
        super().__init__(f"{who}: model declined (category={category!r}): {explanation or ''}".strip())


def make_client(max_retries: int = 4) -> anthropic.AsyncAnthropic:
    """Async client using the key in ANTHROPIC_API_KEY."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Run `export ANTHROPIC_API_KEY=...` and try again."
        )
    return anthropic.AsyncAnthropic(api_key=key, max_retries=max_retries)


@dataclass
class TokenUsage:
    """Running token totals for one role (agent, caller or grader) and one model."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.calls += 1

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total": self.total}


def check_refusal(message: Any, who: str) -> None:
    if getattr(message, "stop_reason", None) == "refusal":
        details = getattr(message, "stop_details", None)
        raise RefusalError(
            who,
            getattr(details, "category", None) if details else None,
            getattr(details, "explanation", None) if details else None,
        )


def text_of(message: Any) -> str:
    """Join the text blocks of a response (skipping thinking and tool_use blocks)."""
    parts = [b.text.strip() for b in message.content if getattr(b, "type", None) == "text"]
    return " ".join(p for p in parts if p)


def block_to_param(block: Any) -> dict[str, Any]:
    """Turn a response content block into a request param, unchanged, for replaying history.
    Thinking blocks keep their (possibly empty) text and signature exactly as returned."""
    if isinstance(block, dict):
        data = dict(block)
    elif hasattr(block, "to_dict"):
        data = block.to_dict()
    else:
        data = block.model_dump(exclude_unset=True)
    return {k: v for k, v in data.items() if v is not None}
