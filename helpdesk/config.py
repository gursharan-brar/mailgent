"""Paths, model IDs, and limits. Environment variables override the defaults noted below."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
STATIC_DIR = PROJECT_ROOT / "static"
PRICES_FILE = PROJECT_ROOT / "prices.toml"

# Models (fixed by the project spec).
AGENT_MODEL = "claude-haiku-5-5"
CALLER_MODEL = "claude-haiku-5-5"
GRADER_MODEL = "claude-sonnet-5-5"

# Short names accepted by `--model` / `--models`. Full model IDs are accepted too.
MODEL_ALIASES = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

# Models that accept mid-conversation {"role": "system"} messages. For any other model the
# agent's per-turn note is sent as a separate text block in the user message instead.
MID_CONVERSATION_SYSTEM_MODELS = {
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-fable-5-1",
}

# The agent must submit its ticket (escalating if not fixed) by this reply.
AGENT_TURN_LIMIT = 12
# The harness stops a call after this many agent replies no matter what.
HARNESS_MAX_TURNS = 15
# Per-run cap on agent + caller tokens (input + output + cache). The grader is not counted.
DEFAULT_RUN_TOKEN_LIMIT = 200_000

# max_tokens per request. Haiku 5.5 / Sonnet 5.5 think by default and thinking counts toward this.
AGENT_MAX_TOKENS = 8_000
CALLER_MAX_TOKENS = 4_000
GRADER_MAX_TOKENS = 16_000

# A run passes only if the deterministic checks pass, the judge says root cause and fix are
# correct, no risky advice was found, and the judge score is at least this.
PASS_SCORE = int(os.environ.get("HELPDESK_PASS_SCORE", "70"))


def resolve_model(name: str) -> str:
    """Map an alias like 'haiku' to a model ID; pass full IDs through unchanged."""
    return MODEL_ALIASES.get(name.strip().lower(), name.strip())


def env_effort(var: str) -> str | None:
    """Optional effort override (low|medium|high|xhigh|max). Unset = the model's default."""
    value = os.environ.get(var, "").strip().lower()
    return value or None


def prompt_cache_enabled() -> bool:
    return os.environ.get("HELPDESK_PROMPT_CACHE", "1").strip() not in ("0", "false", "no", "off")
