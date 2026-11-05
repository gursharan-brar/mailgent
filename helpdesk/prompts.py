"""Load prompt files from prompts/ and fill {{placeholders}}."""

from __future__ import annotations

import re
from functools import lru_cache

from .config import PROMPTS_DIR

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the text of prompts/<name>. Cached: prompts don't change during a run, which
    keeps the agent's system prompt byte-identical across turns (prompt caching, thinking replay)."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip() + "\n"


def render(template: str, values: dict[str, str]) -> str:
    """Replace {{key}} with values[key]. Unknown keys raise so typos are caught early."""

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"prompt placeholder {{{{{key}}}}} has no value")
        return str(values[key])

    return _PLACEHOLDER.sub(sub, template)
