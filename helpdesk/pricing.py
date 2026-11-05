"""Cost from token counts and prices.toml. Blank prices mean unknown, never zero."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PRICES_FILE
from .llm import TokenUsage


@dataclass(frozen=True)
class ModelPrice:
    input: float | None
    output: float | None
    cache_write: float | None
    cache_read: float | None


def _number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def load_prices(path: Path = PRICES_FILE) -> dict[str, ModelPrice]:
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        model: ModelPrice(
            input=_number(row.get("input")),
            output=_number(row.get("output")),
            cache_write=_number(row.get("cache_write")),
            cache_read=_number(row.get("cache_read")),
        )
        for model, row in (data.get("models") or {}).items()
    }


def cost_usd(usage: TokenUsage | dict[str, Any], prices: dict[str, ModelPrice]) -> float | None:
    """USD cost of `usage`, or None if any price it needs is blank."""
    if isinstance(usage, dict):
        usage = TokenUsage(**{k: v for k, v in usage.items() if k != "total"})
    if usage.total == 0:
        return 0.0
    price = prices.get(usage.model)
    if price is None or price.input is None or price.output is None:
        return None
    if usage.cache_creation_input_tokens and price.cache_write is None:
        return None
    if usage.cache_read_input_tokens and price.cache_read is None:
        return None
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_creation_input_tokens * (price.cache_write or 0.0)
        + usage.cache_read_input_tokens * (price.cache_read or 0.0)
    ) / 1_000_000


def sum_costs(costs: list[float | None]) -> float | None:
    """Total, or None if any part is unknown."""
    if any(c is None for c in costs):
        return None
    return float(sum(c for c in costs if c is not None))
