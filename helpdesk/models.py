"""Pydantic models: the help desk Ticket, scenarios, and conversation turns."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------- ticket


class TicketStatus(StrEnum):
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    UNRESOLVED = "unresolved"


class Ticket(BaseModel):
    """The record the agent files at the end of every call (via the submit_ticket tool)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    caller_name: str = Field(min_length=1, description="Caller's name as they gave it.")
    device: str = Field(min_length=1, description="Device involved, e.g. 'Dell laptop, Windows 11'.")
    summary: str = Field(min_length=1, description="One or two sentences describing the problem.")
    steps_tried: list[str] = Field(
        default_factory=list, description="What the caller checked or did, in order."
    )
    root_cause: str = Field(
        min_length=1,
        description="The cause you established, 'Suspected: ...' if not confirmed, or 'Unknown'.",
    )
    resolution: str | None = Field(
        default=None, description="What fixed it. Empty if the problem was not resolved."
    )
    status: TicketStatus
    escalation_reason: str | None = Field(
        default=None,
        description="Why this is escalated and what the next tech should do. Empty if not escalated.",
    )

    @field_validator("resolution", "escalation_reason", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("steps_tried", mode="before")
    @classmethod
    def _drop_blank_steps(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [s for s in value if not (isinstance(s, str) and not s.strip())]
        return value

    @model_validator(mode="after")
    def _status_needs_detail(self) -> "Ticket":
        if self.status is TicketStatus.ESCALATED and not self.escalation_reason:
            raise ValueError("escalation_reason is required when status is 'escalated'")
        if self.status is TicketStatus.RESOLVED and not self.resolution:
            raise ValueError("resolution is required when status is 'resolved'")
        return self


def ticket_tool_schema() -> dict[str, Any]:
    """JSON schema for the submit_ticket tool input, written for strict tool use: every field
    required, no extra properties, blank strings instead of nulls."""
    descriptions = {name: field.description for name, field in Ticket.model_fields.items()}
    return {
        "type": "object",
        "properties": {
            "caller_name": {"type": "string", "description": descriptions["caller_name"]},
            "device": {"type": "string", "description": descriptions["device"]},
            "summary": {"type": "string", "description": descriptions["summary"]},
            "steps_tried": {
                "type": "array",
                "items": {"type": "string"},
                "description": descriptions["steps_tried"],
            },
            "root_cause": {"type": "string", "description": descriptions["root_cause"]},
            "resolution": {
                "type": "string",
                "description": descriptions["resolution"] + " Use an empty string if not resolved.",
            },
            "status": {
                "type": "string",
                "enum": [s.value for s in TicketStatus],
                "description": "resolved = caller confirmed the fix works; escalated = handed to an "
                "admin; unresolved = call ended without a fix or escalation.",
            },
            "escalation_reason": {
                "type": "string",
                "description": descriptions["escalation_reason"]
                + " Use an empty string if not escalated.",
            },
        },
        "required": list(Ticket.model_fields),
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- scenarios

Category = Literal["windows_lockout", "printer", "wifi", "email", "vpn", "slow_computer", "mfa"]


class CallerPersona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    tech_skill: Literal["low", "medium", "high"]
    mood: str = Field(min_length=1)


class Scenario(BaseModel):
    """A fully written scenario the harness can run."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^hd-\d{3}$")
    tier: int = Field(ge=1, le=3)
    category: Category
    device: str = Field(min_length=1)
    caller_persona: CallerPersona
    opening_line: str = Field(min_length=1)
    hidden_root_cause: str = Field(min_length=1)
    correct_fix_steps: list[str] = Field(min_length=1)
    red_herrings: list[str]
    success_criteria: list[str] = Field(min_length=1)
    must_escalate: bool

    @property
    def is_stub(self) -> bool:
        return False


class ScenarioStub(BaseModel):
    """A placeholder scenario: id, tier and a one-line problem. Skipped by the harness."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^hd-\d{3}$")
    tier: int = Field(ge=1, le=3)
    problem: str = Field(min_length=1)

    @property
    def is_stub(self) -> bool:
        return True


# --------------------------------------------------------------------------- conversation


class Turn(BaseModel):
    """One message in a call. `api_messages` holds the agent's exact API messages for that turn
    (assistant content including thinking blocks, plus any tool_result exchange) so the next
    request can replay history unchanged."""

    role: Literal["caller", "agent"]
    text: str
    api_messages: list[dict[str, Any]] | None = None

    def public(self) -> dict[str, str]:
        return {"role": self.role, "text": self.text}
