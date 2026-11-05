import pytest
from pydantic import ValidationError

from helpdesk.agent import SUBMIT_TICKET_TOOL
from helpdesk.models import Ticket, TicketStatus, ticket_tool_schema
from tests.fake_api import ESCALATED_TICKET, GOOD_TICKET


def test_valid_resolved_ticket():
    t = Ticket.model_validate(GOOD_TICKET)
    assert t.status is TicketStatus.RESOLVED
    assert t.escalation_reason is None  # blank string normalised


def test_valid_escalated_ticket():
    t = Ticket.model_validate(ESCALATED_TICKET)
    assert t.status is TicketStatus.ESCALATED
    assert t.resolution is None


def test_escalated_requires_reason():
    with pytest.raises(ValidationError, match="escalation_reason"):
        Ticket.model_validate({**ESCALATED_TICKET, "escalation_reason": "  "})


def test_resolved_requires_resolution():
    with pytest.raises(ValidationError, match="resolution"):
        Ticket.model_validate({**GOOD_TICKET, "resolution": ""})


def test_unresolved_needs_neither():
    t = Ticket.model_validate({**GOOD_TICKET, "status": "unresolved", "resolution": ""})
    assert t.status is TicketStatus.UNRESOLVED


@pytest.mark.parametrize(
    "patch",
    [{"status": "closed"}, {"caller_name": ""}, {"extra_field": 1}, {"steps_tried": "one step"}],
)
def test_rejects_bad_fields(patch):
    with pytest.raises(ValidationError):
        Ticket.model_validate({**GOOD_TICKET, **patch})


def test_blank_steps_are_dropped():
    t = Ticket.model_validate({**GOOD_TICKET, "steps_tried": ["a", " ", "b"]})
    assert t.steps_tried == ["a", "b"]


def test_tool_schema_matches_ticket_model():
    schema = ticket_tool_schema()
    assert set(schema["properties"]) == set(Ticket.model_fields)
    assert set(schema["required"]) == set(Ticket.model_fields)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["status"]["enum"] == [s.value for s in TicketStatus]
    assert SUBMIT_TICKET_TOOL["input_schema"] == schema
    assert SUBMIT_TICKET_TOOL["strict"] is True
