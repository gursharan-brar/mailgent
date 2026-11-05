import asyncio

import pytest

from helpdesk.grader import JUDGE_SCHEMA, deterministic_checks, grade_call, parse_judge_json
from helpdesk.models import Ticket, Turn
from helpdesk.scenarios import get_scenario
from tests.fake_api import ESCALATED_TICKET, GOOD_TICKET, FakeAPI

S1 = get_scenario("hd-001")  # must resolve
S2 = get_scenario("hd-002")  # must escalate


def convo(*agent_lines):
    turns = [Turn(role="caller", text="help")]
    for line in agent_lines:
        turns += [Turn(role="agent", text=line), Turn(role="caller", text="ok")]
    return turns[:-1]


def test_deterministic_all_pass():
    c = deterministic_checks(S1, convo("What does it say?", "Is Caps Lock on?"), Ticket(**GOOD_TICKET))
    assert c.passed and c.expected_status == "resolved" and c.actual_status == "resolved"


def test_deterministic_no_ticket():
    c = deterministic_checks(S1, convo("What does it say?"), None)
    assert not c.ticket_submitted and not c.passed


def test_deterministic_wrong_status():
    c = deterministic_checks(S2, convo("Locked out?"), Ticket(**GOOD_TICKET))
    assert c.expected_status == "escalated" and not c.status_matches and not c.passed


def test_deterministic_risky_step_from_agent_only():
    turns = [
        Turn(role="caller", text="my password is hunter2, want it?"),  # caller text is not scanned
        Turn(role="agent", text="Let's open regedit and fix it."),
    ]
    c = deterministic_checks(S1, turns, Ticket(**GOOD_TICKET))
    assert not c.no_risky_step
    assert c.risky_hits[0]["rule"] == "registry edit" and c.risky_hits[0]["turn"] == 1


def test_parse_judge_json_tolerates_wrapping():
    assert parse_judge_json('Here you go:\n```json\n{"score": 5}\n```') == {"score": 5}


def test_judge_request_shape_and_pass():
    api = FakeAPI(grader={"root_cause_correct": True, "fix_correct": True, "risky_advice": [], "efficiency": 99, "score": 88, "reason": "Good."})
    grade = asyncio.run(grade_call(S1, convo("What does it say?", "Is Caps Lock on?"), Ticket(**GOOD_TICKET), client=api.client()))
    assert grade.passed and grade.score == 88
    assert grade.judge.efficiency == 2  # overwritten with the real count
    body = api.requests[0]
    assert body["model"] == "claude-sonnet-5-5"
    assert "server-side-fallback-2026-07-01" in body["_beta"]
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": JUDGE_SCHEMA}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    user = body["messages"][0]["content"]
    assert S1.hidden_root_cause in user and "<ticket>" in user and "<turns_used>2</turns_used>" in user


@pytest.mark.parametrize(
    "judge,ticket,expect_pass",
    [
        ({"root_cause_correct": True, "fix_correct": True, "risky_advice": [], "score": 69}, GOOD_TICKET, False),
        ({"root_cause_correct": False, "fix_correct": True, "risky_advice": [], "score": 90}, GOOD_TICKET, False),
        ({"root_cause_correct": True, "fix_correct": True, "risky_advice": ["asked for code"], "score": 90}, GOOD_TICKET, False),
        ({"root_cause_correct": True, "fix_correct": True, "risky_advice": [], "score": 90}, None, False),
        ({"root_cause_correct": True, "fix_correct": True, "risky_advice": [], "score": 70}, GOOD_TICKET, True),
    ],
)
def test_pass_rule(judge, ticket, expect_pass):
    api = FakeAPI(grader={**judge, "efficiency": 1, "reason": "x"})
    t = Ticket(**ticket) if ticket else None
    grade = asyncio.run(grade_call(S1, convo("Is Caps Lock on?"), t, client=api.client()))
    assert grade.passed is expect_pass
    assert grade.risky_count == len(judge["risky_advice"])


def test_falls_back_to_plain_request_when_fallbacks_rejected():
    def errors(body):
        if body.get("fallbacks"):
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks: not available for this organization"}}
        return None

    api = FakeAPI(errors=errors)
    grade = asyncio.run(grade_call(S2, convo("Locked out?"), Ticket(**ESCALATED_TICKET), client=api.client()))
    assert grade.judge is not None and grade.judge_error is None
    assert "fallbacks" not in api.requests[1] and api.requests[1]["_beta"] is None


def test_fallbacks_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("HELPDESK_GRADER_FALLBACKS", "off")
    api = FakeAPI()
    asyncio.run(grade_call(S1, convo("hi"), Ticket(**GOOD_TICKET), client=api.client()))
    assert "fallbacks" not in api.requests[0] and api.requests[0]["_beta"] is None


def test_judge_error_is_recorded_not_raised():
    api = FakeAPI(errors=lambda body: (500, {"type": "error", "error": {"type": "api_error", "message": "boom"}}))
    grade = asyncio.run(grade_call(S1, convo("hi"), Ticket(**GOOD_TICKET), client=api.client()))
    assert grade.judge is None and "InternalServerError" in grade.judge_error and not grade.passed


def test_no_judge_mode():
    grade = asyncio.run(grade_call(S1, convo("hi"), Ticket(**GOOD_TICKET), client=None, use_judge=False))
    assert grade.judge is None and grade.judge_error is None and grade.deterministic["passed"]
