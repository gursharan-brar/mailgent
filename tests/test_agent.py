import asyncio

import pytest

from helpdesk.agent import (
    AgentReply,
    build_agent_messages,
    next_agent_message,
    reply_note,
)
from helpdesk.llm import RefusalError, TokenUsage
from helpdesk.models import Turn
from helpdesk.prompts import load_prompt
from tests.fake_api import ESCALATED_TICKET, GOOD_TICKET, FakeAPI, refusal, text, ticket_call


def run(coro):
    return asyncio.run(coro)


def caller(t):
    return Turn(role="caller", text=t)


# ------------------------------------------------------------------ message building


def test_first_request_is_caller_text_then_note():
    msgs = build_agent_messages([caller("my computer won't let me in")])
    assert msgs == [
        {"role": "user", "content": "my computer won't let me in"},
        {"role": "system", "content": reply_note(1)},
    ]


def test_note_goes_in_a_text_block_for_models_without_mid_conversation_system():
    msgs = build_agent_messages([caller("hi")], model="claude-haiku-4-5")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    blocks = msgs[0]["content"]
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert reply_note(1) in blocks[1]["text"]


def test_agent_turns_replay_stored_api_messages_and_count_replies():
    stored = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "Q1?"}]}]
    convo = [caller("a"), Turn(role="agent", text="Q1?", api_messages=stored), caller("b")]
    msgs = build_agent_messages(convo)
    assert msgs[2] == stored[0]
    assert msgs[-1] == {"role": "system", "content": reply_note(2)}


def test_consecutive_caller_turns_merge_and_leading_greeting_is_skipped():
    convo = [Turn(role="agent", text="Help desk, this is Alex."), caller("hello"), caller("are you there?")]
    msgs = build_agent_messages(convo)
    assert msgs[0] == {"role": "user", "content": "hello\nare you there?"}
    assert len(msgs) == 2


def test_reply_note_escalates_at_the_limit():
    assert "3 of at most 12" in reply_note(3)
    assert "your last" in reply_note(12) and "escalated" in reply_note(12)
    assert "past the 12-reply limit" in reply_note(13)


# ------------------------------------------------------------------ agent calls


def test_agent_asks_a_question_using_the_real_sdk():
    api = FakeAPI(agent=[text("Sorry about that! What does the screen say exactly?")])
    deltas = []
    usage = TokenUsage()
    reply = run(
        next_agent_message(
            [caller("my computer won't let me in")], client=api.client(), on_text=deltas.append, usage=usage
        )
    )
    assert isinstance(reply, AgentReply)
    assert reply.text == "Sorry about that! What does the screen say exactly?"
    assert "".join(deltas) == reply.text
    assert reply.ticket is None
    assert reply.first_token_ms is not None
    assert usage.calls == 1 and usage.output_tokens > 0

    body = api.requests[0]
    assert body["_path"] == "/v1/messages"
    assert body["model"] == "claude-haiku-5-5"
    assert body["stream"] is True
    assert body["system"] == load_prompt("agent.md")
    assert body["tools"][0]["name"] == "submit_ticket" and body["tools"][0]["strict"] is True
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][-1] == {"role": "system", "content": reply_note(1)}
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in body

    # The thinking block is kept, unchanged, for replay.
    blocks = reply.api_messages[0]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}


def test_agent_submits_ticket_then_gives_closing_line():
    api = FakeAPI(agent=[ticket_call(GOOD_TICKET), text("All set, Maria. Have a good day!")])
    reply = run(next_agent_message([caller("it worked!")], client=api.client()))
    assert reply.ticket is not None and reply.ticket.status.value == "resolved"
    assert reply.ticket_id.startswith("HD-")
    assert reply.text == "All set, Maria. Have a good day!"
    second = api.requests[1]["messages"]
    tool_result = second[-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "is_error" not in tool_result
    assert reply.ticket_id in tool_result["content"]
    # api_messages: assistant(tool_use), user(tool_result), assistant(closing)
    assert [m["role"] for m in reply.api_messages] == ["assistant", "user", "assistant"]


def test_invalid_ticket_is_returned_as_tool_error_and_retried():
    bad = {**ESCALATED_TICKET, "escalation_reason": ""}
    api = FakeAPI(agent=[ticket_call(bad), ticket_call(ESCALATED_TICKET), text("I've passed this to our admin.")])
    reply = run(next_agent_message([caller("still locked")], client=api.client()))
    assert reply.ticket is not None and reply.ticket.status.value == "escalated"
    assert len(reply.ticket_errors) == 1 and "escalation_reason" in reply.ticket_errors[0]
    error_result = api.requests[1]["messages"][-1]["content"][0]
    assert error_result["is_error"] is True and "escalation_reason" in error_result["content"]


def test_history_is_append_only_across_turns():
    """Each request's messages must start with the previous request's messages plus the reply
    exactly as returned (thinking blocks included); otherwise the API rejects replayed thinking."""
    api = FakeAPI(agent=[text("What does it say?"), text("Is Caps Lock on?"), ticket_call(GOOD_TICKET), text("Bye!")])
    client = api.client()
    convo = [caller("can't log in")]
    for line in ["password is incorrect", "oh, it was on. I'm in!"]:
        reply = run(next_agent_message(convo, client=client))
        convo += [reply.as_turn(), caller(line)]
    final = run(next_agent_message(convo, client=client))
    assert final.ticket is not None

    agent_requests = api.requests
    for earlier, later in zip(agent_requests, agent_requests[1:]):
        prefix = earlier["messages"]
        assert later["messages"][: len(prefix)] == prefix
        assert later["system"] == earlier["system"] and later["tools"] == earlier["tools"]


def test_refusal_raises():
    api = FakeAPI(agent=[refusal()])
    with pytest.raises(RefusalError):
        run(next_agent_message([caller("hello")], client=api.client()))


def test_conversation_must_end_with_caller():
    with pytest.raises(ValueError):
        run(next_agent_message([], client=FakeAPI().client()))
    with pytest.raises(ValueError):
        run(next_agent_message([caller("hi"), Turn(role="agent", text="hello")], client=FakeAPI().client()))


def test_effort_override(monkeypatch):
    monkeypatch.setenv("HELPDESK_AGENT_EFFORT", "low")
    api = FakeAPI()
    run(next_agent_message([caller("hi")], client=api.client()))
    assert api.requests[0]["output_config"] == {"effort": "low"}
