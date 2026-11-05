import asyncio
import re

import pytest

from helpdesk.caller import (
    caller_system_prompt,
    caller_user_message,
    clean_caller_text,
    next_caller_message,
)
from helpdesk.models import Turn
from helpdesk.scenarios import get_scenario, load_full
from tests.fake_api import FakeAPI


@pytest.mark.parametrize("scenario", load_full(), ids=lambda s: s.id)
def test_system_prompt_is_fully_rendered(scenario):
    prompt = caller_system_prompt(scenario)
    assert "{{" not in prompt and "}}" not in prompt
    assert scenario.caller_persona.name in prompt
    assert scenario.hidden_root_cause in prompt  # the caller needs the truth to stay consistent
    for herring in scenario.red_herrings:
        assert herring in prompt


def test_user_message_shows_transcript_from_callers_view():
    s = get_scenario("hd-001")
    convo = [Turn(role="caller", text=s.opening_line), Turn(role="agent", text="What does the screen say?")]
    msg = caller_user_message(s, convo)
    assert "YOU are Maria Lopez" in msg
    assert f"YOU: {s.opening_line}" in msg
    assert "HELP DESK: What does the screen say?" in msg
    # The hidden cause is only in the system prompt, not repeated here.
    assert s.hidden_root_cause not in msg


@pytest.mark.parametrize(
    "raw,clean",
    [
        ('"It says the password is incorrect."', "It says the password is incorrect."),
        ("Maria: Um, it says incorrect?", "Um, it says incorrect?"),
        ("YOU:   okay   done", "okay done"),
        ("CALLER: hi", "hi"),
    ],
)
def test_clean_caller_text(raw, clean):
    assert clean_caller_text(raw, "Maria Lopez") == clean


def test_caller_request_uses_haiku_no_tools_and_cached_system():
    s = get_scenario("hd-003")
    api = FakeAPI(caller=["Gloria: It pops up a little box asking where to save it!"])
    convo = [Turn(role="caller", text=s.opening_line), Turn(role="agent", text="What happens when you click Print?")]
    line = asyncio.run(next_caller_message(s, convo, client=api.client()))
    assert line == "It pops up a little box asking where to save it!"
    body = api.requests[0]
    assert body["model"] == "claude-haiku-5-5"
    assert "tools" not in body and "stream" not in body
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert re.search(r"Gloria Abernathy", body["system"][0]["text"])
    assert body["messages"][-1]["role"] == "user"


def test_caller_needs_agent_last():
    s = get_scenario("hd-001")
    with pytest.raises(ValueError):
        asyncio.run(next_caller_message(s, [Turn(role="caller", text="hi")], client=FakeAPI().client()))
