import io
import json

from helpdesk import chat
from tests.fake_api import GOOD_TICKET, FakeAPI, text, ticket_call


def test_chat_cli_runs_a_call_and_saves_transcript(tmp_path):
    api = FakeAPI(
        agent=[
            text("Hi Maria, what does the screen say?"),
            ticket_call(GOOD_TICKET),
            text("Great, you're all set."),
        ]
    )
    lines = iter(["", "it says the password is incorrect, oh wait caps lock, I'm in!"])
    out = io.StringIO()
    code = chat.main(
        ["--scenario", "hd-001", "--save-dir", str(tmp_path)],
        client=api.client(),
        input_fn=lambda prompt: next(lines),
        out=out,
    )
    assert code == 0
    shown = out.getvalue()
    assert "Hidden root cause" in shown  # brief is for the human caller
    assert "Hi Maria, what does the screen say?" in shown
    assert "=== Ticket HD-" in shown
    saved = list(tmp_path.glob("*-hd-001.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text())
    assert record["ended"] == "ticket"
    assert record["transcript"][0]["text"].startswith("Hi, um, yeah")  # Enter used the opening line
    assert record["ticket"]["status"] == "resolved"


def test_chat_cli_without_key_exits_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert chat.main(["--scenario", "hd-001"]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_chat_cli_stub_scenario_needs_typed_opening(tmp_path):
    api = FakeAPI(agent=[text("Okay, when did that start?")])
    lines = iter(["", "my scans stopped arriving", "/quit"])
    out = io.StringIO()
    chat.main(
        ["--scenario", "hd-015", "--save-dir", str(tmp_path)],
        client=api.client(),
        input_fn=lambda prompt: next(lines),
        out=out,
    )
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["transcript"][0] == {"role": "caller", "text": "my scans stopped arriving"}
    assert record["ended"] == "hangup"
