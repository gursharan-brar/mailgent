import json

import pytest

from helpdesk import report, run
from helpdesk.config import PRICES_FILE
from helpdesk.pricing import ModelPrice, cost_usd, load_prices
from helpdesk.llm import TokenUsage
from tests.fake_api import ESCALATED_TICKET, GOOD_TICKET, FakeAPI, text, ticket_call

REQUIRED_KEYS = {
    "run_id", "scenario_id", "tier", "agent_model", "caller_model", "grader_model", "run_index",
    "status", "error", "turns", "transcript", "ticket", "grade", "tokens", "cost_usd", "wall_time_s",
}


def scripted_agent(body):
    """Ask twice, then file a ticket whose status follows what the caller said."""
    agent_turns = sum(1 for m in body["messages"] if m["role"] == "assistant")
    last = body["messages"][-1]
    if last["role"] == "user" and isinstance(last["content"], list) and last["content"][0].get("type") == "tool_result":
        return text("Thanks, you're all set.")
    if agent_turns < 2:
        return text(f"Question {agent_turns + 1}: what do you see?")
    opening = body["messages"][0]["content"]
    return ticket_call(ESCALATED_TICKET if "laptop" in opening else GOOD_TICKET)


def test_harness_writes_jsonl_and_report(tmp_path, capsys):
    api = FakeAPI(agent=scripted_agent, caller=["It says incorrect.", "Oh, that fixed it!"])
    code = run.main(
        ["--runs", "2", "--models", "haiku", "--scenarios", "hd-001,hd-002", "--concurrency", "3", "--out-dir", str(tmp_path)],
        client=api.client(),
    )
    assert code == 0
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    records = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert len(records) == 4
    for rec in records:
        assert REQUIRED_KEYS <= set(rec)
        assert rec["status"] == "ticket_submitted"
        assert rec["agent_model"] == "claude-haiku-5-5" and rec["grader_model"] == "claude-sonnet-5-5"
        assert rec["turns"] == 3
        assert rec["transcript"][0]["role"] == "caller"
        assert rec["tokens"]["agent"]["total"] > 0 and rec["tokens"]["grader"]["calls"] == 1
        assert rec["cost_usd"]["total"] is None  # prices.toml is blank
        assert rec["grade"]["deterministic"]["passed"]
    assert files[0].with_suffix(".md").exists()
    out = capsys.readouterr().out
    assert "| hd-001 |" in out and "| hd-002 |" in out and "### Overall" in out


def test_turn_cap(tmp_path):
    api = FakeAPI(agent=[text("Hmm, what else do you see?")], caller=["Nothing new."])
    run.main(["--runs", "1", "--scenarios", "hd-005", "--max-turns", "4", "--out-dir", str(tmp_path)], client=api.client())
    rec = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert rec["status"] == "turn_limit" and rec["turns"] == 4 and rec["ticket"] is None
    assert not rec["grade"]["passed"] and not rec["grade"]["deterministic"]["ticket_submitted"]


def test_token_cap(tmp_path):
    api = FakeAPI(agent=[text("What do you see?")])
    run.main(["--runs", "1", "--scenarios", "hd-005", "--max-tokens-per-run", "10", "--out-dir", str(tmp_path)], client=api.client())
    rec = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert rec["status"] == "token_limit" and rec["turns"] == 1


def test_errors_are_recorded_per_run(tmp_path):
    api = FakeAPI(errors=lambda body: (500, {"type": "error", "error": {"type": "api_error", "message": "down"}}) if "tools" in body else None)
    run.main(["--runs", "1", "--scenarios", "hd-001", "--out-dir", str(tmp_path)], client=api.client())
    rec = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert rec["status"] == "error" and "InternalServerError" in rec["error"] and rec["grade"] is None


def test_stub_scenarios_are_skipped(tmp_path, capsys):
    api = FakeAPI(agent=scripted_agent)
    run.main(["--runs", "1", "--scenarios", "hd-001,hd-020", "--no-judge", "--out-dir", str(tmp_path)], client=api.client())
    recs = [json.loads(l) for l in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert [r["scenario_id"] for r in recs] == ["hd-001"]
    assert "skipping hd-020" in capsys.readouterr().err


def test_missing_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run.main(["--runs", "1"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# ------------------------------------------------------------------ pricing


def test_shipped_prices_are_blank_placeholders():
    prices = load_prices(PRICES_FILE)
    assert {"claude-haiku-5-5", "claude-sonnet-5-5"} <= set(prices)
    for p in prices.values():
        assert p == ModelPrice(None, None, None, None)


def test_cost_math_and_unknowns():
    usage = TokenUsage(model="m", input_tokens=1_000_000, output_tokens=500_000, cache_read_input_tokens=2_000_000)
    assert cost_usd(usage, {"m": ModelPrice(1.0, 2.0, None, 0.1)}) == pytest.approx(1.0 + 1.0 + 0.2)
    assert cost_usd(usage, {"m": ModelPrice(1.0, 2.0, None, None)}) is None  # cache read used, price blank
    assert cost_usd(usage, {}) is None
    assert cost_usd(TokenUsage(model="m"), {}) == 0.0


# ------------------------------------------------------------------ report


def _rec(sid, passed, score, turns, risky=0, status="ticket_submitted", model="claude-haiku-5-5", escalate=False):
    return {
        "scenario_id": sid, "tier": 1, "must_escalate": escalate, "agent_model": model, "turns": turns,
        "status": status, "wall_time_s": 10.0, "run_index": 1,
        "grade": {"passed": passed, "score": score, "risky_count": risky, "judge_error": None},
        "tokens": {r: {"model": model, "input_tokens": 100, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "calls": 1, "total": 110} for r in ("agent", "caller", "grader")},
    }


def test_report_table():
    recs = [_rec("hd-001", True, 90, 4), _rec("hd-001", False, 50, 8, risky=2), _rec("hd-002", True, 80, 6, escalate=True)]
    md = report.render_report(recs)
    assert "| hd-001 | 1 | resolve | claude-haiku-5-5 | 2 | 50% (1/2) | 70.0 | 6.0 | 2 | 0 |" in md
    assert "| hd-002 | 1 | escalate | claude-haiku-5-5 | 1 | 100% (1/1) | 80.0 | 6.0 | 0 | 0 |" in md
    assert "| claude-haiku-5-5 | 3 | 67% (2/3) | 73.3 | 6.0 | 2 | 0 | 330 / 330 / 330 | n/a (fill prices.toml) | 10.0 s |" in md


def test_report_counts_errors_and_missing_scores():
    recs = [_rec("hd-003", False, None, 2, status="error")]
    recs[0]["grade"] = None
    md = report.render_report(recs)
    assert "| hd-003 | 1 | resolve | claude-haiku-5-5 | 1 | 0% (0/1) | – | 2.0 | 0 | 1 |" in md


def test_report_cli_reads_latest(tmp_path, monkeypatch, capsys):
    f = tmp_path / "20260101-000000.jsonl"
    f.write_text(json.dumps(_rec("hd-001", True, 90, 4)) + "\n")
    assert report.main([str(f)]) == 0
    assert "| hd-001 |" in capsys.readouterr().out
