"""Eval harness: every full scenario, N runs each, simulated caller + grader, in parallel.

    python -m helpdesk.run --runs 3 --models haiku

Each run: the caller opens with the scenario's opening line, the agent and the simulated
caller alternate until the agent files a ticket, the turn cap (15 agent replies) or the
per-run token cap is hit. Then the grader scores the call. Every run is written as one JSON
line to results/<timestamp>.jsonl, and the report table is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import count_agent_replies, next_agent_message
from .caller import next_caller_message
from .config import (
    CALLER_MODEL,
    DEFAULT_RUN_TOKEN_LIMIT,
    GRADER_MODEL,
    HARNESS_MAX_TURNS,
    MODEL_ALIASES,
    RESULTS_DIR,
    resolve_model,
)
from .grader import grade_call
from .llm import MissingAPIKeyError, TokenUsage, make_client
from .models import Scenario, Ticket, Turn
from .pricing import cost_usd, load_prices, sum_costs
from .report import render_report
from .scenarios import get_scenario, load_full


@dataclass(frozen=True)
class Limits:
    max_turns: int = HARNESS_MAX_TURNS
    max_tokens_per_run: int = DEFAULT_RUN_TOKEN_LIMIT
    judge: bool = True


async def run_one(
    scenario: Scenario,
    agent_model: str,
    run_index: int,
    *,
    client: Any,
    limits: Limits,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    agent_usage = TokenUsage(model=agent_model)
    caller_usage = TokenUsage(model=CALLER_MODEL)
    grader_usage = TokenUsage(model=GRADER_MODEL)
    conversation: list[Turn] = [Turn(role="caller", text=scenario.opening_line)]
    ticket: Ticket | None = None
    ticket_id: str | None = None
    ticket_errors: list[str] = []
    status = "error"
    error: str | None = None

    def spent() -> int:
        return agent_usage.total + caller_usage.total

    try:
        while True:
            if spent() >= limits.max_tokens_per_run:
                status = "token_limit"
                break
            reply = await next_agent_message(
                conversation, client=client, model=agent_model, usage=agent_usage
            )
            conversation.append(reply.as_turn())
            ticket_errors.extend(reply.ticket_errors)
            if reply.ticket is not None:
                ticket, ticket_id, status = reply.ticket, reply.ticket_id, "ticket_submitted"
                break
            if count_agent_replies(conversation) >= limits.max_turns:
                status = "turn_limit"
                break
            if spent() >= limits.max_tokens_per_run:
                status = "token_limit"
                break
            line = await next_caller_message(scenario, conversation, client=client, usage=caller_usage)
            conversation.append(Turn(role="caller", text=line))
    except Exception as exc:  # recorded in the run; the batch carries on
        status, error = "error", f"{type(exc).__name__}: {exc}"

    grade = None
    if count_agent_replies(conversation) > 0:
        graded = await grade_call(
            scenario, conversation, ticket, client=client, usage=grader_usage, use_judge=limits.judge
        )
        grade = graded.model_dump(mode="json")

    prices = load_prices()
    costs = {
        "agent": cost_usd(agent_usage, prices),
        "caller": cost_usd(caller_usage, prices),
        "grader": cost_usd(grader_usage, prices),
    }
    costs["total"] = sum_costs(list(costs.values()))
    return {
        "run_id": f"{scenario.id}-{agent_model}-{run_index}",
        "started_at": started_at,
        "scenario_id": scenario.id,
        "tier": scenario.tier,
        "category": scenario.category,
        "must_escalate": scenario.must_escalate,
        "agent_model": agent_model,
        "caller_model": CALLER_MODEL,
        "grader_model": GRADER_MODEL,
        "run_index": run_index,
        "status": status,
        "error": error,
        "turns": count_agent_replies(conversation),
        "transcript": [t.public() for t in conversation],
        "ticket_id": ticket_id,
        "ticket": ticket.model_dump(mode="json") if ticket else None,
        "ticket_errors": ticket_errors,
        "grade": grade,
        "tokens": {
            "agent": agent_usage.to_dict(),
            "caller": caller_usage.to_dict(),
            "grader": grader_usage.to_dict(),
            "total": agent_usage.total + caller_usage.total + grader_usage.total,
        },
        "cost_usd": costs,
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "limits": {"max_turns": limits.max_turns, "max_tokens_per_run": limits.max_tokens_per_run},
    }


def _progress_line(done: int, total: int, rec: dict[str, Any]) -> str:
    grade = rec.get("grade") or {}
    score = grade.get("score")
    verdict = "PASS" if grade.get("passed") else "fail"
    ticket = (rec.get("ticket") or {}).get("status", "-")
    extra = f" ({rec['error']})" if rec.get("error") else ""
    if grade.get("judge_error"):
        extra += f" (judge: {grade['judge_error']})"
    return (
        f"[{done:>3}/{total}] {rec['scenario_id']} {rec['agent_model']} #{rec['run_index']}: "
        f"{rec['status']}, ticket={ticket}, score={score if score is not None else '-'} {verdict}, "
        f"{rec['turns']} turns, {rec['wall_time_s']}s{extra}"
    )


async def run_batch(
    scenarios: list[Scenario],
    models: list[str],
    runs: int,
    *,
    client: Any,
    limits: Limits,
    concurrency: int,
    out_path: Path,
    log=print,
) -> list[dict[str, Any]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m, i) for m in models for s in scenarios for i in range(1, runs + 1)]
    semaphore = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    records: list[dict[str, Any]] = []

    async def worker(scenario: Scenario, model: str, index: int) -> None:
        async with semaphore:
            rec = await run_one(scenario, model, index, client=client, limits=limits)
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            records.append(rec)
            log(_progress_line(len(records), len(jobs), rec))

    await asyncio.gather(*(worker(*job) for job in jobs))
    order = {(s.id, m, i): n for n, (s, m, i) in enumerate(jobs)}
    records.sort(key=lambda r: order[(r["scenario_id"], r["agent_model"], r["run_index"])])
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m helpdesk.run", description="Run the eval harness.")
    p.add_argument("--runs", type=int, default=3, help="runs per scenario (default 3)")
    p.add_argument(
        "--models",
        default="haiku",
        help=f"comma-separated agent models: aliases {sorted(MODEL_ALIASES)} or full ids (default haiku)",
    )
    p.add_argument("--scenarios", default="", help="comma-separated scenario ids (default: all full scenarios)")
    p.add_argument("--concurrency", type=int, default=5, help="runs in flight at once (default 5)")
    p.add_argument("--max-turns", type=int, default=HARNESS_MAX_TURNS, help="hard cap on agent replies (default 15)")
    p.add_argument(
        "--max-tokens-per-run",
        type=int,
        default=DEFAULT_RUN_TOKEN_LIMIT,
        help=f"cap on agent+caller tokens per run (default {DEFAULT_RUN_TOKEN_LIMIT})",
    )
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (deterministic checks only)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return p


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scenarios:
        scenarios = []
        for sid in args.scenarios.split(","):
            s = get_scenario(sid.strip())
            if not isinstance(s, Scenario):
                print(f"skipping {s.id}: stub scenario", file=sys.stderr)
                continue
            scenarios.append(s)
    else:
        scenarios = load_full()
    models = [resolve_model(m) for m in args.models.split(",") if m.strip()]
    if not scenarios or not models or args.runs < 1:
        print("error: nothing to run", file=sys.stderr)
        return 2
    try:
        client = client or make_client()
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out_dir / f"{stamp}.jsonl"
    limits = Limits(args.max_turns, args.max_tokens_per_run, judge=not args.no_judge)
    total = len(scenarios) * len(models) * args.runs
    print(
        f"Running {total} calls ({len(scenarios)} scenarios x {len(models)} model(s) x {args.runs} runs), "
        f"concurrency {args.concurrency}, caps: {limits.max_turns} turns / "
        f"{limits.max_tokens_per_run:,} tokens per run. Writing {out_path}"
    )
    records = asyncio.run(
        run_batch(
            scenarios, models, args.runs, client=client, limits=limits,
            concurrency=args.concurrency, out_path=out_path,
        )
    )
    report = render_report(records, source=str(out_path))
    out_path.with_suffix(".md").write_text(report, encoding="utf-8")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
