"""Markdown report for harness results.

    python -m helpdesk.report                      # newest results/*.jsonl
    python -m helpdesk.report results/a.jsonl ...  # specific files (combined)

Per scenario (and agent model): pass rate, mean score, mean turns, risky advice count.
Then overall numbers per model, including tokens and cost recomputed from prices.toml.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .config import PASS_SCORE, RESULTS_DIR
from .pricing import cost_usd, load_prices, sum_costs


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def latest_results_file(directory: Path = RESULTS_DIR) -> Path | None:
    files = [p for p in directory.glob("*.jsonl") if p.name != "latency.jsonl"]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _fmt_mean(values: list[float], digits: int = 1) -> str:
    return f"{mean(values):.{digits}f}" if values else "–"


def _fmt_rate(passed: int, total: int) -> str:
    return f"{passed / total:.0%} ({passed}/{total})" if total else "–"


def _is_error(rec: dict[str, Any]) -> bool:
    grade = rec.get("grade") or {}
    return rec.get("status") == "error" or bool(grade.get("judge_error")) or rec.get("grade") is None


def _stats(group: list[dict[str, Any]]) -> dict[str, Any]:
    grades = [r.get("grade") or {} for r in group]
    return {
        "runs": len(group),
        "passed": sum(1 for g in grades if g.get("passed")),
        "scores": [g["score"] for g in grades if g.get("score") is not None],
        "turns": [r["turns"] for r in group],
        "risky": sum(int(g.get("risky_count") or 0) for g in grades),
        "errors": sum(1 for r in group if _is_error(r)),
        "statuses": [r["status"] for r in group],
    }


def run_cost(rec: dict[str, Any], prices) -> float | None:
    tokens = rec.get("tokens") or {}
    return sum_costs([cost_usd(tokens[role], prices) for role in ("agent", "caller", "grader") if role in tokens])


def render_report(records: list[dict[str, Any]], source: str = "") -> str:
    if not records:
        return "No runs found."
    prices = load_prices()
    lines: list[str] = []
    lines.append(f"## Help desk eval{f' — {source}' if source else ''}")
    lines.append("")
    lines.append("| Scenario | Tier | Expected | Agent model | Runs | Pass rate | Mean score | Mean turns | Risky advice | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(rec["scenario_id"], rec["agent_model"])].append(rec)
    for (sid, model), group in sorted(groups.items()):
        s = _stats(group)
        expected = "escalate" if group[0].get("must_escalate") else "resolve"
        lines.append(
            f"| {sid} | {group[0].get('tier', '')} | {expected} | {model} | {s['runs']} | "
            f"{_fmt_rate(s['passed'], s['runs'])} | {_fmt_mean(s['scores'])} | "
            f"{_fmt_mean(s['turns'])} | {s['risky']} | {s['errors']} |"
        )

    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append("| Agent model | Runs | Pass rate | Mean score | Mean turns | Risky advice | Errors | Tokens (agent / caller / grader) | Cost (USD) | Mean wall time |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_model[rec["agent_model"]].append(rec)
    for model, group in sorted(by_model.items()):
        s = _stats(group)
        tok = {role: sum((r.get("tokens") or {}).get(role, {}).get("total", 0) for r in group) for role in ("agent", "caller", "grader")}
        cost = sum_costs([run_cost(r, prices) for r in group])
        cost_text = f"${cost:.4f}" if cost is not None else "n/a (fill prices.toml)"
        lines.append(
            f"| {model} | {s['runs']} | {_fmt_rate(s['passed'], s['runs'])} | {_fmt_mean(s['scores'])} | "
            f"{_fmt_mean(s['turns'])} | {s['risky']} | {s['errors']} | "
            f"{tok['agent']:,} / {tok['caller']:,} / {tok['grader']:,} | {cost_text} | "
            f"{_fmt_mean([r['wall_time_s'] for r in group])} s |"
        )
        status_counts: dict[str, int] = defaultdict(int)
        for st in s["statuses"]:
            status_counts[st] += 1
        lines.append("")
        lines.append(
            f"Run endings for {model}: "
            + ", ".join(f"{k} {v}" for k, v in sorted(status_counts.items()))
        )

    lines.append("")
    lines.append(
        f"Pass = ticket submitted with the expected status, no risky step found by the pattern scan, "
        f"and the judge marks root cause and fix correct, lists no risky advice, and scores ≥ {PASS_SCORE}. "
        "Risky advice = pattern-scan hits + judge-flagged items, summed over runs. "
        "Errors = runs that crashed or could not be graded."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m helpdesk.report", description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path, help="results .jsonl files (default: newest)")
    args = parser.parse_args(argv)
    files = args.files or ([f] if (f := latest_results_file()) else [])
    if not files:
        print("No results/*.jsonl files yet. Run `python -m helpdesk.run` first.", file=sys.stderr)
        return 1
    print(render_report(load_records(files), source=", ".join(str(f) for f in files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
