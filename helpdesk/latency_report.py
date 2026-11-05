"""Summarize results/latency.jsonl: per source and adapter combination, mean / p50 / p90.

    python -m helpdesk.latency_report [results/latency.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles

from .config import RESULTS_DIR

FIELDS = ("stt_ms", "llm_first_token_ms", "tts_first_audio_ms", "first_audio_ms", "total_ms")


def _pct(values: list[float], q: int) -> float:
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[q - 1]


def summarize(lines: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in lines:
        groups[(rec.get("source"), rec.get("stt"), rec.get("tts"), rec.get("agent_model"))].append(rec)
    out = []
    for (source, stt, tts, model), recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out.append(f"### {source}: STT={stt}, TTS={tts}, agent={model} ({len(recs)} turns)")
        out.append("")
        out.append("| Metric | Mean | p50 | p90 |")
        out.append("|---|---|---|---|")
        for field in FIELDS:
            vals = [float(r[field]) for r in recs if r.get(field) is not None]
            if vals:
                out.append(f"| {field} | {mean(vals):.0f} | {_pct(vals, 50):.0f} | {_pct(vals, 90):.0f} |")
            else:
                out.append(f"| {field} | – | – | – |")
        out.append("")
    return "\n".join(out) if out else "No latency records."


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else RESULTS_DIR / "latency.jsonl"
    if not path.exists():
        print(f"{path} not found. Run the voice server or `python -m helpdesk.voice_test` first.", file=sys.stderr)
        return 1
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(summarize(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
