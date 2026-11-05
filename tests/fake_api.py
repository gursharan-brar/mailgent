"""A fake Anthropic Messages API behind httpx2.MockTransport.

Tests use the real `anthropic.AsyncAnthropic` client, so request building, SSE stream parsing,
tool-use accumulation and `get_final_message()` all run through the actual SDK. Only the HTTP
server is fake. Responses are scripted per role: the agent (requests that carry the
submit_ticket tool), the grader (requests with an output_config format) and the caller
(everything else).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterable
from typing import Any

import anthropic
import httpx2

Spec = dict[str, Any]


def text(t: str) -> Spec:
    return {"content": [{"type": "thinking"}, {"type": "text", "text": t}], "stop_reason": "end_turn"}


def ticket_call(ticket: dict[str, Any], preface: str | None = None) -> Spec:
    content: list[dict[str, Any]] = [{"type": "thinking"}]
    if preface:
        content.append({"type": "text", "text": preface})
    content.append({"type": "tool_use", "name": "submit_ticket", "input": ticket})
    return {"content": content, "stop_reason": "tool_use"}


def refusal() -> Spec:
    return {
        "content": [],
        "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": "test refusal"},
    }


def role_of(body: dict[str, Any]) -> str:
    tools = body.get("tools") or []
    if any(t.get("name") == "submit_ticket" for t in tools):
        return "agent"
    if (body.get("output_config") or {}).get("format"):
        return "grader"
    return "caller"


class FakeAPI:
    """Scripted fake API. Pass lists (consumed in order; the last item repeats) or callables
    (body -> Spec) for each role."""

    def __init__(
        self,
        agent: Iterable[Spec] | Callable[[dict[str, Any]], Spec] | None = None,
        caller: Iterable[str] | Callable[[dict[str, Any]], str] | None = None,
        grader: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        errors: Callable[[dict[str, Any]], tuple[int, dict[str, Any]] | None] | None = None,
    ):
        self._agent = agent if callable(agent) else list(agent or [text("What do you see on the screen?")])
        self._caller = caller if callable(caller) else list(caller or ["Okay, I did that."])
        self._grader = grader
        self._errors = errors
        self.requests: list[dict[str, Any]] = []
        self._ids = itertools.count(1)
        self._agent_i = 0
        self._caller_i = 0

    # --------------------------------------------------------------- client
    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test-key",
            base_url="https://fake-anthropic.test",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle)),
            max_retries=0,
        )

    def requests_for(self, role: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if role_of(r) == role]

    # --------------------------------------------------------------- scripting
    def _next(self, script: Any, index_attr: str, body: dict[str, Any]) -> Any:
        if callable(script):
            return script(body)
        i = getattr(self, index_attr)
        setattr(self, index_attr, i + 1)
        return script[min(i, len(script) - 1)]

    def _spec_for(self, body: dict[str, Any]) -> Spec:
        role = role_of(body)
        if role == "agent":
            return self._next(self._agent, "_agent_i", body)
        if role == "caller":
            return text(self._next(self._caller, "_caller_i", body))
        grade = self._grader(body) if callable(self._grader) else self._grader
        grade = grade or {
            "root_cause_correct": True,
            "fix_correct": True,
            "risky_advice": [],
            "efficiency": 0,
            "score": 85,
            "reason": "Found the cause and fixed it safely.",
        }
        return {"content": [{"type": "thinking"}, {"type": "text", "text": json.dumps(grade)}], "stop_reason": "end_turn"}

    # --------------------------------------------------------------- HTTP
    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        body["_path"] = request.url.path
        body["_query"] = request.url.query.decode() if isinstance(request.url.query, bytes) else str(request.url.query)
        body["_beta"] = request.headers.get("anthropic-beta")
        self.requests.append(body)
        if self._errors and (failure := self._errors(body)):
            status, payload = failure
            return httpx2.Response(status, json=payload)
        message = self._build_message(self._spec_for(body), body)
        if body.get("stream"):
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(message)
            )
        return httpx2.Response(200, json=message)

    def _build_message(self, spec: Spec, body: dict[str, Any]) -> dict[str, Any]:
        n = next(self._ids)
        content = []
        for block in spec["content"]:
            block = dict(block)
            if block["type"] == "thinking":
                block = {"type": "thinking", "thinking": "", "signature": f"sig-{n}"}
            elif block["type"] == "tool_use":
                block.setdefault("id", f"toolu_{n:04d}")
            content.append(block)
        out_text = " ".join(b.get("text", "") for b in content) + json.dumps(
            [b.get("input") for b in content if b["type"] == "tool_use"]
        )
        message = {
            "id": f"msg_{n:04d}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "unknown"),
            "content": content,
            "stop_reason": spec["stop_reason"],
            "stop_sequence": None,
            "usage": {
                "input_tokens": len(json.dumps(body.get("messages", []))) // 4 + 50,
                "output_tokens": len(out_text) // 4 + 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        if "stop_details" in spec:
            message["stop_details"] = spec["stop_details"]
        return message


def _sse(message: dict[str, Any]) -> bytes:
    events: list[tuple[str, dict[str, Any]]] = []
    start = {
        **message,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {**message["usage"], "output_tokens": 1},
    }
    start.pop("stop_details", None)
    events.append(("message_start", {"type": "message_start", "message": start}))
    for i, block in enumerate(message["content"]):
        if block["type"] == "text":
            events.append(_block_start(i, {"type": "text", "text": ""}))
            words = block["text"].split(" ")
            for j, word in enumerate(words):
                chunk = word if j == 0 else " " + word
                events.append(_delta(i, {"type": "text_delta", "text": chunk}))
        elif block["type"] == "thinking":
            events.append(_block_start(i, {"type": "thinking", "thinking": "", "signature": ""}))
            events.append(_delta(i, {"type": "signature_delta", "signature": block["signature"]}))
        elif block["type"] == "tool_use":
            events.append(
                _block_start(i, {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}})
            )
            raw = json.dumps(block["input"])
            half = len(raw) // 2
            for part in (raw[:half], raw[half:]):
                events.append(_delta(i, {"type": "input_json_delta", "partial_json": part}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    delta: dict[str, Any] = {"stop_reason": message["stop_reason"], "stop_sequence": None}
    if "stop_details" in message:
        delta["stop_details"] = message["stop_details"]
    events.append(
        (
            "message_delta",
            {"type": "message_delta", "delta": delta, "usage": {"output_tokens": message["usage"]["output_tokens"]}},
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _block_start(index: int, block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return ("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})


def _delta(index: int, delta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})


GOOD_TICKET = {
    "caller_name": "Maria Lopez",
    "device": "Dell OptiPlex desktop, Windows 11",
    "summary": "Could not sign in; password rejected.",
    "steps_tried": ["Read the sign-in error", "Revealed the typed password", "Turned off Caps Lock"],
    "root_cause": "Caps Lock was on; the new keyboard has no indicator light.",
    "resolution": "Turned off Caps Lock and signed in.",
    "status": "resolved",
    "escalation_reason": "",
}

ESCALATED_TICKET = {
    **GOOD_TICKET,
    "root_cause": "Account locked out by an old phone retrying the old password.",
    "resolution": "",
    "status": "escalated",
    "escalation_reason": "Needs an admin to unlock the account in AD; old phone has been powered off.",
}
