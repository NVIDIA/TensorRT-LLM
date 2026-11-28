# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A second, independent reading of every reply by the official SDKs (check C-12).

The protocol adapters (``proto_*.py``) read the wire with rules written from the
protocol documents. Here the same recorded bytes go through the vendors' own
Python SDKs instead: their SSE decoder, their event dispatch and their stream
accumulators, exactly as a client built on them runs them. Nothing here uses an
adapter or ``tracelib``'s SSE reader, so a mistake in those is not repeated here.

When the two readings disagree, a client built on that SDK acts on something
other than what the adapters judged. Rules (references/protocols.md section 5):

    SDK-01  the SDK raised while reading a reply with a success status
    SDK-02  the SDK assembled other content: text, reasoning, calls, or their order
    SDK-03  the SDK ends on another stop reason or terminal
    SDK-04  the SDK reads another message or response id, or model
    SDK-05  the SDK reads other usage counts

Anthropic Messages is read with ``anthropic`` (the beta stream, as Claude Code
calls ``/v1/messages?beta=true``); Chat Completions and Responses with
``openai``. An SDK that is not installed makes C-12 BLOCKED for its protocols.
The versions are part of the validator's identity: ``requirements.txt`` pins
them and ``verdict.py`` records them.

Reports carry rule ids, field names and exception class names only. An SDK
exception message can quote generated content, so it is never kept.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from turnlib import ReplyView, Violation, _canonical

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "http://sdk-reading.invalid"
SDK_OF = {"anthropic-messages": "anthropic", "chat-completions": "openai", "responses": "openai"}
RESPONSES_TERMINAL = {
    "response.completed": "completed",
    "response.incomplete": "incomplete",
    "response.failed": "failed",
}
# Item and block types the adapters read. Any other type is check C-11's (an
# element without a rule), so it is left out of both readings here.
RESPONSES_ITEMS = {
    "message": "text",
    "reasoning": "reasoning",
    "function_call": "call",
    "custom_tool_call": "call",
}
ANTHROPIC_BLOCKS = {
    "text": "text",
    "thinking": "reasoning",
    "redacted_thinking": "reasoning",
    "tool_use": "call",
}
USAGE_KEYS = {
    "anthropic-messages": ("input_tokens", "output_tokens"),
    "chat-completions": ("prompt_tokens", "completion_tokens", "total_tokens"),
    "responses": ("input_tokens", "output_tokens", "total_tokens"),
}
_KITS: Dict[str, Any] = {}


# --------------------------------------------------------------------------- SDKs


def pinned() -> Dict[str, Optional[str]]:
    """The SDK versions ``requirements.txt`` pins."""
    pins: Dict[str, Optional[str]] = {"anthropic": None, "openai": None}
    try:
        with open(os.path.join(HERE, "requirements.txt")) as fh:
            for line in fh:
                match = re.match(r"^\s*(anthropic|openai)==([^\s#;]+)", line)
                if match:
                    pins[match.group(1)] = match.group(2)
    except OSError:
        pass
    return pins


def _kit(name: str) -> Optional[dict]:
    """One SDK, loaded once per process, with a client that never opens a connection."""
    if name in _KITS:
        return _KITS[name]
    kit = None
    try:
        if name == "anthropic":
            import anthropic
            import httpx2
            from anthropic.lib.streaming import BetaMessageStream
            from anthropic.types.beta import BetaMessage, BetaRawMessageStreamEvent

            kit = {
                "version": anthropic.__version__,
                "client": anthropic.Anthropic(
                    api_key="sdk-reading", base_url=BASE_URL, max_retries=0
                ),
                "stream": anthropic.Stream,
                "response": httpx2.Response,
                "request": httpx2.Request,
                "message_stream": BetaMessageStream,
                "event": BetaRawMessageStreamEvent,
                "message": BetaMessage,
                "not_given": anthropic.not_given,
                "api_errors": (anthropic.APIError,),
            }
        elif name == "openai":
            import httpx
            import openai
            from openai.lib.streaming.chat import ChatCompletionStream
            from openai.lib.streaming.responses import ResponseStream
            from openai.types.chat import ChatCompletion, ChatCompletionChunk
            from openai.types.responses import Response, ResponseStreamEvent

            kit = {
                "version": openai.__version__,
                "client": openai.OpenAI(api_key="sdk-reading", base_url=BASE_URL, max_retries=0),
                "stream": openai.Stream,
                "response": httpx.Response,
                "request": httpx.Request,
                "chat_stream": ChatCompletionStream,
                "chunk": ChatCompletionChunk,
                "completion": ChatCompletion,
                "response_stream": ResponseStream,
                "event": ResponseStreamEvent,
                "response_object": Response,
                "omit": openai.omit,
                "api_errors": (openai.APIError,),
            }
    except ImportError:
        kit = None
    _KITS[name] = kit
    return kit


def versions() -> Dict[str, Optional[str]]:
    """The installed SDK versions (None: not installed)."""
    return {name: (_kit(name) or {}).get("version") for name in ("anthropic", "openai")}


def unavailable(protocol: str) -> Optional[str]:
    """Why ``protocol`` cannot be read by its SDK here, or None when it can."""
    name = SDK_OF.get(protocol)
    if name is None:
        return "no SDK reading for protocol " + str(protocol)
    if _kit(name) is None:
        return f"the {name} SDK is not installed (scripts/validate/requirements.txt)"
    return None


# --------------------------------------------------------------------------- entry


def compare(protocol: str, record: dict, reply: ReplyView) -> Optional[List[Violation]]:
    """The SDK reading of one reply against the adapter's reading ``reply``.

    Returns None when the reply is not read here: an error status (C-10 and the
    protocol's error rules judge those), or a body that is neither SSE nor JSON
    (WIRE-01 and WIRE-02). Otherwise the disagreements, empty when none.
    """
    kit = _kit(SDK_OF.get(protocol, ""))
    if kit is None:
        return None
    if isinstance(reply.http_status, int) and reply.http_status >= 400:
        return None
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    body = response.get("body")
    if reply.kind == "sse" and isinstance(body, str):
        reader = {
            "anthropic-messages": _anthropic_stream,
            "chat-completions": _chat_stream,
            "responses": _responses_stream,
        }[protocol]
    elif reply.kind == "json" and isinstance(body, dict):
        if body.get("type") == "error" or body.get("error") is not None:
            return None
        reader = {
            "anthropic-messages": _anthropic_json,
            "chat-completions": _chat_json,
            "responses": _responses_json,
        }[protocol]
    else:
        return None
    return _differences(protocol, _adapter_reading(protocol, reply), reader(kit, body))


# --------------------------------------------------------------------------- SDK readings


def _http_response(kit: dict, text: str, path: str):
    return kit["response"](
        200,
        headers={"content-type": "text/event-stream"},
        content=text.encode("utf-8", "surrogatepass"),
        request=kit["request"]("POST", BASE_URL + path),
    )


def _raised(kit: dict, stage: str, error: BaseException) -> dict:
    return {"raised": (stage, type(error).__name__, isinstance(error, kit["api_errors"]))}


def _attr(obj: Any, *names: str) -> Any:
    """A chained attribute of a leniently built SDK object, None when it is missing."""
    for name in names:
        if obj is None:
            return None
        obj = getattr(obj, name, None)
    return obj


def _text_of(parts: Any, types: Optional[tuple] = None) -> str:
    out = []
    for part in parts or []:
        text = getattr(part, "text", None)
        if isinstance(text, str) and (types is None or getattr(part, "type", None) in types):
            out.append(text)
    return "".join(out)


def _usage(protocol: str, usage: Any) -> dict:
    if usage is None:
        return {}
    values = {key: getattr(usage, key, None) for key in USAGE_KEYS[protocol]}
    return {key: value for key, value in values.items() if value is not None}


def _anthropic_blocks(content: Any) -> List[dict]:
    blocks = []
    for block in content or []:
        kind = ANTHROPIC_BLOCKS.get(getattr(block, "type", None))
        if kind is None:
            continue
        if block.type == "text":
            blocks.append({"kind": "text", "text": _attr(block, "text")})
        elif block.type == "thinking":
            blocks.append({"kind": "reasoning", "text": _attr(block, "thinking"),
                           "carrier": bool(_attr(block, "signature"))})  # fmt: skip
        elif block.type == "redacted_thinking":
            blocks.append({"kind": "reasoning", "text": None, "carrier": True})
        else:
            blocks.append({"kind": "call", "call_id": _attr(block, "id"),
                           "name": _attr(block, "name"),
                           "payload": _canonical(_attr(block, "input"))})  # fmt: skip
    return blocks


def _anthropic_reading(message: Any) -> dict:
    return {
        "raised": None,
        "blocks": _anthropic_blocks(_attr(message, "content")),
        "stop": _attr(message, "stop_reason"),
        "ident": {"id": _attr(message, "id"), "model": _attr(message, "model")},
        "usage": _usage("anthropic-messages", _attr(message, "usage")),
    }


def _anthropic_stream(kit: dict, text: str) -> dict:
    raw = kit["stream"](cast_to=kit["event"], response=_http_response(kit, text, "/v1/messages"),
                        client=kit["client"])  # fmt: skip
    stream = kit["message_stream"](raw, output_format=kit["not_given"])
    try:
        for _ in stream:
            pass
        message = stream.get_final_message()
    # Any exception the SDK raises on these bytes is the finding itself; its class is kept.
    except Exception as error:  # noqa: BLE001
        return _raised(kit, "read", error)
    return _anthropic_reading(message)


def _anthropic_json(kit: dict, body: dict) -> dict:
    try:
        message = kit["client"]._process_response_data(
            data=body, cast_to=kit["message"], response=_http_response(kit, "", "/v1/messages")
        )
    except Exception as error:  # noqa: BLE001 - see _anthropic_stream
        return _raised(kit, "construct", error)
    return _anthropic_reading(message)


def _chat_reading(protocol: str, completion: Any) -> dict:
    choices = _attr(completion, "choices") or []
    choice = next((c for c in choices if getattr(c, "index", None) == 0), None)
    message = _attr(choice, "message")
    calls = []
    for call in _attr(message, "tool_calls") or []:
        calls.append((_attr(call, "id"), _attr(call, "function", "name"),
                      _attr(call, "function", "arguments")))  # fmt: skip
    reasoning = _attr(message, "reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = _attr(message, "reasoning")
    return {
        "raised": None,
        "text": _attr(message, "content") or "",
        "reasoning": reasoning if isinstance(reasoning, str) else "",
        "calls": calls,
        "stop": _attr(choice, "finish_reason"),
        "ident": {"id": _attr(completion, "id"), "model": _attr(completion, "model")},
        "usage": _usage(protocol, _attr(completion, "usage")),
    }


def _chat_stream(kit: dict, text: str) -> dict:
    raw = kit["stream"](cast_to=kit["chunk"],
                        response=_http_response(kit, text, "/v1/chat/completions"),
                        client=kit["client"])  # fmt: skip
    stream = kit["chat_stream"](raw_stream=raw, response_format=kit["omit"],
                                input_tools=kit["omit"])  # fmt: skip
    try:
        for _ in stream:
            pass
        # The accumulated snapshot, as a streaming client holds it. get_final_completion() is
        # the structured-output helper: it raises on every `length` or `content_filter` finish.
        completion = stream.current_completion_snapshot
    except Exception as error:  # noqa: BLE001 - see _anthropic_stream
        return _raised(kit, "read", error)
    return _chat_reading("chat-completions", completion)


def _chat_json(kit: dict, body: dict) -> dict:
    try:
        completion = kit["client"]._process_response_data(
            data=body, cast_to=kit["completion"],
            response=_http_response(kit, "", "/v1/chat/completions"))  # fmt: skip
    except Exception as error:  # noqa: BLE001 - see _anthropic_stream
        return _raised(kit, "construct", error)
    return _chat_reading("chat-completions", completion)


def _responses_items(items: Any) -> List[dict]:
    out = []
    for item in items or []:
        kind = RESPONSES_ITEMS.get(getattr(item, "type", None))
        if kind is None:
            continue
        entry = {"kind": kind, "id": _attr(item, "id")}
        if item.type == "message":
            entry["text"] = _text_of(_attr(item, "content"))
        elif item.type == "reasoning":
            entry["text"] = _text_of(_attr(item, "content") or _attr(item, "summary"))
        else:
            payload = _attr(item, "arguments" if item.type == "function_call" else "input")
            entry.update(call_id=_attr(item, "call_id"), name=_attr(item, "name"),
                         payload=payload if isinstance(payload, str) else None)  # fmt: skip
        out.append(entry)
    return out


def _responses_accumulated(snapshot: Any) -> List[dict]:
    """The SDK's own snapshot, built from ``output_index``-addressed events.

    The SDK accumulates message text and function arguments; for other item
    types only their identity and position are its own.
    """
    out = []
    for item in _attr(snapshot, "output") or []:
        kind = RESPONSES_ITEMS.get(getattr(item, "type", None))
        if kind is None:
            continue
        entry = {"kind": kind, "id": _attr(item, "id")}
        if item.type == "message":
            entry["text"] = _text_of(_attr(item, "content"), ("output_text",))
        elif item.type == "function_call":
            entry["payload"] = _attr(item, "arguments")
        out.append(entry)
    return out


def _responses_stream(kit: dict, text: str) -> dict:
    raw = kit["stream"](cast_to=kit["event"],
                        response=_http_response(kit, text, "/v1/responses"),
                        client=kit["client"])  # fmt: skip
    stream = kit["response_stream"](raw_stream=raw, text_format=kit["omit"],
                                    input_tools=kit["omit"], starting_after=None)  # fmt: skip
    done, terminal = [], "none"
    try:
        for event in stream:
            kind = getattr(event, "type", None)
            if kind == "response.output_item.done":
                done.append(_attr(event, "item"))
            elif kind in RESPONSES_TERMINAL:
                terminal = RESPONSES_TERMINAL[kind]
        final = stream.get_final_response() if terminal == "completed" else None
    except Exception as error:  # noqa: BLE001 - see _anthropic_stream
        return _raised(kit, "read", error)
    # The pinned SDK keeps its accumulated snapshot here; a version that moves it fails the
    # control matrix, which binds the version.
    snapshot = getattr(stream._state, "_ResponseStreamState__current_snapshot", None)
    return {
        "raised": None,
        "blocks": _responses_items(done),
        "accumulated": _responses_accumulated(snapshot),
        "terminal": terminal,
        "ident": {"id": _attr(snapshot, "id"), "model": _attr(snapshot, "model")},
        "usage": _usage("responses", _attr(final, "usage")) if final is not None else None,
    }


def _responses_json(kit: dict, body: dict) -> dict:
    try:
        response = kit["client"]._process_response_data(
            data=body, cast_to=kit["response_object"],
            response=_http_response(kit, "", "/v1/responses"))  # fmt: skip
    except Exception as error:  # noqa: BLE001 - see _anthropic_stream
        return _raised(kit, "construct", error)
    status = _attr(response, "status")
    return {
        "raised": None,
        "blocks": _responses_items(_attr(response, "output")),
        "terminal": {"completed": "completed", "incomplete": "incomplete", "failed": "failed"}.get(
            status, "none"
        ),  # fmt: skip
        "ident": {"id": _attr(response, "id"), "model": _attr(response, "model")},
        "usage": _usage("responses", _attr(response, "usage")),
    }


# --------------------------------------------------------------------------- adapter reading


def _payload_value(payload: Optional[str]) -> Any:
    """An Anthropic call input as a client reads it: the joined JSON, parsed."""
    text = payload if isinstance(payload, str) and payload.strip() else "{}"
    try:
        return _canonical(json.loads(text))
    except json.JSONDecodeError:
        return ["unparsable"]


def _adapter_reading(protocol: str, reply: ReplyView) -> dict:
    items = reply.delivered()
    usage = reply.usage if isinstance(reply.usage, dict) else {}
    reading = {
        "terminal": reply.terminal.get("kind"),
        "stop": reply.terminal.get("stop_reason"),
        "ident": reply.extras.get("ident") or {},
        "usage": {k: usage[k] for k in USAGE_KEYS[protocol] if usage.get(k) is not None},
    }
    if protocol == "chat-completions":
        reading["text"] = "".join(i.text or "" for i in items if i.kind == "text")
        reading["reasoning"] = "".join(i.text or "" for i in items if i.kind == "reasoning")
        reading["calls"] = [(i.call_id, i.name, i.payload) for i in items if i.kind == "call"]
    elif protocol == "anthropic-messages":
        blocks = []
        for item in items:
            if item.kind == "call":
                blocks.append({"kind": "call", "call_id": item.call_id, "name": item.name,
                               "payload": _payload_value(item.payload)})  # fmt: skip
            elif item.kind == "reasoning":
                blocks.append({"kind": "reasoning", "text": item.text,
                               "carrier": item.carrier == "present"})  # fmt: skip
            else:
                blocks.append({"kind": "text", "text": item.text})
        reading["blocks"] = blocks
    else:
        blocks = []
        for item in items:
            entry = {"kind": item.kind, "id": item.id}
            if item.kind == "call":
                entry.update(call_id=item.call_id, name=item.name, payload=item.payload)
            else:
                entry["text"] = item.text
            blocks.append(entry)
        reading["blocks"] = blocks
    return reading


# --------------------------------------------------------------------------- comparison


def _differences(protocol: str, mine: dict, theirs: dict) -> List[Violation]:
    raised = theirs.get("raised")
    if raised:
        stage, name, api_error = raised
        if api_error and mine["terminal"] == "error":
            return []  # both read the error event the server sent
        return [Violation("SDK-01", {"stage": stage, "error": name})]
    fields: Dict[str, List[str]] = {"SDK-02": [], "SDK-03": [], "SDK-04": [], "SDK-05": []}
    if protocol == "chat-completions":
        for key in ("text", "reasoning", "calls"):
            if mine[key] != theirs[key]:
                fields["SDK-02"].append(key)
        if mine["terminal"] != "error" and mine["stop"] != theirs["stop"]:
            fields["SDK-03"].append("finish_reason")
    elif protocol == "anthropic-messages":
        fields["SDK-02"] += _block_fields(mine["blocks"], theirs["blocks"])
        if mine["terminal"] != "error" and mine["stop"] != theirs["stop"]:
            fields["SDK-03"].append("stop_reason")
    else:
        fields["SDK-02"] += _block_fields(mine["blocks"], theirs["blocks"])
        if "accumulated" in theirs:
            fields["SDK-02"] += _accumulated_fields(mine["blocks"], theirs["accumulated"])
        if mine["terminal"] != theirs["terminal"]:
            fields["SDK-03"].append("terminal")
    if mine["terminal"] == "error":
        fields["SDK-03"].append("error_not_raised")
    for key in ("id", "model"):
        if theirs["ident"].get(key) != mine["ident"].get(key):
            fields["SDK-04"].append(key)
    if theirs.get("usage") is not None and theirs["usage"] != mine["usage"]:
        fields["SDK-05"].append("usage")
    return [Violation(rule, {"fields": ",".join(sorted(set(names)))})
            for rule, names in fields.items() if names]  # fmt: skip


def _block_fields(mine: List[dict], theirs: List[dict]) -> List[str]:
    if [b["kind"] for b in mine] != [b["kind"] for b in theirs]:
        return ["order"]
    out = []
    for a, b in zip(mine, theirs):
        for key in ("id", "text", "call_id", "name", "payload", "carrier"):
            if key in a and key in b and a[key] != b[key]:
                out.append(key)
    return out


def _accumulated_fields(mine: List[dict], accumulated: List[dict]) -> List[str]:
    """Responses: the SDK's index-addressed snapshot against the adapter's id-addressed items."""
    if [b.get("id") for b in mine] != [b.get("id") for b in accumulated]:
        return ["accumulated_order"]
    out = []
    for a, b in zip(mine, accumulated):
        if b["kind"] == "text" and a.get("text") != b.get("text"):
            out.append("accumulated_text")
        elif "payload" in b and a.get("payload") != b.get("payload"):
            out.append("accumulated_arguments")
    return out
