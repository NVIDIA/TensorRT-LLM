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
"""Responses (``POST /v1/responses``) adapter: request view, reply view and wire rules.

Rule ids are those of references/protocols.md section 1. The stream rules are
the ones ``check_stream.py`` enforces on recorded traffic.
"""

from collections import Counter, defaultdict
from typing import Any, List, Optional

from tracelib import sse_frames
from turnlib import (
    Item,
    ReplyView,
    RequestView,
    Tool,
    Violation,
    content_text,
    nested_from_descriptions,
    openai_error,
    sha256_text,
    tool_declarations,
    tools_key,
)

PROTOCOL = "responses"
ROUTES = ("/v1/responses",)
TERMINAL = ("response.completed", "response.incomplete", "response.failed")
OPENING = ("response.created", "response.in_progress")
CALL_TYPES = {"function_call": "function", "custom_tool_call": "custom"}
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
# (delta event, done event, payload field on the done event, per content part)
DELTA_KINDS = (
    ("response.output_text.delta", "response.output_text.done", "text", True),
    ("response.reasoning_text.delta", "response.reasoning_text.done", "text", True),
    (
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "arguments",
        False,
    ),
    (
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "input",
        False,
    ),
)
STATUS_REQUIRED = ("message",)
STATUS_NOT_NULL = ("function_call", "reasoning")
KNOWN_FIELDS = {
    "model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls",
    "reasoning", "store", "stream", "stream_options", "temperature", "top_p",
    "max_output_tokens", "max_tool_calls", "include", "metadata", "previous_response_id",
    "text", "truncation", "user", "service_tier", "prompt_cache_key", "prompt_cache_retention",
    "client_metadata", "background", "prompt", "safety_identifier", "top_logprobs",
    "conversation",
}  # fmt: skip
# Which check a broken rule belongs to (validate.md section 4); others are C-01.
RULE_CHECK = {"RSP-02": "C-04", "RSP-03": "C-04", "RSP-10": "C-02", "RSP-11": "C-02"}
ITEM_TYPES = ("message", "reasoning") + tuple(CALL_TYPES)
KNOWN_EVENTS = set(TERMINAL + OPENING) | {
    "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done", "error",
} | {name for kind in DELTA_KINDS for name in kind[:2]}  # fmt: skip


def _unchecked(reply: ReplyView, element: str) -> None:
    """An element this validator has no rule for: recorded (WIRE-05), never dropped silently."""
    reply.extras.setdefault("unchecked", set()).add(element)


def _tool(spec: dict, namespace: Optional[str], declared_in: str) -> Optional[Tool]:
    if spec.get("name") is None and isinstance(spec.get("function"), dict):
        fn = spec["function"]  # chat-completions shaped declaration
        return Tool(fn.get("name"), namespace, "function", declared_in, fn.get("parameters"))
    kind = spec.get("type")
    name = spec.get("name")
    if not isinstance(name, str):
        return None
    if kind == "function":
        return Tool(name, namespace, "function", declared_in, spec.get("parameters"), None,
                    spec.get("description") or "")  # fmt: skip
    if kind == "custom":
        return Tool(name, namespace, "custom", declared_in, None, spec.get("format"),
                    spec.get("description") or "")  # fmt: skip
    return Tool(name, namespace, "builtin", declared_in)


def parse_request(body: dict, headers: dict) -> RequestView:
    view = RequestView(protocol=PROTOCOL, route=ROUTES[0])
    view.stream = bool(body.get("stream"))
    view.model_field = body.get("model") if isinstance(body.get("model"), str) else None
    raw_sources = []

    def walk(tools, namespace, source):
        for spec in tools or []:
            if not isinstance(spec, dict):
                continue
            if spec.get("type") == "namespace":
                walk(spec.get("tools"), spec.get("name"), source)
                continue
            tool = _tool(spec, namespace, source)
            if tool is not None and tool.name:
                view.tools[tool.name] = tool

    if isinstance(body.get("tools"), list):
        raw_sources.append(body["tools"])
        walk(body["tools"], None, "tools")
    items = body.get("input") if isinstance(body.get("input"), list) else []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            raw_sources.append(item.get("tools"))
            walk(item.get("tools"), None, "additional_tools")
    view.nested = nested_from_descriptions(view.tools)
    view.tools_key = tools_key(raw_sources)
    view.declarations = tuple(d for source in raw_sources for d in tool_declarations(source))

    choice = body.get("tool_choice")
    if choice in (None, "auto"):
        view.tool_choice = {"mode": "auto", "name": None}
    elif choice == "none":
        view.tool_choice = {"mode": "none", "name": None}
    elif choice == "required":
        view.tool_choice = {"mode": "required", "name": None}
    elif isinstance(choice, dict) and choice.get("name"):
        view.tool_choice = {"mode": "named", "name": choice.get("name")}
    else:
        view.tool_choice = {"mode": "other", "name": None}

    ptc = body.get("parallel_tool_calls", None)
    view.parallel = "single" if ptc is False else ("allowed" if ptc is True else "unspecified")
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    view.reasoning = {"effort": reasoning.get("effort")}
    if isinstance(body.get("max_output_tokens"), int):
        view.max_output_tokens = body["max_output_tokens"]
    view.unknown_fields = sorted(k for k in body if k not in KNOWN_FIELDS)

    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in CALL_TYPES and item.get("call_id"):
            payload = item.get("arguments") if kind == "function_call" else item.get("input")
            view.echo[str(item["call_id"])] = {
                "kind": "call",
                "name": item.get("name"),
                "call_kind": CALL_TYPES[kind],
                "payload": payload if isinstance(payload, str) else None,
            }
        elif kind == "message" and item.get("role") == "assistant" and item.get("id"):
            view.echo[str(item["id"])] = {"kind": "text", "text": content_text(item.get("content"))}
        elif kind == "reasoning" and item.get("id"):
            view.echo[str(item["id"])] = {
                "kind": "reasoning",
                "carrier": "present" if item.get("encrypted_content") else "absent",
                "carrier_digest": _carrier_digest(item.get("encrypted_content")),
            }
        elif kind in OUTPUT_TYPES and item.get("call_id"):
            view.tool_results.append((str(item["call_id"]), content_text(item.get("output")), None))
        if kind == "message" or kind in CALL_TYPES or kind in OUTPUT_TYPES:
            view.history_messages += 1
    return view


def _carrier_digest(value: Any) -> Optional[str]:
    """sha256 of a reasoning item's ``encrypted_content``: the echo compares the carrier by value."""
    return sha256_text(value) if isinstance(value, str) and value else None


def _item_from(obj: dict, index: int) -> Optional[Item]:
    kind = obj.get("type")
    if kind == "message":
        return Item("text", index, id=obj.get("id"), text=content_text(obj.get("content")),
                    status=obj.get("status"))  # fmt: skip
    if kind == "reasoning":
        parts = obj.get("content") or obj.get("summary") or []
        return Item("reasoning", index, id=obj.get("id"), text=content_text(parts),
                    carrier="present" if obj.get("encrypted_content") else "absent",
                    carrier_digest=_carrier_digest(obj.get("encrypted_content")),
                    status=obj.get("status"))  # fmt: skip
    if kind in CALL_TYPES:
        payload = obj.get("arguments") if kind == "function_call" else obj.get("input")
        return Item("call", index, id=obj.get("id"), call_id=obj.get("call_id"),
                    name=obj.get("name"), namespace=obj.get("namespace"),
                    call_kind=CALL_TYPES[kind], payload=payload if isinstance(payload, str) else None,
                    status=obj.get("status"))  # fmt: skip
    return None


def _items(output: Any) -> List[Item]:
    items = []
    for index, obj in enumerate(output if isinstance(output, list) else []):
        if isinstance(obj, dict):
            item = _item_from(obj, index)
            if item is not None:
                items.append(item)
    return items


def _item_view(obj: dict) -> dict:
    """What a client may read from an item, in either place: the stream's and the snapshot's must match.

    ``status`` and a reasoning item's ``encrypted_content`` are part of it: a client that keeps
    the snapshot replays them.
    """
    kind = obj.get("type")
    if kind in ("message", "reasoning"):
        parts = obj.get("content") or []
        view = {"text": [p.get("text") for p in parts if isinstance(p, dict)],
                "status": obj.get("status")}  # fmt: skip
        if kind == "reasoning":
            view["encrypted_content"] = obj.get("encrypted_content")
        return view
    if kind == "function_call":
        return {k: obj.get(k) for k in ("name", "call_id", "arguments", "namespace", "status")}
    if kind == "custom_tool_call":
        return {k: obj.get(k) for k in ("name", "call_id", "input", "namespace")}
    return {}


def parse_reply(record: dict, request: Optional[RequestView]) -> ReplyView:
    reply = ReplyView()
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    reply.http_status = record.get("http_status")
    reply.status = record.get("status")
    body = response.get("body")
    if response.get("kind") == "sse_text" and isinstance(body, str):
        reply.kind = "sse"
        _parse_stream(body, reply)
    elif isinstance(body, dict):
        reply.kind = "json"
        if body.get("object") == "response":
            reply.final_items = _items(body.get("output"))
            reply.extras["ident"] = {"id": body.get("id"), "model": body.get("model")}
            reply.usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            for item in body.get("output") or []:
                if isinstance(item, dict) and item.get("type") not in ITEM_TYPES:
                    _unchecked(reply, "item:" + str(item.get("type"))[:40])
            reply.terminal = {"kind": _terminal_kind(body.get("status")),
                              "error": _error(body.get("error"))}  # fmt: skip
        elif "error" in body or body.get("object") == "error":
            error = openai_error(body)
            if error is None:
                reply.grammar.append(Violation("RSP-16", {"why": "error_envelope"}))
            reply.terminal = {"kind": "error", "error": _error(error)}
        else:
            # Neither a Response object nor an error: never passed silently.
            reply.grammar.append(Violation("RSP-16", {"why": "unknown_body"}))
    elif body is not None:
        reply.kind = "text"
    return reply


def _terminal_kind(status: Any) -> str:
    return {"completed": "completed", "incomplete": "incomplete", "failed": "failed"}.get(
        status, "none"
    )


def _error(error: Any) -> Optional[dict]:
    if not isinstance(error, dict):
        return None
    return {"type": error.get("type"), "code": error.get("code")}


def _indices(name: str, data: dict, state: dict) -> None:
    """RSP-15: ``output_index`` and ``content_index`` address what the stream announced.

    SDK clients keep the output as a list and put every event at ``output_index``
    (and a text delta at ``content_index`` of that item's parts), not by item id.
    """
    if not name or not name.startswith("response."):
        return
    if name in ("response.output_item.added", "response.output_item.done"):
        item_id = (data.get("item") or {}).get("id") if isinstance(data.get("item"), dict) else None
    else:
        item_id = data.get("item_id")
    positions, parts, faults = state["positions"], state["parts"], state["faults"]
    if name == "response.output_item.added":
        if data.get("output_index") != state["added"]:
            faults.add("added_out_of_order")
        positions.setdefault(item_id, state["added"])
        state["added"] += 1
        return
    if item_id is None:
        return
    if "output_index" not in data:
        faults.add("no_output_index")
    elif item_id in positions and data.get("output_index") != positions[item_id]:
        faults.add("output_index_differs")
    if name == "response.content_part.added":
        if data.get("content_index") != parts[item_id]:
            faults.add("part_out_of_order")
        parts[item_id] += 1
    elif name in ("response.output_text.delta", "response.output_text.done"):
        index = data.get("content_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < parts[item_id]:
            faults.add("text_without_part")


def _parse_stream(body: str, reply: ReplyView) -> None:
    frames = [f for f in sse_frames(body) if f.kind in ("json", "unparsable")]
    events = [(f.name or (f.obj or {}).get("type"), f.obj) for f in frames]
    reply.events = len(events)
    grammar = reply.grammar
    if any(f.kind == "unparsable" for f in frames):
        grammar.append(Violation("RSP-01", {"why": "unparsable_frame"}))
    names = [n for n, _ in events]
    for name in set(names) - KNOWN_EVENTS:
        _unchecked(reply, "event:" + str(name)[:60])
    if names and names[0] != "response.created":
        grammar.append(Violation("RSP-01", {"why": "first_event", "event": str(names[0])[:60]}))
    elif names:
        created = events[0][1] if isinstance(events[0][1], dict) else {}
        opened = created.get("response") if isinstance(created.get("response"), dict) else {}
        # SDK clients start their snapshot from this object and append every added item to it.
        if not opened.get("id") or not opened.get("model") or opened.get("output") != []:
            grammar.append(Violation("RSP-01", {"why": "created_shape"}))
    terminal_at = [i for i, n in enumerate(names) if n in TERMINAL]
    if not terminal_at:
        grammar.append(Violation("RSP-02", {"why": "no_terminal_event"}))
    elif len(terminal_at) > 1:
        grammar.append(Violation("RSP-02", {"why": "multiple_terminal_events",
                                            "count": len(terminal_at)}))  # fmt: skip
    if terminal_at and terminal_at[-1] != len(events) - 1:
        grammar.append(Violation("RSP-03", {"frames_after": len(events) - 1 - terminal_at[-1]}))
    seqs = [d.get("sequence_number") for _, d in events
            if isinstance(d, dict) and "sequence_number" in d]  # fmt: skip
    if any(b is None or a is None or b <= a for a, b in zip(seqs, seqs[1:])):
        grammar.append(Violation("RSP-04", {"why": "not_increasing"}))
    elif seqs and seqs[0] != 0:
        grammar.append(Violation("RSP-04", {"why": "not_from_zero"}))
    elif any(b != a + 1 for a, b in zip(seqs, seqs[1:])):
        grammar.append(Violation("RSP-04", {"why": "gap"}))
    # A repeated response.in_progress keeps the stream alive: it must say nothing new.
    in_progress = [d for n, d in events if n == "response.in_progress" and isinstance(d, dict)]
    first = {
        k: v for k, v in (in_progress[0] if in_progress else {}).items() if k != "sequence_number"
    }
    if any(
        {k: v for k, v in d.items() if k != "sequence_number"} != first for d in in_progress[1:]
    ):
        grammar.append(Violation("RSP-17", {"why": "keepalive_differs"}))
    final = None
    if terminal_at:
        data = events[terminal_at[-1]][1] or {}
        final = data.get("response") if isinstance(data.get("response"), dict) else {}
        name = names[terminal_at[-1]]
        reply.terminal = {
            "kind": {"response.completed": "completed", "response.incomplete": "incomplete",
                     "response.failed": "failed"}[name],
            "error": _error(final.get("error")),
        }  # fmt: skip
        reply.usage = final.get("usage") if isinstance(final.get("usage"), dict) else {}
    opening = [d.get("response") or {} for n, d in events if n in OPENING and isinstance(d, dict)]
    first = opening[0] if opening and isinstance(opening[0], dict) else (final or {})
    reply.extras["ident"] = {"id": first.get("id"), "model": first.get("model")}
    ids = {r.get("id") for r in opening + ([final] if final else []) if isinstance(r, dict)}
    models = {r.get("model") for r in opening + ([final] if final else []) if isinstance(r, dict)}
    if len({i for i in ids if i}) > 1:
        grammar.append(Violation("RSP-05", {"why": "response_id_changes"}))
    if len({m for m in models if m}) > 1:
        grammar.append(Violation("RSP-05", {"why": "model_changes"}))

    added_items, done_items = Counter(), Counter()
    added_parts, done_parts = Counter(), Counter()
    deltas, dones = defaultdict(list), {}
    done_list, done_by_id = [], {}
    indices = {"added": 0, "positions": {}, "parts": Counter(), "faults": set()}
    for n, d in events:
        if not isinstance(d, dict):
            continue
        _indices(n, d, indices)
        if n == "response.output_item.added":
            added_items[(d.get("item") or {}).get("id")] += 1
        elif n == "response.output_item.done":
            item = d.get("item") or {}
            done_items[item.get("id")] += 1
            done_list.append(item)
            done_by_id[item.get("id")] = item
        elif n == "response.content_part.added":
            added_parts[(d.get("item_id"), d.get("content_index"))] += 1
        elif n == "response.content_part.done":
            done_parts[(d.get("item_id"), d.get("content_index"))] += 1
        else:
            for delta_name, done_name, field_name, per_part in DELTA_KINDS:
                key = (done_name, d.get("item_id"), d.get("content_index") if per_part else None)
                if n == delta_name:
                    deltas[key].append(d.get("delta") or "")
                elif n == done_name:
                    dones[key] = d.get(field_name)
    for why in sorted(indices["faults"]):
        grammar.append(Violation("RSP-15", {"why": why}))
    if added_items != done_items:
        grammar.append(Violation("RSP-06", {"added": sum(added_items.values()),
                                            "done": sum(done_items.values())}))  # fmt: skip
    if added_parts != done_parts:
        grammar.append(Violation("RSP-07", {"added": sum(added_parts.values()),
                                            "done": sum(done_parts.values())}))  # fmt: skip
    for key, payload in dones.items():
        done_name, item_id, content_index = key
        if key in deltas and "".join(deltas[key]) != payload:
            grammar.append(Violation("RSP-08", {"event": done_name}))
        item = done_by_id.get(item_id)
        if item is None:
            continue
        if done_name.endswith("arguments.done"):
            other = item.get("arguments")
        elif done_name.endswith("input.done"):
            other = item.get("input")
        else:
            parts = item.get("content") or []
            part = (parts[content_index] if isinstance(parts, list)
                    and isinstance(content_index, int) and content_index < len(parts) else None)  # fmt: skip
            other = part.get("text") if isinstance(part, dict) else None
        if other is not None and other != payload:
            grammar.append(Violation("RSP-09", {"event": done_name}))

    reply.stream_items = _items(done_list)
    for item in done_list:
        if item.get("type") not in ITEM_TYPES:
            _unchecked(reply, "item:" + str(item.get("type"))[:40])
    if final is None:
        reply.final_na_reason = "no terminal snapshot"
        return
    final_output = [i for i in final.get("output") or [] if isinstance(i, dict)]
    reply.final_items = _items(final_output)
    if reply.terminal["kind"] == "failed" and not final_output:
        reply.final_na_reason = "failed snapshot carries no output by design"
        return
    if [(i.get("type"), i.get("id")) for i in done_list] != [
        (i.get("type"), i.get("id")) for i in final_output
    ]:
        grammar.append(Violation("RSP-10", {"stream": len(done_list), "final": len(final_output)}))
    else:
        for streamed, snap in zip(done_list, final_output):
            seen, kept = _item_view(streamed), _item_view(snap)
            if kept.get("status") is None:
                # An item the snapshot may leave without a status is judged by RSP-12.
                seen.pop("status", None)
                kept.pop("status", None)
            if seen != kept:
                grammar.append(Violation("RSP-11", {"item_type": snap.get("type"),
                                                    "fields": sorted(k for k in set(seen) | set(kept)
                                                                     if seen.get(k) != kept.get(k))}))  # fmt: skip
    for item in final_output:
        kind = item.get("type")
        if (kind in STATUS_REQUIRED and item.get("status") is None) or (
            kind in STATUS_NOT_NULL and "status" in item and item["status"] is None
        ):
            grammar.append(Violation("RSP-12", {"item_type": kind}))
