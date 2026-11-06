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
"""Anthropic Messages (``POST /v1/messages``) adapter: request view, reply view, wire rules.

Rule ids are those of references/protocols.md section 2 (ANT-01 .. ANT-15).
"""

import json
from typing import Dict, List, Optional

from tracelib import sse_frames
from turnlib import (
    Item,
    ReplyView,
    RequestView,
    Tool,
    Violation,
    content_text,
    json_object,
    message_digest,
    sha256_text,
    text_blocks,
    tool_declarations,
    tools_key,
)

PROTOCOL = "anthropic-messages"
ROUTES = ("/v1/messages",)
STOP_REASONS = ("end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal")
ERROR_TYPES = (
    "invalid_request_error", "authentication_error", "permission_error", "not_found_error",
    "request_too_large", "rate_limit_error", "api_error", "overloaded_error",
)  # fmt: skip
DELTA_FOR_BLOCK = {
    "text": ("text_delta",),
    "tool_use": ("input_json_delta",),
    "thinking": ("thinking_delta", "signature_delta"),
    "redacted_thinking": (),
}
# Anthropic-defined tools that the CLIENT executes (schema known to the model).
CLIENT_DEFINED = ("bash_", "text_editor_", "computer_", "str_replace_")
KNOWN_FIELDS = {
    "model", "messages", "system", "max_tokens", "metadata", "stop_sequences", "stream",
    "temperature", "top_p", "top_k", "tools", "tool_choice", "thinking", "output_config",
    "context_management", "container", "mcp_servers", "service_tier",
}  # fmt: skip
RULE_CHECK = {"ANT-10": "C-04"}
KNOWN_EVENTS = {"message_start", "content_block_start", "content_block_delta", "content_block_stop",
                "message_delta", "message_stop", "ping", "error"}  # fmt: skip


def _unchecked(reply: ReplyView, element: str) -> None:
    """An element this validator has no rule for: recorded (WIRE-05), never dropped silently."""
    reply.extras.setdefault("unchecked", set()).add(element)


def _tool(spec: dict) -> Optional[Tool]:
    name = spec.get("name")
    if not isinstance(name, str):
        return None
    kind = spec.get("type")
    if kind in (None, "custom"):
        return Tool(name, None, "function", "tools", spec.get("input_schema"), None,
                    spec.get("description") or "")  # fmt: skip
    if isinstance(kind, str) and kind.startswith(CLIENT_DEFINED):
        return Tool(name, None, "builtin", "tools")
    return Tool(name, None, "server", "tools")


def parse_request(body: dict, headers: dict) -> RequestView:
    view = RequestView(protocol=PROTOCOL, route=ROUTES[0])
    view.stream = bool(body.get("stream"))
    view.model_field = body.get("model") if isinstance(body.get("model"), str) else None
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for spec in tools:
        if isinstance(spec, dict):
            tool = _tool(spec)
            if tool is not None:
                view.tools[tool.name] = tool
    view.tools_key = tools_key(tools)
    view.declarations = tool_declarations(tools)

    choice = body.get("tool_choice") if isinstance(body.get("tool_choice"), dict) else {}
    mode = {"auto": "auto", "any": "required", "tool": "named", "none": "none"}.get(
        choice.get("type") or "auto", "other"
    )
    view.tool_choice = {"mode": mode, "name": choice.get("name") if mode == "named" else None}
    view.parallel = "single" if choice.get("disable_parallel_tool_use") is True else "allowed"
    thinking = body.get("thinking") if isinstance(body.get("thinking"), dict) else {}
    output_config = body.get("output_config") if isinstance(body.get("output_config"), dict) else {}
    view.reasoning = {
        "thinking": thinking.get("type"),
        "budget_tokens": thinking.get("budget_tokens"),
        "effort": output_config.get("effort"),
    }
    if isinstance(body.get("max_tokens"), int):
        view.max_output_tokens = body["max_tokens"]
    view.unknown_fields = sorted(k for k in body if k not in KNOWN_FIELDS)

    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    view.history_messages = len(messages)
    if messages:
        view.last_message_digest = message_digest(messages[-1])
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and position:
            items, call_ids = [], []
            content = message.get("content")
            for index, block in enumerate(content if isinstance(content, list)
                                          else [{"type": "text", "text": content}]):  # fmt: skip
                if isinstance(block, dict) and block.get("type") in ("text", "thinking",
                                                                     "redacted_thinking"):  # fmt: skip
                    item = _block_item(block, index, None, False)
                    if item is not None:
                        items.append(item)
                elif (
                    isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
                ):
                    call_ids.append(str(block["id"]))
            view.assistant_turns.append((position, message_digest(messages[position - 1]),
                                         text_blocks(items), tuple(call_ids)))  # fmt: skip
        blocks = message.get("content")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_use" and message.get("role") == "assistant" and block.get("id"):
                view.echo[str(block["id"])] = {
                    "kind": "call",
                    "name": block.get("name"),
                    "call_kind": "function",
                    "payload": json.dumps(block.get("input"), ensure_ascii=False)
                    if isinstance(block.get("input"), dict)
                    else None,
                }
            elif kind == "tool_result" and block.get("tool_use_id"):
                view.tool_results.append(
                    (
                        str(block["tool_use_id"]),
                        content_text(block.get("content")),
                        block.get("is_error") if isinstance(block.get("is_error"), bool) else None,
                    )
                )
    return view


def _block_item(block: dict, index: int, payload: Optional[str], signed: bool) -> Optional[Item]:
    kind = block.get("type")
    if kind == "text":
        return Item("text", index, id=str(index), text=block.get("text") or "")
    if kind == "thinking":
        signature = block.get("signature")
        return Item("reasoning", index, id=str(index), text=block.get("thinking") or "",
                    carrier="present" if signed or signature else "absent",
                    carrier_digest=sha256_text(signature)
                    if isinstance(signature, str) and signature else None)  # fmt: skip
    if kind == "redacted_thinking":
        return Item("reasoning", index, id=str(index), text=None, carrier="present")
    if kind == "tool_use":
        if payload is None:
            payload = json.dumps(block.get("input") or {}, ensure_ascii=False)
        return Item("call", index, id=block.get("id"), call_id=block.get("id"),
                    name=block.get("name"), call_kind="function", payload=payload)  # fmt: skip
    return None


def parse_reply(record: dict, request: Optional[RequestView]) -> ReplyView:
    reply = ReplyView()
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    reply.http_status = record.get("http_status")
    reply.status = record.get("status")
    body = response.get("body")
    if response.get("kind") == "sse_text" and isinstance(body, str):
        reply.kind = "sse"
        reply.final_na_reason = "protocol has no streamed snapshot"
        _parse_stream(body, reply)
    elif isinstance(body, dict):
        reply.kind = "json"
        if body.get("type") == "message":
            _parse_message(body, reply)
        elif body.get("type") == "error" or isinstance(body.get("error"), dict):
            _parse_error_body(body, reply)
        else:
            reply.grammar.append(Violation("ANT-13", {"why": "unknown_body"}))
    elif body is not None:
        reply.kind = "text"
    return reply


def _parse_message(body: dict, reply: ReplyView) -> None:
    grammar = reply.grammar
    if body.get("role") != "assistant" or not isinstance(body.get("content"), list):
        grammar.append(Violation("ANT-13", {"why": "shape"}))
    stop = body.get("stop_reason")
    if stop not in STOP_REASONS:
        grammar.append(Violation("ANT-13", {"why": "stop_reason"}))
    if not isinstance(body.get("usage"), dict):
        grammar.append(Violation("ANT-13", {"why": "usage"}))
    items = []
    for index, block in enumerate(body.get("content") or []):
        if isinstance(block, dict):
            if block.get("type") not in DELTA_FOR_BLOCK:
                _unchecked(reply, "block:" + str(block.get("type"))[:40])
            item = _block_item(block, index, None, False)
            if item is not None:
                items.append(item)
    reply.final_items = items
    reply.usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    reply.extras["ident"] = {"id": body.get("id"), "model": body.get("model")}
    has_tool = any(i.kind == "call" for i in items)
    if (stop == "tool_use") != has_tool and stop != "max_tokens":
        grammar.append(Violation("ANT-09", {"stop_reason": stop, "tool_use_blocks": has_tool}))
    reply.terminal = {"kind": "incomplete" if stop == "max_tokens" else "completed",
                      "stop_reason": stop}  # fmt: skip


def _parse_error_body(body: dict, reply: ReplyView) -> None:
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    if (
        body.get("type") != "error"
        or error.get("type") not in ERROR_TYPES
        or not isinstance(error.get("message"), str)
    ):
        reply.grammar.append(Violation("ANT-14", {"why": "envelope"}))
    expected = {
        400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
        404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
        500: "api_error", 529: "overloaded_error",
    }.get(reply.http_status)  # fmt: skip
    if expected and error.get("type") != expected:
        reply.grammar.append(
            Violation("ANT-14", {"why": "status_type", "status": reply.http_status})
        )
    reply.terminal = {"kind": "error", "error": {"type": error.get("type"), "code": None},
                      "message": error.get("message") if isinstance(error.get("message"), str)
                      else None}  # fmt: skip


def _parse_stream(body: str, reply: ReplyView) -> None:
    grammar = reply.grammar
    frames = sse_frames(body)
    events = []
    for frame in frames:
        if frame.kind == "comment":
            continue
        if frame.kind != "json":
            grammar.append(Violation("ANT-01", {"why": "unparsable_or_empty_frame"}))
            continue
        data_type = frame.obj.get("type")
        # SDK clients dispatch on the `event:` line and drop frames without one (ANT-15).
        if not frame.has_event_line:
            grammar.append(Violation("ANT-15", {"why": "no_event_line"}))
        elif frame.name != data_type:
            grammar.append(Violation("ANT-15", {"why": "event_name_differs_from_type"}))
        events.append((data_type, frame.obj))
    reply.events = len(events)
    names = [n for n, _ in events]
    for name in set(names) - KNOWN_EVENTS:
        _unchecked(reply, "event:" + str(name)[:60])
    non_ping = [i for i, n in enumerate(names) if n != "ping"]

    # ANT-01 message_start first and once
    starts = [i for i, n in enumerate(names) if n == "message_start"]
    if not starts or not non_ping or non_ping[0] != starts[0]:
        grammar.append(Violation("ANT-01", {"why": "message_start_first"}))
    if len(starts) > 1:
        grammar.append(Violation("ANT-01", {"why": "message_start_repeated"}))
    if starts:
        message = events[starts[0]][1].get("message")
        message = message if isinstance(message, dict) else {}
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        if (message.get("role") != "assistant" or message.get("content") != []
                or not message.get("id") or not message.get("model")
                or "input_tokens" not in usage):  # fmt: skip
            grammar.append(Violation("ANT-01", {"why": "message_start_shape"}))
        reply.usage = dict(usage)
        reply.extras["ident"] = {"id": message.get("id"), "model": message.get("model")}

    error_at = [i for i, n in enumerate(names) if n == "error"]
    stop_at = [i for i, n in enumerate(names) if n == "message_stop"]
    delta_at = [i for i, n in enumerate(names) if n == "message_delta"]

    blocks: Dict[int, dict] = {}
    order: List[int] = []
    open_index: Optional[int] = None
    expected_next = 0
    for position, (name, data) in enumerate(events):
        if name == "content_block_start":
            index = data.get("index")
            block = data.get("content_block") if isinstance(data.get("content_block"), dict) else {}
            if index != expected_next:
                grammar.append(Violation("ANT-02", {"why": "index_not_next", "index": index}))
            if open_index is not None:
                grammar.append(Violation("ANT-03", {"why": "block_started_while_open"}))
            expected_next = (index + 1) if isinstance(index, int) else expected_next + 1
            if block.get("type") not in DELTA_FOR_BLOCK:
                _unchecked(reply, "block:" + str(block.get("type"))[:40])
            blocks[index] = {"block": dict(block), "deltas": [], "stopped": False,
                             "signature": False, "thinking_after_signature": False}  # fmt: skip
            order.append(index)
            open_index = index
            if block.get("type") == "tool_use":
                if not block.get("id") or not block.get("name"):
                    grammar.append(Violation("ANT-06", {"why": "tool_use_start_shape"}))
                if block.get("input") not in ({}, None):
                    grammar.append(Violation("ANT-06", {"why": "tool_use_start_input_not_empty"}))
        elif name == "content_block_delta":
            index = data.get("index")
            delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
            state = blocks.get(index)
            if state is None:
                grammar.append(Violation("ANT-02", {"why": "delta_before_start"}))
                continue
            if state["stopped"]:
                grammar.append(Violation("ANT-05", {"why": "delta_after_stop"}))
            block_type = state["block"].get("type")
            if delta.get("type") not in DELTA_FOR_BLOCK.get(block_type, ()):
                grammar.append(
                    Violation("ANT-04", {"block": block_type, "delta": delta.get("type")})
                )
                continue
            if delta.get("type") == "signature_delta":
                state["signature"] = True
            elif delta.get("type") == "thinking_delta" and state["signature"]:
                state["thinking_after_signature"] = True
            state["deltas"].append(delta)
        elif name == "content_block_stop":
            index = data.get("index")
            state = blocks.get(index)
            if state is None:
                grammar.append(Violation("ANT-02", {"why": "stop_before_start"}))
                continue
            if state["stopped"]:
                grammar.append(Violation("ANT-05", {"why": "stopped_twice"}))
            state["stopped"] = True
            if open_index == index:
                open_index = None
        elif name == "message_delta":
            if any(not s["stopped"] for s in blocks.values()):
                grammar.append(Violation("ANT-08", {"why": "message_delta_before_block_stop"}))
    for index, state in blocks.items():
        if not state["stopped"]:
            grammar.append(Violation("ANT-05", {"why": "block_never_stopped"}))
        if state["thinking_after_signature"]:
            grammar.append(Violation("ANT-07", {}))

    # Items, in block order.
    ids = []
    for index in order:
        state = blocks[index]
        block = state["block"]
        kind = block.get("type")
        payload = None
        if kind == "text":
            block["text"] = (block.get("text") or "") + "".join(
                d.get("text") or "" for d in state["deltas"] if d.get("type") == "text_delta"
            )
        elif kind == "thinking":
            block["thinking"] = (block.get("thinking") or "") + "".join(
                d.get("thinking") or "" for d in state["deltas"]
                if d.get("type") == "thinking_delta")  # fmt: skip
            signature = "".join(d.get("signature") or "" for d in state["deltas"]
                                if d.get("type") == "signature_delta")  # fmt: skip
            if signature:
                block["signature"] = (block.get("signature") or "") + signature
        elif kind == "tool_use":
            payload = "".join(d.get("partial_json") or "" for d in state["deltas"]
                              if d.get("type") == "input_json_delta")  # fmt: skip
            text = payload if payload.strip() else "{}"
            if json_object(text) is None:
                grammar.append(Violation("ANT-06", {"why": "partial_json_not_object"}))
            payload = text
            ids.append(block.get("id"))
        item = _block_item(block, index if isinstance(index, int) else 0, payload,
                           state["signature"])  # fmt: skip
        if item is not None:
            reply.stream_items.append(item)
    if len([i for i in ids if i]) != len({i for i in ids if i}):
        grammar.append(Violation("ANT-06", {"why": "duplicate_tool_use_id"}))

    stop_reason = None
    if error_at:
        data = events[error_at[0]][1]
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        if error.get("type") not in ERROR_TYPES or not isinstance(error.get("message"), str):
            grammar.append(Violation("ANT-11", {"why": "error_shape"}))
        if error_at[0] != len(events) - 1:
            grammar.append(Violation("ANT-11", {"why": "events_after_error"}))
        reply.terminal = {"kind": "error", "error": {"type": error.get("type"), "code": None},
                          "message": error.get("message")
                          if isinstance(error.get("message"), str) else None}  # fmt: skip
        return
    if len(delta_at) != 1:
        grammar.append(Violation("ANT-08", {"why": "message_delta_count", "count": len(delta_at)}))
    if delta_at:
        data = events[delta_at[-1]][1]
        delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
        stop_reason = delta.get("stop_reason")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        if stop_reason not in STOP_REASONS:
            grammar.append(Violation("ANT-08", {"why": "stop_reason"}))
        if "output_tokens" not in usage:
            grammar.append(Violation("ANT-08", {"why": "usage_output_tokens"}))
        # message_delta counts are cumulative; a null one leaves message_start's value in place.
        reply.usage.update({k: v for k, v in usage.items() if v is not None})
        has_tool = any(i.kind == "call" for i in reply.stream_items)
        if (stop_reason == "tool_use") != has_tool and stop_reason != "max_tokens":
            grammar.append(Violation("ANT-09", {"stop_reason": stop_reason,
                                                "tool_use_blocks": has_tool}))  # fmt: skip
    if len(stop_at) != 1:
        grammar.append(Violation("ANT-10", {"why": "message_stop_count", "count": len(stop_at)}))
    elif stop_at[0] != len(events) - 1:
        grammar.append(Violation("ANT-10", {"why": "events_after_message_stop"}))
    if stop_at and delta_at and delta_at[-1] > stop_at[0]:
        grammar.append(Violation("ANT-08", {"why": "message_delta_after_stop"}))
    if stop_at:
        reply.terminal = {"kind": "incomplete" if stop_reason == "max_tokens" else "completed",
                          "stop_reason": stop_reason}  # fmt: skip
    else:
        reply.terminal = {"kind": "none", "stop_reason": stop_reason}
