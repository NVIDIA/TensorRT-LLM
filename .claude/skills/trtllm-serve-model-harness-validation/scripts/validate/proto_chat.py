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
"""Chat Completions (``POST /v1/chat/completions``) adapter: request view, reply view, wire rules.

Rule ids are those of references/protocols.md section 3 (CHT-01 .. CHT-15).
"""

import json
from typing import Dict, Optional

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
    openai_error,
    text_blocks,
    tool_declarations,
    tools_key,
)

PROTOCOL = "chat-completions"
ROUTES = ("/v1/chat/completions",)
FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "function_call")
KNOWN_FIELDS = {
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream",
    "stream_options", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
    "n", "stop", "seed", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs",
    "response_format", "reasoning_effort", "chat_template_kwargs", "user", "store", "metadata",
    "logit_bias", "service_tier", "prompt_cache_key", "min_p", "repetition_penalty",
}  # fmt: skip
RULE_CHECK = {"CHT-09": "C-04"}
DELTA_KEYS = {"role", "content", "reasoning_content", "reasoning", "tool_calls"}
CHOICE_KEYS = {"index", "delta", "finish_reason", "logprobs", "stop_reason"}
# trtllm-serve's own choice fields (openai_protocol.py): speculative-decoding statistics (CHT-13).
EXTENSION_KEYS = {"avg_decoded_tokens_per_iter": (int, float), "speculative_decoding": (dict,)}


def _stop_reason_ok(choice: dict) -> bool:
    """CHT-16: a choice's ``stop_reason`` has its type and its place.

    trtllm-serve and vLLM report there the stop token id or the stop string that ended the
    choice: null, an integer or a string, set only beside the choice's ``finish_reason``.
    """
    value = choice.get("stop_reason")
    if value is None:
        return True
    typed = isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))
    return typed and choice.get("finish_reason") is not None


def _unchecked(reply: ReplyView, element: str) -> None:
    """An element this validator has no rule for: recorded (WIRE-05), never dropped silently."""
    reply.extras.setdefault("unchecked", set()).add(element)


def parse_request(body: dict, headers: dict) -> RequestView:
    view = RequestView(protocol=PROTOCOL, route=ROUTES[0])
    view.stream = bool(body.get("stream"))
    view.model_field = body.get("model") if isinstance(body.get("model"), str) else None
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for spec in tools:
        if not isinstance(spec, dict):
            continue
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else None
        if spec.get("type") == "function" and fn and isinstance(fn.get("name"), str):
            view.tools[fn["name"]] = Tool(fn["name"], None, "function", "tools",
                                          fn.get("parameters"), None, fn.get("description") or "")  # fmt: skip
        elif isinstance(spec.get("name"), str):
            view.tools[spec["name"]] = Tool(spec["name"], None, "builtin", "tools")
    view.tools_key = tools_key(tools)
    view.declarations = tool_declarations(tools)

    choice = body.get("tool_choice")
    if choice in (None, "auto"):
        view.tool_choice = {"mode": "auto", "name": None}
    elif choice in ("none", "required"):
        view.tool_choice = {"mode": choice, "name": None}
    elif isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        view.tool_choice = {"mode": "named", "name": choice["function"].get("name")}
    else:
        view.tool_choice = {"mode": "other", "name": None}
    ptc = body.get("parallel_tool_calls", None)
    view.parallel = "single" if ptc is False else ("allowed" if ptc is True else "unspecified")
    kwargs = (
        body.get("chat_template_kwargs")
        if isinstance(body.get("chat_template_kwargs"), dict)
        else {}
    )
    view.reasoning = {"effort": body.get("reasoning_effort"), "template_kwargs": sorted(kwargs)}
    for key in ("max_completion_tokens", "max_tokens"):
        if isinstance(body.get(key), int):
            view.max_output_tokens = body[key]
            break
    options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
    view.include_usage = bool(options.get("include_usage"))
    n = body.get("n")
    view.choices = n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else 1
    view.unknown_fields = sorted(k for k in body if k not in KNOWN_FIELDS)

    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    view.history_messages = len(messages)
    if messages:
        view.last_message_digest = message_digest(messages[-1])
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and position:
            items = []
            reasoning = message.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                items.append(Item("reasoning", 0, text=reasoning, carrier="absent"))
            text = content_text(message.get("content"))
            if text:
                items.append(Item("text", 1, text=text))
            call_ids = tuple(str(c["id"]) for c in message.get("tool_calls") or []
                             if isinstance(c, dict) and c.get("id"))  # fmt: skip
            view.assistant_turns.append((position, message_digest(messages[position - 1]),
                                         text_blocks(items), call_ids))  # fmt: skip
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict) or not call.get("id"):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                args = fn.get("arguments")
                view.echo[str(call["id"])] = {
                    "kind": "call",
                    "name": fn.get("name"),
                    "call_kind": "function",
                    "payload": args if isinstance(args, str) else (
                        json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else None),
                }  # fmt: skip
        elif message.get("role") == "tool" and message.get("tool_call_id"):
            view.tool_results.append(
                (str(message["tool_call_id"]), content_text(message.get("content")), None)
            )
    return view


def parse_reply(record: dict, request: Optional[RequestView]) -> ReplyView:
    reply = ReplyView()
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    reply.http_status = record.get("http_status")
    reply.status = record.get("status")
    body = response.get("body")
    if response.get("kind") == "sse_text" and isinstance(body, str):
        reply.kind = "sse"
        reply.final_na_reason = "protocol has no streamed snapshot"
        _parse_stream(body, reply, request)
    elif isinstance(body, dict):
        reply.kind = "json"
        if body.get("object") == "chat.completion":
            _parse_completion(body, reply, request)
        elif "error" in body or body.get("object") == "error":
            error = openai_error(body)
            if error is None:
                reply.grammar.append(Violation("CHT-15", {"why": "error_envelope"}))
                error = {}
            reply.terminal = {"kind": "error", "error": {"type": error.get("type"),
                              "code": error.get("code")},
                              "message": error.get("message")
                              if isinstance(error.get("message"), str) else None}  # fmt: skip
        else:
            reply.grammar.append(Violation("CHT-12", {"why": "unknown_body"}))
    elif body is not None:
        reply.kind = "text"
    return reply


def _parse_completion(body: dict, reply: ReplyView, request: Optional[RequestView] = None) -> None:
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    if not choices or not isinstance(choices[0], dict):
        reply.grammar.append(Violation("CHT-12", {"why": "no_choice"}))
        return
    expected = request.choices if request is not None else 1
    indices = sorted(
        c.get("index") for c in choices if isinstance(c, dict) and isinstance(c.get("index"), int)
    )
    if indices != list(range(expected)):
        reply.grammar.append(Violation("CHT-14", {"why": "choices_seen", "count": len(choices)}))
    for other in choices[1:]:
        if not isinstance(other, dict) or other.get("finish_reason") not in FINISH_REASONS:
            reply.grammar.append(Violation("CHT-12", {"why": "finish_reason"}))
    for each in choices:
        if isinstance(each, dict) and not _stop_reason_ok(each):
            reply.grammar.append(Violation("CHT-16", {"index": each.get("index")}))
    reply.extras["choices"] = len(choices)
    choice = next((c for c in choices if isinstance(c, dict) and c.get("index") == 0), choices[0])
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    items, index = [], 0
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        items.append(Item("reasoning", index, id="reasoning", text=reasoning, carrier="absent"))
        index += 1
    if isinstance(message.get("content"), str) and message["content"]:
        items.append(Item("text", index, id="content", text=message["content"]))
        index += 1
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            items.append(Item("call", index, id=call.get("id"), call_id=call.get("id"),
                              name=fn.get("name"), call_kind="function",
                              payload=fn.get("arguments") if isinstance(fn.get("arguments"), str)
                              else None))  # fmt: skip
            index += 1
    reply.final_items = items
    finish = choice.get("finish_reason")
    if finish not in FINISH_REASONS:
        reply.grammar.append(Violation("CHT-12", {"why": "finish_reason"}))
    reply.terminal = {"kind": "incomplete" if finish == "length" else "completed",
                      "stop_reason": finish}  # fmt: skip
    reply.usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    reply.extras["ident"] = {"id": body.get("id"), "model": body.get("model")}


def _choice_state() -> dict:
    return {"text": [], "reasoning": [], "calls": {}, "order": [], "finish": None,
            "finish_at": None, "role_seen": False}  # fmt: skip


def _parse_stream(body: str, reply: ReplyView, request: Optional[RequestView]) -> None:
    grammar = reply.grammar
    frames = [f for f in sse_frames(body) if f.kind != "comment"]
    done_at = [i for i, f in enumerate(frames) if f.kind == "done"]
    if len(done_at) != 1:
        grammar.append(Violation("CHT-09", {"why": "done_count", "count": len(done_at)}))
    elif done_at[0] != len(frames) - 1:
        grammar.append(Violation("CHT-09", {"why": "frames_after_done"}))
    chunks = []
    error = None
    for frame in frames:
        if frame.kind == "done":
            continue
        if frame.kind != "json":
            grammar.append(Violation("CHT-01", {"why": "unparsable_frame"}))
            continue
        obj = frame.obj
        if isinstance(obj.get("error"), dict):
            error = obj["error"]
            continue
        if obj.get("object") != "chat.completion.chunk":
            grammar.append(Violation("CHT-01", {"why": "not_a_chunk"}))
            continue
        chunks.append(obj)
    reply.events = len(chunks)
    if chunks:
        reply.extras["ident"] = {"id": chunks[0].get("id"), "model": chunks[0].get("model")}
    for key in ("id", "model", "created"):
        values = {c.get(key) for c in chunks if c.get(key) is not None}
        if len(values) > 1:
            grammar.append(Violation("CHT-02", {"field": key}))

    states: Dict[int, dict] = {}
    reasoning_fields = set()
    usage_chunks = []
    for position, chunk in enumerate(chunks):
        choices = chunk.get("choices")
        if choices == [] or choices is None:
            if isinstance(chunk.get("usage"), dict):
                usage_chunks.append(position)
                reply.usage = chunk["usage"]
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                grammar.append(Violation("CHT-14", {"why": "choice_index"}))
                continue
            state = states.setdefault(index, _choice_state())
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if not _stop_reason_ok(choice):
                grammar.append(Violation("CHT-16", {"index": index}))
            for key in set(choice) - CHOICE_KEYS:
                value = choice[key]
                if value is None:
                    continue
                if key in EXTENSION_KEYS:
                    if isinstance(value, bool) or not isinstance(value, EXTENSION_KEYS[key]):
                        grammar.append(Violation("CHT-13", {"field": key}))
                else:
                    _unchecked(reply, "choice:" + str(key)[:40])
            for key in set(delta) - DELTA_KEYS:
                if delta[key] is not None:
                    _unchecked(reply, "delta:" + str(key)[:40])
            if state["finish_at"] is not None and (delta or choice.get("finish_reason")):
                grammar.append(Violation("CHT-06", {"why": "after_finish"}))
            if not state["role_seen"]:
                if delta.get("role") != "assistant":
                    grammar.append(Violation("CHT-03", {}))
                state["role_seen"] = True
            if isinstance(delta.get("content"), str):
                state["text"].append(delta["content"])
            for field_name in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(field_name), str):
                    state["reasoning"].append(delta[field_name])
                    reasoning_fields.add(field_name)
            _fragments(delta, state, grammar)
            if choice.get("finish_reason") is not None:
                if state["finish"] is not None:
                    grammar.append(Violation("CHT-06", {"why": "second_finish_reason"}))
                state["finish"] = choice["finish_reason"]
                state["finish_at"] = position
    expected = request.choices if request is not None else 1
    if chunks and sorted(states) != list(range(max(expected, 1))):
        grammar.append(Violation("CHT-14", {"why": "choices_seen", "count": len(states)}))
    for index, state in states.items():
        if state["finish"] is None:
            grammar.append(Violation("CHT-06", {"why": "no_finish_reason"}))
        elif state["finish"] not in FINISH_REASONS:
            grammar.append(Violation("CHT-06", {"why": "unknown_finish_reason"}))
        ids = [state["calls"][i]["id"] for i in state["order"] if state["calls"][i]["id"]]
        if len(ids) != len(set(ids)):
            grammar.append(Violation("CHT-04", {"why": "duplicate_id"}))
        for call_index in state["order"]:
            payload = "".join(state["calls"][call_index]["args"])
            if json_object(payload) is None:
                grammar.append(Violation("CHT-05", {"index": call_index, "choice": index}))
        has_calls = bool(state["order"])
        if state["finish"] is not None and (state["finish"] == "tool_calls") != has_calls \
                and state["finish"] != "length":  # fmt: skip
            grammar.append(
                Violation("CHT-07", {"finish_reason": state["finish"], "tool_calls": has_calls})
            )
    if not states:
        grammar.append(Violation("CHT-06", {"why": "no_finish_reason"}))
    last_finish = max(
        (s["finish_at"] for s in states.values() if s["finish_at"] is not None), default=None
    )
    if len(usage_chunks) > 1:
        grammar.append(Violation("CHT-08", {"why": "usage_chunks", "count": len(usage_chunks)}))
    if usage_chunks and request is not None and request.include_usage is False:
        grammar.append(Violation("CHT-08", {"why": "usage_not_requested"}))
    if usage_chunks and last_finish is not None and usage_chunks[0] < last_finish:
        grammar.append(Violation("CHT-08", {"why": "usage_before_finish"}))

    # The turn record is choice 0's; other choices are held to the same rules above.
    first = states.get(0) or _choice_state()
    items, position = [], 0
    if first["reasoning"]:
        items.append(Item("reasoning", position, id="reasoning", text="".join(first["reasoning"]),
                          carrier="absent"))  # fmt: skip
        position += 1
    if first["text"]:
        items.append(Item("text", position, id="content", text="".join(first["text"])))
        position += 1
    for call_index in first["order"]:
        call = first["calls"][call_index]
        items.append(Item("call", position, id=call["id"], call_id=call["id"], name=call["name"],
                          call_kind="function", payload="".join(call["args"])))  # fmt: skip
        position += 1
    reply.stream_items = items
    reply.extras["reasoning_fields"] = sorted(reasoning_fields)
    reply.extras["choices"] = len(states)
    finished = bool(states) and all(s["finish"] is not None for s in states.values())
    finish = first["finish"]
    if error is not None:
        reply.terminal = {"kind": "error", "error": {"type": error.get("type"),
                          "code": error.get("code")},
                          "message": error.get("message")
                          if isinstance(error.get("message"), str) else None}  # fmt: skip
    elif finished and done_at:
        reply.terminal = {"kind": "incomplete" if finish == "length" else "completed",
                          "stop_reason": finish}  # fmt: skip
    else:
        reply.terminal = {"kind": "none", "stop_reason": finish}


def _fragments(delta: dict, state: dict, grammar: list) -> None:
    """CHT-04: one choice's tool-call fragments, keyed by index."""
    calls = state["calls"]
    for fragment in delta.get("tool_calls") or []:
        if not isinstance(fragment, dict):
            continue
        index = fragment.get("index")
        fn = fragment.get("function") if isinstance(fragment.get("function"), dict) else {}
        if not isinstance(index, int):
            grammar.append(Violation("CHT-04", {"why": "no_index"}))
            continue
        if index not in calls:
            # SDK clients address the tool_calls list by index: 0, 1, 2 in order.
            if index != len(calls):
                grammar.append(Violation("CHT-04", {"why": "index_not_next", "index": index}))
            if not fragment.get("id") or fragment.get("type") not in ("function", None) \
                    or not fn.get("name"):  # fmt: skip
                grammar.append(Violation("CHT-04", {"why": "first_fragment_shape"}))
            calls[index] = {"id": fragment.get("id"), "name": fn.get("name"), "args": []}
            state["order"].append(index)
        elif fragment.get("id") is not None or fn.get("name") is not None:
            # openai-python concatenates every string it receives again.
            grammar.append(Violation("CHT-04", {"why": "later_fragment_repeats_id_or_name"}))
        if isinstance(fn.get("arguments"), str):
            calls[index]["args"].append(fn["arguments"])
