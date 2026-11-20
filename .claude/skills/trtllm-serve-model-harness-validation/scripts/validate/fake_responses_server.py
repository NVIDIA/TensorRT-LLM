#!/usr/bin/env python3
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
"""A fake /v1/responses server that answers with canned streams, for controls (test-only).

It exercises the replay -> validate pipeline without a GPU, and it produces the
POSITIVE controls the validator needs: every mode except ``ok`` breaks exactly
one rule that a conformance check must catch.

  ok                     reasoning + message + one call to the first declared tool
  drop_part_done         one content_part.done missing          -> unpaired_content_part
  snapshot_text_differs  completed text differs from the stream -> item_content_differs
  delta_differs          output_text.done != concatenated deltas -> delta_done_mismatch
  no_terminal            the stream stops before response.completed -> no_terminal_event
  custom_as_function     a custom tool delivered as function_call -> type_mismatch
  two_calls              two call items (fails under parallel_tool_calls=false)
  prefixed_name          the call name carries its namespace prefix -> prefixed_name_of_declared
  overlength             response.failed with context_length_exceeded (a correct answer)
  unknown_event          an event type no rule covers               -> C-11
  created_without_output response.created carries no output list  -> SDK clients cannot accumulate
  wrong_output_index     the message's events carry the reasoning item's output_index
  usage_missing          response.completed without usage           -> USG-01
  usage_total_mismatch   total_tokens is not input + output           -> USG-02
  error_string           HTTP 400 whose `error` is a bare string      -> RSP-16

Measurement modes (a harness's timeouts and error handling, not wire faults; --gap-seconds sets
the wait):

  slow_first_event       nothing at all, headers included, for the gap
  silent_gap             the stream goes silent for the gap after the reasoning item
  silent_gap_comments    the silent_gap stream, with an SSE comment line every 5 s during the gap
  cut_mid_stream         the connection is dropped after the first text delta, no terminal event
  http_400, http_429, http_500
                         an HTTP error in OpenAI's error envelope (429 with ``retry-after: 1``)
  overlength_http_400    over-length as an HTTP 400 in OpenAI's envelope, code context_length_exceeded
  overlength_top_level   the same message in trtllm-serve's own top-level error body
                         (``{"object": "error", "message", "type", "param", "code"}``)
  undeclared_tool        a call to a tool the request did not declare
  invalid_input          a call with ``{}`` for arguments
  length_with_call       a call, then ``response.incomplete`` for ``max_output_tokens``

The mode comes from ``--mode`` or, per request, from the header ``x-fake-mode``. With
``--fault-requests N`` the ``--mode`` applies to N requests only, after the first ``--fault-skip``
ones, and the others are answered ``ok``: a harness that recovers is seen recovering, and a fault
can wait until the conversation has a history.
Tools are read from the request (``tools``, namespace groups, ``additional_tools``); a request
that declares none (a harness's summary request, say) is answered without a call.

Function-call arguments satisfy the declared schema's required fields (``--arguments`` JSON is
merged over them), and with
--max-tool-turns N the server answers without a call once the input holds N tool
outputs, so a real harness (e.g. `codex exec`) finishes its turn: a wiring test
of harness -> recorder/server -> trace -> checks, without a model.

usage: fake_responses_server.py [--host 127.0.0.1] [--port 8399] [--mode ok] [--max-tool-turns N]
                                [--gap-seconds S] [--fault-requests N] [--fault-skip N]
                                [--arguments JSON]
"""

import argparse
import asyncio
import json
import time
import uuid

from aiohttp import web


def first_tool(body):
    def walk(tools, namespace):
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "namespace":
                found = walk(tool.get("tools"), tool.get("name"))
                if found:
                    return found
            elif tool.get("type") in ("function", "custom") and tool.get("name"):
                return tool, namespace
        return None

    found = walk(body.get("tools"), None)
    for item in body.get("input") or []:
        if found:
            break
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            found = walk(item.get("tools"), None)
    return found or (None, None)  # nothing declared: the answer has no call


def schema_arguments(tool):
    """A JSON object that satisfies the tool's required parameters, so a real harness accepts the call."""
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    args = {}
    for key in params.get("required") or []:
        kind = (props.get(key) or {}).get("type")
        if kind in ("integer", "number"):
            args[key] = 1
        elif kind == "boolean":
            args[key] = False
        elif kind == "array":
            args[key] = []
        elif kind == "object":
            args[key] = {}
        else:
            args[key] = "echo hi" if any(w in key for w in ("cmd", "command")) else "x"
    return json.dumps(args)


def tool_outputs(body):
    return sum(
        1
        for i in body.get("input") or []
        if isinstance(i, dict)
        and i.get("type") in ("function_call_output", "custom_tool_call_output")
    )


def frames(body, mode, max_tool_turns=0, arguments=None):
    rid, model = "resp_" + uuid.uuid4().hex, body.get("model", "fake")
    arguments_over = arguments or {}
    seq = iter(range(0, 10_000))  # from 0, as OpenAI and trtllm-serve number a stream
    out = []

    def emit(kind, **data):
        data = dict(type=kind, sequence_number=next(seq), **data)
        out.append(f"event: {kind}\ndata: {json.dumps(data)}\n\n")

    base = {"id": rid, "object": "response", "model": model, "status": "in_progress", "output": []}
    opening = base
    if mode == "created_without_output":
        opening = {k: v for k, v in base.items() if k != "output"}
    emit("response.created", response=opening)
    emit("response.in_progress", response=opening)
    if mode == "overlength":
        failed = dict(
            base,
            status="failed",
            error={
                "code": "context_length_exceeded",
                "message": "input exceeds the context window",
            },
        )
        emit("response.failed", response=failed)
        return out
    items = []
    rs_id, msg_id = "rs_" + uuid.uuid4().hex, "msg_" + uuid.uuid4().hex
    reasoning = {
        "id": rs_id,
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Plan the call."}],
    }
    emit(
        "response.output_item.added",
        output_index=0,
        item=dict(reasoning, content=[], status="in_progress"),
    )
    emit(
        "response.reasoning_text.delta",
        item_id=rs_id,
        output_index=0,
        content_index=0,
        delta="Plan the call.",
    )
    emit(
        "response.reasoning_text.done",
        item_id=rs_id,
        output_index=0,
        content_index=0,
        text="Plan the call.",
    )
    emit("response.output_item.done", output_index=0, item=reasoning)
    items.append(reasoning)
    text = "Running it now."
    # wrong_output_index: the message's part and text events point at the reasoning item (0).
    msg_index = 0 if mode == "wrong_output_index" else 1
    message = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    emit(
        "response.output_item.added",
        output_index=1,
        item=dict(message, content=[], status="in_progress"),
    )
    emit(
        "response.content_part.added",
        item_id=msg_id,
        output_index=msg_index,
        content_index=0,
        part={"type": "output_text", "text": "", "annotations": []},
    )
    for piece in ("Running ", "it now."):
        emit(
            "response.output_text.delta",
            item_id=msg_id,
            output_index=msg_index,
            content_index=0,
            delta=piece,
        )
    emit(
        "response.output_text.done",
        item_id=msg_id,
        output_index=msg_index,
        content_index=0,
        text=text + ("!" if mode == "delta_differs" else ""),
    )
    if mode != "drop_part_done":
        emit(
            "response.content_part.done",
            item_id=msg_id,
            output_index=msg_index,
            content_index=0,
            part={"type": "output_text", "text": text, "annotations": []},
        )
    emit("response.output_item.done", output_index=1, item=message)
    items.append(
        dict(
            message,
            content=[
                {
                    "type": "output_text",
                    "annotations": [],
                    "text": text + (" (rewritten)" if mode == "snapshot_text_differs" else ""),
                }
            ],
        )
    )
    tool, namespace = first_tool(body)
    n_calls = 2 if mode == "two_calls" else 1
    if tool is None or (max_tool_turns and tool_outputs(body) >= max_tool_turns):
        n_calls = 0  # end the conversation: a real harness stops when a turn has no call
    for k in range(n_calls):
        index = 2 + k
        name = (namespace + "." if mode == "prefixed_name" and namespace else "") + tool["name"]
        if mode == "undeclared_tool":
            name = "definitely_not_declared"
        call_id = "call_" + uuid.uuid4().hex
        as_custom = tool.get("type") == "custom" and mode != "custom_as_function"
        if as_custom:
            item = {
                "id": "ctc_" + uuid.uuid4().hex,
                "type": "custom_tool_call",
                "call_id": call_id,
                "name": name,
                "input": "echo hi",
            }
            if namespace and mode != "prefixed_name":
                item["namespace"] = namespace
            emit("response.output_item.added", output_index=index, item=dict(item, input=""))
            emit(
                "response.custom_tool_call_input.delta",
                item_id=item["id"],
                output_index=index,
                delta="echo hi",
            )
            emit(
                "response.custom_tool_call_input.done",
                item_id=item["id"],
                output_index=index,
                input="echo hi",
            )
        else:
            arguments = schema_arguments(tool) if tool.get("type") == "function" else "{}"
            if arguments_over:
                arguments = json.dumps(dict(json.loads(arguments), **arguments_over))
            if mode == "invalid_input":
                arguments = "{}"
            item = {
                "id": "fc_" + uuid.uuid4().hex,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            }
            if namespace and mode != "prefixed_name":
                item["namespace"] = namespace
            emit(
                "response.output_item.added",
                output_index=index,
                item=dict(item, arguments="", status="in_progress"),
            )
            emit(
                "response.function_call_arguments.delta",
                item_id=item["id"],
                output_index=index,
                delta=arguments,
            )
            emit(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=index,
                arguments=arguments,
            )
        emit("response.output_item.done", output_index=index, item=item)
        items.append(item)
    if mode == "unknown_event":
        # An event type the validator has no rule for (a reasoning summary): must be recorded.
        emit("response.reasoning_summary_text.delta", item_id="rs_x", output_index=0, delta="sum")
    if mode != "no_terminal":
        done = dict(base, status="completed", output=items)
        if mode == "length_with_call":
            done.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        if mode != "usage_missing":
            output_tokens = 5 + len(items)
            total = 10 + output_tokens + (1 if mode == "usage_total_mismatch" else 0)
            done["usage"] = {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0},
                             "output_tokens": output_tokens,
                             "output_tokens_details": {"reasoning_tokens": 2},
                             "total_tokens": total}  # fmt: skip
        emit("response.incomplete" if mode == "length_with_call" else "response.completed",
             response=done)  # fmt: skip
    return out


# HTTP errors a harness may retry or give up on, in OpenAI's error envelope.
HTTP_ERRORS = {
    "http_400": (400, {"message": "Unrecognized request argument supplied: frequency_bias",
                       "type": "invalid_request_error", "param": None, "code": None}),
    "http_429": (429, {"message": "Rate limit reached for requests", "type": "requests",
                       "param": None, "code": "rate_limit_exceeded"}),
    "http_500": (500, {"message": "The server had an error while processing your request.",
                       "type": "server_error", "param": None, "code": None}),
    "overlength_http_400": (400, {"message": "Your input exceeds the context window of this model.",
                                  "type": "invalid_request_error", "param": "input",
                                  "code": "context_length_exceeded"}),
}  # fmt: skip


def request_mode(request):
    """This request's mode.

    The ``x-fake-mode`` header, else ``--mode`` for the requests after the first ``--fault-skip``
    ones, ``--fault-requests`` of them (all when 0), else ok.
    """
    header = request.headers.get("x-fake-mode")
    if header:
        return header
    app = request.app
    app["served"]["requests"] += 1
    position = app["served"]["requests"] - app["fault_skip"]
    limit = app["fault_requests"]
    return app["mode"] if position >= 1 and (not limit or position <= limit) else "ok"


async def responses(request):
    body = await request.json()
    mode = request_mode(request)
    if mode == "error_string":
        return web.json_response({"error": "1 validation error: store: Extra inputs are not permitted"},
                                 status=400)  # fmt: skip
    if mode in HTTP_ERRORS:
        status, error = HTTP_ERRORS[mode]
        return web.json_response({"error": error}, status=status,
                                 headers={"retry-after": "1"} if status == 429 else None)  # fmt: skip
    if mode == "overlength_top_level":
        # trtllm-serve's ErrorResponse: the error's fields at the top level, no `error` object.
        return web.json_response({"object": "error",
                                  "message": "Your input exceeds the context window of this model.",
                                  "type": "BadRequestError", "param": None, "code": 400}, status=400)  # fmt: skip
    turns = request.app["max_tool_turns"]
    if not body.get("stream"):
        chunks = frames(body, mode, turns, request.app["arguments"])
        last = json.loads(chunks[-1].split("data: ", 1)[1])
        return web.json_response(last.get("response") or {})
    if mode == "slow_first_event":
        await asyncio.sleep(request.app["gap"])
    resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    chunks = frames(body, mode, turns, request.app["arguments"])
    # Where the stream stalls (silent_gap) or is dropped (cut_mid_stream): before the message
    # item, or after its first text delta.
    message_at = next((i for i, c in enumerate(chunks) if '"type": "message"' in c), None)
    first_delta = next((i for i, c in enumerate(chunks) if "response.output_text.delta" in c), None)
    for position, chunk in enumerate(chunks):
        if mode == "silent_gap" and position == message_at:
            await asyncio.sleep(request.app["gap"])
        if mode == "silent_gap_comments" and position == message_at:
            until = time.monotonic() + request.app["gap"]
            while time.monotonic() < until:
                await asyncio.sleep(min(5.0, max(0.0, until - time.monotonic())))
                await resp.write(b": keepalive\n\n")
        await resp.write(chunk.encode())
        if mode == "cut_mid_stream" and position == first_delta:
            if request.transport is not None:
                request.transport.abort()
            return resp
    await resp.write_eof()
    return resp


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--mode", default="ok")
    ap.add_argument(
        "--max-tool-turns",
        type=int,
        default=0,
        help="answer without a call once the input holds N tool outputs (lets a real harness finish)",
    )
    ap.add_argument("--gap-seconds", type=float, default=1.5)
    ap.add_argument(
        "--fault-requests",
        type=int,
        default=0,
        help="apply --mode to N requests only, then answer ok (0: all)",
    )
    ap.add_argument(
        "--fault-skip",
        type=int,
        default=0,
        help="answer the first N requests ok before --mode applies",
    )
    ap.add_argument(
        "--arguments",
        type=json.loads,
        default={},
        help="a JSON object merged over the arguments of every function call",
    )
    args = ap.parse_args()
    if not isinstance(args.arguments, dict):
        ap.error("--arguments must be a JSON object")
    app = web.Application(client_max_size=1 << 30)
    app["mode"] = args.mode
    app["max_tool_turns"] = args.max_tool_turns
    app["gap"] = args.gap_seconds
    app["fault_requests"] = args.fault_requests
    app["fault_skip"] = args.fault_skip
    app["arguments"] = args.arguments
    app["served"] = {"requests": 0}
    app.router.add_post("/v1/responses", responses)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
