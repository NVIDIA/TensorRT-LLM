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
"""The control matrix that qualifies this validator (references/acceptance-cases.md).

For each protocol, a fake server is started and a synthetic corpus is replayed
once per fault mode; ``verdict.py`` must trip exactly the mode's expected set
(rules plus declared consequences) and nothing else, and every normal control
must trip nothing. Then the cross-protocol fixture cases (X-..) run without a
server. No real trace is read.

The receipt (``<out-dir>/controls.json``) binds the result to the validator tree
SHA-256; ``verdict.py --tier full --controls-receipt`` accepts only a passing
receipt for the exact tree that issues the verdict (check V-01).

usage: controls.py --out-dir DIR [--only responses,anthropic,chat,cross] [--keep]
Exit 0 when every case passes, 1 otherwise.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
SKILL = os.path.dirname(SCRIPTS)
sys.path.insert(0, HERE)
import pipeline as pipeline_mod  # noqa: E402
import profiles as profiles_mod  # noqa: E402
import verdict as verdict_mod  # noqa: E402

PY = sys.executable
GLM_PROFILE = os.path.join(SKILL, "profiles", "model", "glm-5.yaml")
CODEX_OUTCOMES = {
    "aborted": {"exact": "aborted"},
    "unsupported": {"prefix": "unsupported call:"},
    "arguments_rejected": {"prefix": "failed to parse function arguments"},
    "unavailable_in_mode": {"regex": r"is unavailable in (\w+) mode"},
}
# Claude Code's tool results for a call it did not run (2.1.289-2.1.309 binaries): rejected by the
# user or a permission rule, or interrupted.
CLAUDE_ABORTED = [
    "The user doesn't want to proceed with this tool use.",
    "Permission for this tool use was denied.",
    "Permission to use ",
    "[Request interrupted by user for tool use]",
    "[Tool call did not complete:",
]
CONTROL_PROFILES = {
    "responses": {
        "id": "control-responses", "status": "measured", "protocol": "responses",
        "identity": {"session": {"body": "prompt_cache_key"},
                     "thread": {"body": "client_metadata.thread_id"}},
        "outcomes": CODEX_OUTCOMES,
        "errors": {"overlength": {"shape": "in_stream", "event": "response.failed",
                                  "code": "context_length_exceeded"}},
        "timeouts": {"first_event_s": 30, "idle_s": 1.0},
    },
    "anthropic": {
        "id": "control-anthropic", "status": "measured", "protocol": "anthropic-messages",
        "identity": {"session": {"header": "x-claude-code-session-id"}},
        "outcomes": {"aborted": {"prefix": CLAUDE_ABORTED},
                     "unsupported": {"contains": "No such tool available"},
                     "arguments_rejected": {"contains": "InputValidationError"},
                     "error": {"is_error": True}},
        "echo": {"text": "exact", "reasoning": "exact"},
        "errors": {"overlength": {"shape": "http_400", "error_type": "invalid_request_error",
                                  "wording": "prompt is too long"}},
        "timeouts": {"first_event_s": 30, "idle_s": 1.0},
    },
    "chat": {
        "id": "control-chat", "status": "measured", "protocol": "chat-completions",
        "identity": {"thread": {"header": "x-thread"}},
        "outcomes": {"aborted": {"none": "control: the chat client never cancels a call"},
                     "arguments_rejected": {"none": "control: the chat client runs every call"},
                     "error": {"prefix": "Error:"}},
        "echo": {"text": "exact", "reasoning": "exact"},
        "errors": {"overlength": {"shape": "http_400", "error_type": "invalid_request_error",
                                  "wording": "maximum context length"}},
        "timeouts": {"first_event_s": 30, "idle_s": 1.0},
    },
}  # fmt: skip

RESPONSES_EXPECT = {
    "ok": set(), "overlength": set(),
    "drop_part_done": {"C-01", "RSP-07"},
    "snapshot_text_differs": {"C-02", "RSP-11"},
    "delta_differs": {"C-01", "RSP-08", "RSP-09"},
    "no_terminal": {"C-04", "RSP-02"},
    "custom_as_function": {"C-06"},
    "two_calls": {"C-07"},
    "prefixed_name": {"C-06"},
    "unknown_event": {"C-11", "WIRE-05"},
    "created_without_output": {"C-01", "RSP-01", "C-12", "SDK-01"},
    "wrong_output_index": {"C-01", "RSP-15", "C-12", "SDK-01"},
    "usage_missing": {"C-14", "USG-01"},
    "usage_total_mismatch": {"C-14", "USG-02"},
    "error_string": {"C-01", "RSP-16", "C-16"},  # the store rejection pi met on main
}  # fmt: skip
ANTHROPIC_EXPECT = {
    "ok": set(), "ok_thinking": set(), "overlength_profile": set(),
    "overlength_400_wrong_wording": {"C-10"},
    "no_message_delta": {"C-01", "ANT-08", "C-04", "ANT-10"},
    "no_message_stop": {"C-04", "ANT-10"},
    "delta_before_start": {"C-01", "ANT-02", "C-12", "SDK-01"},
    "interleaved_blocks": {"C-01", "ANT-03"},
    "delta_type_mismatch": {"C-01", "ANT-04"},
    "bad_partial_json": {"C-01", "ANT-06", "M-01:payload_invalid"},
    "duplicate_tool_use_id": {"C-01", "ANT-06", "C-05"},
    "stop_reason_mismatch": {"C-01", "ANT-09"},
    "two_tools_disable_parallel": {"C-07"},
    "undeclared_tool": {"M-01:undeclared"},
    "invalid_input": {"M-01:payload_invalid"},
    "error_untyped": {"C-01", "ANT-11"},
    "content_after_stop": {"C-04", "ANT-10"},
    "silent_gap": {"R-02"},
    "html_body": {"C-01", "WIRE-01"},
    "json_for_stream": {"C-01", "WIRE-03"},
    "unknown_block": {"C-11", "WIRE-05"},
    "no_event_lines": {"C-01", "ANT-15", "C-12", "SDK-01"},
    "usage_zero_output": {"C-14", "USG-04"},
    "count_tokens_off": {"C-14", "USG-05"},
}  # fmt: skip
CHAT_EXPECT = {
    "ok": set(), "ok_reasoning": set(), "overlength_profile": set(),
    "no_done": {"C-04", "CHT-09"},
    "two_finish_reasons": {"C-01", "CHT-06"},
    "fragment_without_id": {"C-01", "CHT-04", "C-05"},
    "arguments_not_object": {"C-01", "CHT-05", "M-01:payload_invalid"},
    "finish_reason_mismatch": {"C-01", "CHT-07"},
    "duplicate_tool_call_id": {"C-01", "CHT-04", "C-05"},
    "two_calls_ptc_false": {"C-07"},
    "reasoning_in_content": {"C-09", "M-03"},
    "model_changes": {"C-01", "CHT-02"},
    "invalid_input": {"M-01:payload_invalid"},
    "silent_gap": {"R-02"},
    "html_body": {"C-01", "WIRE-01"},
    "json_for_stream": {"C-01", "WIRE-03"},
    "unknown_delta_field": {"C-11", "WIRE-05"},
    "bad_extension_field": {"C-01", "CHT-13"},
    "bad_stop_reason": {"C-01", "CHT-16"},
    "tool_index_gap": {"C-01", "CHT-04", "C-12", "SDK-01"},
    "repeated_call_id": {"C-01", "CHT-04", "C-12", "SDK-02"},
    "usage_cached_exceeds": {"C-14", "USG-03"},
    "usage_requested_missing": {"C-14", "USG-01"},
    "two_choices": set(),
    "choice_missing": {"C-01", "CHT-14"},
    "choice_without_finish": {"C-01", "CHT-06"},
    "error_string": {"C-01", "CHT-15", "C-16"},  # the store rejection pi met on main
}  # fmt: skip
READ_SCHEMA = {"type": "object", "properties": {"file_path": {"type": "string"}},
               "required": ["file_path"]}  # fmt: skip


# --------------------------------------------------------------------------- corpora


def responses_body(i, mode):
    grammar = "start: CMD\nCMD: /[a-z ]+/\n"
    tools = [{"type": "namespace", "name": "functions", "tools": [
        {"type": "custom", "name": "exec", "description": "run",
         "format": {"type": "grammar", "syntax": "lark", "definition": grammar}},
        {"type": "function", "name": "wait", "description": "wait",
         "parameters": {"type": "object", "properties": {"ms": {"type": "integer"}},
                        "required": ["ms"]}}]}]  # fmt: skip
    return "/v1/responses", {
        "model": "control", "stream": True, "parallel_tool_calls": False,
        "prompt_cache_key": f"s{i}", "client_metadata": {"thread_id": f"t{i}"},
        "input": [{"type": "additional_tools", "tools": tools},
                  {"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": f"task {i}"}]}],
    }  # fmt: skip


def anthropic_body(i, mode):
    body = {"model": "control", "max_tokens": 256, "stream": True,
            "system": [{"type": "text", "text": "control"}],
            "messages": [{"role": "user", "content": f"task {i}"}],
            "tools": [{"name": "Read", "description": "read a file", "input_schema": READ_SCHEMA}]}  # fmt: skip
    if mode == "two_tools_disable_parallel":
        body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    return "/v1/messages", body


def chat_body(i, mode):
    body = {"model": "control", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"task {i}"}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "read",
                                                        "parameters": READ_SCHEMA}}]}  # fmt: skip
    if mode == "two_calls_ptc_false":
        body["parallel_tool_calls"] = False
    if mode in ("two_choices", "choice_missing", "choice_without_finish"):
        body["n"] = 2
    return "/v1/chat/completions", body


# --------------------------------------------------------------------------- servers


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(argv, port, log):
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"server did not start: {' '.join(argv)}")


def write_profile(path, data):
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)
    return path


def run_verdict(out, paths, harness, population="controls", claims=None, model=GLM_PROFILE,
                matrix=None, raw=None, responses=None, build_id=None):  # fmt: skip
    argv = ["--harness", harness, "--model", model, "--population", population]
    if responses:
        argv += ["--responses"] + list(responses)
    if claims:
        argv += ["--claims", claims]
    if matrix:
        argv += ["--matrix", matrix]
    if build_id is not None:
        argv += ["--build-id", build_id]
    if raw:
        argv += ["--raw"] + list(raw)
    argv += ["--out-dir", out]  # also ends --raw's list before the positional paths
    code = verdict_mod.main(argv + list(paths))
    with open(os.path.join(out, "verdict.json")) as fh:
        return code, json.load(fh)


def count_tokens_body(body):
    """What a client sends to count a Messages request: its prompt-shaping fields."""
    keep = ("model", "messages", "system", "tools", "tool_choice", "thinking", "output_config",
            "context_management")  # fmt: skip
    return {k: v for k, v in body.items() if k in keep}


def protocol_matrix(name, expect, body_fn, server_argv, work, n=3):
    results = []
    port = free_port()
    log = open(os.path.join(work, f"{name}-server.log"), "w")
    proc = start(server_argv + ["--port", str(port)], port, log)
    harness = write_profile(os.path.join(work, f"{name}-profile.yaml"), CONTROL_PROFILES[name])
    try:
        for mode, expected in expect.items():
            corpus = os.path.join(work, f"{name}-{mode}.corpus.jsonl")
            with open(corpus, "w") as fh:
                for i in range(n):
                    route, body = body_fn(i, mode)
                    fh.write(json.dumps({"corpus_id": f"{name}-{mode}-{i}", "route": route,
                                         "thread": f"{name}-{mode}-{i}", "body": body}) + "\n")  # fmt: skip
                    if route == "/v1/messages":  # a count before the turn, as Claude Code may send
                        fh.write(json.dumps({"corpus_id": f"{name}-{mode}-{i}-count",
                                             "route": "/v1/messages/count_tokens",
                                             "body": count_tokens_body(body)}) + "\n")  # fmt: skip
            out = os.path.join(work, f"{name}-{mode}")
            shutil.rmtree(out, ignore_errors=True)
            replay = subprocess.run(
                [PY, os.path.join(SCRIPTS, "replay_requests.py"), "--base-url",
                 f"http://127.0.0.1:{port}", "--out-dir", out, "--concurrency", "4",
                 "--first-event-timeout", "30", "--header", f"x-fake-mode: {mode}",
                 "--header", "x-claude-code-session-id: control", corpus],
                capture_output=True, text=True, timeout=120)  # fmt: skip
            if replay.returncode != 0:
                results.append({"protocol": name, "mode": mode, "ok": False,
                                "error": "replay failed: " + replay.stderr[-300:]})  # fmt: skip
                continue
            _, verdict = run_verdict(os.path.join(out, "verdict"), [out], harness)
            tripped = set(verdict["tripped"])
            ok = tripped == expected
            results.append({"protocol": name, "mode": mode, "ok": ok,
                            "expected": sorted(expected), "tripped": sorted(tripped)})  # fmt: skip
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()
    return results


# --------------------------------------------------------------------------- cross cases


def _write_trace(directory, pairs):
    """pairs: [(request_record, response_record_or_None)] -> request/response shards."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "requests-x.jsonl"), "w") as req, \
            open(os.path.join(directory, "responses-x.jsonl"), "w") as resp:  # fmt: skip
        for request, response in pairs:
            req.write(json.dumps(request) + "\n")
            if response is not None:
                resp.write(json.dumps(response) + "\n")


def _anthropic_sse(blocks, stop_reason="tool_use"):
    events = [("message_start", {"message": {"id": "m", "type": "message", "role": "assistant",
                                             "model": "x", "content": [],
                                             "usage": {"input_tokens": 1}}})]  # fmt: skip
    for index, block in enumerate(blocks):
        if block["type"] == "thinking":
            events += [("content_block_start", {"index": index, "content_block": {"type": "thinking", "thinking": ""}}),
                       ("content_block_delta", {"index": index, "delta": {"type": "thinking_delta",
                                                                          "thinking": block["text"]}})]  # fmt: skip
            if block.get("signature"):
                events.append(("content_block_delta", {"index": index, "delta": {
                    "type": "signature_delta", "signature": block["signature"]}}))  # fmt: skip
        elif block["type"] == "text":
            events += [("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}}),
                       ("content_block_delta", {"index": index, "delta": {"type": "text_delta",
                                                                          "text": block["text"]}})]  # fmt: skip
        else:
            events += [("content_block_start", {"index": index, "content_block": {
                "type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}),
                ("content_block_delta", {"index": index, "delta": {"type": "input_json_delta",
                                                                   "partial_json": block["json"]}})]  # fmt: skip
        events.append(("content_block_stop", {"index": index}))
    events += [("message_delta", {"delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 1}}),
               ("message_stop", {})]  # fmt: skip
    return "".join(f"event: {n}\ndata: {json.dumps(dict(d, type=n))}\n\n" for n, d in events)


def _anth_request(trace_id, messages, at):
    return {"event": "request", "trace_id": trace_id, "route": "/v1/messages", "recorded_at": at,
            "headers": [["x-claude-code-session-id", "s1"]],
            "body": {"model": "x", "max_tokens": 64, "stream": True, "messages": messages,
                     "tools": [{"name": "Read", "input_schema": READ_SCHEMA}]}}  # fmt: skip


def _sse_response(trace_id, body, status="completed", engine_id=None):
    record = {"event": "response", "trace_id": trace_id, "status": status, "http_status": 200,
              "response": {"kind": "sse_text", "body": body}}  # fmt: skip
    if engine_id is not None:
        record["disagg_request_id"] = engine_id
    return record


GLM_RAW = "Plan.</think>Reading.<tool_call>Read<arg_key>file_path</arg_key><arg_value>a</arg_value></tool_call>"
RAW_CASES = {
    # case: (raw text, delivered blocks, disable_parallel_tool_use, expected C-13 rules, M-07 expected)
    "X-13": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."},
                       {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}],
             False, set(), False),
    "X-14": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."}],
             False, {"RAW-01"}, False),
    "X-15": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."},
                       {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "b"}'}],
             False, {"RAW-02"}, False),
    "X-16": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading"},
                       {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}],
             False, {"RAW-03"}, False),
    "X-17": (GLM_RAW.replace("</arg_value></tool_call>", ""),
             [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."}],
             False, set(), True),
    "X-18": (GLM_RAW + GLM_RAW.split("Reading.", 1)[1],
             [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."},
              {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}],
             True, set(), False),
}  # fmt: skip


def echo_cases(work, anth):
    """C-03 for texts: Messages and Chat echo our text and reasoning at the appended position."""
    results = []
    user = {"role": "user", "content": "go"}
    delivered = [{"type": "thinking", "text": "Plan.", "signature": "c2ln"},
                 {"type": "text", "text": "Reading."},
                 {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]  # fmt: skip
    use = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}
    result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
    }
    variants = {
        "X-21": ([{"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
                  {"type": "text", "text": "Reading."}, use], anth, "PASS", None),
        "X-22": ([{"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
                  {"type": "text", "text": "Reading!"}, use], anth, "FAIL", "text"),
        "X-23": ([{"type": "thinking", "thinking": "Plan."}, {"type": "text", "text": "Reading."}, use],
                 anth, "FAIL", "reasoning"),
        "X-24": ([{"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
                  {"type": "text", "text": "Reading."}, use],
                 write_profile(os.path.join(work, "x24.yaml"), dict(CONTROL_PROFILES["anthropic"],
                                                                     echo={"text": "TODO"})),
                 "BLOCKED", None),
    }  # fmt: skip
    for case, (echoed, profile, status, field) in variants.items():
        directory = os.path.join(work, case.lower())
        turn2 = [user, {"role": "assistant", "content": echoed}, result]
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(delivered))),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(
            os.path.join(directory, "v"), [directory], profile, "window", "conformance"
        )
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        ok = c03["status"] == status and (status != "PASS" or c03["text_echo_equal"] == 1)
        if field:
            with open(os.path.join(directory, "v", "report.json")) as fh:
                fields = {e.get("field") for e in json.load(fh)["examples"].get("C-03", [])}
            ok = ok and field in fields
        results.append({"case": case, "ok": ok})
        if case == "X-21":
            # X-101 rides on this case: the signed thinking block that came back verbatim is a
            # reasoning echo, the text block a text echo.
            results.append({"case": "X-101", "ok": c03["reasoning_echo_equal"] == 1
                            and c03["text_echo_equal"] == 1})  # fmt: skip
    # X-47: the same request answered twice (a retry after the network dropped the first answer);
    # the client goes on with the second answer, so only the second is held against the echo.
    directory = os.path.join(work, "x-47")
    first = [{"type": "text", "text": "First answer."}]
    second = [{"type": "text", "text": "Second answer."}]
    turn2 = [user, {"role": "assistant", "content": [{"type": "text", "text": "Second answer."}]},
             {"role": "user", "content": "go on"}]  # fmt: skip
    _write_trace(directory, [
        (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(first, "end_turn"))),
        (_anth_request("t2", [user], "2026-01-01T00:00:02Z"), _sse_response("t2", _anthropic_sse(second, "end_turn"))),
        (_anth_request("t3", turn2, "2026-01-01T00:00:03Z"),
         _sse_response("t3", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
    ])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-47", "ok": c03["status"] == "PASS" and c03["superseded_answers"] == 1
                    and c03["text_echo_equal"] == 1})  # fmt: skip
    # X-48: the harness dropped an empty answer and asked again at the same position, with a text
    # block appended to its last user message (Claude Code after an empty turn). The client went
    # on with the second answer, so only the second is held against the echo.
    # X-49: the second request changed the last user message instead of appending to it: not a
    # re-ask, so the first answer is still judged, and its echo differs.
    reminder = {"role": "system", "content": "reminder"}
    asked = {"role": "user", "content": [{"type": "text", "text": "go"}]}
    nudge = {"type": "text", "text": "Your reply was empty. Continue."}
    empty = [{"type": "thinking", "text": "Hmm.", "signature": "c2ln"}]
    answer = [{"type": "thinking", "text": "Plan.", "signature": "c2ln"},
              {"type": "text", "text": "Second answer."}]  # fmt: skip
    for case, again, status in (
        ("X-48", {"role": "user", "content": asked["content"] + [nudge]}, "PASS"),
        ("X-49", {"role": "user", "content": [{"type": "text", "text": "go!"}, nudge]}, "FAIL"),
    ):
        directory = os.path.join(work, case.lower())
        turn3 = [again, reminder, {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
            {"type": "text", "text": "Second answer."}]}, {"role": "user", "content": "go on"}]  # fmt: skip
        _write_trace(directory, [
            (_anth_request("t1", [asked, reminder], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(empty, "end_turn"))),
            (_anth_request("t2", [again, reminder], "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse(answer, "end_turn"))),
            (_anth_request("t3", turn3, "2026-01-01T00:00:03Z"),
             _sse_response("t3", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        results.append({"case": case, "ok": c03["status"] == status and c03["text_echo_equal"] == 1
                        and c03["reasked_answers"] == (1 if status == "PASS" else 0)})  # fmt: skip
    # X-25: Chat Completions; the echo is the assistant message's content at the appended position.
    chat = write_profile(os.path.join(work, "x25.yaml"), CONTROL_PROFILES["chat"])
    directory = os.path.join(work, "x-25")
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Reading."}}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]  # fmt: skip
    body = "".join("data: " + json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m",
                                              created=1)) + "\n\n" for c in chunks) + "data: [DONE]\n\n"  # fmt: skip

    def chat_request(trace_id, messages, at):
        return {"event": "request", "trace_id": trace_id, "route": "/v1/chat/completions",
                "recorded_at": at, "headers": [["x-thread", "th"]],
                "body": {"model": "m", "stream": True, "messages": messages}}  # fmt: skip

    _write_trace(directory, [
        (chat_request("c1", [user], "2026-01-01T00:00:01Z"), _sse_response("c1", body)),
        (chat_request("c2", [user, {"role": "assistant", "content": "Reading?"},
                             {"role": "user", "content": "and?"}], "2026-01-01T00:00:02Z"),
         _sse_response("c2", body)),
    ])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], chat, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-25", "ok": c03["status"] == "FAIL"})
    return results


def unechoed_by_design_cases(work, anth):
    """C-03 for replies a harness never appends: counted under their own label, not as missing.

    X-102: a structured-output side request (``output_config.format``, no tools) comes between a
    turn and the request that sends it back. X-103: the reply to a compaction request is the
    summary the next request starts from, with a shorter history. X-103b: a sub-agent's last
    reply, followed by its parent's next request, which is shorter than the sub-agent's but
    continues the parent's conversation: missing, not compacted.
    """
    results = []
    user = {"role": "user", "content": "go"}
    done = _anthropic_sse([{"type": "text", "text": "done"}], "end_turn")

    def turn(call_id, text, path):
        """A reply's blocks (a text and a call), and the assistant message that echoes them."""
        blocks = [{"type": "text", "text": text},
                  {"type": "tool_use", "id": call_id, "name": "Read",
                   "json": json.dumps({"file_path": path})}]  # fmt: skip
        message = {"role": "assistant", "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": call_id, "name": "Read", "input": {"file_path": path}}]}  # fmt: skip
        return blocks, message

    def result(call_id, *more):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                                             "content": "ok"}, *more]}  # fmt: skip

    def judged(case, pairs):
        directory = os.path.join(work, case.lower())
        _write_trace(directory, pairs)
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
        return next(r for r in v["checks"] if r["id"] == "C-03")

    reading, read = turn("toolu_1", "Reading.", "a")
    side = _anth_request("s1", [{"role": "user", "content": "a title?"}], "2026-01-01T00:00:02Z")
    side["body"].pop("tools")
    side["body"]["output_config"] = {"format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"title": {"type": "string"}}}}}  # fmt: skip
    c03 = judged("X-102", [
        (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(reading))),
        (side, _sse_response("s1", _anthropic_sse([{"type": "text", "text": '{"title": "T"}'}], "end_turn"))),
        (_anth_request("t2", [user, read, result("toolu_1")], "2026-01-01T00:00:03Z"),
         _sse_response("t2", done)),
    ])  # fmt: skip
    results.append({"case": "X-102", "ok": c03["status"] == "PASS"
                    and c03.get("text_not_echoed_side_request") == 1 and c03["text_echo_missing"] == 0})  # fmt: skip
    summarize = {"type": "text", "text": "Summarize the conversation."}
    summary = [{"type": "thinking", "text": "S.", "signature": "c2ln"},
               {"type": "text", "text": "Summary."}]  # fmt: skip
    c03 = judged("X-103", [
        (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(reading))),
        (_anth_request("t2", [user, read, result("toolu_1", summarize)], "2026-01-01T00:00:02Z"),
         _sse_response("t2", _anthropic_sse(summary, "end_turn"))),
        (_anth_request("t3", [{"role": "user", "content": "Summary. Continue."}], "2026-01-01T00:00:03Z"),
         _sse_response("t3", done)),
    ])  # fmt: skip
    results.append({"case": "X-103", "ok": c03["status"] == "PASS"
                    and c03.get("text_not_echoed_compacted") == 1
                    and c03.get("reasoning_not_echoed_compacted") == 1
                    and c03["text_echo_missing"] == 0 and c03["reasoning_echo_missing"] == 0})  # fmt: skip
    task, delegated = turn("toolu_t", "Delegating.", "t")
    first, looked = turn("toolu_r1", "Looking.", "b")
    second, looked_more = turn("toolu_r2", "Looking more.", "c")
    sub = {"role": "user", "content": "sub task"}
    c03 = judged("X-103b", [
        (_anth_request("m1", [user], "2026-01-01T00:00:01Z"), _sse_response("m1", _anthropic_sse(task))),
        (_anth_request("s1", [sub], "2026-01-01T00:00:02Z"), _sse_response("s1", _anthropic_sse(first))),
        (_anth_request("s2", [sub, looked, result("toolu_r1")], "2026-01-01T00:00:03Z"),
         _sse_response("s2", _anthropic_sse(second))),
        (_anth_request("s3", [sub, looked, result("toolu_r1"), looked_more, result("toolu_r2")],
                       "2026-01-01T00:00:04Z"),
         _sse_response("s3", _anthropic_sse([{"type": "text", "text": "Found it."}], "end_turn"))),
        (_anth_request("m2", [user, delegated, result("toolu_t")], "2026-01-01T00:00:05Z"),
         _sse_response("m2", done)),
    ])  # fmt: skip
    results.append({"case": "X-103b", "ok": c03["status"] == "PASS" and c03["text_echo_missing"] == 1
                    and c03.get("text_not_echoed_compacted") == 0})  # fmt: skip
    return results


EDIT_SCHEMA = {"type": "object", "required": ["file_path"],
               "properties": {"file_path": {"type": "string"},
                              "replace_all": {"type": "boolean", "default": False}}}  # fmt: skip
BASH_SCHEMA = {"type": "object", "properties": {"command": {"type": "string"}},
               "required": ["command"]}  # fmt: skip


def rewrite_cases(work, anth):
    """C-03 for calls a harness sends back changed as its profile declares (echo.tool_input)."""
    results = []
    user = {"role": "user", "content": "go"}
    tools = [
        {"name": "Edit", "input_schema": EDIT_SCHEMA},
        {"name": "Bash", "input_schema": BASH_SCHEMA},
    ]
    rewriting = write_profile(os.path.join(work, "x-rewrites.yaml"), dict(
        CONTROL_PROFILES["anthropic"], echo={"text": "exact", "reasoning": "exact",
                                             "tool_input": ["schema_defaults", "cd_cwd_prefix"]}))  # fmt: skip

    def request(trace_id, messages, at):
        return {"event": "request", "trace_id": trace_id, "route": "/v1/messages", "recorded_at": at,
                "headers": [["x-claude-code-session-id", "s1"]],
                "body": {"model": "x", "max_tokens": 64, "stream": True, "messages": messages,
                         "tools": tools}}  # fmt: skip

    # (case, tool, delivered arguments, echoed arguments, profile, C-03 status)
    variants = (
        # X-39: an argument filled in at its declared default, declared by the profile: equal.
        ("X-39", "Edit", '{"file_path": "a"}', {"file_path": "a", "replace_all": False}, rewriting, "PASS"),
        # X-40: the same echo where the profile declares no rewrite: it differs.
        ("X-40", "Edit", '{"file_path": "a"}', {"file_path": "a", "replace_all": False}, anth, "FAIL"),
        # X-41: an added argument that is not its declared default is a change.
        ("X-41", "Edit", '{"file_path": "a"}', {"file_path": "a", "replace_all": True}, rewriting, "FAIL"),
        # X-42: a leading `cd <dir> &&` cut from what we delivered, declared: equal.
        ("X-42", "Bash", '{"command": "cd /w && ls"}', {"command": "ls"}, rewriting, "PASS"),
        # X-43: the cut is one-sided: a `cd` the harness added is a change.
        ("X-43", "Bash", '{"command": "ls"}', {"command": "cd /w && ls"}, rewriting, "FAIL"),
        # X-44: and so is a `cd` into another directory.
        ("X-44", "Bash", '{"command": "cd /w && ls"}', {"command": "cd /v && ls"}, rewriting, "FAIL"),
    )  # fmt: skip
    for case, name, delivered, echoed, profile, status in variants:
        directory = os.path.join(work, case.lower())
        reply = _anthropic_sse(
            [{"type": "tool_use", "id": "toolu_1", "name": name, "json": delivered}]
        )
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": name, "input": echoed}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": "ok"}]}]  # fmt: skip
        _write_trace(directory, [
            (request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", reply)),
            (request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(
            os.path.join(directory, "v"), [directory], profile, "window", "conformance"
        )
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        ok = c03["status"] == status and c03["echo_rewritten"] == (1 if status == "PASS" else 0)
        results.append({"case": case, "ok": ok})
    # X-45: a rewrite the validator does not know fails the run instead of passing silently.
    unknown = write_profile(os.path.join(work, "x45.yaml"), dict(
        CONTROL_PROFILES["anthropic"], echo={"text": "exact", "tool_input": ["strip_everything"]}))  # fmt: skip
    try:
        pipeline_mod.Pipeline(
            profiles_mod.load_profile(unknown), profiles_mod.load_profile(GLM_PROFILE)
        )
        refused = False
    except ValueError:
        refused = True
    results.append({"case": "X-45", "ok": refused})
    return results


K3_PROFILE = os.path.join(SKILL, "profiles", "model", "kimi-k3.yaml")


def usage_cases(work, anth):
    """C-14 against the server's own counts: its prompt count (trace) and the engine's (raw)."""
    results = []
    user = {"role": "user", "content": "go"}
    reply = _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")  # bills input 1, output 1
    for case, prompt, model, expected in (
        ("X-26", 99, GLM_PROFILE, True),
        ("X-27", 4, K3_PROFILE, False),
    ):
        directory = os.path.join(work, case.lower())
        record = dict(
            _sse_response("t1", reply), prompt_tokens={"count": prompt, "source": "control"}
        )
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), record)])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           model=model)  # fmt: skip
        c14 = next(r for r in v["checks"] if r["id"] == "C-14")
        results.append({"case": case, "ok": ("USG-07" in v["tripped"]) == expected
                        and c14["usage_vs_trace_prompt"] == 1})  # fmt: skip
    directory = os.path.join(work, "x-28")
    _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                              _sse_response("t1", reply, engine_id="e1"))])  # fmt: skip
    with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
        fh.write(
            json.dumps({"disagg_request_id": "e1", "text": "x</think>ok", "num_output_tokens": 7})
            + "\n"
        )
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                       raw=[directory])  # fmt: skip
    results.append({"case": "X-28", "ok": "USG-06" in v["tripped"] and "C-14" in v["tripped"]})
    # X-29 / X-30: the turn after an echoed call billed less input; allowed only when the client
    # asked the server to clear history (context_management).
    for case, clearing in (("X-29", False), ("X-30", True)):
        directory = os.path.join(work, case.lower())
        call = [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]
        first = _anthropic_sse(call).replace('"input_tokens": 1', '"input_tokens": 50')
        turn2 = [
            user,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "a"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
        ]
        second = _anth_request("t2", turn2, "2026-01-01T00:00:02Z")
        if clearing:
            second["body"]["context_management"] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", first)),
                                 (second, _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}],
                                                                             "end_turn")))])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
        results.append({"case": case, "ok": ("USG-08" in v["tripped"]) != clearing})
    return results


def diff_runs(directory, system="sys", billed=7, text="ok", blocks=None):
    """One turn over Chat and over Messages: the same conversation, configurable server answers."""
    user = {"role": "user", "content": "go"}
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}},
    ]
    chat_sse = (
        "".join(
            "data: "
            + json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m", created=1))
            + "\n\n"
            for c in chunks
        )
        + "data: [DONE]\n\n"
    )
    chat = {
        "event": "request",
        "trace_id": "c1",
        "route": "/v1/chat/completions",
        "recorded_at": "2026-01-01T00:00:01Z",
        "body": {
            "model": "m",
            "stream": True,
            "messages": [{"role": "system", "content": "sys"}, user],
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": READ_SCHEMA}}
            ],
        },
    }
    anth_sse = _anthropic_sse(blocks or [{"type": "text", "text": text}], "end_turn").replace(
        '"input_tokens": 1', f'"input_tokens": {billed}'
    )
    anth = {"event": "request", "trace_id": "a1", "route": "/v1/messages", "recorded_at": "2026-01-01T00:00:01Z",
            "body": {"model": "m", "stream": True, "max_tokens": 64, "system": system, "messages": [user],
                     "tools": [{"name": "read", "input_schema": READ_SCHEMA}]}}  # fmt: skip
    _write_trace(os.path.join(directory, "chat"), [(chat, _sse_response("c1", chat_sse))])
    _write_trace(os.path.join(directory, "anth"), [(anth, _sse_response("a1", anth_sse))])
    return os.path.join(directory, "chat"), os.path.join(directory, "anth")


def diff_runs_two_turns(directory, first=("r", "r"), replay=("r", "r"), billed=(9, 9)):
    """Two turns over Chat and over Messages: turn 1 answers with reasoning, turn 2 replays it.

    first: the reasoning each path generated on turn 1; replay: the reasoning each request of
    turn 2 sends back; billed: each path's input bill on turn 2 (turn 1 bills 7 on both).
    """
    user, more = {"role": "user", "content": "go"}, {"role": "user", "content": "more"}

    def chat_sse(reasoning, prompt):
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": reasoning}}
                ]
            },
            {"choices": [{"index": 0, "delta": {"content": "ok"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": 2,
                    "total_tokens": prompt + 2,
                },
            },
        ]
        frames = [
            json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m", created=1))
            for c in chunks
        ]
        return "".join(f"data: {frame}\n\n" for frame in frames) + "data: [DONE]\n\n"

    tools = [{"type": "function", "function": {"name": "read", "parameters": READ_SCHEMA}}]

    def chat_request(trace_id, messages, at):
        body = {
            "model": "m",
            "stream": True,
            "messages": [{"role": "system", "content": "sys"}] + messages,
            "tools": tools,
        }
        return {"event": "request", "trace_id": trace_id, "route": "/v1/chat/completions",
                "recorded_at": at, "body": body}  # fmt: skip

    def anth_request(trace_id, messages, at):
        body = {
            "model": "m",
            "stream": True,
            "max_tokens": 64,
            "system": "sys",
            "messages": messages,
            "tools": [{"name": "read", "input_schema": READ_SCHEMA}],
        }
        return {"event": "request", "trace_id": trace_id, "route": "/v1/messages",
                "recorded_at": at, "body": body}  # fmt: skip

    def anth_sse(reasoning, prompt):
        blocks = [
            {"type": "thinking", "text": reasoning, "signature": "c2ln"},
            {"type": "text", "text": "ok"},
        ]
        sse = _anthropic_sse(blocks, "end_turn")
        return sse.replace('"input_tokens": 1', f'"input_tokens": {prompt}')

    chat_history = [
        user,
        {"role": "assistant", "content": "ok", "reasoning_content": replay[0]},
        more,
    ]
    thinking = {"type": "thinking", "thinking": replay[1], "signature": "c2ln"}
    anth_history = [
        user,
        {"role": "assistant", "content": [thinking, {"type": "text", "text": "ok"}]},
        more,
    ]
    t1, t2 = "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"
    chat_pairs = [
        (chat_request("c1", [user], t1), _sse_response("c1", chat_sse(first[0], 7))),
        (chat_request("c2", chat_history, t2), _sse_response("c2", chat_sse(first[0], billed[0]))),
    ]
    anth_pairs = [
        (anth_request("a1", [user], t1), _sse_response("a1", anth_sse(first[1], 7))),
        (anth_request("a2", anth_history, t2), _sse_response("a2", anth_sse(first[1], billed[1]))),
    ]
    _write_trace(os.path.join(directory, "chat"), chat_pairs)
    _write_trace(os.path.join(directory, "anth"), anth_pairs)
    return os.path.join(directory, "chat"), os.path.join(directory, "anth")


def protocol_diff_cases(work):
    """protocol_diff.py: an equal pair passes; a server-side difference fails; a harness one does not."""
    import protocol_diff

    results = []
    for case, kwargs, code, attribution in (("X-31", {}, 0, "equal"),
                                            ("X-32", {"billed": 9}, 1, "server_render"),
                                            ("X-33", {"system": "other"}, 0, "harness"),
                                            ("X-34", {"text": "no"}, 2, "divergent_generation"),
                                            ("X-35", {"blocks": [{"type": "thinking", "text": "ok"}]},
                                             1, "server_output")):  # fmt: skip
        directory = os.path.join(work, case.lower())
        chat, anth = diff_runs(directory, **kwargs)
        out = os.path.join(directory, "diff.json")
        rc = protocol_diff.main(["--out", out, f"chat={chat}", f"messages={anth}"])
        with open(out) as fh:
            row = json.load(fh)["rows"][0]["messages"]
        results.append({"case": case, "ok": rc == code and row["attribution"] == attribution})
    # X-36: turn 1 diverged in its reasoning; turn 2 replays each path's own reasoning and is
    # billed differently. The bill follows the replayed reasoning: after_divergence, not the server.
    # X-37: turn 1 agreed, but the harness replays other reasoning over one path: harness.
    # X-38: everything replayed agrees and the bills differ: the server rendered it differently.
    for case, kwargs, code, attribution in (
            ("X-36", {"first": ("ra", "rb"), "replay": ("ra", "rb"), "billed": (9, 10)}, 2, "after_divergence"),
            ("X-37", {"replay": ("r", "other"), "billed": (9, 11)}, 0, "harness"),
            ("X-38", {"billed": (9, 12)}, 1, "server_render")):  # fmt: skip
        directory = os.path.join(work, case.lower())
        chat, anth = diff_runs_two_turns(directory, **kwargs)
        out = os.path.join(directory, "diff.json")
        rc = protocol_diff.main(["--out", out, f"chat={chat}", f"messages={anth}"])
        with open(out) as fh:
            rows = json.load(fh)["rows"]
        second = rows[1]["messages"]["attribution"] if len(rows) > 1 else None
        results.append({"case": case, "ok": rc == code and second == attribution})
    return results


def raw_cases(work, anth):
    """C-13: what was delivered against what the model wrote (GLM's native format)."""
    results = []
    user = {"role": "user", "content": "go"}
    for case, (text, blocks, single, rules, anomaly) in RAW_CASES.items():
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        if single:
            request["body"]["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(blocks, stop),
                                                         engine_id="e1"))])  # fmt: skip
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(
                json.dumps({"disagg_request_id": "e1", "text": text, "finish_reason": "stop"})
                + "\n"
            )
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        tripped = set(v["tripped"])
        # A reply the model broke its format in is joined but not compared.
        ok = ({t for t in tripped if t.startswith("RAW-")} == rules
              and ("C-13" in tripped) == bool(rules) and ("M-07" in tripped) == anomaly
              and c13.get("joined") == 1 and c13["denominator"] == (0 if anomaly else 1))  # fmt: skip
        results.append({"case": case, "ok": ok})
        if case == "X-17":
            # X-100 rides on this case: its only joined reply was not compared, so C-13 tested
            # nothing and is BLOCKED, with the reason, not PASS over 0 of 1.
            results.append({"case": "X-100", "ok": c13["status"] == "BLOCKED"
                            and c13.get("not_judged") == 1
                            and "none was compared" in (c13["reason"] or "")})  # fmt: skip

    def judged(case, text, blocks, model=GLM_PROFILE):
        """One Messages reply delivering ``blocks``, judged against a raw record of ``text``."""
        directory = os.path.join(work, case.lower())
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                                  _sse_response("t1", _anthropic_sse(blocks, stop), engine_id="e1"))])  # fmt: skip
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(json.dumps({"disagg_request_id": "e1", "text": text}) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           model=model, raw=[directory])  # fmt: skip
        with open(os.path.join(directory, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        tripped = set(v["tripped"])
        return next(r for r in v["checks"] if r["id"] == "C-13"), counts, tripped

    # X-99: GLM opened its call inside the reasoning and wrote the </think> it owed after the call.
    # Serving ends the reasoning at the call and drops that close; GLM's profile declares it
    # (`owed_think_close`), so the reply is compared, and equal, under the repair.
    # X-99b: the same under a profile that does not declare the repair: the model's slip
    # (`call_in_reasoning`), not compared, flagged by M-07.
    # X-99c: a call inside the reasoning and no close at all stays an unclosed think.
    call = "<tool_call>Read<arg_key>file_path</arg_key><arg_value>a</arg_value></tool_call>"
    owed = [{"type": "thinking", "text": "Plan."},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]  # fmt: skip
    with open(GLM_PROFILE) as fh:
        glm = yaml.safe_load(fh)
    glm["native"] = dict(glm["native"], repairs=[r for r in glm["native"].get("repairs") or ()
                                                 if r != "owed_think_close"])  # fmt: skip
    unrepaired = write_profile(os.path.join(work, "x-99b-glm.yaml"), glm)
    c13, counts, tripped = judged("X-99", "Plan." + call + "</think>", owed)
    results.append({"case": "X-99", "ok": c13["status"] == "PASS" and c13["denominator"] == 1
                    and counts.get("C-13:repair:owed_think_close") == 1
                    and not {t for t in tripped if t.startswith(("RAW-", "M-07"))}})  # fmt: skip
    c13, counts, tripped = judged("X-99b", "Plan." + call + "</think>", owed, model=unrepaired)
    results.append({"case": "X-99b", "ok": counts.get("C-13:anomaly:call_in_reasoning") == 1
                    and c13["denominator"] == 0 and c13.get("not_judged") == 1 and "M-07" in tripped
                    and not {t for t in tripped if t.startswith("RAW-")}})  # fmt: skip
    c13, counts, tripped = judged("X-99c", "Plan." + call, owed)
    results.append({"case": "X-99c", "ok": counts.get("C-13:anomaly:unclosed_think") == 1
                    and not counts.get("C-13:repair:owed_think_close") and c13.get("not_judged") == 1
                    and c13["denominator"] == 0 and "M-07" in tripped})  # fmt: skip
    # X-100b: of two joined replies, one is compared (equal) and one has a model format anomaly:
    # C-13 PASS over the one compared, and the reason says how many were not.
    directory = os.path.join(work, "x-100b")
    _write_trace(directory, [
        (_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
         _sse_response("t1", _anthropic_sse(RAW_CASES["X-13"][1]), engine_id="e1")),
        (_anth_request("t2", [{"role": "user", "content": "again"}], "2026-01-01T00:00:02Z"),
         _sse_response("t2", _anthropic_sse(RAW_CASES["X-17"][1], "end_turn"), engine_id="e2")),
    ])  # fmt: skip
    with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
        for engine_id, case in (("e1", "X-13"), ("e2", "X-17")):
            fh.write(
                json.dumps({"disagg_request_id": engine_id, "text": RAW_CASES[case][0]}) + "\n"
            )
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                       raw=[directory])  # fmt: skip
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-100b", "ok": c13["status"] == "PASS" and c13["denominator"] == 1
                    and c13.get("joined") == 2 and c13.get("not_judged") == 1
                    and "1 of 2 joined replies" in (c13["reason"] or "")})  # fmt: skip
    # X-20: an engine id that names two raw records joins nothing (fail closed).
    directory = os.path.join(work, "x-20")
    request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
    _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(RAW_CASES["X-13"][1]),
                                                     engine_id="e1"))])  # fmt: skip
    with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
        for _ in range(2):
            fh.write(json.dumps({"disagg_request_id": "e1", "text": GLM_RAW}) + "\n")
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                       raw=[directory])  # fmt: skip
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-20", "ok": c13["status"] == "BLOCKED" and c13["denominator"] == 0})
    # X-46: on an aggregated server the raw record has no orchestrator id; the reply joins it by
    # the engine request id its Anthropic message id carries (msg_<id>_<hex>).
    directory = os.path.join(work, "x-46")
    reply = _anthropic_sse(RAW_CASES["X-13"][1]).replace(
        '"id": "m"', '"id": "msg_5_0123456789abcdef"', 1
    )
    _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                              _sse_response("t1", reply))])  # fmt: skip
    with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
        fh.write(
            json.dumps({"request_id": 5, "text": RAW_CASES["X-13"][0], "finish_reason": "stop"})
            + "\n"
        )
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                       raw=[directory])  # fmt: skip
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-46", "ok": c13["status"] == "PASS" and c13["denominator"] == 1})
    # X-19: without raw output the raw view is not applicable, never PASS.
    directory = os.path.join(work, "x-13")
    _, v = run_verdict(os.path.join(directory, "v19"), [directory], anth, "window", "conformance")
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-19", "ok": c13["status"] == "NOT_APPLICABLE"})
    return results


def shared_engine_id_cases(work, anth):
    """C-13 when one engine id names several raw records: one per generation worker.

    Each generation worker of a disaggregated server numbers its requests the same way, and a
    reply's id (``msg_<id>_<hex>``, ``chatcmpl-<id>``) carries the generation worker's engine id.
    The reply joins the record that ran inside its window (request arrival 00:00:10Z to response
    completion 00:00:20Z, plus pipeline.RAW_CLOCK_SKEW_S) with the output count it reports, or
    none. In each case another request's text is on offer: joined, it fails C-13 (RAW-03).
    """
    results = []
    user = {"role": "user", "content": "go"}
    reply = _anthropic_sse(RAW_CASES["X-13"][1]).replace(
        '"id": "m"', '"id": "msg_5_0123456789abcdef"', 1
    )  # reports 1 output token
    other = GLM_RAW.replace("Reading.", "Waiting.")

    def raw(text, first, last, tokens, disagg, request_id="5", output_index=0):
        """A generation worker's raw record as the hook writes it, run from `first` to `last` s."""
        return {"request_id": request_id, "disagg_request_id": disagg, "output_index": output_index,
                "text": text, "num_output_tokens": tokens, "finish_reason": "stop",
                "started_at": f"2026-01-01T00:{first // 60:02d}:{first % 60:02d}.250000+00:00",
                "finished_at": f"2026-01-01T00:{last // 60:02d}:{last % 60:02d}.750000+00:00"}  # fmt: skip

    def judged(case, records, request=None, response=reply, profile=anth):
        directory = os.path.join(work, case.lower())
        request = request or _anth_request("t1", [user], "2026-01-01T00:00:10Z")
        answer = dict(
            _sse_response(request["trace_id"], response), finished_at="2026-01-01T00:00:20Z"
        )
        _write_trace(directory, [(request, answer)])
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], profile, "window",
                           "conformance", raw=[directory])  # fmt: skip
        return {r["id"]: r for r in v["checks"]}, set(v["tripped"])

    def compared(checks, tripped, resolved, engine_counts):
        # A record joined by resolving already matched the reply's output count: USG-06 does not
        # compare it again (C-14 usage_vs_engine_output), as it does a record its id names alone.
        return (checks["C-13"]["status"] == "PASS" and checks["C-13"]["denominator"] == 1
                and checks["C-13"].get("resolved_ambiguous_id", 0) == resolved
                and checks["C-14"].get("usage_vs_engine_output") == engine_counts
                and not {t for t in tripped if t.startswith("RAW-")})  # fmt: skip

    def unjoined(checks, tripped):
        return (checks["C-13"]["status"] == "BLOCKED" and checks["C-13"].get("joined") == 0
                and checks["C-13"].get("unjoined_ambiguous_id") == 1
                and not {t for t in tripped if t.startswith("RAW-")})  # fmt: skip

    # X-105: the other worker's record ran minutes later: the reply joins its own, whose start
    # (00:00:09.25, on a worker clock behind the recorder's) lies within the skew, and compares.
    checks, tripped = judged(
        "X-105", [raw(other, 300, 305, 1, "9001"), raw(GLM_RAW, 9, 19, 1, "9002")]
    )
    results.append({"case": "X-105", "ok": compared(checks, tripped, 1, 0)})
    # X-105b: the only record inside the window wrote 9 tokens where the reply reports 1, and the
    # other ran outside it: neither is the reply's generation, so nothing joins.
    checks, tripped = judged(
        "X-105b", [raw(other, 12, 18, 9, "9001"), raw(GLM_RAW, 300, 305, 1, "9002")]
    )
    results.append({"case": "X-105b", "ok": unjoined(checks, tripped)})
    # X-105c: both records fit the window and the count: which one is the reply's is unknown.
    checks, tripped = judged(
        "X-105c", [raw(other, 12, 18, 1, "9001"), raw(GLM_RAW, 13, 19, 1, "9002")]
    )
    results.append({"case": "X-105c", "ok": unjoined(checks, tripped)})
    # X-105d: an engine id that names one record joins it, wherever its times fall.
    checks, tripped = judged("X-105d", [raw(GLM_RAW, 300, 305, 1, "9002")])
    results.append({"case": "X-105d", "ok": compared(checks, tripped, 0, 1)})
    # X-105e: a Chat request for two choices (1 output token each, 2 reported) has a record per
    # choice under one engine id, which neither the window nor the reply's total count tells
    # apart; another worker's record that wrote 2 tokens is not picked for them.
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Reading."}}]},
              {"choices": [{"index": 1, "delta": {"role": "assistant", "content": "Waiting."}}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
              {"choices": [{"index": 1, "delta": {}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}]  # fmt: skip
    chat_reply = "".join("data: " + json.dumps(dict(c, id="chatcmpl-5", object="chat.completion.chunk",
                                                    model="m", created=1)) + "\n\n"
                         for c in chunks) + "data: [DONE]\n\n"  # fmt: skip
    chat_request = {"event": "request", "trace_id": "c1", "route": "/v1/chat/completions",
                    "recorded_at": "2026-01-01T00:00:10Z", "headers": [["x-thread", "th"]],
                    "body": {"model": "m", "stream": True, "n": 2, "messages": [user],
                             "stream_options": {"include_usage": True}}}  # fmt: skip
    chat = write_profile(os.path.join(work, "x105e.yaml"), CONTROL_PROFILES["chat"])
    checks, tripped = judged("X-105e", [raw("</think>Reading.", 12, 18, 1, "9001"),
                                        raw("</think>Waiting.", 12, 18, 1, "9001", output_index=1),
                                        raw("</think>Writing.", 13, 19, 2, "9002")],
                             chat_request, chat_reply, chat)  # fmt: skip
    results.append({"case": "X-105e", "ok": unjoined(checks, tripped)})
    # X-106: a server that names its messages after the orchestrator's disagg_request_id: the id
    # in the reply joins the record that carries it.
    renamed = reply.replace("msg_5_", "msg_9001_", 1)
    checks, tripped = judged("X-106", [raw(GLM_RAW, 12, 18, 1, "9001", "3")], response=renamed)
    results.append({"case": "X-106", "ok": compared(checks, tripped, 0, 1)})
    # X-106b: when the id in the reply is the engine id of one record and the disagg_request_id of
    # another, the engine id joins, as before.
    checks, tripped = judged(
        "X-106b", [raw(other, 12, 18, 1, "5", "8"), raw(GLM_RAW, 12, 18, 1, None)]
    )
    results.append({"case": "X-106b", "ok": compared(checks, tripped, 0, 1)})
    return results


V41_PROFILE = os.path.join(SKILL, "profiles", "model", "deepseek-v41.yaml")


def pairing_cases(work, anth):
    """Pairing by call ids (C-03 texts, USG-08) and the raw view per thinking mode.

    The request a reply is held against carries its call ids, and a raw record is read in the
    mode its request asked for.
    """
    results = []

    def use(call_id, text):
        return {"role": "assistant", "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": call_id, "name": "Read", "input": {"file_path": "a"}}]}  # fmt: skip

    def result(call_id):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                                             "content": "ok"}]}  # fmt: skip

    def billed(sse, tokens):
        return sse.replace('"input_tokens": 1', f'"input_tokens": {tokens}', 1)

    def reply(call_id, text, tokens=1):
        return billed(_anthropic_sse([{"type": "text", "text": text},
                                      {"type": "tool_use", "id": call_id, "name": "Read",
                                       "json": '{"file_path": "a"}'}]), tokens)  # fmt: skip

    done = _anthropic_sse([{"type": "text", "text": "done"}], "end_turn")
    user = {"role": "user", "content": "go"}
    # X-62: two sub-agents of one session ask with different first messages and the same last
    # one, so both replies sit at the same position after the same message, and the second's
    # echo arrives first. Matched by call ids, both texts echo equal (matched by position, the
    # first reply was held against the second's echo: a false C-03).
    directory = os.path.join(work, "x-62")
    tail = {"role": "user", "content": "continue"}
    task_a, task_b = {"role": "user", "content": "task a"}, {"role": "user", "content": "task b"}
    _write_trace(directory, [
        (_anth_request("a1", [task_a, tail], "2026-01-01T00:00:01Z"), _sse_response("a1", reply("toolu_a", "A."))),
        (_anth_request("b1", [task_b, tail], "2026-01-01T00:00:02Z"), _sse_response("b1", reply("toolu_b", "B."))),
        (_anth_request("b2", [task_b, tail, use("toolu_b", "B."), result("toolu_b")], "2026-01-01T00:00:03Z"),
         _sse_response("b2", done)),
        (_anth_request("a2", [task_a, tail, use("toolu_a", "A."), result("toolu_a")], "2026-01-01T00:00:04Z"),
         _sse_response("a2", done)),
    ])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-62", "ok": c03["status"] == "PASS" and c03["text_echo_equal"] == 2})

    def usage_case(
        case, second_messages, second_tokens, context_management=None, side=False, second_tools=True
    ):
        directory = os.path.join(work, case.lower())
        pairs = [(_anth_request("t1", second_messages[0], "2026-01-01T00:00:01Z"),
                  _sse_response("t1", reply("toolu_1", "R.", 100)))]  # fmt: skip
        if side:
            request = _anth_request(
                "s", [{"role": "user", "content": "a title?"}], "2026-01-01T00:00:02Z"
            )
            request["body"].pop("tools")
            pairs.append((request, _sse_response("s", billed(_anthropic_sse(
                [{"type": "text", "text": "T"}], "end_turn"), 10))))  # fmt: skip
        second = _anth_request("t2", second_messages[1], "2026-01-01T00:00:03Z")
        if context_management is not None:
            second["body"]["context_management"] = context_management
        if not second_tools:
            second["body"].pop("tools")
        pairs.append((second, _sse_response("t2", billed(done, second_tokens))))
        _write_trace(directory, pairs)
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
        with open(os.path.join(directory, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        return v, counts

    echo = [user, use("toolu_1", "R."), result("toolu_1")]
    # X-63: a side request of the session (no tools, small) comes between a turn and the request
    # that sends its call back. USG-08 holds the turn against the latter, which grew.
    v, counts = usage_case("X-63", ([user], echo), 120, side=True)
    results.append({"case": "X-63", "ok": "USG-08" not in v["tripped"]
                    and counts.get("den:next_input_pairs") == 1})  # fmt: skip
    # X-64: the request that sends the call back replaced the history with a summary
    # (compaction): its prompt is smaller, which is the harness's doing, not a usage error.
    earlier = [user, use("toolu_0", "Z."), result("toolu_0"), {"role": "user", "content": "more"}]
    summary = {"role": "user", "content": "the conversation so far, summarized"}
    v, counts = usage_case(
        "X-64", (earlier, [summary, use("toolu_1", "R."), result("toolu_1")]), 40
    )
    results.append({"case": "X-64", "ok": "USG-08" not in v["tripped"]
                    and counts.get("C-14:next_input_shrank_by_compaction") == 1})  # fmt: skip
    # X-115: the request that sends the call back declares no tools (Codex asks for its compaction
    # summary that way, with the whole history): its prompt may be smaller without any usage error.
    # X-115b: the same request with the turn's tools and a smaller prompt is still USG-08.
    for case, tools_kept, trips in (("X-115", False, False), ("X-115b", True, True)):
        v, counts = usage_case(case, ([user], echo), 40, second_tools=tools_kept)
        results.append({"case": case, "ok": ("USG-08" in v["tripped"]) == trips
                        and (trips or counts.get("C-14:next_input_shrank_by_tools_change") == 1)})  # fmt: skip
    # X-65: context_management that clears thinking with keep "all" removes nothing: a smaller
    # next prompt is still USG-08. X-66: clearing tool uses may shrink it.
    for case, edit, trips in (
        ("X-65", {"type": "clear_thinking_20251015", "keep": "all"}, True),
        ("X-66", {"type": "clear_tool_uses_20250919"}, False),
    ):
        v, _ = usage_case(case, ([user], echo), 50, context_management={"edits": [edit]})
        results.append({"case": case, "ok": ("USG-08" in v["tripped"]) == trips})
    # X-67: V4.1 renders a Messages request without `thinking` in chat mode, so the raw output has
    # no </think>; read from the content it is judged by C-13 (PASS) and is no M-07. X-68: the
    # same output for a request that asked for thinking is an unclosed think (M-07).
    for case, thinking, unclosed in (("X-67", None, False), ("X-68", {"type": "adaptive"}, True)):
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        if thinking is not None:
            request["body"]["thinking"] = thinking
        answer = _anthropic_sse([{"type": "text", "text": "OK"}], "end_turn")
        _write_trace(directory, [(request, _sse_response("t1", answer, engine_id="e1"))])
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(
                json.dumps({"disagg_request_id": "e1", "text": "OK", "finish_reason": "stop"})
                + "\n"
            )
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           model=V41_PROFILE, raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        ok = ("M-07" in v["tripped"]) == unclosed and "C-13" not in v["tripped"]
        if not unclosed:
            ok = ok and c13["status"] == "PASS" and c13["denominator"] == 1
        results.append({"case": case, "ok": ok})
    return results


def applicability_cases(work, anth):
    """A check whose situation never arose tested nothing: NOT_APPLICABLE, never PASS."""
    results = []
    user = {"role": "user", "content": "go"}
    call = [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]
    # X-69: no request limits calls or sets tool_choice, and nothing ran over length.
    # X-70: one request sets a named tool_choice and at most one call, and is answered by one call.
    for case, choice, expect in (
        ("X-69", None, {"C-07": "NOT_APPLICABLE", "C-08": "NOT_APPLICABLE", "H-04": "NOT_APPLICABLE"}),
        ("X-70", {"type": "tool", "name": "Read", "disable_parallel_tool_use": True},
         {"C-07": "PASS", "C-08": "PASS", "H-04": "NOT_APPLICABLE"}),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        if choice is not None:
            request["body"]["tool_choice"] = choice
        _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(call)))])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window")
        status = {r["id"]: r["status"] for r in v["checks"]}
        results.append({"case": case, "ok": all(status.get(k) == x for k, x in expect.items())})
    return results


def rate_unit_cases(work, anth):
    """The matrix's unit for rates: sessions, or each check's own unit (turns)."""
    results = []
    directory = os.path.join(work, "x-71")
    pairs = []
    for session in ("s1", "s2"):
        for i in range(80):
            trace_id = f"{session}-{i}"
            request = _anth_request(trace_id, [{"role": "user", "content": f"step {i}"}],
                                    f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z")  # fmt: skip
            request["headers"] = [["x-claude-code-session-id", session]]
            pairs.append((request, _sse_response(trace_id, _anthropic_sse(
                [{"type": "text", "text": "Done."}], "end_turn"))))  # fmt: skip
    _write_trace(directory, pairs)
    # X-71: 160 clean turns from 2 sessions, 2% thresholds counted per session: insufficient n.
    # X-72: the same counted per turn: PASS (0 of 160, upper bound 1.7%), sessions still reported.
    # X-73: a unit the validator does not know is not guessed: BLOCKED.
    for case, unit, expect in (("X-71", "session", "BLOCKED"), ("X-72", "turn", "PASS"),
                               ("X-73", "turns", "BLOCKED")):  # fmt: skip
        matrix = os.path.join(work, case.lower() + "-matrix.yaml")
        with open(matrix, "w") as fh:
            json.dump({"thresholds": {"model_behavior": {"markup_leak_rate": 0.02,
                                                         "empty_turn_rate": 0.02},
                                      "unit_for_rates": unit}}, fh)  # fmt: skip
        _, v = run_verdict(os.path.join(directory, case), [directory], anth, "window",
                           "model-behavior", matrix=matrix)  # fmt: skip
        rates = [r for r in v["checks"] if r["id"] in ("M-03", "M-06")]
        ok = len(rates) == 2 and all(r["status"] == expect for r in rates)
        if case == "X-71":
            ok = ok and all(r["denominator"] == 2 for r in rates)
        if case == "X-72":
            ok = ok and all(r["denominator"] == 160 and r["sessions"] == 2 for r in rates)
        if case == "X-73":
            ok = ok and all("unknown unit_for_rates" in (r["reason"] or "") for r in rates)
        results.append({"case": case, "ok": ok})
    return results


def cut_bound_cases(work):
    """R-01/R-02 against a client cut known only from below (`{at_least: s}`) or exactly."""
    results = []
    user = {"role": "user", "content": "go"}

    def profile(name, first, idle):
        data = json.loads(json.dumps(CONTROL_PROFILES["anthropic"]))
        timeouts = dict(data.get("timeouts") or {})
        timeouts.update(first_event_s=first, idle_s=idle)
        data["timeouts"] = timeouts
        return write_profile(os.path.join(work, name), data)

    bound = profile("x-bound.yaml", {"at_least": 10}, {"at_least": 30})
    exact = profile("x-exact.yaml", 10, 30)
    # X-74: first event at 0.5 s, longest silence 5 s, under bounds of 10 s and 30 s: PASS.
    # X-75: a 34.5 s silence, above the known 30 s: whether the client cuts it is unknown, BLOCKED.
    # X-76: the same silence against an exact 30 s cut: FAIL.
    # X-77: a first event at 12 s against a known 10 s: BLOCKED.
    for case, chunks, first, harness, expect in (
        ("X-74", [0.5, 1.0, 6.0], 0.5, bound, {"R-01": "PASS", "R-02": "PASS"}),
        ("X-75", [0.5, 6.0, 40.5], 0.5, bound, {"R-01": "PASS", "R-02": "BLOCKED"}),
        ("X-76", [0.5, 6.0, 40.5], 0.5, exact, {"R-01": "PASS", "R-02": "FAIL"}),
        ("X-77", [12.0, 12.5], 12.0, bound, {"R-01": "BLOCKED", "R-02": "PASS"}),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        response = _sse_response(
            "t1", _anthropic_sse([{"type": "text", "text": "Done."}], "end_turn")
        )
        response["timing"] = {"headers_s": 0.01, "first_byte_s": first,
                              "chunks": [[t, 1] for t in chunks]}  # fmt: skip
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), response)])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], harness, "window", "runtime")
        status = {r["id"]: r["status"] for r in v["checks"]}
        results.append({"case": case, "ok": all(status.get(k) == x for k, x in expect.items())})
    return results


def outcome_and_signal_cases(work, anth):
    """Outcomes, applicability, signatures and M-02 on raw text.

    Rated outcomes the profile cannot see, checks with nothing to compare, signatures by value,
    and M-02 on the raw text the model wrote.
    """
    results = []
    user = {"role": "user", "content": "go"}
    call = [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]
    rejected = ("The user doesn't want to proceed with this tool use. The tool use was rejected (eg. "
                "if it was a file edit, the new_string was NOT written to the file).")  # fmt: skip
    matrix = os.path.join(work, "x-78-matrix.yaml")
    with open(matrix, "w") as fh:
        json.dump({"thresholds": {"harness_outcome": {"aborted_rate": 0.02,
                                                      "arguments_rejected_rate": 0.02},
                                  "model_behavior": {"degenerate_rate": 0.02},
                                  "unit_for_rates": "turn"}}, fh)  # fmt: skip

    def answered(name, result, is_error=True):
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": result, "is_error": is_error}]}]  # fmt: skip
        directory = os.path.join(work, name)
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(call))),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn"))),
        ])  # fmt: skip
        return directory

    def profile(name, outcomes):
        data = json.loads(json.dumps(CONTROL_PROFILES["anthropic"]))
        data["outcomes"] = outcomes
        return write_profile(os.path.join(work, name), data)

    # X-78: Claude Code's rejection of a call is an abort, a rated outcome, not a tool error.
    d = answered("x-78", rejected)
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "harness-outcome", matrix=matrix)
    h = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-78", "ok": h["H-01:aborted"]["count"] == 1
                    and h["H-01:aborted"]["status"] != "PASS"})  # fmt: skip
    # X-79: a profile that does not name the rated `aborted` outcome cannot count one: BLOCKED.
    blind = profile("x-79.yaml", {"unsupported": {"contains": "No such tool available"},
                                  "arguments_rejected": {"contains": "InputValidationError"},
                                  "error": {"is_error": True}})  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v79"), [d], blind, "window", "harness-outcome",
                       matrix=matrix)  # fmt: skip
    h = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-79", "ok": h["H-01:aborted"]["status"] == "BLOCKED"
                    and "declares no `aborted`" in (h["H-01:aborted"]["reason"] or "")
                    and v["claims"]["harness-outcome"] == "BLOCKED"})  # fmt: skip
    # X-80: declared absent with a reason, it is NOT_APPLICABLE, and the claim can pass.
    absent = profile("x-80.yaml", {"aborted": {"none": "this client never cancels a call"},
                                   "arguments_rejected": {"contains": "InputValidationError"},
                                   "error": {"is_error": True}})  # fmt: skip
    d80 = answered("x-80", "done", is_error=False)
    _, v = run_verdict(os.path.join(d80, "v"), [d80], absent, "window", "harness-outcome",
                       matrix=matrix)  # fmt: skip
    h = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-80", "ok": h["H-01:aborted"]["status"] == "NOT_APPLICABLE"
                    and h["H-01:aborted"]["reason"] == "this client never cancels a call"})  # fmt: skip
    # X-104: an outcome flag whose outcome the measured profile does not declare (Codex's
    # `unavailable_in_mode`) cannot fire: NOT_APPLICABLE with that reason, while a declared one
    # (`unsupported`) passes. X-104b: under a profile that is not measured, BLOCKED.
    unmeasured = write_profile(os.path.join(work, "x-104b.yaml"),
                               dict(CONTROL_PROFILES["anthropic"], status="partial"))  # fmt: skip
    for case, harness, expect in (
        ("X-104", anth, "NOT_APPLICABLE"),
        ("X-104b", unmeasured, "BLOCKED"),
    ):
        _, v = run_verdict(os.path.join(d80, case.lower()), [d80], harness, "window", "harness-outcome",
                           matrix=matrix)  # fmt: skip
        h = {r["id"]: r for r in v["checks"]}
        flag = h["H-01:unavailable_in_mode"]
        results.append({"case": case, "ok": flag["status"] == expect
                        and "declares no `unavailable_in_mode`" in (flag["reason"] or "")
                        and h["H-01:unsupported"]["status"] == "PASS"})  # fmt: skip
    # X-81: Anthropic streams carry no final snapshot, so C-02 compared nothing.
    _, v = run_verdict(os.path.join(d80, "v81"), [d80], anth, "window", "conformance")
    c02 = next(r for r in v["checks"] if r["id"] == "C-02")
    results.append({"case": "X-81", "ok": c02["status"] == "NOT_APPLICABLE"})
    # X-82: a thinking signature echoed with another value is not an exact echo; the same value is.
    for case, echoed, expect in (("X-82", "c2lnQg==", "FAIL"), ("X-82b", "c2lnQQ==", "PASS")):
        delivered = [{"type": "thinking", "text": "Plan.", "signature": "c2lnQQ=="}] + call
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Plan.", "signature": echoed},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": "ok"}]}]  # fmt: skip
        directory = os.path.join(work, case.lower())
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(delivered))),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window",
                           "conformance")  # fmt: skip
        results.append({"case": case, "ok": v["claims"]["conformance"] == expect})
    # X-83: with raw output, M-02 reads what the model wrote: a run of pad tokens the server kept
    # out of the delivered text is found, in a reply whose raw text kept special tokens.
    # X-84: a raw text without special tokens (the server dropped them) cannot show the run: the
    # reply is not counted, and with no reply left M-02 is BLOCKED.
    pad, eos = "<\uff5c\u2581pad\u2581\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>"
    for case, text, count, denominator, status in (
        ("X-83", "</think>Done." + pad * 25 + eos, 1, 1, None),
        ("X-84", "</think>Done.", 0, 0, "BLOCKED"),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        request["body"]["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(
            [{"type": "thinking", "text": ""}, {"type": "text", "text": "Done."}], "end_turn"),
            engine_id="e1"))])  # fmt: skip
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(json.dumps({"disagg_request_id": "e1", "text": text}) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window",
                           "model-behavior", model=V41_PROFILE, matrix=matrix,
                           raw=[directory])  # fmt: skip
        m02 = next(r for r in v["checks"] if r["id"] == "M-02")
        ok = m02["count"] == count and m02["denominator"] == denominator
        if status:
            ok = ok and m02["status"] == status and m02.get("raw_unobservable") == 1
        results.append({"case": case, "ok": ok})
    return results


def responses_reasoning_echo_cases(work):
    """C-03 on Responses reasoning items: echoed by id, the carrier compared by value.

    The Codex CLI asks for ``encrypted_content`` and sends each reasoning item back with its id;
    what it sends must be the carrier we delivered.
    """
    results = []
    carrier = "trtllm-reasoning-v1:UGxhbi4="
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [], "status": "completed",
                 "content": [{"type": "reasoning_text", "text": "Plan."}],
                 "encrypted_content": carrier}  # fmt: skip
    call = {"id": "fc_1", "type": "function_call", "name": "wait", "call_id": "c1",
            "arguments": '{"ms": 1}', "status": "completed"}  # fmt: skip
    tools = [{"type": "function", "name": "wait", "parameters": {"type": "object"}}]

    def sse(items):
        frames = [("response.created", {"response": {"id": "r", "model": "x", "status": "in_progress",
                                                     "output": []}})]  # fmt: skip
        for index, item in enumerate(items):
            frames += [("response.output_item.added", {"output_index": index, "item": item}),
                       ("response.output_item.done", {"output_index": index, "item": item})]  # fmt: skip
        frames.append(("response.completed", {"response": {
            "id": "r", "model": "x", "status": "completed", "output": items,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}}))  # fmt: skip
        return "".join(f"event: {n}\ndata: {json.dumps(dict(d, type=n, sequence_number=i))}\n\n"
                       for i, (n, d) in enumerate(frames))  # fmt: skip

    def request(trace_id, items, at):
        return {"event": "request", "trace_id": trace_id, "route": "/v1/responses", "recorded_at": at,
                "body": {"model": "x", "stream": True, "tools": tools, "store": False,
                         "include": ["reasoning.encrypted_content"], "prompt_cache_key": "s1",
                         "client_metadata": {"thread_id": "th"}, "input": items}}  # fmt: skip

    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}
    output = {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    echoed = {k: v for k, v in reasoning.items() if k != "status"}
    exact = write_profile(os.path.join(work, "x85.yaml"), dict(
        CONTROL_PROFILES["responses"], echo={"text": "exact", "reasoning": "exact"}))  # fmt: skip
    todo = write_profile(os.path.join(work, "x88.yaml"), dict(
        CONTROL_PROFILES["responses"], echo={"text": "exact", "reasoning": "TODO"}))  # fmt: skip
    # X-85: the carrier comes back as delivered. X-86: another value. X-87: without a carrier.
    # X-88: the profile does not say how reasoning is echoed: not compared, so BLOCKED.
    # X-89: the next request leaves the reasoning item out: counted as missing, not a failure.
    cases = (
        ("X-85", [echoed], exact, "PASS", "reasoning_echo_equal"),
        ("X-86", [dict(echoed, encrypted_content=carrier[:-4] + "QQ==")], exact, "FAIL", None),
        ("X-87", [{k: v for k, v in echoed.items() if k != "encrypted_content"}], exact, "FAIL", None),
        ("X-88", [echoed], todo, "BLOCKED", "reasoning_echo_unchecked"),
        ("X-89", [], exact, "PASS", "reasoning_echo_missing"),
    )  # fmt: skip
    for case, replayed, profile, status, counter in cases:
        directory = os.path.join(work, case.lower())
        second = [user] + replayed + [{k: v for k, v in call.items() if k not in ("id", "status")},
                                      output]  # fmt: skip
        _write_trace(directory, [
            (request("r1", [user], "2026-01-01T00:00:01Z"), _sse_response("r1", sse([reasoning, call]))),
            (request("r2", second, "2026-01-01T00:00:02Z"), _sse_response("r2", sse([
                {"id": "msg_2", "type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "done", "annotations": []}]}]))),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], profile, "window",
                           "conformance")  # fmt: skip
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        ok = c03["status"] == status
        if counter:
            ok = ok and c03.get(counter) == 1
        if status == "FAIL":
            with open(os.path.join(directory, "v", "report.json")) as fh:
                fields = {e.get("field") for e in json.load(fh)["examples"].get("C-03", [])}
            ok = ok and "reasoning" in fields
        results.append({"case": case, "ok": ok})
    # X-90: trtllm-serve names a Responses reply resp_<engine id>_<hex>; the number joins the raw
    # record with that request_id, and C-13 reads it. X-90b: the same reply under a random
    # resp_<hex> id, which names no request: the record is there, nothing joins, C-13 is BLOCKED.
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    for case, name, status in (
        ("X-90", "resp_7_00ff00ff00ff00ff", "PASS"),
        ("X-90b", "resp_0123456789abcdef0123456789abcdef", "BLOCKED"),
    ):
        directory = os.path.join(work, case.lower())
        items = [{"id": "rs_9", "type": "reasoning", "summary": [], "status": "completed",
                  "content": [{"type": "reasoning_text", "text": "Plan."}]},
                 {"id": "msg_9", "type": "message", "role": "assistant", "status": "completed",
                  "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]  # fmt: skip
        reply = _sse_response("r1", sse(items).replace('"id": "r"', f'"id": "{name}"'))
        req = request("r1", [user], "2026-01-01T00:00:01Z")
        req["body"]["reasoning"] = {"summary": "auto"}
        _write_trace(directory, [(req, reply)])
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(json.dumps({"request_id": "7", "text": "Plan.</think>Done." + eos}) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], exact, "window",
                           "conformance", model=V41_PROFILE, raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        results.append({"case": case, "ok": c13["status"] == status
                        and (status != "PASS" or c13["denominator"] == 1)})  # fmt: skip
    # X-90c: a disaggregated server names a Responses reply resp_<disagg_request_id>_<hex>; the
    # number is read as a disagg_request_id before it is read as an engine id, so the name joins
    # the record that carries that disagg_request_id, not the one whose engine id is the same
    # number. X-90d: a random resp_<hex> name carries no id, and nothing joins.
    for case, name, status in (
        ("X-90c", "resp_9001_00ff00ff00ff00ff", "PASS"),
        ("X-90d", "resp_0123456789abcdef0123456789abcdef", "BLOCKED"),
    ):
        directory = os.path.join(work, case.lower())
        items = [{"id": "rs_9", "type": "reasoning", "summary": [], "status": "completed",
                  "content": [{"type": "reasoning_text", "text": "Plan."}]},
                 {"id": "msg_9", "type": "message", "role": "assistant", "status": "completed",
                  "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]  # fmt: skip
        reply = _sse_response("r1", sse(items).replace('"id": "r"', f'"id": "{name}"'))
        req = request("r1", [user], "2026-01-01T00:00:01Z")
        req["body"]["reasoning"] = {"summary": "auto"}
        _write_trace(directory, [(req, reply)])
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(json.dumps({"request_id": "5", "disagg_request_id": "9001",
                                 "text": "Plan.</think>Done." + eos}) + "\n")  # fmt: skip
            fh.write(json.dumps({"request_id": "9001", "text": "Other.</think>Text." + eos})
                     + "\n")  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], exact, "window",
                           "conformance", model=V41_PROFILE, raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        results.append({"case": case, "ok": c13["status"] == status
                        and (status != "PASS" or c13["denominator"] == 1)})  # fmt: skip
    return results


def full_tier_fix_cases(work):
    """Responses snapshot, sequence and keepalive rules, claim blockers and tolerated payloads.

    X-91: an argument the tool does not declare, under a harness that tolerates it, is counted
    (``tolerated``), not rated in M-01; X-91b: the same without the tolerance is payload_invalid;
    X-91c: tolerated only when every violation is. X-92/X-93: a snapshot item's status or
    encrypted_content that differs from the stream's (RSP-11). X-94/X-94b: sequence numbers with
    a gap, or not from 0 (RSP-04). X-95: a keepalive that differs from the first
    response.in_progress (RSP-17); X-95b: an equal one passes. X-96: a claim BLOCKED by a check
    names it in claim_blockers. X-97: a delivered text with no next turn is counted in C-03.
    """
    results = []
    schema = {"type": "object", "properties": {"cmd": {"type": "string"}},
              "required": ["cmd"], "additionalProperties": False}  # fmt: skip
    tools = [{"type": "function", "name": "exec_command", "parameters": schema}]

    def frames_sse(frames):
        return "".join(f"event: {n}\ndata: {json.dumps(dict(d, type=n))}\n\n" for n, d in frames)

    def numbered(frames, start=0, gap_at=None):
        out, seq = [], start
        for i, (n, d) in enumerate(frames):
            if gap_at is not None and i == gap_at:
                seq += 1
            out.append((n, dict(d, sequence_number=seq)))
            seq += 1
        return out

    def stream(items, snapshot=None, extra_in_progress=None):
        opened = {"response": {"id": "r", "model": "x", "status": "in_progress", "output": []}}
        frames = [("response.created", opened), ("response.in_progress", opened)]
        if extra_in_progress is not None:
            frames.append(("response.in_progress", extra_in_progress))
        for index, item in enumerate(items):
            frames += [("response.output_item.added", {"output_index": index, "item": item}),
                       ("response.output_item.done", {"output_index": index, "item": item})]  # fmt: skip
        frames.append(("response.completed", {"response": {
            "id": "r", "model": "x", "status": "completed",
            "output": snapshot if snapshot is not None else items,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}}))  # fmt: skip
        return frames

    def request(trace_id, at, input_items=None):
        return {"event": "request", "trace_id": trace_id, "route": "/v1/responses", "recorded_at": at,
                "body": {"model": "x", "stream": True, "tools": tools, "prompt_cache_key": "s1",
                         "client_metadata": {"thread_id": "th"},
                         "input": input_items or [{"type": "message", "role": "user",
                                                   "content": [{"type": "input_text", "text": "go"}]}]}}  # fmt: skip

    tolerant = write_profile(os.path.join(work, "x91.yaml"), dict(
        CONTROL_PROFILES["responses"], tools={"tolerated_payload": ["additionalProperties"]}))  # fmt: skip
    strict = write_profile(os.path.join(work, "x91b.yaml"), CONTROL_PROFILES["responses"])
    call = {"id": "fc_1", "type": "function_call", "name": "exec_command", "call_id": "c1",
            "status": "completed"}  # fmt: skip
    for case, arguments, profile, invalid, tolerated in (
        ("X-91", '{"cmd": "ls", "timeout": 5}', tolerant, 0, 1),
        ("X-91b", '{"cmd": "ls", "timeout": 5}', strict, 1, 0),
        ("X-91c", '{"cmd": 5, "timeout": 5}', tolerant, 1, 0),
    ):
        directory = os.path.join(work, case.lower())
        body = frames_sse(numbered(stream([dict(call, arguments=arguments)])))
        _write_trace(
            directory, [(request("r1", "2026-01-01T00:00:01Z"), _sse_response("r1", body))]
        )
        _, v = run_verdict(
            os.path.join(directory, "v"), [directory], profile, "window", "model-behavior"
        )
        m01 = next(r for r in v["checks"] if r["id"] == "M-01:payload_invalid")
        ok = m01["count"] == invalid and m01.get("tolerated", 0) == tolerated
        if case == "X-91b":
            # X-96 rides on this case: the BLOCKED rate check is named in claim_blockers.
            blockers = v["claim_blockers"].get("model-behavior") or []
            results.append({"case": "X-96", "ok": v["claims"]["model-behavior"] == "BLOCKED"
                            and any(b.startswith("M-01:payload_invalid") for b in blockers)})  # fmt: skip
        results.append({"case": case, "ok": ok})
    message = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}  # fmt: skip
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [], "status": "completed",
                 "content": [{"type": "reasoning_text", "text": "Plan."}],
                 "encrypted_content": "trtllm-reasoning-v1:UGxhbi4="}  # fmt: skip
    opened = {"response": {"id": "r", "model": "x", "status": "in_progress", "output": []}}
    variants = {
        # case: (frames, expected rule or None)
        "X-92": (numbered(stream([message], snapshot=[dict(message, status="incomplete")])), "RSP-11"),
        "X-93": (numbered(stream([reasoning, message], snapshot=[
            dict(reasoning, encrypted_content="trtllm-reasoning-v1:UGxhbiE="), message])), "RSP-11"),
        "X-94": (numbered(stream([message]), gap_at=3), "RSP-04"),
        "X-94b": (numbered(stream([message]), start=1), "RSP-04"),
        "X-95": (numbered(stream([message], extra_in_progress={"response": dict(
            opened["response"], status="queued")})), "RSP-17"),
        "X-95b": (numbered(stream([message], extra_in_progress=opened)), None),
    }  # fmt: skip
    for case, (frames, rule) in variants.items():
        directory = os.path.join(work, case.lower())
        _write_trace(
            directory,
            [(request("r1", "2026-01-01T00:00:01Z"), _sse_response("r1", frames_sse(frames)))],
        )
        _, v = run_verdict(
            os.path.join(directory, "v"), [directory], strict, "window", "conformance"
        )
        tripped = set(v["tripped"])
        rules = {"RSP-04", "RSP-11", "RSP-17"}
        results.append({"case": case, "ok": (rule in tripped) if rule else not (tripped & rules)})
    # X-98: a raw record with token ids: a run of a declared id is found although the decoded
    # text kept no special token; X-98b: ids without a run count as observed and clean.
    user = {"role": "user", "content": "go"}
    for case, ids, count in (("X-98", [100] + [2] * 25 + [1], 1), ("X-98b", [100, 2, 101, 1], 0)):
        directory = os.path.join(work, case.lower())
        req = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        req["body"]["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        _write_trace(directory, [(req, _sse_response("t1", _anthropic_sse(
            [{"type": "thinking", "text": ""}, {"type": "text", "text": "Done."}], "end_turn"),
            engine_id="e1"))])  # fmt: skip
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(
                json.dumps({"disagg_request_id": "e1", "text": "</think>Done.", "token_ids": ids})
                + "\n"
            )
        anth = write_profile(os.path.join(work, "x98.yaml"), CONTROL_PROFILES["anthropic"])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "model-behavior",
                           model=V41_PROFILE, raw=[directory])  # fmt: skip
        m02 = next(r for r in v["checks"] if r["id"] == "M-02")
        results.append({"case": case, "ok": m02["count"] == count and m02["denominator"] == 1
                        and not m02.get("raw_unobservable")})  # fmt: skip
    # X-97: one reply, no later request: its text is counted as having no next turn.
    directory = os.path.join(work, "x-97")
    _write_trace(directory, [(request("r1", "2026-01-01T00:00:01Z"),
                              _sse_response("r1", frames_sse(numbered(stream([message])))))])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], strict, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-97", "ok": c03.get("text_no_next_turn") == 1})
    return results


K3_PAD = 163839  # kimi-k3.yaml: [PAD], the last id of its 163840-token vocabulary
K3_THINK = "<|close|>think<|sep|><|open|>response<|sep|>"  # the reasoning ends, the response opens


def _named_reply(sse):
    """A Messages stream named after engine request 5 (msg_5_<hex>), which joins raw record 5."""
    return sse.replace('"id": "m"', '"id": "msg_5_0123456789abcdef"', 1)


def _failed_stream(blocks):
    """A Messages stream the server failed after it began: blocks closed, then an error event.

    The way trtllm-serve ends a stream whose generation degenerated (no message_delta, no
    message_stop): the client drops what it received.
    """
    error = {"type": "error", "error": {"type": "api_error", "message": "Generation failed: the "
                                        "output degenerated into a run of token id 163839"}}  # fmt: skip
    head = _anthropic_sse(blocks, "end_turn").split("event: message_delta")[0]
    return _named_reply(head + "event: error\ndata: " + json.dumps(error) + "\n\n")


def degenerate_stop_cases(work, anth):
    """R-04 and M-02 on Kimi-K3's [PAD] run, in the delivered text or in the raw token ids.

    R-04 fails a run that reached the client: in what was delivered, or in the generation of a
    reply delivered as an answer (decoding drops the run's special tokens, so the client keeps a
    truncated reply). A run only the raw record shows, in a reply the server ended with an error
    and no answer, was stopped as it should be: the model's (M-02), not the runtime's.
    """
    results = []
    user = {"role": "user", "content": "go"}
    matrix = os.path.join(work, "x-09-matrix.yaml")
    with open(matrix, "w") as fh:
        json.dump({"thresholds": {"model_behavior": {"degenerate_rate": 0.9},
                                  "unit_for_rates": "turn"}}, fh)  # fmt: skip
    # Every reply streams within the profile's cuts and names its build, so the runtime claim
    # rests on its checks alone.
    timing = {"headers_s": 0.1, "first_byte_s": 0.2, "chunks": [[0.2, 100], [0.4, 100]],
              "total_s": 0.5}  # fmt: skip

    def judged(case, body, raw_text=None, ids=None, **ended):
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        request["body"]["thinking"] = {"type": "adaptive"}
        request["build_id"] = "b1"
        if ids is not None:  # the reply reports the engine's output count
            body = body.replace('"usage": {"output_tokens": 1}',
                                f'"usage": {{"output_tokens": {len(ids)}}}')  # fmt: skip
        _write_trace(directory, [(request, dict(_sse_response("t1", body), timing=timing,
                                                build_id="b1"))])  # fmt: skip
        raw = []
        if raw_text is not None:
            record = {"event": "raw_output", "request_id": "5", "output_index": 0,
                      "streaming": True, "text": raw_text, "token_ids": ids,
                      "num_output_tokens": len(ids), **ended}  # fmt: skip
            with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
                fh.write(json.dumps(record) + "\n")
            raw = [directory]
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "smoke",
                           "runtime,model-behavior,conformance", model=K3_PROFILE, matrix=matrix,
                           raw=raw)  # fmt: skip
        return v, {r["id"]: r for r in v["checks"]}

    def stopped(v, checks):
        """R-04 passes and counts the stop; M-02 counts the run; nothing else fails."""
        return (checks["R-04"]["status"] == "PASS" and checks["R-04"]["count"] == 0
                and checks["R-04"].get("stopped") == 1 and checks["M-02"]["count"] == 1
                and checks["M-02"]["denominator"] == 1 and v["claims"]["runtime"] == "PASS"
                and v["claims"]["conformance"] == "PASS")  # fmt: skip

    def reached(v, checks):
        """R-04 fails, M-02 counts the run."""
        return (checks["R-04"]["status"] == "FAIL" and checks["R-04"]["count"] == 1
                and checks["R-04"].get("stopped", 0) == 0 and checks["M-02"]["count"] == 1
                and v["claims"]["runtime"] == "FAIL")  # fmt: skip

    run = [163587, 1000, 163589] + [K3_PAD] * 25  # think close, response open, then the run
    # X-09: the server stopped a [PAD] run and failed the reply before any output: the stream
    # opens, then ends with an error event. The hook wrote what the generation had when the
    # request ended (partial, aborted, end_reason): a well-formed prefix, and the run in its ids.
    v, checks = judged("X-09", _failed_stream([]), K3_THINK + "<|close|>response<|sep|>", run,
                       aborted=True, partial=True, end_reason="error")  # fmt: skip
    results.append({"case": "X-09", "ok": stopped(v, checks)
                    and checks["C-13"].get("stopped_raw") == 1})  # fmt: skip
    # X-09b: the same stop after the stream delivered some text, which the client drops with the
    # failed reply: the run itself never reached it.
    reading = [{"type": "text", "text": "Reading."}]
    v, checks = judged("X-09b", _failed_stream(reading),
                       K3_THINK + "Reading.<|close|>response<|sep|>", run + [K3_PAD] * 5,
                       aborted=True, partial=True, end_reason="error")  # fmt: skip
    results.append({"case": "X-09b", "ok": stopped(v, checks)})
    # X-107: the run reached the client in the delivered text, as the server kept the tokens.
    v, checks = judged("X-107", _named_reply(_anthropic_sse([{"type": "text", "text": "[PAD]" * 25}],
                                                            "end_turn")))  # fmt: skip
    results.append({"case": "X-107", "ok": reached(v, checks)})
    # X-107b: the server let the run go on to the token budget and delivered what decoding left
    # ("Reading.") as an answer cut at max_tokens: the run is only in the raw ids, and the reply
    # reached the client all the same.
    v, checks = judged("X-107b", _named_reply(_anthropic_sse(reading, "max_tokens")),
                       K3_THINK + "Reading.<|close|>response<|sep|>", run)  # fmt: skip
    results.append({"case": "X-107b", "ok": reached(v, checks)})
    # X-107c: the same run in a generation that then ended, delivered as a finished answer.
    v, checks = judged("X-107c", _named_reply(_anthropic_sse(reading, "end_turn")),
                       K3_THINK + "Reading.<|close|>response<|sep|><|close|>message<|sep|>", run)  # fmt: skip
    results.append({"case": "X-107c", "ok": reached(v, checks)})
    return results


def stopped_raw_cases(work, anth):
    """C-13 against the raw record of a generation the server stopped (partial or aborted).

    Such a record holds what the model wrote until the stop, so what was delivered must be a
    prefix of it (RAW-05), whatever the reply's terminal says; a finished record is compared
    whole (RAW-01 to RAW-04).
    """
    results = []
    user = {"role": "user", "content": "go"}

    def judged(case, delivered_text, **ended):
        directory = os.path.join(work, case.lower())
        blocks = [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": delivered_text}]
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                                  _sse_response("t1", _anthropic_sse(blocks, "end_turn"),
                                                engine_id="e1"))])  # fmt: skip
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(json.dumps({"disagg_request_id": "e1", "text": "Plan.</think>Reading.",
                                 **ended}) + "\n")  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           raw=[directory])  # fmt: skip
        with open(os.path.join(directory, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        return c13, {t for t in v["tripped"] if t.startswith("RAW-")}, counts

    # X-108: a reply that completed with a prefix of what an aborted generation wrote (a final
    # record of a request aborted in the worker).
    c13, rules, counts = judged("X-108", "Read", aborted=True)
    results.append({"case": "X-108", "ok": c13["status"] == "PASS" and c13["denominator"] == 1
                    and c13.get("stopped_raw") == 1 and not rules
                    and counts.get("C-13:stopped_raw:aborted") == 1})  # fmt: skip
    # X-108b: the same reply against a finished generation lost what the model wrote.
    c13, rules, _ = judged("X-108b", "Read", aborted=False)
    results.append({"case": "X-108b", "ok": c13["status"] == "FAIL" and rules == {"RAW-03"}
                    and c13.get("stopped_raw", 0) == 0})  # fmt: skip
    # X-108c: a reply that delivered more than the partial record of a request that ended in an
    # error holds (the hook's shape: partial, aborted, end_reason): not a prefix of what was
    # written.
    c13, rules, counts = judged("X-108c", "Reading. More.", partial=True, aborted=True,
                                end_reason="error")  # fmt: skip
    results.append({"case": "X-108c", "ok": c13["status"] == "FAIL" and rules == {"RAW-05"}
                    and c13.get("stopped_raw") == 1
                    and counts.get("C-13:stopped_raw:partial") == 1})  # fmt: skip
    return results


def revision_cases(work, anth):
    """The revision a verdict checks the records against (I-05), and the evidence it read (V-02).

    A reply's build_id names the build that served it. With no revision declared (--build-id
    not given, or empty), nothing was checked against it: inventory is BLOCKED and says so. The
    verdict records the declared id and the SHA-256 of every shard it read.
    """
    results = []
    user = {"role": "user", "content": "go"}
    done = _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")

    def trace(directory, build_id):
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        answer = _sse_response("t1", done)
        if build_id:
            request["build_id"] = answer["build_id"] = build_id
        _write_trace(directory, [(request, answer)])
        return directory

    stamped = trace(os.path.join(work, "x-109"), "b1")
    unstamped = trace(os.path.join(work, "x-109e"), None)
    for case, directory, declared, inventory, extra in (
        ("X-109", stamped, None, "BLOCKED", "--build-id not given"),
        ("X-109b", stamped, "", "BLOCKED", "--build-id empty"),
        ("X-109c", stamped, "b1", "PASS", None),
        ("X-109d", stamped, "b2", "BLOCKED", "not matching --build-id"),
        ("X-109e", unstamped, None, "PASS", None),
    ):
        _, v = run_verdict(os.path.join(directory, case.lower()), [directory], anth, "window",
                           "inventory,runtime", build_id=declared)  # fmt: skip
        revision = v.get("revision") or {}
        blockers = v["claim_blockers"]["inventory"]
        ok = (v["claims"]["inventory"] == inventory and revision.get("declared_build_id") == declared
              and revision.get("checked") == bool(declared)
              and (extra is None or any(extra in b for b in blockers)))  # fmt: skip
        if directory == stamped:
            ok = ok and revision.get("build_ids_seen") == {"b1": 1} \
                and revision.get("replies_with_other_build_id") == (1 if declared == "b2" else 0)  # fmt: skip
        else:  # no build_id anywhere: inventory as before, the runtime claim BLOCKED by R-05
            ok = ok and revision.get("replies_without_build_id") == 1 \
                and v["claims"]["runtime"] == "BLOCKED"  # fmt: skip
        results.append({"case": case, "ok": ok})

    def sha256_of(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    # X-110: the verdict binds the bytes it read: each shard with its role and SHA-256, a digest
    # over them that does not depend on where they lie, and the snapshot's MANIFEST.sha256.
    snapshot = os.path.join(work, "x-110", "snap")
    trace(os.path.join(snapshot, "trace"), "b1")
    os.makedirs(os.path.join(snapshot, "raw"), exist_ok=True)
    with open(os.path.join(snapshot, "raw", "raw-x.jsonl"), "w") as fh:
        fh.write(json.dumps({"request_id": "7", "text": "Plan.</think>ok"}) + "\n")
    shards = {os.path.join(snapshot, "trace", "requests-x.jsonl"): "population",
              os.path.join(snapshot, "trace", "responses-x.jsonl"): "population",
              os.path.join(snapshot, "raw", "raw-x.jsonl"): "raw"}  # fmt: skip
    manifest = os.path.join(snapshot, "MANIFEST.sha256")
    with open(manifest, "w") as fh:
        fh.writelines(f"{sha256_of(p)}  {os.path.relpath(p, snapshot)}\n" for p in shards)
    expected = hashlib.sha256("".join(sorted(f"{role} {sha256_of(p)}\n"
                                             for p, role in shards.items())).encode()).hexdigest()  # fmt: skip

    def bound(root, out):
        _, v = run_verdict(os.path.join(work, "x-110", out), [os.path.join(root, "trace")], anth,
                           "window", "inventory", raw=[os.path.join(root, "raw")], build_id="b1")  # fmt: skip
        return v["population"]

    first = bound(snapshot, "v1")
    moved = os.path.join(work, "x-110", "moved")
    shutil.copytree(snapshot, moved)
    same = bound(moved, "v2")  # the same bytes elsewhere
    with open(os.path.join(moved, "trace", "responses-x.jsonl"), "a") as fh:
        fh.write("\n")  # one byte more
    changed = bound(moved, "v3")
    listed = {(s["path"], s["role"], s["sha256"]) for s in first.get("evidence_shards") or ()}
    results.append({"case": "X-110", "ok": first.get("evidence_sha256") == expected
                    and listed == {(p, role, sha256_of(p)) for p, role in shards.items()}
                    and first.get("manifests") == [{"path": manifest, "sha256": sha256_of(manifest)}]
                    and same.get("evidence_sha256") == expected
                    and changed.get("evidence_sha256") not in (None, expected)})  # fmt: skip
    return results


PI_PROFILES = {name: os.path.join(SKILL, "profiles", "harness", f"{name}.yaml")
               for name in ("pi-chat", "pi-messages", "pi-responses")}  # fmt: skip


def _chat_sse(calls=(), text="Working.", finish=None, tool="read"):
    """A Chat Completions stream: text, then one call per (call id, arguments JSON) to `tool`."""
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}]
    for index, (call_id, arguments) in enumerate(calls):
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": index, "id": call_id, "type": "function",
            "function": {"name": tool, "arguments": arguments}}]}}]})  # fmt: skip
    chunks.append({"choices": [{"index": 0, "delta": {},
                                "finish_reason": finish or ("tool_calls" if calls else "stop")}]})  # fmt: skip
    return "".join("data: " + json.dumps(dict(c, id="chatcmpl-1", object="chat.completion.chunk",
                                              model="m", created=1)) + "\n\n"
                   for c in chunks) + "data: [DONE]\n\n"  # fmt: skip


def _chat_request(trace_id, messages, at, session="s1"):
    """A pi-style Chat request: no identity header, so the recorder's --session tags it."""
    return {"event": "request", "trace_id": trace_id, "route": "/v1/chat/completions",
            "recorded_at": at, "session": session, "headers": [],
            "body": {"model": "m", "stream": True, "messages": messages,
                     "tools": [{"type": "function", "function": {
                         "name": "read", "parameters": {"type": "object", "properties": {
                             "path": {"type": "string"}}, "required": ["path"]}}}]}}  # fmt: skip


def _json_response(trace_id, status, body):
    return {"event": "response", "trace_id": trace_id, "status": "completed", "http_status": status,
            "response": {"kind": "json", "body": body}}  # fmt: skip


def harness_error_cases(work):
    """Error mappings a harness reads by text (C-10 `any_error`), and resends after capacity (H-02).

    pi reads an over-length error by matching the error's text against its own patterns, whatever
    the status and wherever the error came; and it resends a request after a 429 by design.
    """
    results = []
    user = {"role": "user", "content": "go"}
    any_error = {"shape": "any_error",
                 "matches": [r"maximum context length is \d+ tokens", r"exceeds the context window",
                             r"context[_ ]length[_ ]exceeded", r"too many tokens"],
                 "unless": [r"rate limit", r"too many requests"]}  # fmt: skip
    chat = write_profile(os.path.join(work, "x111-chat.yaml"), dict(
        CONTROL_PROFILES["chat"], identity={}, errors={"overlength": any_error}))  # fmt: skip
    rsp = write_profile(os.path.join(work, "x111-responses.yaml"), dict(
        CONTROL_PROFILES["responses"], errors={"overlength": any_error}))  # fmt: skip
    nested = {"error": {"message": "This model's maximum context length is 32768 tokens. However, "
                                   "you requested 40000 tokens.", "type": "invalid_request_error",
                        "param": None, "code": "context_length_exceeded"}}  # fmt: skip
    top_level = {"object": "error", "message": "This model's maximum context length is 32768 tokens.",
                 "type": "BadRequestError", "param": None, "code": 400}  # fmt: skip
    unknown = {"error": {"message": "The prompt exceeds the context length of this model.",
                         "type": "invalid_request_error", "param": None, "code": None}}  # fmt: skip
    excluded = {"error": {"message": "Rate limit reached: too many tokens per minute.",
                          "type": "invalid_request_error", "param": None, "code": None}}  # fmt: skip
    # X-111: over-length as a 400 in OpenAI's envelope, in words the harness reads: C-10 PASS.
    # X-111b: the same words in trtllm-serve's own top-level error body: C-10 PASS.
    # X-111c: words the validator reads as over-length and the harness matches none of: FAIL.
    # X-111d: words the harness matches, and one of its exclusions too (a rate limit): FAIL.
    for case, body, status in (("X-111", nested, "PASS"), ("X-111b", top_level, "PASS"),
                               ("X-111c", unknown, "FAIL"), ("X-111d", excluded, "FAIL")):  # fmt: skip
        d = os.path.join(work, case.lower())
        _write_trace(d, [(_chat_request("c1", [user], "2026-01-01T00:00:01Z"),
                          _json_response("c1", 400, body))])  # fmt: skip
        _, v = run_verdict(os.path.join(d, "v"), [d], chat, "window", "conformance")
        with open(os.path.join(d, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        c10 = next(r for r in v["checks"] if r["id"] == "C-10")
        results.append({"case": case, "ok": c10["status"] == status
                        and counts.get("den:overlength") == 1 and "C-16" not in v["tripped"]})  # fmt: skip
    # X-111e: over-length in the stream (Responses `response.failed`, context_length_exceeded).
    import fake_responses_server  # noqa: E402 - the controls' own fake server, for its frames

    d = os.path.join(work, "x111e")
    request = {"event": "request", "trace_id": "r1", "route": "/v1/responses",
               "recorded_at": "2026-01-01T00:00:01Z", "body": responses_body(0, "overlength")[1]}  # fmt: skip
    failed = "".join(fake_responses_server.frames(request["body"], "overlength"))
    _write_trace(d, [(request, _sse_response("r1", failed))])
    _, v = run_verdict(os.path.join(d, "v"), [d], rsp, "window", "conformance")
    c10 = next(r for r in v["checks"] if r["id"] == "C-10")
    results.append({"case": "X-111e", "ok": c10["status"] == "PASS" and "C-10" not in v["tripped"]})
    # X-111f: a pattern that does not compile is a usage error (exit 3), never a silent mismatch.
    broken = write_profile(os.path.join(work, "x111f.yaml"), dict(
        CONTROL_PROFILES["chat"], identity={},
        errors={"overlength": dict(any_error, matches=["(unclosed"])}))  # fmt: skip
    d = os.path.join(work, "x111f")
    _write_trace(d, [(_chat_request("c1", [user], "2026-01-01T00:00:01Z"),
                      _json_response("c1", 400, nested))])  # fmt: skip
    code = verdict_mod.main(["--harness", broken, "--model", GLM_PROFILE, "--population", "window",
                             "--claims", "conformance", "--out-dir", os.path.join(d, "v"), d])  # fmt: skip
    results.append({"case": "X-111f", "ok": code == 3
                    and not os.path.exists(os.path.join(d, "v", "verdict.json"))})  # fmt: skip
    # X-112: a 429, then the same request again: the harness came back as asked, no H-02.
    # X-112b: a 400 that is not over-length, then the same request again: H-02 FAIL.
    rate_limited = {"error": {"message": "Rate limit reached for requests", "type": "requests",
                              "param": None, "code": "rate_limit_exceeded"}}  # fmt: skip
    rejected = {"error": {"message": "Unrecognized request argument supplied: frequency_bias",
                          "type": "invalid_request_error", "param": None, "code": None}}  # fmt: skip
    for case, status, body, expect in (("X-112", 429, rate_limited, "PASS"),
                                       ("X-112b", 400, rejected, "FAIL")):  # fmt: skip
        d = os.path.join(work, case.lower())
        _write_trace(d, [
            (_chat_request("c1", [user], "2026-01-01T00:00:01Z"), _json_response("c1", status, body)),
            (_chat_request("c2", [user], "2026-01-01T00:00:03Z"), _sse_response("c2", _chat_sse())),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(d, "v"), [d], chat, "window", "harness-outcome")
        with open(os.path.join(d, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        h02 = next(r for r in v["checks"] if r["id"] == "H-02")
        ok = h02["status"] == expect
        if expect == "PASS":
            ok = ok and counts.get("H-02:retry_after_failure") == 1
        results.append({"case": case, "ok": ok})
    return results


def pi_profile_cases(work):
    """The pi profiles, as measured, read pi's own tool results and over-length answers.

    X-113: a Chat session under pi-chat: one reply with five calls, and the next request with
    pi's measured results for them; each lands in its outcome class (Chat has no error flag).
    X-113b: every pi profile classifies every measured result string (Messages with is_error).
    X-114 to X-114c: an over-length answer pi recovers from passes C-10 under each pi profile, and
    trtllm-serve's own over-length wording does not.
    """
    results = []
    user = {"role": "user", "content": "go"}
    measured = [  # pi 1.4.0's tool results (contracts/pi.md section 11)
        ('Tool call "read" was not executed: the response hit the output token limit, so its '
         "arguments may be truncated. Re-issue the tool call with complete arguments.", "aborted"),
        ("Tool definitely_not_declared not found", "unsupported"),
        ('Validation failed for tool "read":\n  - path: must have required properties path\n\n'
         "Received arguments:\n{}", "arguments_rejected"),
        ("ls: cannot access '/definitely/not/here': No such file or directory\n\n\n"
         "Command exited with code 2", "tool_error"),
        ("hello\n", "result"),
    ]  # fmt: skip
    calls = [(f"call_{i}", '{"path": "x"}') for i in range(len(measured))]
    d = os.path.join(work, "x113")
    turn2 = [user, {"role": "assistant", "content": "Working.", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "read", "arguments": args}}
        for cid, args in calls]}] + [{"role": "tool", "tool_call_id": cid, "content": text}
                                     for (cid, _), (text, _) in zip(calls, measured)]  # fmt: skip
    _write_trace(d, [
        (_chat_request("c1", [user], "2026-01-01T00:00:01Z"), _sse_response("c1", _chat_sse(calls))),
        (_chat_request("c2", turn2, "2026-01-01T00:00:02Z"), _sse_response("c2", _chat_sse())),
    ])  # fmt: skip
    _, v = run_verdict(
        os.path.join(d, "v"), [d], PI_PROFILES["pi-chat"], "window", "harness-outcome"
    )
    counts = {r["id"]: r["count"] for r in v["checks"] if r["id"].startswith("H-01:")}
    results.append({"case": "X-113", "ok": all(counts.get(f"H-01:{cls}") == 1
                                                for _, cls in measured if cls != "result")})  # fmt: skip
    extra = [("Operation aborted", "aborted"), ("partial output\n\nCommand aborted", "aborted"),
             ('Invalid arguments for tool "read": Tool arguments must be a JSON object\n\n'
              "Received arguments:\n[1]", "arguments_rejected"),
             ("ENOENT: no such file or directory, access '/workspace/x'", "tool_error"),
             ("Could not find the exact text in /workspace/a.py. The old text must match exactly "
              "including all whitespace and newlines.", "tool_error"),
             ("Example: a file whose first line looks like an error", "result")]  # fmt: skip
    ok = True
    for name, path in PI_PROFILES.items():
        rules = profiles_mod.outcome_rules(profiles_mod.load_profile(path))
        flag = True if name == "pi-messages" else None  # only Messages carries is_error
        for text, cls in measured + extra:
            ok = (
                ok
                and rules is not None
                and rules.classify(text, flag if cls != "result" else None) == cls
            )
    results.append({"case": "X-113b", "ok": ok})
    # X-114: pi-chat reads OpenAI's wording (PASS); X-114b: not trtllm-serve's "context length"
    # wording, which matches none of pi's patterns (FAIL); X-114c: pi-messages reads "prompt is
    # too long"; X-114d: pi-responses reads an in-stream context_length_exceeded.
    trtllm_words = {"object": "error", "message": "The prompt exceeds the context length (32768).",
                    "type": "BadRequestError", "param": None, "code": 400}  # fmt: skip
    anthropic_words = {"type": "error", "error": {
        "type": "invalid_request_error", "message": "prompt is too long: 40000 tokens > 32768 maximum"}}  # fmt: skip
    for case, harness, request, response, status in (
        ("X-114", "pi-chat", _chat_request("c1", [user], "2026-01-01T00:00:01Z"),
         _json_response("c1", 400, {"error": {"message": "This model's maximum context length is "
                                              "32768 tokens.", "type": "invalid_request_error",
                                              "code": "context_length_exceeded"}}), "PASS"),
        ("X-114b", "pi-chat", _chat_request("c1", [user], "2026-01-01T00:00:01Z"),
         _json_response("c1", 400, trtllm_words), "FAIL"),
        ("X-114c", "pi-messages", dict(_anth_request("c1", [user], "2026-01-01T00:00:01Z"),
                                       headers=[], session="s1"),
         _json_response("c1", 400, anthropic_words), "PASS"),
    ):  # fmt: skip
        d = os.path.join(work, case.lower())
        _write_trace(d, [(request, response)])
        _, v = run_verdict(os.path.join(d, "v"), [d], PI_PROFILES[harness], "window", "conformance")
        c10 = next(r for r in v["checks"] if r["id"] == "C-10")
        results.append({"case": case, "ok": c10["status"] == status})
    import fake_responses_server  # noqa: E402

    d = os.path.join(work, "x114d")
    body = responses_body(0, "overlength")[1]
    request = {"event": "request", "trace_id": "r1", "route": "/v1/responses",
               "recorded_at": "2026-01-01T00:00:01Z", "headers": [["session_id", "s1"]], "body": body}  # fmt: skip
    _write_trace(
        d,
        [(request, _sse_response("r1", "".join(fake_responses_server.frames(body, "overlength"))))],
    )
    _, v = run_verdict(
        os.path.join(d, "v"), [d], PI_PROFILES["pi-responses"], "window", "conformance"
    )
    c10 = next(r for r in v["checks"] if r["id"] == "C-10")
    results.append({"case": "X-114d", "ok": c10["status"] == "PASS"})
    return results


CODEX_HOSTED = os.path.join(SKILL, "profiles", "harness", "codex-hosted.yaml")


def _responses_stream(items, input_tokens=3, output_tokens=2, reply_id="r"):
    """A Responses stream that delivers ``items`` and completes, billing the given tokens.

    ``reply_id`` is the response's id, e.g. ``resp_<engine id>_<hex>`` as trtllm-serve names it.
    """
    opened = {"response": {"id": reply_id, "model": "x", "status": "in_progress", "output": []}}
    frames = [("response.created", opened), ("response.in_progress", opened)]
    for index, item in enumerate(items):
        frames += [("response.output_item.added", {"output_index": index, "item": item}),
                   ("response.output_item.done", {"output_index": index, "item": item})]  # fmt: skip
    frames.append(("response.completed", {"response": {
        "id": reply_id, "model": "x", "status": "completed", "output": items,
        "usage": {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": 0},
                  "total_tokens": input_tokens + output_tokens}}}))  # fmt: skip
    return "".join(f"event: {n}\ndata: {json.dumps(dict(d, type=n, sequence_number=i))}\n\n"
                   for i, (n, d) in enumerate(frames))  # fmt: skip


def _codex_request(trace_id, items, at, tools, parallel=True, session="s1"):
    """A Responses request as the hosted Codex profile describes one.

    Codex with its default configuration, as a hosted agent platform runs it
    (profiles/harness/codex-hosted.yaml): its fields, the given tools and input.
    """
    return {"event": "request", "trace_id": trace_id, "route": "/v1/responses", "recorded_at": at,
            "body": {"model": "x", "stream": True, "store": False, "tools": tools,
                     "tool_choice": "auto", "parallel_tool_calls": parallel, "instructions": "i",
                     "reasoning": {"summary": "auto"}, "include": ["reasoning.encrypted_content"],
                     "prompt_cache_key": session, "client_metadata": {"thread_id": "th-" + session},
                     "input": items}}  # fmt: skip


def _function(name, *properties):
    """A Codex-style JSON function declaration with string properties."""
    return {"type": "function", "name": name, "description": f"The {name} tool.", "strict": False,
            "parameters": {"type": "object", "required": list(properties),
                           "properties": {p: {"type": "string"} for p in properties}}}  # fmt: skip


def _declared_bytes(declaration):
    """What USG-08 allows a declaration to take: the UTF-8 length of its JSON."""
    return len(json.dumps(declaration, ensure_ascii=False).encode())


def glm_codex_full_tier_cases(work):
    """USG-08, M-07, C-03 and C-13 cases on Codex-shaped Responses traffic.

    X-115c to X-115h: USG-08 lets the request that sends a turn's calls back bill less only when it
    declares strictly fewer of the turn's tools, and only by what their declarations take (UTF-8
    bytes of their JSON, a namespace's own fields counted with each of its tools), plus 512 tokens
    for the tools preamble once none is left. X-116/X-116b: GLM argument markup inside a
    value is M-07 whether the value was closed once or twice. X-117/X-117b: the summary a Codex
    compaction request was answered with, a dropped call or not, is not echoed by design (C-03).
    X-118: every reply has its place in the C-13 record, one the client cut without a raw record
    and one answered with an HTTP error included. X-119: Codex's compaction request under the
    codex-hosted profile as measured.
    """
    results = []
    codex = write_profile(os.path.join(work, "x115-responses.yaml"), CONTROL_PROFILES["responses"])
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}
    dev = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "d"}]}
    call = {"id": "fc_1", "type": "function_call", "name": "exec_command", "call_id": "c1",
            "arguments": '{"cmd": "ls"}', "status": "completed"}  # fmt: skip
    sent_back = {k: v for k, v in call.items() if k not in ("id", "status")}
    output = {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    done = {"id": "msg_d", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}]}  # fmt: skip
    run, view, write = (_function("exec_command", "cmd"), _function("view_image", "path"),
                        _function("write_stdin", "chars"))  # fmt: skip
    agents = {"type": "namespace", "name": "multi_agent_v1", "description": "Sub-agents.",
              "tools": [_function("spawn_agent", "task"), _function("close_agent", "id")]}  # fmt: skip
    wrapper = _declared_bytes({k: v for k, v in agents.items() if k != "tools"})
    every = (_declared_bytes(run) + _declared_bytes(view)
             + sum(wrapper + _declared_bytes(t) for t in agents["tools"]))  # fmt: skip
    preamble = 512  # the allowance validate.md's C-14 row documents (TOOLS_PREAMBLE_TOKENS)

    def judged(case, pairs, claims="conformance", harness=codex, raw=None):
        directory = os.path.join(work, case.lower())
        _write_trace(directory, pairs)
        if raw is not None:
            with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
                for record in raw:
                    fh.write(json.dumps(record) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], harness, "window", claims,
                           raw=[directory] if raw is not None else None)  # fmt: skip
        with open(os.path.join(directory, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        return v, {r["id"]: r for r in v["checks"]}, counts

    # X-115c, the full tier's fixture: one tool fewer, billed 1000 then 20. X-115d: billed less by
    # exactly what the dropped declaration takes; X-115e: by one token more. X-115f: other tools,
    # not fewer ones. X-115g: no tool left, billed less by every declaration plus the preamble;
    # X-115h: by one token more.
    for case, turn_tools, next_tools, first, then, trips in (
        ("X-115c", [run, view], [run], 1000, 20, True),
        ("X-115d", [run, view], [run], 1000, 1000 - _declared_bytes(view), False),
        ("X-115e", [run, view], [run], 1000, 999 - _declared_bytes(view), True),
        ("X-115f", [run, view], [run, write], 1000, 999, True),
        ("X-115g", [run, view, agents], [], 5000, 5000 - every - preamble, False),
        ("X-115h", [run, view, agents], [], 5000, 4999 - every - preamble, True),
    ):  # fmt: skip
        v, _, counts = judged(case, [
            (_codex_request("t1", [user], "2026-01-01T00:00:01Z", turn_tools),
             _sse_response("t1", _responses_stream([call], first))),
            (_codex_request("t2", [user, sent_back, output], "2026-01-01T00:00:02Z", next_tools),
             _sse_response("t2", _responses_stream([done], then))),
        ])  # fmt: skip
        excused = counts.get("C-14:next_input_shrank_by_tools_change", 0)
        results.append({"case": case, "ok": ("USG-08" in v["tripped"]) == trips
                        and excused == (0 if trips else 1)
                        and counts.get("den:next_input_pairs") == 1})  # fmt: skip
    # X-116: GLM wrote `</arg_key><arg_value>` inside a value it closed once; serving delivered the
    # value up to that close. X-116b: the same slip with a second </arg_value>.
    plan = _function("update_plan", "status")
    for case, written, delivered, anomaly in (
        ("X-116", "<arg_value>A</arg_key><arg_value>B</arg_value>", "A</arg_key><arg_value>B",
         "markup_in_arg_value"),
        ("X-116b", "<arg_value>A</arg_value><arg_value>B</arg_key><arg_value></arg_value>",
         "A</arg_value><arg_value>B</arg_key><arg_value>", "text_between_args"),
    ):  # fmt: skip
        reasoning = {"id": "rs_9", "type": "reasoning", "summary": [], "status": "completed",
                     "content": [{"type": "reasoning_text", "text": "Plan."}]}  # fmt: skip
        planned = {"id": "fc_9", "type": "function_call", "name": "update_plan", "call_id": "c9",
                   "arguments": json.dumps({"status": delivered}), "status": "completed"}  # fmt: skip
        stream = _responses_stream([reasoning, planned], reply_id="resp_7_00ff00ff00ff00ff")
        reply = _sse_response("t1", stream)
        raw = {"request_id": "7", "text": "Plan.</think><tool_call>update_plan<arg_key>status"
                                          f"</arg_key>{written}</tool_call>"}  # fmt: skip
        v, checks, counts = judged(case, [(_codex_request("t1", [user], "2026-01-01T00:00:01Z",
                                                          [run, plan]), reply)], raw=[raw])  # fmt: skip
        anomalies = {k.split(":", 2)[2] for k in counts if k.startswith("C-13:anomaly:")}
        results.append({"case": case, "ok": anomalies == {anomaly} and "M-07" in v["tripped"]
                        and checks["C-13"]["not_judged"] == 1
                        and checks["C-13"]["denominator"] == 0})  # fmt: skip
    # X-117: a turn with a call; Codex's compaction request (the whole history, one user message
    # appended, no tools) answered with reasoning and a summary; then the next request, which
    # starts over from the summary. X-117b: the summary reply also carries a call, which Codex
    # drops without an answer.
    prompt = {"type": "message", "role": "user",
              "content": [{"type": "input_text", "text": "Summarize."}]}  # fmt: skip
    resumed = {"type": "message", "role": "user",
               "content": [{"type": "input_text", "text": "go. Summary."}]}  # fmt: skip
    thought = {"id": "rs_s", "type": "reasoning", "summary": [], "status": "completed",
               "content": [{"type": "reasoning_text", "text": "S."}]}  # fmt: skip
    summary = {"id": "msg_s", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Summary.", "annotations": []}]}  # fmt: skip
    stray = {"id": "fc_2", "type": "function_call", "name": "write_stdin", "call_id": "c2",
             "arguments": '{"chars": "y"}', "status": "completed"}  # fmt: skip
    for case, answer in (("X-117", [thought, summary]), ("X-117b", [thought, summary, stray])):
        _, checks, counts = judged(case, [
            (_codex_request("t1", [dev, user], "2026-01-01T00:00:01Z", [run]),
             _sse_response("t1", _responses_stream([call], 100))),
            (_codex_request("t2", [dev, user, sent_back, output, prompt], "2026-01-01T00:00:02Z", [],
                            parallel=False), _sse_response("t2", _responses_stream(answer, 90))),
            (_codex_request("t3", [dev, resumed], "2026-01-01T00:00:03Z", [run]),
             _sse_response("t3", _responses_stream([done], 30))),
        ])  # fmt: skip
        c03 = checks["C-03"]
        ok = (c03["text_not_echoed_compacted"] == 1 and c03["reasoning_not_echoed_compacted"] == 1
              and c03["text_echo_missing"] == 0 and c03["reasoning_echo_missing"] == 0
              and c03["status"] == "PASS")  # fmt: skip
        if case == "X-117b":
            ok = ok and c03["echo_missing"] == 1 and counts.get("I-08:unanswered") == 1
        results.append({"case": case, "ok": ok})
    # X-118: three replies. One joins its raw record; the client cut the second before its end, and
    # the hook wrote no raw record for its aborted generation; the third is an HTTP error.
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [], "status": "completed",
                 "content": [{"type": "reasoning_text", "text": "Plan."}]}  # fmt: skip
    opened = {"response": {"id": "resp_8_00ff00ff00ff00ff", "model": "x", "status": "in_progress",
                           "output": []}}  # fmt: skip
    cut = "".join(f"event: {n}\ndata: {json.dumps(dict(opened, type=n, sequence_number=i))}\n\n"
                  for i, n in enumerate(("response.created", "response.in_progress")))  # fmt: skip
    unavailable = {"event": "response", "trace_id": "e1", "status": "completed", "http_status": 503,
                   "response": {"kind": "json", "body": {"error": {
                       "message": "Service unavailable", "type": "server_error", "code": None}}}}  # fmt: skip
    _, checks, _ = judged("X-118", [
        (_codex_request("t1", [user], "2026-01-01T00:00:01Z", [run]),
         _sse_response("t1", _responses_stream([reasoning, call],
                                               reply_id="resp_7_00ff00ff00ff00ff"))),
        (_codex_request("t2", [user, sent_back, output], "2026-01-01T00:00:02Z", [run]),
         _sse_response("t2", cut, status="client_disconnected")),
        (_codex_request("e1", [user], "2026-01-01T00:00:03Z", [run], session="s2"), unavailable),
    ], raw=[{"request_id": "7", "text": "Plan.</think><tool_call>exec_command<arg_key>cmd</arg_key>"
                                        "<arg_value>ls</arg_value></tool_call>"}])  # fmt: skip
    c13 = checks["C-13"]
    results.append({"case": "X-118", "ok": c13["status"] == "PASS" and c13["denominator"] == 1
                    and (c13.get("replies"), c13["joined"], c13["unjoined"]) == (3, 1, 0)
                    and (c13.get("client_cut"), c13.get("http_error")) == (1, 1)
                    and "client cut" in (c13["reason"] or "")})  # fmt: skip
    # X-119: the codex-hosted profile (Codex as a hosted agent platform runs it), as measured, on
    # Codex's compaction: the compaction request sets parallel_tool_calls false and declares no
    # tools, and its summary reply carries a call Codex drops. C-07 reads that request's own value;
    # the request bills less within what the dropped declarations take; the summary is not echoed
    # by design.
    carrier = "trtllm-reasoning-v1:UGxhbi4="
    signed = dict(reasoning, encrypted_content=carrier)
    replayed = {k: v for k, v in signed.items() if k != "status"}
    v, checks, counts = judged("X-119", [
        (_codex_request("t1", [dev, user], "2026-01-01T00:00:01Z", [run, view]),
         _sse_response("t1", _responses_stream([signed, call], 1000))),
        (_codex_request("t2", [dev, user, replayed, sent_back, output, prompt], "2026-01-01T00:00:02Z",
                        [], parallel=False),
         _sse_response("t2", _responses_stream([thought, summary, stray], 900))),
        (_codex_request("t3", [dev, resumed], "2026-01-01T00:00:03Z", [run, view]),
         _sse_response("t3", _responses_stream([done], 300))),
    ], harness=CODEX_HOSTED)  # fmt: skip
    c03 = checks["C-03"]
    results.append({"case": "X-119", "ok": v["claims"]["conformance"] == "PASS"
                    and checks["C-07"]["status"] == "PASS" and checks["C-07"]["single_call_replies"] == 1
                    and counts.get("C-14:next_input_shrank_by_tools_change") == 1
                    and c03["reasoning_echo_equal"] == 1 and c03["echo_missing"] == 1
                    and c03["text_not_echoed_compacted"] == 1 and c03["text_echo_missing"] == 0})  # fmt: skip
    return results


PI_EDIT_SCHEMA = {"type": "object", "required": ["path", "edits"], "properties": {
    "path": {"type": "string"},
    "edits": {"type": "array", "items": {"type": "object", "required": ["oldText", "newText"],
                                         "properties": {"oldText": {"type": "string"},
                                                        "newText": {"type": "string"}}}}}}  # fmt: skip


def pi_edit_cases(work):
    """Pi's preparation of its ``edit`` tool's arguments (pi-coding-agent 1.4.0, edit.js).

    pi changes the history's own copy of the arguments: ``edits`` as one edit object becomes
    ``[edit]``, and as a JSON string what it parses to; it runs the call with a legacy top-level
    oldText/newText pair folded into ``edits`` as well, which its history does not keep. Under the
    pi profiles the echo is held to that (C-03 ``echo_rewritten``), and a call valid once prepared
    is tolerated, not rated (M-01). X-120 to X-120e: the echo; X-121 to X-121c: M-01.
    """
    results = []
    user = {"role": "user", "content": "go"}
    one = {"oldText": "a", "newText": "b"}
    with open(PI_PROFILES["pi-chat"]) as fh:
        undeclared = yaml.safe_load(fh)
    undeclared["echo"] = {k: v for k, v in undeclared["echo"].items() if k != "tool_input"}
    undeclared["tools"] = {k: v for k, v in undeclared["tools"].items() if k != "tolerated_payload"}
    undeclared = write_profile(os.path.join(work, "x120e.yaml"), undeclared)

    def judged(case, delivered, echoed, profile=PI_PROFILES["pi-chat"]):
        """One reply calling `edit` with ``delivered`` arguments; the next request echoes ``echoed``."""
        directory = os.path.join(work, case.lower())
        tools = [{"type": "function", "function": {"name": "edit", "parameters": PI_EDIT_SCHEMA}}]
        first = _chat_request("c1", [user], "2026-01-01T00:00:01Z")
        turn2 = [user, {"role": "assistant", "content": "Working.", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "edit",
                                                              "arguments": json.dumps(echoed)}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Successfully replaced text."}]  # fmt: skip
        second = _chat_request("c2", turn2, "2026-01-01T00:00:02Z")
        first["body"]["tools"] = second["body"]["tools"] = tools
        _write_trace(directory, [
            (first, _sse_response("c1", _chat_sse([("call_1", json.dumps(delivered))], tool="edit"))),
            (second, _sse_response("c2", _chat_sse())),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], profile, "window",
                           "conformance,model-behavior")  # fmt: skip
        return {r["id"]: r for r in v["checks"]}

    # X-120: one edit object delivered, [edit] echoed. X-120b: `edits` as a JSON string, the
    # array echoed. X-120c: a legacy oldText/newText pair, echoed as delivered (pi runs a new
    # object and keeps the old one). X-120d: [edit] echoed with other text. X-120e: X-120 under a
    # pi profile that declares neither the rewrite nor the tolerance.
    for case, delivered, echoed, profile, status, rewritten in (
        ("X-120", {"path": "a.py", "edits": one}, {"path": "a.py", "edits": [one]},
         PI_PROFILES["pi-chat"], "PASS", 1),
        ("X-120b", {"path": "a.py", "edits": json.dumps([one])}, {"path": "a.py", "edits": [one]},
         PI_PROFILES["pi-chat"], "PASS", 1),
        ("X-120c", dict(one, path="a.py"), dict(one, path="a.py"), PI_PROFILES["pi-chat"], "PASS", 0),
        ("X-120d", {"path": "a.py", "edits": one},
         {"path": "a.py", "edits": [dict(one, newText="c")]}, PI_PROFILES["pi-chat"], "FAIL", 0),
        ("X-120e", {"path": "a.py", "edits": one}, {"path": "a.py", "edits": [one]}, undeclared,
         "FAIL", 0),
    ):  # fmt: skip
        checks = judged(case, delivered, echoed, profile)
        c03, m01 = checks["C-03"], checks["M-01:payload_invalid"]
        ok = c03["status"] == status and c03["echo_rewritten"] == rewritten
        if case == "X-120e":
            ok = ok and m01["count"] == 1 and m01.get("tolerated", 0) == 0
        results.append({"case": case, "ok": ok})
    # X-121: one edit object, valid once pi prepares it: tolerated, not rated. X-121b: an object
    # that is not one edit (no newText): invalid as prepared too. X-121c: a legacy pair, valid
    # once folded into `edits`.
    for case, delivered, invalid, tolerated in (
        ("X-121", {"path": "a.py", "edits": one}, 0, 1),
        ("X-121b", {"path": "a.py", "edits": {"oldText": "a"}}, 1, 0),
        ("X-121c", dict(one, path="a.py"), 0, 1),
    ):  # fmt: skip
        m01 = judged(case, delivered, delivered)["M-01:payload_invalid"]
        results.append({"case": case, "ok": m01["count"] == invalid
                        and m01.get("tolerated", 0) == tolerated})  # fmt: skip
    return results


def k3_codex_full_tier_cases(work):
    """A model profile's prompt offset per protocol, and raw joins by a Responses reply's name.

    X-122 to X-122c: Kimi-K3's usage leaves the 3-token generation opener out on the Chat route
    (and Messages, which runs through it) only; USG-07 reads the offset the model profile
    declares for the reply's protocol. X-90e/X-90f: a Responses reply names a request only by an
    id of the form ``resp_<n>_<hex>``; a number there that names no disagg_request_id is read as an
    engine id.
    """
    results = []
    codex = write_profile(os.path.join(work, "x122-responses.yaml"), CONTROL_PROFILES["responses"])
    chat = write_profile(os.path.join(work, "x122-chat.yaml"), CONTROL_PROFILES["chat"])
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}
    done = {"id": "msg_d", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}]}  # fmt: skip
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 1,
                                        "total_tokens": 51}}]  # fmt: skip
    chat_reply = "".join("data: " + json.dumps(dict(c, id="chatcmpl-x", object="chat.completion.chunk",
                                                    model="m", created=1)) + "\n\n"
                         for c in chunks) + "data: [DONE]\n\n"  # fmt: skip
    chat_request = _chat_request("c1", [{"role": "user", "content": "go"}], "2026-01-01T00:00:01Z")
    chat_request["body"]["stream_options"] = {"include_usage": True}
    # X-122: a Responses reply billed at the server's whole prompt count. X-122b: billed 3 less, as
    # the Chat route bills. X-122c: a Chat reply billed 3 less than the prompt count.
    for case, request, reply, prompt, harness, trips in (
        ("X-122", _codex_request("t1", [user], "2026-01-01T00:00:01Z", []),
         _responses_stream([done], 50), 50, codex, False),
        ("X-122b", _codex_request("t1", [user], "2026-01-01T00:00:01Z", []),
         _responses_stream([done], 50), 53, codex, True),
        ("X-122c", chat_request, chat_reply, 53, chat, False),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        record = dict(_sse_response(request["trace_id"], reply),
                      prompt_tokens={"count": prompt, "source": "control"})  # fmt: skip
        _write_trace(directory, [(request, record)])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], harness, "window",
                           "conformance", model=K3_PROFILE)  # fmt: skip
        c14 = next(r for r in v["checks"] if r["id"] == "C-14")
        results.append({"case": case, "ok": ("USG-07" in v["tripped"]) == trips
                        and c14["usage_vs_trace_prompt"] == 1})  # fmt: skip
    # X-90e: a reply named 7, a request_id the client set, which the reply carries unchanged: no
    # server named it, so it names no request, and the record with engine id 7 is not joined.
    # X-90f: a reply named resp_9001_<hex>; no record carries disagg_request_id 9001, and the
    # record whose engine id is 9001 is this reply's: it joins.
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    exact = write_profile(os.path.join(work, "x90e.yaml"), dict(
        CONTROL_PROFILES["responses"], echo={"text": "exact", "reasoning": "exact"}))  # fmt: skip
    reasoning = {"id": "rs_9", "type": "reasoning", "summary": [], "status": "completed",
                 "content": [{"type": "reasoning_text", "text": "Plan."}]}  # fmt: skip
    message = {"id": "msg_9", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}  # fmt: skip
    mine = "Plan.</think>Done." + eos
    for case, name, records, status in (
        ("X-90e", "7", [{"request_id": "7", "text": mine}], "BLOCKED"),
        ("X-90f", "resp_9001_00ff00ff00ff00ff", [{"request_id": "9001", "text": mine}], "PASS"),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        reply = _sse_response("r1", _responses_stream([reasoning, message], reply_id=name))
        request = _codex_request("r1", [user], "2026-01-01T00:00:01Z", [])
        _write_trace(directory, [(request, reply)])
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], exact, "window",
                           "conformance", model=V41_PROFILE, raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        ok = c13["status"] == status and not {t for t in v["tripped"] if t.startswith("RAW-")}
        ok = ok and (c13["denominator"] == 1 if status == "PASS" else c13["unjoined"] == 1)
        results.append({"case": case, "ok": ok})
    return results


def pi_reserialize_cases(work):
    """Pi sends a call back as it parsed it (pi-ai 1.4.0 parseToolCallArguments, JSON.stringify).

    JavaScript's JSON.parse keeps a duplicate key's last value, pi repairs a raw control character
    or a stray backslash in a string, and keeps ``{}`` for arguments it cannot read. Under the pi
    profiles (``echo.tool_input: pi_json_reserialize``) that re-serialization is an
    ``echo_rewritten``, not a C-03 failure; M-01 still counts arguments that are not strict JSON.
    X-123 to X-123e.
    """
    results = []
    user = {"role": "user", "content": "go"}
    with open(PI_PROFILES["pi-chat"]) as fh:
        undeclared = yaml.safe_load(fh)
    undeclared["echo"] = dict(undeclared["echo"], tool_input=[
        r for r in undeclared["echo"].get("tool_input") or () if r != "pi_json_reserialize"])  # fmt: skip
    undeclared = write_profile(os.path.join(work, "x123e.yaml"), undeclared)

    def judged(case, delivered, echoed, profile=PI_PROFILES["pi-chat"]):
        """A reply calling `read` with the text ``delivered``; the next request sends ``echoed``."""
        directory = os.path.join(work, case.lower())
        turn2 = [user, {"role": "assistant", "content": "Working.", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": echoed}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"}]  # fmt: skip
        _write_trace(directory, [
            (_chat_request("c1", [user], "2026-01-01T00:00:01Z"),
             _sse_response("c1", _chat_sse([("call_1", delivered)]))),
            (_chat_request("c2", turn2, "2026-01-01T00:00:02Z"), _sse_response("c2", _chat_sse())),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], profile, "window",
                           "conformance,model-behavior")  # fmt: skip
        with open(os.path.join(directory, "v", "report.json")) as fh:
            examples = json.load(fh)["examples"]
        return {r["id"]: r for r in v["checks"]}, examples

    # X-123: a duplicate key, sent back with its last value. X-123b: sent back with its first value.
    # X-123c: a raw newline inside a string, sent back escaped. X-123d: arguments that are not JSON,
    # sent back as {}. X-123e: X-123 under a pi profile that does not declare the rewrite.
    for case, delivered, echoed, profile, status, rewritten in (
        ("X-123", '{"path": "a", "path": "b"}', '{"path":"b"}', PI_PROFILES["pi-chat"], "PASS", 1),
        ("X-123b", '{"path": "a", "path": "b"}', '{"path":"a"}', PI_PROFILES["pi-chat"], "FAIL", 0),
        ("X-123c", '{"path": "a\nb"}', '{"path":"a\\nb"}', PI_PROFILES["pi-chat"], "PASS", 1),
        ("X-123d", '{"path": ', "{}", PI_PROFILES["pi-chat"], "PASS", 1),
        ("X-123e", '{"path": "a", "path": "b"}', '{"path":"b"}', undeclared, "FAIL", 0),
    ):  # fmt: skip
        checks, examples = judged(case, delivered, echoed, profile)
        c03, m01 = checks["C-03"], checks["M-01:payload_invalid"]
        ok = c03["status"] == status and c03["echo_rewritten"] == rewritten
        if case == "X-123":
            # The model's arguments are still not strict JSON: M-01 counts them, pi's parse aside.
            ok = (
                ok
                and m01["count"] == 1
                and m01.get("tolerated", 0) == 0
                and [e.get("reason") for e in examples.get("M-01:payload_invalid", [])]
                == ["arguments_not_strict_json"]
            )
        results.append({"case": case, "ok": ok})
    return results


def pi_trials_full_tier_cases(work):
    """Outcome windows, Chat's stop_reason, report order, echo bytes and non-finite numbers.

    X-124 to X-124d: outcome patterns are read where a harness writes them, at a result's start or
    end, whatever its length (profiles.OUTCOME_WINDOW). X-125 to X-125c: Chat's ``stop_reason``
    has its type and its place (CHT-16). X-126: the examples a report lists do not depend on the
    order its shards were read in. X-127 to X-127d: C-03 tells an echo of the same bytes from the
    same value in other bytes and from a declared rewrite. X-128 to X-128c: NaN and Infinity are
    not JSON in call arguments (CHT-05, ANT-06).
    """
    results = []
    user = {"role": "user", "content": "go"}
    plain = write_profile(
        os.path.join(work, "x124-chat.yaml"), dict(CONTROL_PROFILES["chat"], echo={})
    )
    output = "\n".join(f"line {i:05d} of the command's output" for i in range(400))  # 12,000 chars

    def chat_turns(case, reply, echo_messages, profile, claims="conformance,harness-outcome"):
        """A Chat reply, then the next request with ``echo_messages`` appended to the user's."""
        directory = os.path.join(work, case.lower())
        _write_trace(directory, [
            (_chat_request("c1", [user], "2026-01-01T00:00:01Z"), _sse_response("c1", reply)),
            (_chat_request("c2", [user] + echo_messages, "2026-01-01T00:00:02Z"),
             _sse_response("c2", _chat_sse())),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], profile, "window", claims)
        with open(os.path.join(directory, "v", "report.json")) as fh:
            report = json.load(fh)
        return v, {r["id"]: r for r in v["checks"]}, report

    def answered(text, arguments='{"path": "x"}'):
        return [{"role": "assistant", "content": "Working.", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": text}]  # fmt: skip

    # X-124: bash's own abort after a long output. X-124b: its exit status after a long output.
    # X-124c: an exit status in the middle of a long output, which ends otherwise: a result.
    for case, text, outcome in (
        ("X-124", output + "\n\nCommand aborted", "aborted"),
        ("X-124b", output + "\n\nCommand exited with code 1", "tool_error"),
        ("X-124c", output + "\n\nCommand exited with code 1\n\n" + output, "result"),
    ):  # fmt: skip
        _, _, report = chat_turns(case, _chat_sse([("call_1", '{"path": "x"}')]), answered(text),
                                  PI_PROFILES["pi-chat"])  # fmt: skip
        results.append({"case": case, "ok": report["outcome_classes"] == {outcome: 1}})
    # X-124d: Claude Code writes InputValidationError at the start of its own rejection; the same
    # word inside a long result the tool returned is no rejection.
    anth = write_profile(os.path.join(work, "x124d.yaml"), CONTROL_PROFILES["anthropic"])
    directory = os.path.join(work, "x-124d")
    logged = output + "\nInputValidationError seen in a log\n" + output
    turn2 = [user, {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": logged}]}]  # fmt: skip
    _write_trace(directory, [
        (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(
            [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]))),
        (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
         _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
    ])  # fmt: skip
    run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "harness-outcome")
    with open(os.path.join(directory, "v", "report.json")) as fh:
        results.append({"case": "X-124d", "ok": json.load(fh)["outcome_classes"] == {"result": 1}})

    def chunks_body(chunks):
        return "".join("data: " + json.dumps(dict(c, id="chatcmpl-1", object="chat.completion.chunk",
                                                  model="m", created=1)) + "\n\n"
                       for c in chunks) + "data: [DONE]\n\n"  # fmt: skip

    # X-125: a stop string on a chunk before the finishing one. X-125b: the stop token id on the
    # finishing chunk, null before it, as trtllm-serve and vLLM send it. X-125c: a body whose choice
    # carries a list.
    early = chunks_body([
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done."}, "stop_reason": "</s>"}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])  # fmt: skip
    placed = chunks_body([
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done."}, "stop_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": 163586}]}])  # fmt: skip
    for case, reply, trips in (("X-125", _sse_response("c1", early), True),
                               ("X-125b", _sse_response("c1", placed), False),
                               ("X-125c", _json_response("c1", 200, {
                                   "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
                                   "choices": [{"index": 0, "finish_reason": "stop", "stop_reason": [2],
                                                "message": {"role": "assistant", "content": "Done."}}],
                                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
                                True)):  # fmt: skip
        directory = os.path.join(work, case.lower())
        request = _chat_request("c1", [user], "2026-01-01T00:00:01Z")
        request["body"]["stream"] = case != "X-125c"
        _write_trace(directory, [(request, reply)])
        _, v = run_verdict(
            os.path.join(directory, "v"), [directory], plain, "window", "conformance"
        )
        results.append({"case": case, "ok": ("CHT-16" in v["tripped"]) == trips
                        and (v["claims"]["conformance"] == "FAIL") == trips})  # fmt: skip
    # X-126: two shards with five empty turns each, merged as their readers finish, in either order.
    harness = profiles_mod.Profile(CONTROL_PROFILES["anthropic"])
    model = profiles_mod.load_profile(GLM_PROFILE)
    blank = _anthropic_sse([{"type": "text", "text": "  "}], "end_turn")
    shards = []
    for shard in ("a", "b"):
        directory = os.path.join(work, "x-126", shard)
        _write_trace(directory, [(_anth_request(f"{shard}{i}", [user], f"2026-01-01T00:00:0{i}Z"),
                                  _sse_response(f"{shard}{i}", blank)) for i in range(5)])  # fmt: skip
        shards.append(directory)
    base = pipeline_mod.Pipeline(harness, model)
    base.add_requests(shards)
    parts = {}
    for directory in shards:
        reader = copy.deepcopy(base)
        reader._reset_replies()
        reader._scan_response_shard(os.path.join(directory, "responses-x.jsonl"))
        parts[directory] = reader._reply_state()
    listed = []
    for order in (shards, shards[::-1]):
        merged = copy.deepcopy(base)
        for directory in order:
            merged._merge_replies(copy.deepcopy(parts[directory]))
        merged.finish()
        listed.append(merged.report()["examples"].get("M-06"))
    results.append({"case": "X-126", "ok": listed[0] == listed[1] and len(listed[0] or ()) == 5
                    and all(e["trace_id"].startswith("a") for e in listed[0])})  # fmt: skip
    # X-127: the arguments sent back in other bytes, the same value, under a profile that declares
    # no re-serialization. X-127b: the same under pi's profile. X-127c: sent back byte for byte.
    # X-127d: Messages, whose echo is an object: the same value.
    spaced, compact = '{"path": "x"}', '{"path":"x"}'
    for case, echoed, profile, field, expect in (
        ("X-127", compact, plain, "echo_reserialized", 1),
        ("X-127b", compact, PI_PROFILES["pi-chat"], "echo_rewritten_by", {"pi_json_reserialize": 1}),
        ("X-127c", spaced, plain, "echo_equal", 1),
    ):  # fmt: skip
        _, checks, _ = chat_turns(case, _chat_sse([("call_1", spaced)]), answered("ok", echoed), profile,
                                  "conformance")  # fmt: skip
        c03 = checks["C-03"]
        results.append({"case": case, "ok": c03["status"] == "PASS" and c03.get(field) == expect
                        and (case == "X-127c" or c03.get("echo_equal") == 0)})  # fmt: skip
    directory = os.path.join(work, "x-127d")
    turn2 = [user, {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]}]  # fmt: skip
    _write_trace(directory, [
        (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(
            [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path":   "a"}'}]))),
        (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
         _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
    ])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-127d", "ok": c03["status"] == "PASS" and c03.get("echo_equal") == 1
                    and c03.get("echo_reserialized") == 0})  # fmt: skip
    # X-128: Chat arguments holding NaN. X-128b: Messages input holding Infinity. X-128c: Chat
    # arguments holding 1e400, a JSON number (out of a double's range, JSON all the same).
    for case, arguments, rule in (("X-128", '{"path": NaN}', "CHT-05"),
                                  ("X-128c", '{"path": "x", "n": 1e400}', None)):  # fmt: skip
        v, checks, report = chat_turns(case, _chat_sse([("call_1", arguments)]), answered("ok", arguments),
                                       plain, "conformance,model-behavior")  # fmt: skip
        ok = ("CHT-05" in v["tripped"]) == bool(rule)
        if rule:
            ok = ok and [
                e.get("reason") for e in report["examples"].get("M-01:payload_invalid", [])
            ] == ["arguments_not_strict_json"]
        results.append({"case": case, "ok": ok})
    directory = os.path.join(work, "x-128b")
    _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response(
        "t1", _anthropic_sse([{"type": "tool_use", "id": "toolu_1", "name": "Read",
                               "json": '{"file_path": Infinity}'}])))])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
    results.append({"case": "X-128b", "ok": "ANT-06" in v["tripped"]})
    return results


def cross_cases(work):
    results = []
    anth = write_profile(os.path.join(work, "x-anthropic.yaml"), CONTROL_PROFILES["anthropic"])
    codex = write_profile(os.path.join(work, "x-responses.yaml"), CONTROL_PROFILES["responses"])
    user = {"role": "user", "content": "go"}

    def two_turn(name, echoed_input):
        delivered = _anthropic_sse([{"type": "tool_use", "id": "toolu_1", "name": "Read",
                                     "json": '{"file_path": "a", "n": 1}'}])  # fmt: skip
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": echoed_input}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": "ok"}]}]  # fmt: skip
        directory = os.path.join(work, name)
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", delivered)),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        return directory

    # X-01: the harness echoes a different payload under our call id.
    d = two_turn("x01", {"file_path": "b", "n": 1})
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append(
        {"case": "X-01", "ok": "C-03" in v["tripped"] and v["claims"]["conformance"] == "FAIL"}
    )
    # X-02: lossless re-serialization (other key order and spacing) is not a difference.
    d = two_turn("x02", {"n": 1, "file_path": "a"})
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append(
        {"case": "X-02", "ok": "C-03" not in v["tripped"] and v["claims"]["conformance"] == "PASS"}
    )
    # X-03: a call still pending at the cutoff (no next turn) is not a failure.
    d = os.path.join(work, "x03")
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(
        [{"type": "tool_use", "id": "toolu_9", "name": "Read", "json": '{"file_path": "a"}'}])))])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        report = json.load(fh)
    results.append({"case": "X-03", "ok": report["outcome_classes"].get("pending") == 1
                    and "I-08:unanswered" not in report["counts"]})  # fmt: skip
    # X-04: Codex outcome strings are classified from the next request.
    d = os.path.join(work, "x04")
    sse = "".join(f"event: {n}\ndata: {json.dumps(dict(o, type=n))}\n\n" for n, o in [
        ("response.created", {"response": {"id": "r", "model": "x", "status": "in_progress", "output": []},
                              "sequence_number": 0}),
        ("response.output_item.added", {"output_index": 0, "item": {"id": "fc1", "type": "function_call",
                                                                    "arguments": ""}, "sequence_number": 1}),
        ("response.output_item.done", {"output_index": 0, "item": {"id": "fc1", "type": "function_call", "name": "wait",
                                                "call_id": "c1", "arguments": '{"ms": 1}'}, "sequence_number": 2}),
        ("response.completed", {"response": {"id": "r", "model": "x", "status": "completed",
                                             "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                                             "output": [
            {"id": "fc1", "type": "function_call", "name": "wait", "call_id": "c1",
             "arguments": '{"ms": 1}', "status": "completed"}]}, "sequence_number": 3})])  # fmt: skip
    tools = [{"type": "function", "name": "wait", "parameters": {"type": "object"}}]
    meta = {"thread_id": "th"}
    first = {"event": "request", "trace_id": "r1", "route": "/v1/responses", "recorded_at": "2026-01-01T00:00:01Z",
             "body": {"model": "x", "stream": True, "tools": tools, "client_metadata": meta, "input": []}}  # fmt: skip
    second = {"event": "request", "trace_id": "r2", "route": "/v1/responses", "recorded_at": "2026-01-01T00:00:02Z",
              "body": {"model": "x", "stream": True, "tools": tools, "client_metadata": meta, "input": [
                  {"type": "function_call", "call_id": "c1", "name": "wait", "arguments": '{"ms": 1}'},
                  {"type": "function_call_output", "call_id": "c1", "output": "aborted"}]}}  # fmt: skip
    _write_trace(d, [(first, _sse_response("r1", sse)), (second, None)])
    _, v = run_verdict(os.path.join(d, "v"), [d], codex, "window", "conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        report = json.load(fh)
    results.append({"case": "X-04", "ok": report["outcome_classes"].get("aborted") == 1})
    # X-05: a harness profile without measured outcomes blocks the harness-outcome claim.
    unknown = write_profile(
        os.path.join(work, "x05.yaml"), dict(CONTROL_PROFILES["responses"], outcomes="TODO")
    )
    _, v = run_verdict(os.path.join(d, "v5"), [d], unknown, "window", "harness-outcome")
    results.append({"case": "X-05", "ok": v["claims"]["harness-outcome"] == "BLOCKED"
                    and any("outcomes are TODO" in b for b in v["claim_blockers"]["harness-outcome"])})  # fmt: skip
    # X-06: a completed reply with neither text nor a call is a vacuous turn.
    d = os.path.join(work, "x06")
    blank = _anthropic_sse([{"type": "text", "text": "  \n"}], "end_turn")
    _write_trace(
        d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", blank))]
    )
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append({"case": "X-06", "ok": "M-06" in v["tripped"]})
    # X-08: a client disconnect is not a server wire failure; H-03 attributes it.
    d = os.path.join(work, "x08")
    cut = _anthropic_sse([{"type": "text", "text": "partial"}]).split("event: message_delta")[0]
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                      _sse_response("t1", cut, status="client_disconnected"))])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        report = json.load(fh)
    results.append({"case": "X-08", "ok": "C-04" not in v["tripped"]
                    and report["counts"].get("H-03:client_disconnected") == 1})  # fmt: skip
    # X-50: the recorder's own 502 while the server was down, and a stream the recorder lost
    # mid-reply. Neither is the server's reply to judge: R-06 counts both, C-01 and C-04 do not.
    d = os.path.join(work, "x50")
    gone = {"event": "response", "trace_id": "t1", "status": "upstream_error", "http_status": 502,
            "response": {"kind": "text", "body": "ServerDisconnectedError"}}  # fmt: skip
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), gone),
                     (_anth_request("t2", [user], "2026-01-01T00:00:02Z"),
                      _sse_response("t2", cut, status="upstream_error"))])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "smoke", "conformance,runtime")
    checks = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-50", "ok": checks["C-01"]["status"] == "PASS"
                    and checks["C-04"]["status"] == "PASS" and checks["R-06"]["status"] == "FAIL"
                    and checks["R-06"]["count"] == 2})  # fmt: skip
    # X-51: a request with no response record (the capture lost it): inventory is BLOCKED.
    # X-52: the recorder stopped before the reply: pending at the capture's end, not judged, and
    # inventory is not blocked by it.
    done = _sse_response("t1", _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn"))
    stopped = {"event": "response", "trace_id": "t2", "status": "recorder_shutdown",
               "http_status": None, "response": {"kind": "none", "body": None}}  # fmt: skip
    for case, second in (("X-51", None), ("X-52", stopped)):
        d = os.path.join(work, case.lower())
        _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), done),
                         (_anth_request("t2", [user], "2026-01-01T00:00:02Z"), second)])  # fmt: skip
        _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "inventory,conformance")
        with open(os.path.join(d, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        lost = counts.get("I-01:request_without_response", 0)
        if case == "X-51":
            ok = lost == 1 and v["claims"]["inventory"] == "BLOCKED"
        else:
            ok = (lost == 0 and counts.get("inventory:pending_at_capture_end") == 1
                  and v["claims"]["inventory"] == "PASS" and "C-04" not in v["tripped"])  # fmt: skip
        results.append({"case": case, "ok": ok})
    # X-53: the server rejects a capability with the wording the harness profile declares; the
    # harness retries without it. C-16 is flagged, and conformance still passes.
    # X-54: the same rejection with a wording the profile does not declare: C-16 FAIL.
    retrying = write_profile(os.path.join(work, "x53.yaml"), dict(CONTROL_PROFILES["anthropic"], errors=dict(
        CONTROL_PROFILES["anthropic"].get("errors") or {}, capability_retries=[
            {"field": "thinking.display", "wording": "Unknown parameter: ... display"}])))  # fmt: skip
    for case, message, status in (
        ("X-53", "Unknown parameter: 'thinking.adaptive.display'", "FLAGGED"),
        ("X-54", "thinking.display is not supported", "FAIL"),
    ):
        d = os.path.join(work, case.lower())
        rejected = {"event": "response", "trace_id": "t1", "status": "completed", "http_status": 400,
                    "response": {"kind": "json", "body": {"type": "error", "error": {
                        "type": "invalid_request_error", "message": message}}}}  # fmt: skip
        asked = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        asked["body"]["thinking"] = {"type": "adaptive", "display": "updates"}
        retried = _anth_request("t2", [user], "2026-01-01T00:00:02Z")  # the same, without the field
        retried["body"]["thinking"] = {"type": "adaptive"}
        _write_trace(d, [(asked, rejected), (retried, done | {"trace_id": "t2"})])
        _, v = run_verdict(os.path.join(d, "v"), [d], retrying, "window", "conformance")
        c16 = next(r for r in v["checks"] if r["id"] == "C-16")
        ok = c16["status"] == status and "H-02" not in v["tripped"]
        if status == "FLAGGED":
            ok = ok and c16["declared_fields"] == {"thinking.display": 1} and "C-16" in v["tripped"] \
                and v["claims"]["conformance"] == "PASS"  # fmt: skip
        results.append({"case": case, "ok": ok})
    # X-55: an operator probe (a self-test through the recorder) is not the cell's traffic.
    d = os.path.join(work, "x55")
    probe = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
    probe["headers"] = [["x-mh-probe", "1"]]
    _write_trace(d, [(probe, done), (_anth_request("t2", [user], "2026-01-01T00:00:02Z"),
                                     done | {"trace_id": "t2"})])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "inventory,conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        counts = json.load(fh)["counts"]
    results.append({"case": "X-55", "ok": counts.get("inventory:operator_probes") == 1
                    and counts.get("den:replies") == 1 and v["claims"]["inventory"] == "PASS"})  # fmt: skip
    # X-56: count_tokens rejected with a wording the profile does not declare: C-16 FAIL.
    d = os.path.join(work, "x56")
    counting = {"event": "request", "trace_id": "c1", "route": "/v1/messages/count_tokens",
                "recorded_at": "2026-01-01T00:00:01Z", "headers": [["x-claude-code-session-id", "s1"]],
                "body": {"model": "x", "messages": [user]}}  # fmt: skip
    refused = {"event": "response", "trace_id": "c1", "status": "completed", "http_status": 400,
               "response": {"kind": "json", "body": {"type": "error", "error": {
                   "type": "invalid_request_error", "message": "unknown field"}}}}  # fmt: skip
    _write_trace(d, [(counting, refused), (_anth_request("t2", [user], "2026-01-01T00:00:02Z"),
                                           done | {"trace_id": "t2"})])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    c16 = next(r for r in v["checks"] if r["id"] == "C-16")
    results.append({"case": "X-56", "ok": c16["status"] == "FAIL" and c16["count"] == 1})
    # X-57: two clients gave up before the first byte, after 35 s and after 5 s, against a 30 s
    # first-byte cutoff: the first saw the cutoff pass (R-01), the second left early.
    # X-58: each cut is attributed to the client in an H-03 check record (a flag).
    timed = write_profile(os.path.join(work, "x57.yaml"), dict(
        CONTROL_PROFILES["anthropic"], timeouts={"first_event_s": 30, "idle_s": 30}))  # fmt: skip
    d = os.path.join(work, "x57")
    gave_up = [{"event": "response", "trace_id": t, "status": "client_disconnected", "http_status": None,
                "response": {"kind": "none", "body": None},
                "timing": {"client_error": "CancelledError", "before_reply": True, "total_s": w}}
               for t, w in (("t1", 35.0), ("t2", 5.0))]  # fmt: skip
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), gave_up[0]),
                     (_anth_request("t2", [user], "2026-01-01T00:00:40Z"), gave_up[1])])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], timed, "smoke", "runtime,harness-outcome")
    checks = {r["id"]: r for r in v["checks"]}
    with open(os.path.join(d, "v", "report.json")) as fh:
        counts = json.load(fh)["counts"]
    results.append({"case": "X-57", "ok": checks["R-01"]["status"] == "FAIL" and checks["R-01"]["count"] == 1
                    and counts.get("den:cuts_before_reply") == 2})  # fmt: skip
    results.append(
        {
            "case": "X-58",
            "ok": checks["H-03"]["status"] == "FLAGGED" and checks["H-03"]["count"] == 2,
        }
    )
    # X-59: a window's last request is answered in the next hour. Without that hour's responses
    # inventory is BLOCKED (I-01); with them (--responses) the request is answered, and a
    # response there for a request outside the window is ignored.
    d = os.path.join(work, "x59")
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:59:59Z"), None)])
    later = os.path.join(work, "x59-next")
    os.makedirs(later, exist_ok=True)
    with open(os.path.join(later, "responses-x.jsonl"), "w") as fh:
        fh.write(json.dumps(done) + "\n" + json.dumps(done | {"trace_id": "elsewhere"}) + "\n")
    _, alone = run_verdict(os.path.join(d, "v1"), [d], anth, "window", "inventory")
    _, joined = run_verdict(
        os.path.join(d, "v2"), [d], anth, "window", "inventory", responses=[later]
    )
    with open(os.path.join(d, "v2", "report.json")) as fh:
        counts = json.load(fh)["counts"]
    results.append({"case": "X-59", "ok": alone["claims"]["inventory"] == "BLOCKED"
                    and joined["claims"]["inventory"] == "PASS"
                    and counts.get("inventory:boundary_responses_joined") == 1
                    and counts.get("inventory:boundary_responses_ignored") == 1})  # fmt: skip
    # X-60: a model profile that names no degenerate tokens leaves M-02 blind to the model's own
    # runaway output: BLOCKED, not PASS, even under a lenient threshold.
    # X-61: the same trace judged with a profile that names them (K3's): PASS.
    d = os.path.join(work, "x60")
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(
        [{"type": "text", "text": "Done."}], "end_turn")))])  # fmt: skip
    lenient = os.path.join(work, "x60-matrix.yaml")
    with open(lenient, "w") as fh:
        json.dump(
            {
                "thresholds": {
                    "model_behavior": {"degenerate_rate": 0.9},
                    "unit_for_rates": "session",
                }
            },
            fh,
        )
    # X-60b: a profile that names its tokens by id only (GLM's) cannot see them without raw
    # token ids: BLOCKED too, with that reason.
    with open(GLM_PROFILE) as fh:
        blind = dict(yaml.safe_load(fh), degenerate=[])
    blind_profile = write_profile(os.path.join(work, "x60-blind.yaml"), blind)
    for case, model, status, reason in (("X-60", blind_profile, "BLOCKED", "no degenerate"),
                                        ("X-60b", GLM_PROFILE, "BLOCKED", "by id only"),
                                        ("X-61", K3_PROFILE, "PASS", None)):  # fmt: skip
        _, v = run_verdict(os.path.join(d, case), [d], anth, "window", "model-behavior", model=model,
                           matrix=lenient)  # fmt: skip
        m02 = next(r for r in v["checks"] if r["id"] == "M-02")
        ok = m02["status"] == status and (reason is None or reason in (m02["reason"] or ""))
        results.append({"case": case, "ok": ok})
    # X-10: zero events in 20 sessions does not establish a rate below 1%.
    bounds = verdict_mod.wilson(0, 20)
    results.append({"case": "X-10", "ok": bounds is not None and bounds[1] > 0.01})
    return (results + raw_cases(work, anth) + shared_engine_id_cases(work, anth) + echo_cases(work, anth)
            + unechoed_by_design_cases(work, anth) + rewrite_cases(work, anth)
            + usage_cases(work, anth) + protocol_diff_cases(work)
            + pairing_cases(work, anth) + applicability_cases(work, anth)
            + rate_unit_cases(work, anth) + cut_bound_cases(work)
            + outcome_and_signal_cases(work, anth) + responses_reasoning_echo_cases(work)
            + full_tier_fix_cases(work) + degenerate_stop_cases(work, anth)
            + stopped_raw_cases(work, anth) + revision_cases(work, anth)
            + harness_error_cases(work) + pi_profile_cases(work)
            + glm_codex_full_tier_cases(work) + pi_edit_cases(work)
            + k3_codex_full_tier_cases(work) + pi_reserialize_cases(work)
            + pi_trials_full_tier_cases(work))  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="responses,anthropic,chat,cross")
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    args = ap.parse_args(argv)
    only = set(args.only.split(","))
    os.makedirs(args.out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="mh-controls-")
    results, cross = [], []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        if "responses" in only:
            results += protocol_matrix("responses", RESPONSES_EXPECT, responses_body,
                                       [PY, os.path.join(HERE, "fake_responses_server.py")], work)  # fmt: skip
        if "anthropic" in only:
            results += protocol_matrix("anthropic", ANTHROPIC_EXPECT, anthropic_body,
                                       [PY, os.path.join(HERE, "fake_agent_server.py"),
                                        "--gap-seconds", "1.5"], work)  # fmt: skip
        if "chat" in only:
            results += protocol_matrix("chat", CHAT_EXPECT, chat_body,
                                       [PY, os.path.join(HERE, "fake_agent_server.py"),
                                        "--gap-seconds", "1.5"], work)  # fmt: skip
        if "cross" in only:
            cross = cross_cases(work)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    passed = all(r["ok"] for r in results + cross) and bool(results or cross)
    receipt = {
        "schema": "trtllm-serve-model-harness-validation/controls@1",
        "validator_tree_sha256": verdict_mod.tree_sha256(),
        "sdk_versions": verdict_mod.sdk_diff.versions(),
        "started_at": started,
        "scope": sorted(only),
        "complete_scope": only >= {"responses", "anthropic", "chat", "cross"},
        "results": results,
        "cross_cases": cross,
        "passed": passed and only >= {"responses", "anthropic", "chat", "cross"},
    }
    with open(os.path.join(args.out_dir, "controls.json"), "w") as fh:
        json.dump(receipt, fh, indent=1)
    for r in results:
        print(
            f"{'OK ' if r['ok'] else 'BAD'} {r['protocol']:9s} {r['mode']:28s} "
            f"expected={r.get('expected')} tripped={r.get('tripped')} {r.get('error', '')}"
        )
    for r in cross:
        print(f"{'OK ' if r['ok'] else 'BAD'} cross     {r['case']}")
    print(json.dumps({"passed": receipt["passed"], "cases": len(results) + len(cross),
                      "failed": [f"{r.get('protocol', 'cross')}:{r.get('mode', r.get('case'))}"
                                 for r in results + cross if not r["ok"]]}))  # fmt: skip
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
