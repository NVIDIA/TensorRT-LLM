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
"""Harmony streaming output must not depend on how tokens are batched.

Speculative decoding (e.g. Eagle3) delivers several accepted tokens per step,
so the same completion reaches the streaming adapter in batches of varying
size. The client-visible result (reasoning, content, and reassembled tool
calls) must equal the one produced when tokens arrive one at a time, both for
well-formed completions and for malformed ones the parser rejects.
"""

import json
import os

import pytest
from utils.llm_data import llm_datasets_root

from tensorrt_llm.serve import harmony_adapter
from tensorrt_llm.serve.harmony_adapter import HarmonyAdapter

pytestmark = pytest.mark.cpu_only

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
TOOL_CALL = ("get_weather", '{"city":"Paris","unit":"c"}')

_ANALYSIS = "<|channel|>analysis<|message|>User wants weather. Call tool.<|end|>"
_CALL = (
    "<|start|>assistant<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"city":"Paris","unit":"c"}<|call|>'
)

_FINAL = "<|start|>assistant<|channel|>final<|message|>It is sunny.<|return|>"

# A header the parser rejects at <|message|>: the channel marker has no name.
_REJECTED_HEADER = "<|start|>assistant<|channel|><|message|>"

# name -> (harmony completion text, expected tool calls, expected content)
CASES = {
    "tool_call": (_ANALYSIS + _CALL, [TOOL_CALL], ""),
    "preamble_then_tool_call": (
        _ANALYSIS + "<|start|>assistant<|channel|>commentary<|message|>Checking.<|end|>" + _CALL,
        [TOOL_CALL],
        # Commentary preambles keep their channel markers in the content stream.
        "<|channel|>commentary<|message|>Checking.<|end|>",
    ),
    "tool_call_on_analysis_channel": (
        _ANALYSIS + _CALL.replace("commentary to=", "analysis to="),
        [TOOL_CALL],
        "",
    ),
    "final_only": (_ANALYSIS + _FINAL, [], "It is sunny."),
    # Malformed: the parser raises partway through. Tokens accepted before the
    # rejected one must still be streamed regardless of batch boundaries.
    "missing_start": (_ANALYSIS + _CALL.replace("<|start|>assistant", ""), [], ""),
    "broken_channel_header": (
        _ANALYSIS + _CALL.replace("<|channel|>commentary", " commentary"),
        [],
        "",
    ),
}

# Same layout as CASES. The parser rejects part of the completion, drops it and
# resumes at the next <|start|>, so a complete message after the rejected part
# must still reach the client. Streaming only: the non-streaming parser fails on
# the whole completion instead of resuming.
RECOVERY_CASES = {
    "stray_channel_then_final": (_ANALYSIS + "<|channel|>" + _FINAL, [], "It is sunny."),
    "rejected_header_then_final": (_ANALYSIS + _REJECTED_HEADER + _FINAL, [], "It is sunny."),
    "rejected_header_then_tool_call": (_ANALYSIS + _REJECTED_HEADER + _CALL, [TOOL_CALL], ""),
    # No <|start|> follows the rejected header, so the rest of that message is
    # dropped at every batch size.
    "rejected_header_not_restarted": (
        _ANALYSIS + _REJECTED_HEADER + "final<|message|>It is sunny.<|return|>",
        [],
        "",
    ),
}
STREAMING_CASES = {**CASES, **RECOVERY_CASES}

# name -> (harmony completion text, expected (channel, recipient, text) messages).
# A stray control token sits between two valid messages; the parser rejects it,
# and the message after it must still be parsed.
MESSAGE_CASES = {
    "stray_channel_before_final": (
        _ANALYSIS + "<|channel|>" + _FINAL,
        [("analysis", None, "User wants weather. Call tool."), ("final", None, "It is sunny.")],
    ),
    "stray_message_before_final": (
        _ANALYSIS + "<|message|>" + _FINAL,
        [("analysis", None, "User wants weather. Call tool."), ("final", None, "It is sunny.")],
    ),
    "stray_channel_before_tool_call": (
        _ANALYSIS + "<|channel|>" + _CALL,
        [
            ("analysis", None, "User wants weather. Call tool."),
            ("commentary", "functions.get_weather", TOOL_CALL[1]),
        ],
    ),
    "rejected_header_then_final": (
        _ANALYSIS + _REJECTED_HEADER + _FINAL,
        [("analysis", None, "User wants weather. Call tool."), ("final", None, "It is sunny.")],
    ),
    # No <|start|> follows the rejected header, so the rest of that message is dropped.
    "rejected_header_not_restarted": (
        _ANALYSIS + _REJECTED_HEADER + "final<|message|>It is sunny.<|return|>",
        [("analysis", None, "User wants weather. Call tool.")],
    ),
}


@pytest.fixture(scope="module")
def _tiktoken_local_cache():
    """Point openai_harmony at the local vocab before any adapter is built."""
    cache_dir = os.path.join(llm_datasets_root(), "tiktoken_vocab")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("TIKTOKEN_RS_CACHE_DIR", cache_dir)
        yield


@pytest.fixture(scope="module")
def adapter(_tiktoken_local_cache):
    """Return one HarmonyAdapter shared by the module, loaded from the local vocab."""
    return HarmonyAdapter()


def _stream(adapter, tokens, batch_sizes):
    """Stream ``tokens`` in the given batch sizes and reassemble like a client."""
    request_id = "batch-invariance"
    adapter.cleanup_stream_state(request_id)
    reasoning, content, tools = "", "", {}
    start = 0
    try:
        for size in batch_sizes:
            batch = tokens[start : start + size]
            start += size
            if not batch:
                break
            responses, should_stop = adapter.create_openai_streaming_response(
                request_id, batch, TOOLS, tool_choice="auto"
            )
            for response in responses:
                delta = json.loads(response.removeprefix("data: "))["choices"][0]["delta"]
                reasoning += delta.get("reasoning_content") or ""
                content += delta.get("content") or ""
                for tool_call in delta.get("tool_calls") or []:
                    entry = tools.setdefault(tool_call["index"], {"name": None, "args": ""})
                    function = tool_call.get("function") or {}
                    entry["name"] = entry["name"] or function.get("name")
                    entry["args"] += function.get("arguments") or ""
            if should_stop:
                break
    finally:
        adapter.cleanup_stream_state(request_id)
    return {
        "reasoning": reasoning,
        "content": content,
        "tool_calls": sorted((t["name"], t["args"]) for t in tools.values()),
    }


def _schedules(num_tokens):
    """Batchings that put every token boundary, incl. control tokens, mid-batch."""
    for batch_size in range(2, 7):
        for offset in range(batch_size):
            head = [offset] if offset else []
            yield f"bs{batch_size}_off{offset}", head + [batch_size] * num_tokens
    for split in range(1, num_tokens):
        yield f"split{split}", [split, num_tokens]
    yield "single_batch", [num_tokens]


def _stream_messages(adapter, tokens, batch_sizes):
    """Feed ``tokens`` to the Responses-API messages path in the given batch sizes."""
    request_id = "batch-invariance-messages"
    adapter.cleanup_stream_state(request_id)
    messages = []
    start = 0
    try:
        for size in batch_sizes:
            batch = tokens[start : start + size]
            start += size
            if batch:
                messages = adapter.stateful_stream_harmony_tokens_to_openai_messages(
                    request_id, batch, TOOLS, tool_choice="auto"
                )
    finally:
        adapter.cleanup_stream_state(request_id)
    return [(m.channel, m.recipient, "".join(part.text for part in m.content)) for m in messages]


@pytest.mark.parametrize("case", list(STREAMING_CASES))
def test_streaming_is_batch_size_invariant(adapter, case):
    """Every batching yields the batch-size-1 output, which matches the expected one."""
    text, expected_tool_calls, expected_content = STREAMING_CASES[case]
    tokens = adapter.encoding.encode(text, allowed_special="all")

    reference = _stream(adapter, tokens, [1] * len(tokens))
    assert reference["tool_calls"] == expected_tool_calls
    assert reference["reasoning"] == "User wants weather. Call tool."
    assert reference["content"] == expected_content

    mismatches = {
        label: result
        for label, schedule in _schedules(len(tokens))
        if (result := _stream(adapter, tokens, schedule)) != reference
    }
    assert not mismatches, f"batch-dependent output vs batch size 1: {mismatches}"


@pytest.mark.parametrize("case", list(CASES))
def test_streaming_matches_non_streaming_tool_calls(adapter, case):
    """Streaming and non-streaming parsing agree on the tool calls produced."""
    text, expected_tool_calls, _ = CASES[case]
    tokens = adapter.encoding.encode(text, allowed_special="all")

    parsed = adapter.harmony_output_to_openai(tokens, TOOLS, "auto")
    non_streaming = sorted(
        (tc["function"]["name"], tc["function"]["arguments"])
        for tc in parsed.get("tool_calls") or []
    )
    assert non_streaming == expected_tool_calls
    assert _stream(adapter, tokens, [len(tokens)])["tool_calls"] == expected_tool_calls


@pytest.mark.parametrize("case", list(MESSAGE_CASES))
def test_messages_recover_after_rejected_token(adapter, case):
    """The messages path keeps parsing after a rejected token, at any batch size."""
    text, expected_messages = MESSAGE_CASES[case]
    tokens = adapter.encoding.encode(text, allowed_special="all")

    assert _stream_messages(adapter, tokens, [1] * len(tokens)) == expected_messages
    mismatches = {
        label: result
        for label, schedule in _schedules(len(tokens))
        if (result := _stream_messages(adapter, tokens, schedule)) != expected_messages
    }
    assert not mismatches, f"batch-dependent messages: {mismatches}"


@pytest.mark.parametrize("stream", [_stream, _stream_messages], ids=["deltas", "messages"])
def test_rejected_tokens_are_reported(adapter, monkeypatch, stream):
    """Both streaming paths log an error for a batch with a rejected token, and only then."""
    errors = []
    monkeypatch.setattr(harmony_adapter.logger, "error", errors.append)

    tokens = adapter.encoding.encode(_ANALYSIS + _FINAL, allowed_special="all")
    stream(adapter, tokens, [len(tokens)])
    assert not errors

    tokens = adapter.encoding.encode(_ANALYSIS + "<|channel|>" + _FINAL, allowed_special="all")
    stream(adapter, tokens, [len(tokens)])
    assert len(errors) == 1
    assert f"rejected 1 of {len(tokens)} token(s)" in errors[0]
