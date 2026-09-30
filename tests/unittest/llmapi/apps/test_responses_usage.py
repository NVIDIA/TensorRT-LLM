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
"""Offline tests for the Responses API usage block.

The usage block was never populated, so every response reported no token
consumption at all. Agentic clients use it to track how much of the context
window a conversation has spent and to decide when to compact; without it a
long session keeps appending turns until it overflows the context window.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.responses_utils import _create_usage

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _generation(prompt_tokens=7, completion_tokens=3, cached_tokens=0):
    return SimpleNamespace(
        prompt_token_ids=list(range(prompt_tokens)),
        outputs=[SimpleNamespace(token_ids=list(range(completion_tokens)))],
        cached_tokens=cached_tokens,
    )


def test_usage_counts_prompt_and_generated_tokens():
    usage = _create_usage(_generation(prompt_tokens=7, completion_tokens=3))
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    assert usage.total_tokens == 10


def test_usage_reports_cached_tokens():
    usage = _create_usage(_generation(cached_tokens=5))
    assert usage.input_tokens_details.cached_tokens == 5


def test_usage_defaults_cached_tokens_when_backend_omits_them():
    """cached_tokens is absent on some result types; it must not raise."""
    result = _generation()
    del result.cached_tokens
    assert _create_usage(result).input_tokens_details.cached_tokens == 0


def test_usage_sums_every_output_sequence():
    result = _generation(completion_tokens=3)
    result.outputs.append(SimpleNamespace(token_ids=[1, 2]))
    assert _create_usage(result).output_tokens == 5


def test_usage_is_omitted_without_prompt_tokens():
    """Reporting zero would look like a real count of no tokens."""
    result = _generation()
    result.prompt_token_ids = None
    assert _create_usage(result) is None


# ---------------------------------------------------------------------------
# Results handed to a postprocessing worker
# ---------------------------------------------------------------------------


def test_usage_uses_the_prompt_token_count_supplied_by_the_caller():
    """Regression: usage was null for every request on the served path.

    Postprocessing runs in a separate worker, and the result it receives has
    no link back to the request that produced it, so its prompt tokens are
    unreachable. The executor records the count on the postprocessing
    arguments instead, and that is what has to be used.
    """
    result = _generation(prompt_tokens=7, completion_tokens=3)
    del result.prompt_token_ids
    usage = _create_usage(result, num_prompt_tokens=11)
    assert usage.input_tokens == 11
    assert usage.total_tokens == 14


def test_supplied_prompt_token_count_wins_over_the_result():
    result = _generation(prompt_tokens=7)
    assert _create_usage(result, num_prompt_tokens=11).input_tokens == 11


# ---------------------------------------------------------------------------
# The usage block has to survive being streamed
# ---------------------------------------------------------------------------


def test_usage_validates_inside_a_streamed_completion_event():
    """Regression: a usage block the SDK rejects silently truncates the stream.

    response.completed embeds the response, and the SDK model re-validates it.
    A missing field raises while the response is already being streamed, so
    the client gets deltas and then nothing - no terminating event, no error -
    and waits indefinitely. Building the event is what catches this; checking
    our own model would not, since ours is what was wrong.
    """
    from openai.types.responses import ResponseCompletedEvent

    usage = _create_usage(_generation(prompt_tokens=7, completion_tokens=3, cached_tokens=6))
    event = ResponseCompletedEvent(
        type="response.completed",
        sequence_number=0,
        response={
            "id": "resp_1",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage.model_dump(),
        },
    )
    assert event.response.usage.input_tokens == 7
    assert event.response.usage.input_tokens_details.cached_tokens == 6


# ---------------------------------------------------------------------------
# reasoning_tokens
#
# Reported as a hardcoded zero until now. The count is rebuilt from the text
# the reasoning parser itself claimed, rather than by searching the generated
# token ids for `</think>`, and these tests are the reason: GLM breaks a
# marker search three separate ways under agent workloads, and each of them
# was observed live before it was tested here.
# ---------------------------------------------------------------------------


class _WordTokenizer:
    """One token per whitespace-separated word. Enough to count with."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


def _reasoning_tokens(text, parser="glm", tokenizer=_WordTokenizer(), tools=None):
    """Run the real parser, then count what it called reasoning."""
    from tensorrt_llm.serve.responses_utils import _count_reasoning_tokens, _create_output_content

    _items, _messages, reasoning_texts = _create_output_content(
        SimpleNamespace(outputs=[SimpleNamespace(index=0, text=text)]),
        reasoning_parser=parser,
        tool_parser=None,
        tools=tools,
    )
    return _count_reasoning_tokens(tokenizer, reasoning_texts, 10_000)


def test_reasoning_tokens_counts_up_to_the_closing_tag():
    # The ordinary shape: reasoning, close, answer.
    assert _reasoning_tokens("one two three</think>four five") == 3


def test_reasoning_tokens_stop_at_the_first_close_not_the_last():
    # GLM closes the block more than once (reasoning_parser.py:370-388). The
    # parser splits at the first close and drops the stray one; a search for
    # `</think>` in token space that took the last match -- which is what
    # ThinkingBudgetLogitsProcessor does, for its own good reasons -- would
    # count the middle stretch as reasoning too.
    text = "one two</think>three four five</think><tool_call>x</tool_call>"
    assert _reasoning_tokens(text) == 2


def test_reasoning_tokens_end_at_an_implicit_tool_call():
    # Observed live: 26,055 characters over 169 frames with no closing tag at
    # all and one well-formed <tool_call>. Counting to the missing `</think>`
    # would have called the entire turn reasoning.
    text = "planning the call<tool_call>{}</tool_call>"
    assert _reasoning_tokens(text) == 3


def test_reasoning_tokens_covers_a_turn_that_never_leaves_the_block():
    # `<think>` is prefilled by the template, so output starts inside the
    # block; with no terminator the whole turn really is reasoning.
    assert _reasoning_tokens("still thinking about it") == 4


def test_reasoning_tokens_are_zero_without_a_reasoning_parser():
    assert _reasoning_tokens("one two three", parser=None) == 0


def test_reasoning_tokens_are_zero_without_a_tokenizer():
    assert _reasoning_tokens("one two three", tokenizer=None) == 0


def test_reasoning_tokens_never_exceed_generated_tokens():
    # Re-encoding a detokenized substring need not reproduce the tokenization
    # it came from, and usage claiming more reasoning than output is visibly
    # wrong to a client budgeting a context window.
    from tensorrt_llm.serve.responses_utils import _count_reasoning_tokens

    assert _count_reasoning_tokens(_WordTokenizer(), ["a b c d e"], 3) == 3


def test_usage_reports_the_reasoning_tokens_it_counts():
    usage = _create_usage(
        _generation(prompt_tokens=7, completion_tokens=9),
        tokenizer=_WordTokenizer(),
        reasoning_texts=["one two three four"],
    )
    assert usage.output_tokens_details.reasoning_tokens == 4
    # Reasoning tokens are part of the generated total, not additional to it.
    assert usage.output_tokens == 9
    assert usage.total_tokens == 16


def test_usage_still_defaults_to_zero_reasoning_tokens():
    usage = _create_usage(_generation())
    assert usage.output_tokens_details.reasoning_tokens == 0


# ---------------------------------------------------------------------------
# input_image degradation
# ---------------------------------------------------------------------------


def test_an_image_part_degrades_to_a_text_placeholder():
    """Regression: an input_image part 400'd the whole request.

    Codex attaches screenshots as `input_image` (base64). The input union has
    no image member, so validation fell through to ResponseInputTextParam and
    rejected the request -- deterministically, so the client's retries all
    failed and the campaign died on its backoff limit. Measured: 2 of the
    first 182 Kernel-Trace campaigns, each with the full base64 body echoed
    into the stop reason.
    """
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    req = ResponsesRequest(
        model="m",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look at this"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            },
            # And as a bare top-level part, which clients also send.
            {"type": "input_image", "image_url": "data:image/png;base64,BB"},
        ],
    )
    dumped = req.model_dump()["input"]
    texts = []
    for item in dumped:
        parts = item.get("content") if isinstance(item.get("content"), list) else [item]
        for p in parts:
            assert p.get("type") != "input_image", "image part survived"
            if isinstance(p.get("text"), str):
                texts.append(p["text"])
    assert any("image omitted" in t and "image/png" in t for t in texts)
    # The placeholder must not carry the base64 payload.
    assert not any("AAAA" in t for t in texts)


def test_an_image_in_a_tool_output_degrades_too():
    """Regression: tool results carry parts under "output", not "content".

    An agent that plots something gets the PNG back through the tool: Codex
    sends `custom_tool_call_output` whose `output` list ends with an
    `input_image` part. The first degrade pass only walked `content` and the
    top level, so this shape still 400'd the whole request -- measured at 114
    Kernel-Trace campaigns in the 24h after the content/top-level fix went
    live (2026-09-23). Shape taken verbatim from a traced rejected request.
    """
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    req = ResponsesRequest(
        model="m",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "plot it"},
                ],
            },
            {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_1",
                "name": "exec",
                "input": "python plot.py",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "Script completed\n"},
                    {"type": "input_text", "text": "layout-a2 image"},
                    {"type": "input_image", "image_url": "data:image/png;base64,CCCC"},
                ],
            },
        ],
    )
    out = req.model_dump()["input"][2]["output"]
    assert all(p.get("type") != "input_image" for p in out), (
        "image part survived inside tool output"
    )
    joined = " ".join(p.get("text", "") for p in out)
    assert "image omitted" in joined and "image/png" in joined
    assert "CCCC" not in joined
    # A plain-string tool output must pass through untouched.
    req2 = ResponsesRequest(
        model="m",
        input=[
            {
                "type": "custom_tool_call",
                "id": "ctc_2",
                "call_id": "call_2",
                "name": "exec",
                "input": "true",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_2",
                "output": "Script completed\n",
            },
        ],
    )
    assert req2.model_dump()["input"][1]["output"] == "Script completed\n"
