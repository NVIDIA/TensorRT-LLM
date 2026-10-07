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

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_protocol import PromptTokensDetails, ResponsesRequest, UsageInfo
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingProcessor,
    _count_reasoning_tokens,
    _create_output_content,
    _create_usage,
)

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
# Disaggregated serving: the context phase's usage comes with the handoff
# ---------------------------------------------------------------------------


def _ctx_usage(prompt_tokens, cached_tokens):
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=1,
        total_tokens=prompt_tokens + 1,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
    )


def _handed_off(ctx_usage, prompt_tokens=100, completion_tokens=5):
    """A generation-worker result: the whole prompt arrived as transferred KV."""
    result = _generation(prompt_tokens, completion_tokens, cached_tokens=prompt_tokens)
    result.outputs[0].disaggregated_params = SimpleNamespace(ctx_usage=ctx_usage)
    return result


@pytest.mark.parametrize(
    "ctx_usage, input_tokens, cached_tokens",
    [
        (_ctx_usage(96, 32), 96, 32),
        (_ctx_usage(100, 0), 100, 0),
        (_ctx_usage(100, 40).model_dump(), 100, 40),
        (UsageInfo(prompt_tokens=100, completion_tokens=1, total_tokens=101), 100, 0),
    ],
    ids=["warm", "cold", "as_dict", "no_details"],
)
def test_the_context_phase_decides_prompt_and_cached_tokens(ctx_usage, input_tokens, cached_tokens):
    usage = _create_usage(_handed_off(ctx_usage), num_prompt_tokens=100)
    assert usage.input_tokens == input_tokens
    assert usage.input_tokens_details.cached_tokens == cached_tokens
    assert usage.output_tokens == 5


# ---------------------------------------------------------------------------
# reasoning_tokens: re-encoded from the text the reasoning parser claimed
# ---------------------------------------------------------------------------


class _WordTokenizer:
    """One token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.mark.parametrize(
    "text, parser, tokenizer, expected",
    [
        ("one two three</think>four five", "glm", _WordTokenizer(), 3),
        ("one two</think>three four</think><tool_call>x</tool_call>", "glm", _WordTokenizer(), 2),
        ("planning the call<tool_call>{}</tool_call>", "glm", _WordTokenizer(), 3),
        ("still thinking about it", "glm", _WordTokenizer(), 4),
        ("one two three", None, _WordTokenizer(), 0),
        ("one two three", "glm", None, 0),
    ],
    ids=["closed", "first_close_counts", "implicit_end", "never_closed", "no_parser", "no_tok"],
)
def test_reasoning_tokens_count_what_the_parser_called_reasoning(text, parser, tokenizer, expected):
    _items, _messages, reasoning_texts = _create_output_content(
        SimpleNamespace(outputs=[SimpleNamespace(index=0, text=text)]), reasoning_parser=parser
    )
    assert _count_reasoning_tokens(tokenizer, reasoning_texts, 10_000) == expected


def test_reasoning_tokens_never_exceed_generated_tokens():
    assert _count_reasoning_tokens(_WordTokenizer(), ["a b c d e"], 3) == 3


def test_usage_reports_reasoning_tokens_within_the_output_tokens():
    usage = _create_usage(
        _generation(prompt_tokens=7, completion_tokens=9),
        tokenizer=_WordTokenizer(),
        reasoning_texts=["one two three four"],
    )
    assert usage.output_tokens_details.reasoning_tokens == 4
    assert (usage.output_tokens, usage.total_tokens) == (9, 16)
    assert _create_usage(_generation()).output_tokens_details.reasoning_tokens == 0


def test_a_streamed_response_counts_reasoning_tokens():
    request = ResponsesRequest(model="m", input="hi", stream=True)
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="m",
        use_harmony=False,
        reasoning_parser="glm",
    )
    output = SimpleNamespace(
        index=0,
        text="one two</think>answer",
        text_diff="one two</think>answer",
        finish_reason="stop",
        token_ids=[1, 2, 3],
        disaggregated_params=None,
    )
    result = SimpleNamespace(outputs=[output], _done=True, prompt_token_ids=[1], cached_tokens=0)
    processor.process_single_output(result)
    frame = processor.get_final_response_non_store(result, tokenizer=_WordTokenizer())
    usage = json.loads(frame.split("data: ", 1)[1])["response"]["usage"]
    assert usage["output_tokens_details"]["reasoning_tokens"] == 2


# ---------------------------------------------------------------------------
# input_image parts degrade to a text placeholder
# ---------------------------------------------------------------------------

_PNG = "data:image/png;base64,AAAA"


@pytest.mark.parametrize(
    "item, key",
    [
        (
            {"role": "user", "content": [{"type": "input_image", "image_url": _PNG}]},
            "content",
        ),
        (
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": {"url": _PNG}}],
            },
            "content",
        ),
        (
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": [{"type": "input_image", "image_url": _PNG}],
            },
            "output",
        ),
        ({"type": "input_image", "image_url": _PNG}, None),
    ],
    ids=["message_part", "url_object", "tool_output", "top_level"],
)
def test_an_image_degrades_to_a_text_placeholder(item, key):
    request = ResponsesRequest(model="m", input=[item])
    dumped = request.model_dump()["input"][0]
    (part,) = dumped[key] if key else [dumped]
    assert part["type"] == "input_text"
    assert part["text"].startswith("[image omitted: image/png, ")
    assert "AAAA" not in part["text"]


def test_a_string_tool_output_is_not_degraded():
    item = {"type": "custom_tool_call_output", "call_id": "call_2", "output": "done\n"}
    request = ResponsesRequest(model="m", input=[item])
    assert request.model_dump()["input"][0]["output"] == "done\n"
