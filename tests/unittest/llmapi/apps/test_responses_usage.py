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

from tensorrt_llm.serve.responses_utils import _create_usage


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
