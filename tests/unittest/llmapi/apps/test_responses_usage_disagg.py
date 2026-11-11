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
"""Usage accounting for the Responses API under disaggregated serving.

The generation worker cannot measure prefix reuse: the whole prompt reaches it
as KV transferred from the context worker, so its own ``cached_tokens`` equals
the prompt length. Reporting that says every request was a complete cache hit,
which is what a client watching its context window would act on. The real
number comes across with the handoff, and chat completions has read it from
there since #14177.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_disagg_service import _ctx_usage_info
from tensorrt_llm.serve.openai_protocol import PromptTokensDetails, UsageInfo
from tensorrt_llm.serve.responses_utils import _create_usage

pytestmark = pytest.mark.cpu_only


def _result(prompt_len, output_len, cached, ctx_usage=None):
    """A finished generation, shaped as the postprocessor sees one."""
    output = SimpleNamespace(
        token_ids=list(range(output_len)),
        disaggregated_params=(
            SimpleNamespace(ctx_usage=ctx_usage) if ctx_usage is not None else None
        ),
    )
    return SimpleNamespace(
        outputs=[output],
        prompt_token_ids=list(range(prompt_len)),
        cached_tokens=cached,
    )


def _ctx(prompt_tokens, cached_tokens):
    """The context phase's usage, as the handoff carries it."""
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=1,
        total_tokens=prompt_tokens + 1,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
    )


# ---------------------------------------------------------------------------
# Aggregated: no handoff, so the local counts are the only ones, and correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cached", [32, 0])
def test_without_a_handoff_the_local_counts_are_used(cached):
    usage = _create_usage(_result(100, 5, cached=cached))

    assert usage.input_tokens == 100
    assert usage.output_tokens == 5
    assert usage.input_tokens_details.cached_tokens == cached


# ---------------------------------------------------------------------------
# Disaggregated: the context phase's usage comes with the handoff
# ---------------------------------------------------------------------------


def test_the_context_phase_decides_the_cached_count():
    """The 100 the generation worker reports is the prompt length, not reuse."""
    usage = _create_usage(_result(100, 5, cached=100, ctx_usage=_ctx(100, 32)))

    assert usage.input_tokens_details.cached_tokens == 32
    assert usage.input_tokens == 100


def test_a_cold_prompt_is_not_reported_as_fully_cached():
    """The failure this guards: cached equal to input on every single request."""
    usage = _create_usage(_result(4096, 10, cached=4096, ctx_usage=_ctx(4096, 0)))

    assert usage.input_tokens_details.cached_tokens == 0
    assert usage.input_tokens != usage.input_tokens_details.cached_tokens


def test_the_context_phase_also_decides_the_prompt_length():
    """The context worker's count wins over the one the request was submitted with."""
    usage = _create_usage(
        _result(100, 5, cached=100, ctx_usage=_ctx(96, 64)), num_prompt_tokens=100
    )

    assert usage.input_tokens == 96
    assert usage.input_tokens_details.cached_tokens == 64
    assert usage.total_tokens == 96 + 5


def test_output_tokens_stay_local():
    """Only the generation worker knows what it generated."""
    usage = _create_usage(_result(100, 7, cached=100, ctx_usage=_ctx(100, 50)))

    assert usage.output_tokens == 7


def test_a_handoff_carrying_no_details_reports_no_reuse():
    bare = UsageInfo(prompt_tokens=100, completion_tokens=1, total_tokens=101)
    usage = _create_usage(_result(100, 5, cached=100, ctx_usage=bare))

    assert usage.input_tokens_details.cached_tokens == 0


def test_a_handoff_arriving_as_a_dict_is_accepted():
    """It crosses a process boundary, so it may not still be a model."""
    usage = _create_usage(_result(100, 5, cached=100, ctx_usage=_ctx(100, 40).model_dump()))

    assert usage.input_tokens_details.cached_tokens == 40


# ---------------------------------------------------------------------------
# The orchestrator converts the context reply's usage into the handoff shape
# ---------------------------------------------------------------------------


def _context_reply(input_tokens_details):
    # Shaped by hand rather than through the SDK model: the two carry the same
    # field names but not the same required set, and these tests are about
    # what the orchestrator copies, not about the SDK's schema.
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=96,
            input_tokens_details=input_tokens_details,
            output_tokens=1,
            total_tokens=97,
        )
    )


def test_cached_tokens_survive_the_conversion():
    carried = _ctx_usage_info(_context_reply(SimpleNamespace(cached_tokens=64)))

    assert carried.prompt_tokens == 96
    assert carried.completion_tokens == 1
    assert carried.total_tokens == 97
    assert carried.prompt_tokens_details.cached_tokens == 64


def test_a_reply_without_details_still_converts():
    carried = _ctx_usage_info(_context_reply(None))

    assert carried.prompt_tokens == 96
    assert carried.prompt_tokens_details.cached_tokens == 0


def test_a_usage_already_in_the_handoff_shape_passes_through():
    already = _ctx(96, 64)

    assert _ctx_usage_info(SimpleNamespace(usage=already)) is already


def test_a_reply_without_usage_carries_none():
    assert _ctx_usage_info(SimpleNamespace(usage=None)) is None


def test_the_conversion_round_trips_into_the_generation_workers_usage():
    """What the orchestrator relays is what the generation worker then reports."""
    carried = _ctx_usage_info(_context_reply(SimpleNamespace(cached_tokens=64)))
    usage = _create_usage(_result(96, 5, cached=96, ctx_usage=carried))

    assert usage.input_tokens == 96
    assert usage.input_tokens_details.cached_tokens == 64
    assert usage.output_tokens == 5
    assert usage.total_tokens == 101
