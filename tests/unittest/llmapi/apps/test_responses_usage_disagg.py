# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Usage accounting for the Responses API under disaggregated serving.

The generation worker cannot measure prefix reuse: the whole prompt reaches it
as KV transferred from the context worker, so its own `cached_tokens` equals the
prompt length. Reporting that says every request was a complete cache hit, which
is what a client watching its context window would act on. The real number comes
across with the handoff, and chat completions has read it from there since
PR #14177.
"""

import unittest
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_protocol import PromptTokensDetails, UsageInfo
from tensorrt_llm.serve.responses_utils import _create_usage

# The CPU-* CI stages run pytest with -m 'cpu_only', and tests/unittest/conftest.py
# drops every file without this marker there.
pytestmark = pytest.mark.cpu_only


def result(prompt_len, output_len, cached, ctx_usage=None):
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


def ctx(prompt_tokens, cached_tokens):
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=1,
        total_tokens=prompt_tokens + 1,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
    )


class Aggregated(unittest.TestCase):
    """No handoff: the local counts are the only ones, and they are correct."""

    def test_local_counts_are_used(self):
        usage = _create_usage(result(100, 5, cached=32))
        self.assertEqual(100, usage.input_tokens)
        self.assertEqual(5, usage.output_tokens)
        self.assertEqual(32, usage.input_tokens_details.cached_tokens)

    def test_no_cache_hit_reports_zero(self):
        usage = _create_usage(result(100, 5, cached=0))
        self.assertEqual(0, usage.input_tokens_details.cached_tokens)


class Disaggregated(unittest.TestCase):
    def test_the_context_phase_decides_the_cached_count(self):
        """The 100 the generation worker reports is the prompt length, not reuse."""
        usage = _create_usage(result(100, 5, cached=100, ctx_usage=ctx(100, 32)))
        self.assertEqual(32, usage.input_tokens_details.cached_tokens)
        self.assertEqual(100, usage.input_tokens)

    def test_a_cold_prompt_is_not_reported_as_fully_cached(self):
        """The failure this fixes: cached == input on every single request."""
        usage = _create_usage(result(4096, 10, cached=4096, ctx_usage=ctx(4096, 0)))
        self.assertEqual(0, usage.input_tokens_details.cached_tokens)
        self.assertNotEqual(usage.input_tokens, usage.input_tokens_details.cached_tokens)

    def test_the_context_phase_also_decides_the_prompt_length(self):
        usage = _create_usage(
            result(100, 5, cached=100, ctx_usage=ctx(96, 64)), num_prompt_tokens=100
        )
        self.assertEqual(96, usage.input_tokens)
        self.assertEqual(64, usage.input_tokens_details.cached_tokens)

    def test_output_tokens_stay_local(self):
        """Only the generation worker knows what it generated."""
        usage = _create_usage(result(100, 7, cached=100, ctx_usage=ctx(100, 50)))
        self.assertEqual(7, usage.output_tokens)

    def test_a_handoff_carrying_no_details_reports_no_reuse(self):
        bare = UsageInfo(prompt_tokens=100, completion_tokens=1, total_tokens=101)
        usage = _create_usage(result(100, 5, cached=100, ctx_usage=bare))
        self.assertEqual(0, usage.input_tokens_details.cached_tokens)

    def test_a_handoff_arriving_as_a_dict_is_accepted(self):
        """It crosses a process boundary, so it may not still be a model."""
        as_dict = ctx(100, 40).model_dump()
        usage = _create_usage(result(100, 5, cached=100, ctx_usage=as_dict))
        self.assertEqual(40, usage.input_tokens_details.cached_tokens)


class ProxyHandoff(unittest.TestCase):
    """The proxy converts the context reply's usage into the handoff shape."""

    def test_cached_tokens_survive_the_conversion(self):
        from tensorrt_llm.serve.openai_disagg_service import _ctx_usage_info

        # Shaped by hand rather than through the SDK model: the two carry the
        # same field names but not the same required set, and this test is
        # about what the proxy copies, not about the SDK's schema.
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=96,
                input_tokens_details=SimpleNamespace(cached_tokens=64),
                output_tokens=1,
                total_tokens=97,
            )
        )
        carried = _ctx_usage_info(response)
        self.assertEqual(96, carried.prompt_tokens)
        self.assertEqual(1, carried.completion_tokens)
        self.assertEqual(64, carried.prompt_tokens_details.cached_tokens)

    def test_a_reply_without_details_still_converts(self):
        from tensorrt_llm.serve.openai_disagg_service import _ctx_usage_info

        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=96,
                input_tokens_details=None,
                output_tokens=1,
                total_tokens=97,
            )
        )
        carried = _ctx_usage_info(response)
        self.assertEqual(0, carried.prompt_tokens_details.cached_tokens)

    def test_a_usage_that_is_already_in_the_handoff_shape_passes_through(self):
        from tensorrt_llm.serve.openai_disagg_service import _ctx_usage_info

        already = ctx(96, 64)
        self.assertIs(already, _ctx_usage_info(SimpleNamespace(usage=already)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
