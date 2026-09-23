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
"""Derivation tests for the per-request speculative_decoding response field.

py_per_pos_accepted is prefix-cumulative -- entry k counts steps that accepted
at least k+1 drafts -- so it is a survival function, and the acceptance
histogram is its negative first difference. These tests pin that derivation and
the three identities a consumer relies on, since a payload that violates any of
them is indistinguishable from corruption on the client side:

* sum(acceptance_histogram) == num_spec_steps
* sum(j * acceptance_histogram[j]) == total_accepted_draft_tokens
* total_accepted_draft_tokens <= total_draft_tokens

Totals come from spec_dec_totals (exact) while the histogram comes from the
per-position vectors, so the second identity is the one that would break first
if the two ever stopped agreeing.

GPU-free logic, but importing postprocess_handlers is unproven on the CPU-only
CI stage, so this file is wired into a GPU list (l0_a10.yml) like its sibling
test_spec_dec_stats_pairing.py.
"""

import json
from types import SimpleNamespace

import pytest
from pytest import param

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import MAX_SPEC_DECODE_POSITIONS
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.serve.openai_server import OpenAIServer, resolve_spec_decode_num_spec_tokens
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    _build_spec_decode_stats,
    chat_response_post_processor,
    chat_stream_post_processor,
    completion_response_post_processor,
    completion_stream_post_processor,
)


def _rsp(survival, num_spec_steps, totals):
    """Result stub. `survival` is the leading non-zero part of per_pos_accepted."""
    accepted = list(survival) + [0] * (MAX_SPEC_DECODE_POSITIONS - len(survival))
    drafted = [num_spec_steps] + [0] * (MAX_SPEC_DECODE_POSITIONS - 1)
    return SimpleNamespace(
        per_pos_accepted=accepted, per_pos_drafted=drafted, spec_dec_totals=totals
    )


def _args(*, enabled=True, num_spec_tokens=None):
    return SimpleNamespace(
        return_spec_decode_stats=enabled, spec_decode_num_spec_tokens=num_spec_tokens
    )


def _assert_identities(stats):
    histogram = stats.acceptance_histogram
    assert sum(histogram) == stats.num_spec_steps
    assert sum(j * c for j, c in enumerate(histogram)) == stats.total_accepted_draft_tokens
    assert stats.total_accepted_draft_tokens <= stats.total_draft_tokens


class TestHistogramDerivation:
    @pytest.mark.parametrize(
        "survival, steps, totals, num_spec_tokens, expected",
        [
            # 20 steps: 8 accepted none, 6 accepted 2, 6 accepted all 3.
            param([12, 12, 6], 20, (30, 60), 3, [8, 0, 6, 6], id="mixed"),
            param([], 10, (0, 30), 3, [10, 0, 0, 0], id="all_rejected"),
            param([5, 5, 5], 5, (15, 15), 3, [0, 0, 0, 5], id="all_accepted"),
        ],
    )  # fmt: skip
    def test_histogram(self, survival, steps, totals, num_spec_tokens, expected):
        stats = _build_spec_decode_stats(
            _rsp(survival, steps, totals), _args(num_spec_tokens=num_spec_tokens), "stop"
        )
        assert stats.acceptance_histogram == expected
        _assert_identities(stats)

    def test_mean_acceptance_length_is_derivable(self):
        # Not a field: consumers derive it, and it must match the counts. The
        # field would otherwise duplicate avg_decoded_tokens_per_iter on the
        # same choice and the two could drift.
        stats = _build_spec_decode_stats(
            _rsp([12, 12, 6], 20, (30, 60)), _args(num_spec_tokens=3), "stop"
        )
        assert 1 + (stats.total_accepted_draft_tokens / stats.num_spec_steps) == 2.5

    def test_histogram_padded_to_configured_budget(self):
        # Length must describe the draft budget, not the depth this particular
        # request happened to reach, so it is stable across requests.
        stats = _build_spec_decode_stats(_rsp([3], 5, (3, 25)), _args(num_spec_tokens=5), "stop")
        assert len(stats.acceptance_histogram) == 6
        _assert_identities(stats)

    def test_adaptive_drafting_reports_no_fixed_bound(self):
        # Under draft_len_schedule there is no fixed per-step bound, so
        # num_spec_tokens is None and the histogram sizes to observed depth.
        stats = _build_spec_decode_stats(
            _rsp([12, 12, 6], 20, (30, 60)), _args(num_spec_tokens=None), "stop"
        )
        assert stats.num_spec_tokens is None
        assert stats.acceptance_histogram == [8, 0, 6, 6]
        _assert_identities(stats)

    def test_deep_drafting_beyond_initial_capacity(self):
        # Tree drafting can exceed MAX_SPEC_DECODE_POSITIONS; the executor grows
        # the vectors rather than truncating, so the identities must still hold.
        depth = MAX_SPEC_DECODE_POSITIONS + 4
        survival = [1] * depth
        stats = _build_spec_decode_stats(
            SimpleNamespace(
                per_pos_accepted=survival,
                per_pos_drafted=[1] + [0] * (depth - 1),
                spec_dec_totals=(depth, depth),
            ),
            _args(num_spec_tokens=depth),
            "stop",
        )
        _assert_identities(stats)


class TestOmission:
    """The field is absent, not null-filled, whenever it cannot be trusted."""

    def test_absent_when_not_opted_in(self):
        assert (
            _build_spec_decode_stats(
                _rsp([12], 20, (30, 60)), _args(enabled=False, num_spec_tokens=3), "stop"
            )
            is None
        )

    def test_absent_on_non_terminal_stream_chunk(self):
        # Streaming carries this only on the chunk bearing finish_reason;
        # intermediate chunks would report a partial request as if complete.
        assert (
            _build_spec_decode_stats(_rsp([12], 20, (30, 60)), _args(num_spec_tokens=3), None)
            is None
        )

    def test_absent_when_nothing_drafted(self):
        assert (
            _build_spec_decode_stats(_rsp([], 0, (0, 0)), _args(num_spec_tokens=3), "stop") is None
        )

    def test_absent_without_per_position_vectors(self):
        # The C++/TRT backend populates spec metrics via
        # updateNumTokensPerIteration and has no per-position vectors.
        assert (
            _build_spec_decode_stats(
                SimpleNamespace(per_pos_accepted=None, per_pos_drafted=None, spec_dec_totals=None),
                _args(num_spec_tokens=3),
                "stop",
            )
            is None
        )


class TestNumSpecTokensResolution:
    """What the server reports as the fixed per-step draft bound.

    This value is emitted as ``num_spec_tokens`` and sizes the acceptance
    histogram, so getting it wrong makes every histogram the wrong width.
    """

    def test_no_speculative_config_is_none(self):
        assert resolve_spec_decode_num_spec_tokens(SimpleNamespace()) is None

    def test_no_args_is_none(self):
        assert resolve_spec_decode_num_spec_tokens(None) is None

    def test_fixed_bound_is_reported(self):
        args = SimpleNamespace(
            speculative_config=SimpleNamespace(max_draft_len=4, draft_len_schedule=None)
        )
        assert resolve_spec_decode_num_spec_tokens(args) == 4

    def test_draft_len_schedule_reports_no_bound(self):
        # The bound varies by batch size, so None is the honest answer rather
        # than whichever max_draft_len happens to be configured alongside it.
        args = SimpleNamespace(
            speculative_config=SimpleNamespace(max_draft_len=4, draft_len_schedule={1: 4, 8: 2})
        )
        assert resolve_spec_decode_num_spec_tokens(args) is None


class TestServerOptIn:
    """``_apply_spec_decode_stats_opt_in`` is what reaches the handlers.

    The handlers read ``return_spec_decode_stats`` and
    ``spec_decode_num_spec_tokens`` off the postproc args; this is the only
    place they are set, so a server that resolved its config correctly but
    failed to apply it would emit nothing.
    """

    @staticmethod
    def _server(enabled, num_spec_tokens=4):
        server = object.__new__(OpenAIServer)
        server._per_request_spec_decode_stats = enabled
        server._spec_decode_num_spec_tokens = num_spec_tokens
        return server

    @staticmethod
    def _args():
        return SimpleNamespace(return_spec_decode_stats=False, spec_decode_num_spec_tokens=None)

    def test_disabled_server_leaves_args_untouched(self):
        args = self._args()
        OpenAIServer._apply_spec_decode_stats_opt_in(self._server(False), args)
        assert args.return_spec_decode_stats is False
        assert args.spec_decode_num_spec_tokens is None

    def test_enabled_server_propagates_fixed_bound(self):
        args = self._args()
        OpenAIServer._apply_spec_decode_stats_opt_in(self._server(True), args)
        assert args.return_spec_decode_stats is True
        assert args.spec_decode_num_spec_tokens == 4

    def test_enabled_server_propagates_adaptive_bound(self):
        args = self._args()
        OpenAIServer._apply_spec_decode_stats_opt_in(self._server(True, num_spec_tokens=None), args)
        assert args.return_spec_decode_stats is True
        assert args.spec_decode_num_spec_tokens is None


def _padded(values):
    """Pad a per-position vector to the executor's initial capacity, as sent."""
    return list(values) + [0] * (MAX_SPEC_DECODE_POSITIONS - len(values))


def _sequence_response(*, sequence_index, is_final, per_pos_drafted, per_pos_accepted, totals):
    """One child request's final response, as GenerationResultBase receives it.

    Mirrors the executor response shape used in test_disaggregated_params.py,
    plus the per-request spec-decode counters the PyTorch executor attaches.
    """
    result = SimpleNamespace(
        is_final=is_final,
        decoding_iter=1,
        avg_decoded_tokens_per_iter=None,
        context_phase_params=None,
        finish_reasons=[tllm.FinishReason.END_ID],
        output_token_ids=[[5, 6]],
        sequence_index=sequence_index,
        cum_log_probs=None,
        log_probs=None,
        generation_logits=None,
        context_logits=None,
        request_perf_metrics=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
        per_pos_drafted=_padded(per_pos_drafted),
        per_pos_accepted=_padded(per_pos_accepted),
        spec_dec_totals=totals,
    )
    return SimpleNamespace(result=result, has_error=lambda: False)


def _stats_by_index_from_response(response):
    return {c.index: c.speculative_decoding.model_dump() for c in response.choices}


def _stats_by_index_from_stream(chunks):
    stats = {}
    for chunk in chunks:
        body = chunk.removeprefix("data: ").strip()
        if body == "[DONE]":
            continue
        for choice in json.loads(body).get("choices", []):
            if "speculative_decoding" in choice:
                stats[choice["index"]] = choice["speculative_decoding"]
    return stats


_SPEC_DECODE_OPT_IN = dict(
    num_choices=2, num_prompt_tokens=3, return_spec_decode_stats=True, spec_decode_num_spec_tokens=2
)


def _chat_args():
    return ChatPostprocArgs(role="assistant", model="m", **_SPEC_DECODE_OPT_IN)


def _completion_args():
    return CompletionPostprocArgs(model="m", detokenize=False, **_SPEC_DECODE_OPT_IN)


class TestPerSequenceAttribution:
    """Each choice of an n > 1 request reports its own sequence's acceptance.

    Every candidate runs as its own child request with its own counters, but
    GenerationResultBase keeps a single request-level copy that each arriving
    response overwrites. Building every choice from that copy would stamp
    whichever candidate reported last onto all of them. Covers all four
    formatters, since each reads the counters at its own call site.
    """

    @staticmethod
    def _two_sequence_result():
        # n > 1 needs non-greedy sampling, as every real multi-candidate
        # request has.
        result = GenerationResultBase(
            id=1, sampling_params=SamplingParams(max_tokens=2, n=2, temperature=0.8)
        )
        # Sequence 0: 4 steps drafting 2 each; 3 steps accepted >= 1 and 1
        # step accepted both -> histogram [1, 2, 1], 4 of 8 accepted.
        result._handle_response(
            _sequence_response(
                sequence_index=0,
                is_final=False,
                per_pos_drafted=[4, 4],
                per_pos_accepted=[3, 1],
                totals=(4, 8),
            )
        )
        # Sequence 1 reports last: 3 steps, only 1 accepted anything ->
        # histogram [2, 1, 0], 1 of 6 accepted.
        result._handle_response(
            _sequence_response(
                sequence_index=1,
                is_final=True,
                per_pos_drafted=[3, 3],
                per_pos_accepted=[1],
                totals=(1, 6),
            )
        )
        return result

    @pytest.mark.parametrize(
        "post_processor, make_args, stats_by_index",
        [
            param(
                completion_response_post_processor,
                _completion_args,
                _stats_by_index_from_response,
                id="completion",
            ),
            param(
                completion_stream_post_processor,
                _completion_args,
                _stats_by_index_from_stream,
                id="completion_stream",
            ),
            param(
                chat_response_post_processor, _chat_args, _stats_by_index_from_response, id="chat"
            ),
            param(
                chat_stream_post_processor,
                _chat_args,
                _stats_by_index_from_stream,
                id="chat_stream",
            ),
        ],
    )
    def test_each_choice_reports_its_own_sequence(self, post_processor, make_args, stats_by_index):
        output = post_processor(self._two_sequence_result(), make_args())

        assert stats_by_index(output) == {
            0: {
                "acceptance_rate": 0.5,
                "total_accepted_draft_tokens": 4,
                "total_draft_tokens": 8,
                "num_spec_steps": 4,
                "acceptance_histogram": [1, 2, 1],
                "num_spec_tokens": 2,
            },
            1: {
                "acceptance_rate": 1 / 6,
                "total_accepted_draft_tokens": 1,
                "total_draft_tokens": 6,
                "num_spec_steps": 3,
                "acceptance_histogram": [2, 1, 0],
                "num_spec_tokens": 2,
            },
        }
