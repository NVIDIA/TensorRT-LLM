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

from types import SimpleNamespace

import pytest
from pytest import param

from tensorrt_llm._torch.pyexecutor.llm_request import MAX_SPEC_DECODE_POSITIONS
from tensorrt_llm.serve.postprocess_handlers import _build_spec_decode_stats


def _rsp(survival, num_spec_steps, totals):
    """Result stub. `survival` is the leading non-zero part of per_pos_accepted."""
    accepted = list(survival) + [0] * (MAX_SPEC_DECODE_POSITIONS - len(survival))
    drafted = [num_spec_steps] + [0] * (MAX_SPEC_DECODE_POSITIONS - 1)
    return SimpleNamespace(per_pos_accepted=accepted,
                           per_pos_drafted=drafted,
                           spec_dec_totals=totals)


def _args(*, enabled=True, num_spec_tokens=None):
    return SimpleNamespace(return_spec_decode_stats=enabled,
                           spec_decode_num_spec_tokens=num_spec_tokens)


def _assert_identities(stats):
    histogram = stats.acceptance_histogram
    assert sum(histogram) == stats.num_spec_steps
    assert sum(j * c for j, c in enumerate(
        histogram)) == stats.total_accepted_draft_tokens
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
    def test_histogram(self, survival, steps, totals, num_spec_tokens,
                       expected):
        stats = _build_spec_decode_stats(_rsp(survival, steps, totals),
                                         _args(num_spec_tokens=num_spec_tokens),
                                         "stop")
        assert stats.acceptance_histogram == expected
        _assert_identities(stats)

    def test_mean_acceptance_length_is_derivable(self):
        # Not a field: consumers derive it, and it must match the counts. The
        # field would otherwise duplicate avg_decoded_tokens_per_iter on the
        # same choice and the two could drift.
        stats = _build_spec_decode_stats(_rsp([12, 12, 6], 20, (30, 60)),
                                         _args(num_spec_tokens=3), "stop")
        assert 1 + (stats.total_accepted_draft_tokens /
                    stats.num_spec_steps) == 2.5

    def test_histogram_padded_to_configured_budget(self):
        # Length must describe the draft budget, not the depth this particular
        # request happened to reach, so it is stable across requests.
        stats = _build_spec_decode_stats(_rsp([3], 5, (3, 25)),
                                         _args(num_spec_tokens=5), "stop")
        assert len(stats.acceptance_histogram) == 6
        _assert_identities(stats)

    def test_adaptive_drafting_reports_no_fixed_bound(self):
        # Under draft_len_schedule there is no fixed per-step bound, so
        # num_spec_tokens is None and the histogram sizes to observed depth.
        stats = _build_spec_decode_stats(_rsp([12, 12, 6], 20, (30, 60)),
                                         _args(num_spec_tokens=None), "stop")
        assert stats.num_spec_tokens is None
        assert stats.acceptance_histogram == [8, 0, 6, 6]
        _assert_identities(stats)

    def test_deep_drafting_beyond_initial_capacity(self):
        # Tree drafting can exceed MAX_SPEC_DECODE_POSITIONS; the executor grows
        # the vectors rather than truncating, so the identities must still hold.
        depth = MAX_SPEC_DECODE_POSITIONS + 4
        survival = [1] * depth
        stats = _build_spec_decode_stats(
            SimpleNamespace(per_pos_accepted=survival,
                            per_pos_drafted=[1] + [0] * (depth - 1),
                            spec_dec_totals=(depth, depth)),
            _args(num_spec_tokens=depth), "stop")
        _assert_identities(stats)


class TestOmission:
    """The field is absent, not null-filled, whenever it cannot be trusted."""

    def test_absent_when_not_opted_in(self):
        assert _build_spec_decode_stats(_rsp([12], 20, (30, 60)),
                                        _args(enabled=False,
                                              num_spec_tokens=3),
                                        "stop") is None

    def test_absent_on_non_terminal_stream_chunk(self):
        # Streaming carries this only on the chunk bearing finish_reason;
        # intermediate chunks would report a partial request as if complete.
        assert _build_spec_decode_stats(_rsp([12], 20, (30, 60)),
                                        _args(num_spec_tokens=3), None) is None

    def test_absent_when_nothing_drafted(self):
        assert _build_spec_decode_stats(_rsp([], 0, (0, 0)),
                                        _args(num_spec_tokens=3),
                                        "stop") is None

    def test_absent_without_per_position_vectors(self):
        # The C++/TRT backend populates spec metrics via
        # updateNumTokensPerIteration and has no per-position vectors.
        assert _build_spec_decode_stats(
            SimpleNamespace(per_pos_accepted=None,
                            per_pos_drafted=None,
                            spec_dec_totals=None), _args(num_spec_tokens=3),
            "stop") is None
