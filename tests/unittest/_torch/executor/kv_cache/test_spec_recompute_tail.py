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
"""The spec-recompute tail: cap reuse so a prompt tail passes the target forward.

Hidden-state-conditioned drafters capture target hidden states only for tokens
that physically run through a target forward, so a prefix-cache hit starves the
drafter of context. ``_spec_recompute_target`` (V2 manager) and
``KVCacheManager._maybe_rewind_reused_context`` (V1 manager) rewind the context
start of a cache-hit request to a block-aligned position that leaves at least
``context_recompute_tail`` prompt tokens to recompute.

``_spec_recompute_target`` is deliberately a module-level function taking the
manager scalars explicitly: the V2 scheduler tests bind the real
``prepare_context`` onto a bare ``Mock`` manager, where a method-dispatched
helper would resolve to an auto-created Mock attribute and feed a Mock into the
cursor arithmetic. The request-only no-op guards run before ``tail`` is read,
so that binding stays safe.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import _spec_recompute_target
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


def _req(prompt_len, is_dummy=False):
    return SimpleNamespace(prompt_len=prompt_len, is_dummy=is_dummy)


class TestSpecRecomputeTarget:
    def test_nothing_committed_is_a_no_op(self):
        req = _req(prompt_len=16)
        assert _spec_recompute_target(req, 0, tail=4, tokens_per_block=4, is_draft=False) == 0

    def test_the_draft_pool_never_rewinds(self):
        req = _req(prompt_len=16)
        assert _spec_recompute_target(req, 8, tail=4, tokens_per_block=4, is_draft=True) == 8

    def test_a_dummy_request_never_rewinds(self):
        req = _req(prompt_len=16, is_dummy=True)
        assert _spec_recompute_target(req, 8, tail=4, tokens_per_block=4, is_draft=False) == 8

    def test_a_zero_tail_disables_the_recompute(self):
        req = _req(prompt_len=16)
        assert _spec_recompute_target(req, 8, tail=0, tokens_per_block=4, is_draft=False) == 8

    def test_a_negative_tail_forces_a_full_re_prefill(self):
        req = _req(prompt_len=10)
        assert _spec_recompute_target(req, 8, tail=-1, tokens_per_block=4, is_draft=False) == 0

    def test_the_rewind_target_is_block_aligned_and_covers_the_tail(self):
        req = _req(prompt_len=13)
        # 13 - 12 = 1 < 6: rewind to floor_block(13 - 6) = 4, leaving 9 >= 6
        # prompt tokens to recompute from a block boundary.
        assert _spec_recompute_target(req, 12, tail=6, tokens_per_block=4, is_draft=False) == 4

    def test_an_uncommitted_span_covering_the_tail_is_left_alone(self):
        req = _req(prompt_len=12)
        # 12 - 4 = 8 >= 4: the natural recompute already covers the tail.
        assert _spec_recompute_target(req, 4, tail=4, tokens_per_block=4, is_draft=False) == 4

    def test_request_only_guards_run_before_the_manager_scalars_are_read(self):
        # The V2 scheduler tests bind the real prepare_context onto a Mock
        # manager, so tail/is_draft arrive as auto-created Mock attributes.
        # A zero-reuse request must pass through before either is inspected.
        req = _req(prompt_len=16)
        assert (
            _spec_recompute_target(req, 0, tail=Mock(), tokens_per_block=Mock(), is_draft=Mock())
            == 0
        )


class _V1Req:
    """The slice of LlmRequest that _maybe_rewind_reused_context touches."""

    def __init__(self, prompt_len, prepopulated):
        self.prompt_len = prompt_len
        self.prepopulated_prompt_len = prepopulated
        self.context_current_position = prepopulated
        self.context_chunk_size = prompt_len - prepopulated
        self.setter_calls = []

    def set_prepopulated_prompt_len(self, value, tokens_per_block):
        self.setter_calls.append((value, tokens_per_block))
        self.prepopulated_prompt_len = value
        # Mirrors the C++ setter: only a nonzero value moves the position.
        if value:
            self.context_current_position = value


def _rewind(reqs, tail, tokens_per_block=4):
    mgr = SimpleNamespace(_spec_recompute_tail=tail, tokens_per_block=tokens_per_block)
    KVCacheManager._maybe_rewind_reused_context(mgr, list(reqs))


class TestV1RewindReusedContext:
    def test_a_request_without_reuse_is_untouched(self):
        req = _V1Req(prompt_len=16, prepopulated=0)
        _rewind([req], tail=4)
        assert req.setter_calls == []
        assert req.context_chunk_size == 16

    def test_a_full_hit_is_rewound_to_a_block_aligned_tail(self):
        req = _V1Req(prompt_len=16, prepopulated=16)
        _rewind([req], tail=4)
        assert req.setter_calls == [(12, 4)]
        assert req.context_current_position == 12
        assert req.context_chunk_size == 4

    def test_a_negative_tail_forces_a_full_re_prefill(self):
        req = _V1Req(prompt_len=16, prepopulated=16)
        _rewind([req], tail=-1)
        assert req.setter_calls == [(0, 4)]
        # The C++ setter ignores zero, so the rewind resets the position
        # explicitly.
        assert req.context_current_position == 0
        assert req.context_chunk_size == 16

    def test_a_hit_already_leaving_the_tail_uncached_is_untouched(self):
        req = _V1Req(prompt_len=16, prepopulated=4)
        _rewind([req], tail=4)
        assert req.setter_calls == []
        assert req.context_chunk_size == 12
