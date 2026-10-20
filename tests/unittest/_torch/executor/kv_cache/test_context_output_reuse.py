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
"""Requests returning context logits bypass KV cache prefix reuse.

Context logits, and the prompt logprobs derived from them, exist only for tokens
that pass through the forward pass. A reused prefix would leave those positions
unscored, so these requests re-prefill their whole prompt: the V2 manager caps
their reuse claim at zero, and the V1 manager rewinds them to position 0 after
the scheduler has admitted them without a reuse discount.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _context_output_claim_limit,
    _tighter_claim_limit,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler as scheduler_module
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    BindMicroBatchScheduler,
    PyMicroBatchScheduler,
    clear_context_output_reuse_estimates,
)

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 4


class TestContextOutputClaimLimit:
    def test_a_context_logits_request_claims_nothing(self):
        req = SimpleNamespace(py_return_context_logits=True, is_dummy=False)
        assert _context_output_claim_limit(req) == 0

    def test_a_plain_request_is_not_capped(self):
        req = SimpleNamespace(py_return_context_logits=False, is_dummy=False)
        assert _context_output_claim_limit(req) is None

    def test_a_dummy_request_is_not_capped(self):
        req = SimpleNamespace(py_return_context_logits=True, is_dummy=True)
        assert _context_output_claim_limit(req) is None


class TestTighterClaimLimit:
    @pytest.mark.parametrize(
        "first, second, expected",
        [(None, None, None), (None, 8, 8), (8, None, 8), (0, 8, 0), (12, 8, 8)],
    )
    def test_the_stricter_cap_wins_and_none_means_uncapped(self, first, second, expected):
        assert _tighter_claim_limit(first, second) == expected


class _V1Req:
    """The slice of LlmRequest the V1 context-output rewind touches."""

    def __init__(self, prompt_len, prepopulated, chunk, *, context_logits=True, is_dummy=False):
        self.py_request_id = id(self)
        self.prompt_len = prompt_len
        self.py_return_context_logits = context_logits
        self.is_dummy = is_dummy
        self.prepopulated_prompt_len = prepopulated
        self.context_current_position = prepopulated
        self.context_chunk_size = chunk

    def set_prepopulated_prompt_len(self, value, tokens_per_block):
        self.prepopulated_prompt_len = value
        # Mirrors the C++ setter: only a nonzero value moves the position.
        if value:
            self.context_current_position = value


def _manager(*, windowed=False, is_draft=False):
    mgr = SimpleNamespace(
        is_draft=is_draft,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=64,
        max_attention_window_vec=[16 if windowed else 64],
    )
    mgr._has_windowed_attention = KVCacheManager._has_windowed_attention.__get__(mgr)
    return mgr


def _rewind(mgr, reqs, admitted_chunks):
    # Mirrors prepare_resources: only context-logits requests record a chunk.
    scheduled_chunks = {
        req.py_request_id: chunk
        for req, chunk in zip(reqs, admitted_chunks)
        if req.py_return_context_logits
    }
    KVCacheManager._maybe_rewind_context_output_requests(mgr, list(reqs), scheduled_chunks)


class TestV1RewindContextOutputRequests:
    def test_a_cache_hit_restarts_at_zero_with_its_admitted_chunk(self):
        # Admitted with a 16-token chunk; add_sequence_batch then shifted it
        # past an 8-token reused prefix.
        req = _V1Req(prompt_len=24, prepopulated=8, chunk=16)
        _rewind(_manager(), [req], [16])
        assert req.prepopulated_prompt_len == 0
        assert req.context_current_position == 0
        assert req.context_chunk_size == 16

    def test_a_full_prompt_chunk_recomputes_the_whole_prompt(self):
        req = _V1Req(prompt_len=24, prepopulated=20, chunk=4)
        _rewind(_manager(), [req], [24])
        assert req.context_current_position == 0
        assert req.context_chunk_size == 24

    def test_a_request_without_context_logits_keeps_its_reuse(self):
        req = _V1Req(prompt_len=24, prepopulated=8, chunk=16, context_logits=False)
        _rewind(_manager(), [req], [24])
        assert req.prepopulated_prompt_len == 8
        assert req.context_current_position == 8

    def test_a_request_without_a_cache_hit_is_untouched(self):
        req = _V1Req(prompt_len=24, prepopulated=0, chunk=24)
        _rewind(_manager(), [req], [24])
        assert req.context_current_position == 0
        assert req.context_chunk_size == 24

    def test_a_dummy_request_is_untouched(self):
        req = _V1Req(prompt_len=24, prepopulated=8, chunk=16, is_dummy=True)
        _rewind(_manager(), [req], [24])
        assert req.context_current_position == 8

    def test_the_draft_manager_does_not_rewind(self):
        req = _V1Req(prompt_len=24, prepopulated=8, chunk=16)
        _rewind(_manager(is_draft=True), [req], [24])
        assert req.context_current_position == 8

    def test_windowed_attention_keeps_the_reuse(self):
        # A rewound cursor would cross detached out-of-window spans; the worker
        # rejects the short context logits instead.
        req = _V1Req(prompt_len=24, prepopulated=8, chunk=16)
        _rewind(_manager(windowed=True), [req], [24])
        assert req.context_current_position == 8


class TestClearContextOutputReuseEstimates:
    def test_only_context_logits_requests_lose_their_reuse_credit(self):
        scored = SimpleNamespace(py_return_context_logits=True, estimated_reusable_tokens=32)
        plain = SimpleNamespace(py_return_context_logits=False, estimated_reusable_tokens=32)
        clear_context_output_reuse_estimates([scored, plain])
        assert scored.estimated_reusable_tokens == 0
        assert plain.estimated_reusable_tokens == 32

    @pytest.mark.parametrize("scheduler_cls", [BindMicroBatchScheduler, PyMicroBatchScheduler])
    def test_every_micro_batch_scheduler_clears_before_budgeting(self, monkeypatch, scheduler_cls):
        # Callers such as the multimodal scheduler invoke the micro-batch
        # scheduler directly, so the clearing has to happen inside it.
        class _Cleared(Exception):
            pass

        seen = []

        def clear(requests):
            seen.append(requests)
            raise _Cleared

        monkeypatch.setattr(scheduler_module, "clear_context_output_reuse_estimates", clear)
        requests = [SimpleNamespace()]
        with pytest.raises(_Cleared):
            scheduler_cls.schedule(object.__new__(scheduler_cls), requests, set())
        assert seen == [requests]
