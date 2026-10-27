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
"""Reuse claims that end on the page boundaries of an unpaired draft pool.

A standalone DFlash/DSpark drafter reading an unpaired draft pool stores the
uncached tail of a prefix hit at its target positions and hides the reused
prefix from its reads in whole pages (``DFlashWorker._ctx_base_for_store``).
With partial reuse a hit can end anywhere inside a page, and the stored tail
tokens of that page are then never read: up to ``page - 1`` of the newest
prompt tokens missing from the drafter's context. The target manager therefore
caps a claim at the draft page boundary at or below its match
(``KVCacheManagerV2.align_reuse_claims_to_draft_pages``), and the target
recomputes the rest of the page.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _draft_page_aligned_claim_limit,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DEFAULT_BEAM_INDEX, ReuseScope


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "depth, page, expected",
    [
        (186, 64, 128),  # a shared system prompt ending inside a page
        (191, 64, 128),
        (192, 64, None),  # already on a boundary
        (63, 64, 0),  # less than a page: nothing worth claiming
        (0, 64, None),
        (186, 0, None),  # alignment off
        (186, 1, None),
    ],
)
def test_claim_limit(depth, page, expected):
    assert _draft_page_aligned_claim_limit(depth, page) == expected


def _manager(*, alignment=64, partial=True, tokens_per_block=128, is_draft=False, depth=186):
    manager = object.__new__(KVCacheManagerV2)
    manager._spec_claim_alignment = alignment
    manager.enable_partial_reuse = partial
    manager.tokens_per_block = tokens_per_block
    manager.is_draft = is_draft
    manager.impl = Mock()
    manager.impl.probe_reuse.return_value = depth
    return manager


def _req(**overrides):
    fields = dict(
        py_request_id=7,
        lora_task_id=None,
        cache_salt=None,
        is_dummy=False,
        is_disagg_generation_init_state=False,
        is_first_context_chunk=True,
        return_perf_metrics=False,
        prompt_len=500,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.cpu_only
class TestAlignedReuseLimit:
    def test_an_unaligned_match_is_capped_at_the_page_boundary(self):
        manager = _manager()
        # The claim evidence is the request's int32 token view, not a list.
        tokens = np.arange(499, dtype=np.int32)
        assert manager._draft_page_aligned_reuse_limit(_req(), tokens) == 128
        manager.impl.probe_reuse.assert_called_once()
        scope, probed = manager.impl.probe_reuse.call_args.args
        assert isinstance(scope, ReuseScope)
        assert (scope.lora_id, scope.salt) == (None, None)
        assert probed is tokens

    def test_the_probe_uses_the_claims_reuse_scope(self):
        manager = _manager()
        manager._draft_page_aligned_reuse_limit(_req(lora_task_id=3, cache_salt="s"), [1, 2])
        scope = manager.impl.probe_reuse.call_args.args[0]
        assert (scope.lora_id, scope.salt) == (3, KVCacheManagerV2._derive_reuse_salt("s"))

    def test_an_aligned_match_is_left_alone(self):
        assert _manager(depth=256)._draft_page_aligned_reuse_limit(_req(), [1]) is None

    @pytest.mark.parametrize(
        "manager_kwargs, req_kwargs, tokens",
        [
            (dict(alignment=0), {}, [1]),
            (dict(is_draft=True), {}, [1]),
            ({}, dict(is_dummy=True), [1]),
            ({}, dict(is_disagg_generation_init_state=True), [1]),
            ({}, {}, []),
            # Without partial reuse a match ends on a 128-token block boundary,
            # which is a 64-token page boundary too: no probe needed.
            (dict(partial=False), {}, [1]),
        ],
        ids=["off", "draft-pool", "dummy", "disagg-gen-init", "no-evidence", "full-block-reuse"],
    )
    def test_no_probe_where_no_cap_can_apply(self, manager_kwargs, req_kwargs, tokens):
        manager = _manager(**manager_kwargs)
        assert manager._draft_page_aligned_reuse_limit(_req(**req_kwargs), tokens) is None
        manager.impl.probe_reuse.assert_not_called()

    def test_full_block_reuse_still_aligns_pages_larger_than_the_block(self):
        manager = _manager(partial=False, tokens_per_block=32, depth=96)
        assert manager._draft_page_aligned_reuse_limit(_req(), [1]) == 64

    def test_the_hook_sets_and_clears_the_alignment(self):
        manager = object.__new__(KVCacheManagerV2)
        assert manager._spec_claim_alignment == 0  # class default: off
        manager.align_reuse_claims_to_draft_pages(64)
        assert manager._spec_claim_alignment == 64
        manager.align_reuse_claims_to_draft_pages(0)
        assert manager._spec_claim_alignment == 0


@pytest.mark.cpu_only
def test_prepare_context_cache_claims_the_aligned_prefix():
    """The capped claim is a shorter match: the evidence handed to the cache is
    the one for the aligned limit, which composes with a recompute-tail limit."""
    manager = _manager(depth=186)
    manager.conversation_manager = None
    manager.kv_cache_map = {}
    manager.enable_block_reuse = True
    manager.is_estimating_kv_cache = False
    manager._has_cp_helix = False
    manager._stream = SimpleNamespace(cuda_stream=0)
    evidence = {None: list(range(499)), 128: list(range(128))}
    manager._context_reuse_tokens = Mock(side_effect=lambda req, limit=None: evidence[limit])
    kv_cache = SimpleNamespace(num_committed_tokens=128)
    manager._create_kv_cache = Mock(return_value=kv_cache)
    manager._record_branch_snapshot_point = Mock()
    manager._connector_may_serve = Mock(return_value=False)
    manager._resume_context_cache = Mock(return_value=True)

    assert manager.prepare_context_cache(_req()) == 128
    assert [
        c.args[1] if len(c.args) > 1 else None for c in manager._context_reuse_tokens.call_args_list
    ] == [
        None,
        128,
    ]
    assert manager._create_kv_cache.call_args.args[2] == evidence[128]
    manager._record_branch_snapshot_point.assert_called_once_with(_req(), kv_cache, 128)


# ---------------------------------------------------------------------------
# A real manager (GPU): the core claims exactly the aligned partial prefix.
# ---------------------------------------------------------------------------

TOKENS_PER_BLOCK = 8
DRAFT_PAGE = 4
MAX_SEQ_LEN = 32


@dataclasses.dataclass
class _ContextRequest:
    """The slice of LlmRequest the V2 manager's context path reads."""

    request_id: int
    tokens: list
    context_remaining_length: int = 0
    lora_task_id: int = None
    cache_salt: str = None
    py_conversation_params: object = None
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    is_disagg_generation_init_state: bool = False
    is_generation_only_request: bool = False
    is_dummy_request: bool = False
    return_perf_metrics: bool = False
    context_current_position: int = 0
    py_served_prefix_end: int = 0
    py_num_connector_matched_tokens: int = 0
    prepopulated_prompt: tuple = None
    multimodal_hashes: object = None
    multimodal_positions: object = None
    multimodal_lengths: object = None

    def __post_init__(self):
        self.py_request_id = self.request_id
        self.context_remaining_length = len(self.tokens)

    @property
    def prompt_len(self):
        return len(self.tokens)

    @property
    def is_dummy(self):
        return self.is_dummy_request

    @property
    def prepopulated_prompt_len(self):
        return 0 if self.prepopulated_prompt is None else self.prepopulated_prompt[0]

    def get_tokens(self, beam_id=DEFAULT_BEAM_INDEX):
        return self.tokens

    def get_tokens_view(self, beam_id=DEFAULT_BEAM_INDEX):
        return np.asarray(self.tokens, dtype=np.int32)

    def set_prepopulated_prompt_len(self, length, tokens_per_block):
        self.prepopulated_prompt = (length, tokens_per_block)
        # Mirrors the C++ setter: only a nonzero value moves the position.
        if length:
            self.context_current_position = length


def _real_manager(tail):
    spec = DSparkDecodingConfig(
        max_draft_len=4, speculative_model="draft", context_recompute_tail=tail
    )
    return KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=True,
            max_gpu_total_bytes=16 << 20,
            max_attention_window=[MAX_SEQ_LEN],
            max_util_for_resume=1.0,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=spec,
        vocab_size=4096,
        enable_stats=False,
    )


def _batch(request):
    batch = ScheduledRequests()
    batch.append_context_request(request)
    return batch


def _run_context(manager, request):
    batch = _batch(request)
    manager.prepare_resources(batch)
    assert manager.prepare_context(request)
    request.context_remaining_length = request.prompt_len - request.context_current_position
    assert manager.resize_context(request, num_tokens=request.context_remaining_length)
    request.context_current_position = request.prompt_len
    request.context_remaining_length = 0
    manager.update_context_resources(batch)


@pytest.mark.parametrize(
    "alignment, tail, expected_claim",
    [
        (0, 0, 21),  # control: partial reuse claims the 21 shared tokens
        (DRAFT_PAGE, 0, 20),  # aligned: the claim stops on the page boundary
        (DRAFT_PAGE, 6, 20),  # tail limit floor_block(32 - 6) = 24 > 21: alignment decides
        (DRAFT_PAGE, 12, 16),  # tail limit floor_block(32 - 12) = 16 < 21: already aligned
    ],
)
def test_a_partial_hit_is_claimed_up_to_the_draft_page_boundary(alignment, tail, expected_claim):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = _real_manager(tail)
    manager.align_reuse_claims_to_draft_pages(alignment)
    shared = list(range(21))
    request_a = _ContextRequest(1, list(range(MAX_SEQ_LEN)))
    request_b = _ContextRequest(2, shared + list(range(1000, 1000 + MAX_SEQ_LEN - len(shared))))
    try:
        _run_context(manager, request_a)
        manager.prepare_resources(_batch(request_b))
        assert manager.prepare_context(request_b)
        claimed = manager.kv_cache_map[request_b.py_request_id].num_committed_tokens
        assert claimed == expected_claim
        assert request_b.context_current_position == expected_claim
        assert request_b.prepopulated_prompt_len == expected_claim
        # The recompute then runs to the end of the prompt and commits.
        request_b.context_remaining_length = request_b.prompt_len - expected_claim
        assert manager.resize_context(request_b, num_tokens=request_b.context_remaining_length)
        request_b.context_current_position = request_b.prompt_len
        request_b.context_remaining_length = 0
        manager.update_context_resources(_batch(request_b))
    finally:
        for request in (request_b, request_a):
            manager.free_resources(request)
        manager.shutdown()
