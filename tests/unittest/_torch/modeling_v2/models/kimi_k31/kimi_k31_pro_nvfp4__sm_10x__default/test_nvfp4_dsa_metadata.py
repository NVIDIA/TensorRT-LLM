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
"""Kimi K3.1 DSA metadata: the TopK-conversion block table of a speculative decode step."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.backend import (  # noqa: E501
    DSATrtllmAttention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.metadata import (  # noqa: E501
    DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import (
    DSAtrtllmAttentionMetadata as EngineDSAtrtllmAttentionMetadata,
)

pytestmark = pytest.mark.cpu_only

_TOKENS_PER_BLOCK = 64
_MAX_DRAFT_LEN = 4
_TOKENS_PER_GEN_STEP = 1 + _MAX_DRAFT_LEN
# One-model speculative decoding reserves max_draft_len - 1 extra KV tokens per request.
_NUM_EXTRA_KV_TOKENS = _MAX_DRAFT_LEN - 1


class _FakeManager:
    """Page tables of a V2 manager that gathers only the batch's allocated block columns."""

    tokens_per_block = _TOKENS_PER_BLOCK

    def __init__(self, capacities: dict[int, int]):
        self.capacities = capacities

    def blocks(self, request_id: int) -> int:
        return -(-self.capacities[request_id] // _TOKENS_PER_BLOCK)

    def get_pool_block_indices(self, request_ids: list[int]) -> torch.Tensor:
        num_cols = max(self.blocks(r) for r in request_ids)
        rows = [
            [100 * (i + 1) + c if c < self.blocks(r) else 0 for c in range(num_cols)]
            for i, r in enumerate(request_ids)
        ]
        return torch.tensor(rows, dtype=torch.int32)

    def get_allocated_block_counts(self, request_ids: list[int]) -> torch.Tensor:
        return torch.tensor([self.blocks(r) for r in request_ids], dtype=torch.int32)


def _metadata(manager: _FakeManager, request_ids, cached, seq_lens_kv, max_blocks: int = 8):
    md = SimpleNamespace(
        kv_cache_manager=manager,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=list(cached)),
        seq_lens_kv=torch.tensor(seq_lens_kv, dtype=torch.int),
        num_seqs=len(request_ids),
        enable_helix=False,
        request_ids=list(request_ids),
        block_table=torch.full((len(request_ids), max_blocks), -7, dtype=torch.int32),
    )
    md._update_indexer_k_cache_block_offsets = lambda: manager.get_pool_block_indices(
        md.request_ids
    )
    return md


def test_engine_dsa_warmups_recognize_target_metadata():
    # The engine's decoder warmup runs its DSA prewarms (paged MQA logits schedule metadata, decode
    # Top-K) only for instances and subclasses of its own DSA metadata.
    assert DSATrtllmAttention.Metadata is DSAtrtllmAttentionMetadata
    assert issubclass(DSAtrtllmAttentionMetadata, EngineDSAtrtllmAttentionMetadata)


@pytest.mark.parametrize("metadata_type", [DSAtrtllmAttentionMetadata])
def test_first_decode_step_host_length_past_capacity(metadata_type):
    # Overlap scheduler, a request's first decode step: the host assumes its in-flight step (its
    # context step) added a full generation step, prompt - 1 + tokens_per_gen_step cached tokens,
    # while the manager reserved prompt + extra KV tokens at the context step and one generation
    # step on top. The host length then needs one token more than the request holds; with this
    # prompt that token opens a page the request does not have.
    prompt = 3 * _TOKENS_PER_BLOCK - _NUM_EXTRA_KV_TOKENS - _TOKENS_PER_GEN_STEP
    capacity = prompt + _NUM_EXTRA_KV_TOKENS + _TOKENS_PER_GEN_STEP
    cached = prompt - 1 + _TOKENS_PER_GEN_STEP
    assert cached + _TOKENS_PER_GEN_STEP == capacity + 1
    assert capacity % _TOKENS_PER_BLOCK == 0

    # A steady decode row whose host length fits its two pages.
    steady_capacity = 2 * _TOKENS_PER_BLOCK - 10
    steady_cached = steady_capacity - _NUM_EXTRA_KV_TOKENS - _TOKENS_PER_GEN_STEP
    manager = _FakeManager({11: capacity, 12: steady_capacity})
    md = _metadata(
        manager,
        request_ids=[11, 12],
        cached=[cached, steady_cached],
        seq_lens_kv=[_TOKENS_PER_GEN_STEP, _TOKENS_PER_GEN_STEP],
    )

    metadata_type.prepare_for_indexer_k_cache(md)

    table = md.block_table
    # Only held pages are listed; the column past the batch's pages is left untouched.
    assert table[0, :3].tolist() == [100, 101, 102]
    assert table[1, :3].tolist() == [200, 201, -1]
    assert bool((table[:, 3:] == -7).all())


@pytest.mark.parametrize("metadata_type", [DSAtrtllmAttentionMetadata])
def test_block_table_masks_columns_past_host_length(metadata_type):
    # A row whose host length needs fewer pages than it holds lists only those pages.
    manager = _FakeManager({5: 3 * _TOKENS_PER_BLOCK, 6: 3 * _TOKENS_PER_BLOCK})
    md = _metadata(manager, request_ids=[5, 6], cached=[60, 130], seq_lens_kv=[1, 1])

    metadata_type.prepare_for_indexer_k_cache(md)

    assert md.block_table[0, :3].tolist() == [100, -1, -1]
    assert md.block_table[1, :3].tolist() == [200, 201, 202]
