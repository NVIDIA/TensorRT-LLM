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
"""Kimi K3.1 DSA metadata: the TopK-conversion block table of a speculative decode step, and the
indexer's verify-row split of a CUDA graph."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
    K31DSAtrtllmAttentionMetadata,
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
        request_ids=list(request_ids),
        block_table=torch.full((len(request_ids), max_blocks), -7, dtype=torch.int32),
    )
    md._update_indexer_k_cache_block_offsets = lambda: manager.get_pool_block_indices(
        md.request_ids
    )
    return md


def test_first_decode_step_host_length_past_capacity():
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

    K31DSAtrtllmAttentionMetadata.prepare_for_indexer_k_cache(md)

    table = md.block_table
    # Only held pages are listed; the column past the batch's pages is left untouched.
    assert table[0, :3].tolist() == [100, 101, 102]
    assert table[1, :3].tolist() == [200, 201, -1]
    assert bool((table[:, 3:] == -7).all())


def test_block_table_masks_columns_past_host_length():
    # A row whose host length needs fewer pages than it holds lists only those pages.
    manager = _FakeManager({5: 3 * _TOKENS_PER_BLOCK, 6: 3 * _TOKENS_PER_BLOCK})
    md = _metadata(manager, request_ids=[5, 6], cached=[60, 130], seq_lens_kv=[1, 1])

    K31DSAtrtllmAttentionMetadata.prepare_for_indexer_k_cache(md)

    assert md.block_table[0, :3].tolist() == [100, -1, -1]
    assert md.block_table[1, :3].tolist() == [200, 201, 202]


def _dsl_spec_metadata(is_cuda_graph: bool, next_n: int, num_generations: int, max_blocks=4):
    """What ``prepare_for_spec_decode`` reads on a generation-only step of the DSL logits path."""
    rows = num_generations * next_n
    md = SimpleNamespace(
        is_cuda_graph=is_cuda_graph,
        _k31_perf=False,
        use_cute_dsl_paged_mqa_logits=True,
        max_draft_tokens=next_n - 1,
        gen_token_stride=next_n,
        kv_cache_manager=object(),
        num_contexts=0,
        num_generations=num_generations,
        num_seqs=num_generations,
        num_sms=148,
        _indexer_compress_ratio=1,
        dsl_atom_ctx_offsets_key=None,
        kv_lens_expanded_host=torch.zeros(rows, dtype=torch.int32),
        kv_lens_expanded_cuda=torch.zeros(rows, dtype=torch.int32),
        # Request i holds pages 100 * (i + 1) + column.
        host_indexer_k_cache_block_offsets=torch.tensor(
            [[100 * (i + 1) + c for c in range(max_blocks)] for i in range(num_generations)],
            dtype=torch.int32,
        ),
        host_block_table_expanded=torch.full((rows, max_blocks), -7, dtype=torch.int32),
        block_table_expanded=torch.full((rows, max_blocks), -7, dtype=torch.int32),
    )
    for name in ("get_indexer_kv_lens", "_refresh_expanded_block_table"):
        setattr(md, name, MethodType(getattr(K31DSAtrtllmAttentionMetadata, name), md))
    return md


def _dsl_split(md) -> tuple:
    return md.dsl_expand_factor, md.dsl_atom, md.dsl_kernel_atoms


@pytest.mark.parametrize(
    "next_n, graph_split, eager_split",
    [(8, (2, 4, 1), (4, 2, 1)), (6, (2, 3, 1), (6, 1, 1))],
    ids=["draft7", "draft5"],
)
def test_graph_replay_keeps_the_captured_dsl_split_and_its_table(next_n, graph_split, eager_split):
    # A graph captured at the served length and replayed with 3,000-token contexts, where an eager
    # step splits the verify rows otherwise. The captured indexer reads expand_factor rows per
    # request, so the table this step writes must hold each request's pages in its own rows.
    graph = _dsl_spec_metadata(True, next_n, num_generations=2)
    K31DSAtrtllmAttentionMetadata.prepare_for_spec_decode(graph, torch.tensor([131072, 131072]))
    assert _dsl_split(graph) == graph_split

    K31DSAtrtllmAttentionMetadata.prepare_for_spec_decode(graph, torch.tensor([3000, 3000]))

    factor = graph_split[0]
    assert _dsl_split(graph) == graph_split
    assert graph.block_table_expanded[: 2 * factor, 0].tolist() == [100] * factor + [200] * factor
    assert graph.kv_lens_expanded_cuda[: 2 * factor].tolist() == [3000] * (2 * factor)

    eager = _dsl_spec_metadata(False, next_n, num_generations=2)
    K31DSAtrtllmAttentionMetadata.prepare_for_spec_decode(eager, torch.tensor([3000, 3000]))
    assert _dsl_split(eager) == eager_split
