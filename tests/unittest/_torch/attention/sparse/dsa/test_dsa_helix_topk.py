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
"""Helix CP: per-rank DSA Top-K rows reduce to the unsharded global Top-K."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import merge_helix_topk


def _shard(total_len: int, cp_size: int, block: int) -> list[list[int]]:
    """Round-robin token blocks over CP ranks (global token ids per rank)."""
    ranks = [[] for _ in range(cp_size)]
    for start in range(0, total_len, block):
        ranks[(start // block) % cp_size].extend(range(start, min(start + block, total_len)))
    return ranks


def _local_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Local Top-K rows as the indexer emits them: local ids, -1 padded."""
    num_tokens, local_len = scores.shape
    rows = torch.full((num_tokens, k), -1, dtype=torch.int32)
    take = min(k, local_len)
    if take:
        rows[:, :take] = torch.topk(scores, take, dim=1).indices.to(torch.int32)
    return rows


@pytest.mark.parametrize(
    "total_len, cp_size, k",
    [(5000, 4, 256), (700, 4, 256), (256, 2, 256), (3000, 8, 64)],
)
def test_merge_matches_global_topk(total_len: int, cp_size: int, k: int) -> None:
    torch.manual_seed(total_len + cp_size)
    num_tokens = 3
    global_scores = torch.randn(num_tokens, total_len)
    owners = _shard(total_len, cp_size, block=64)
    local_scores = [global_scores[:, ids] for ids in owners]
    local_rows = [_local_topk(s, k) for s in local_scores]

    def candidate_scores(rank: int) -> torch.Tensor:
        rows = local_rows[rank]
        gathered = torch.gather(local_scores[rank], 1, rows.clamp(min=0).long())
        return gathered.masked_fill(rows < 0, float("-inf"))

    stacked = torch.stack([candidate_scores(r) for r in range(cp_size)])
    kept_global = [set() for _ in range(num_tokens)]
    for rank in range(cp_size):
        rows = local_rows[rank].clone()
        merge_helix_topk(local_scores[rank], rows, rank, lambda _t: stacked)
        for t in range(num_tokens):
            row = rows[t].tolist()
            n_kept = sum(i >= 0 for i in row)
            # Survivors are compacted to the front of the row.
            assert all(i >= 0 for i in row[:n_kept]) and all(i < 0 for i in row[n_kept:])
            kept_global[t].update(owners[rank][i] for i in row[:n_kept])

    expected_k = min(k, total_len)
    for t in range(num_tokens):
        expected = set(torch.topk(global_scores[t], expected_k).indices.tolist())
        assert kept_global[t] == expected


def test_merge_is_noop_for_single_rank() -> None:
    scores = torch.randn(2, 100)
    rows = _local_topk(scores, 32)
    before = rows.clone()
    merge_helix_topk(scores, rows, 0, lambda t: t.unsqueeze(0))
    assert torch.equal(rows, before)


def _helix_metadata(**fields):
    from types import SimpleNamespace

    num_generations = fields["num_generations"]
    defaults = dict(
        num_contexts=0,
        num_seqs=num_generations,
        _helix_spec_tokens_valid=False,
        helix_owned_new_tokens_cpu=torch.zeros(8, dtype=torch.int),
        helix_is_inactive_rank_cpu=torch.zeros(8, dtype=torch.bool),
    )
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def test_helix_local_kv_lens_counts_only_owned_new_tokens() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )

    cached = torch.tensor([100, 64, 7], dtype=torch.int)
    # Single-token generation: rank 1 of the batch is inactive for request 1.
    md = _helix_metadata(
        num_generations=3,
        seq_lens_kv=torch.tensor([1, 1, 1], dtype=torch.int),
        helix_is_inactive_rank_cpu=torch.tensor([False, True, False] + [False] * 5),
    )
    kv = DSAtrtllmAttentionMetadata._helix_local_kv_lens(md, cached)
    assert kv.tolist() == [101, 64, 8]

    # Verify groups of 6: ownership is a per-sequence count (a group can
    # straddle a page boundary between two CP ranks).
    md = _helix_metadata(
        num_generations=3,
        seq_lens_kv=torch.tensor([6, 6, 6], dtype=torch.int),
        _helix_spec_tokens_valid=True,
        helix_owned_new_tokens_cpu=torch.tensor([6, 0, 2] + [0] * 5, dtype=torch.int),
    )
    kv = DSAtrtllmAttentionMetadata._helix_local_kv_lens(md, cached)
    assert kv.tolist() == [106, 64, 9]


def test_helix_host_gen_slot_mappings_are_row_aligned() -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
        Indexer,
        IndexerParams,
        _compute_slot_mappings,
    )

    tpb, head_dim, qbs = 64, 128, 128
    block_offsets = torch.tensor([[3, 9, -1], [5, 11, -1], [2, 4, -1]], dtype=torch.int32)
    cached = [70, 10, 64]
    params = IndexerParams(
        num_contexts=0,
        num_generations=3,
        num_ctx_tokens=0,
        head_dim=head_dim,
        quant_block_size=qbs,
        tokens_per_block=tpb,
        compress_ratio=1,
        request_ids=[0, 1, 2],
        num_past_tokens=cached,
        seq_lens=torch.tensor([1, 1, 1]),
    )
    n = 3
    md = SimpleNamespace(
        num_ctx_tokens=0,
        num_tokens=n,
        num_contexts=0,
        num_seqs=n,
        num_generations=n,
        _helix_spec_tokens_valid=False,
        helix_is_inactive_rank_cpu=torch.tensor([False, True, False]),
        host_indexer_k_cache_block_offsets=block_offsets,
        host_slot_mapping_fp8=torch.full((8,), -7, dtype=torch.int64),
        host_slot_mapping_scale=torch.full((8,), -7, dtype=torch.int64),
        slot_mapping_fp8=torch.full((8,), -7, dtype=torch.int64),
        slot_mapping_scale=torch.full((8,), -7, dtype=torch.int64),
    )
    Indexer._helix_host_gen_slot_mappings(md, params)
    expected_fp8, expected_scale = _compute_slot_mappings(
        torch.tensor([70, 0, 64]), block_offsets, torch.arange(3), head_dim, tpb, qbs
    )
    # One slot per generation row; the inactive rank's row is skipped (-1).
    assert md.slot_mapping_fp8[:n].tolist() == [int(expected_fp8[0]), -1, int(expected_fp8[2])]
    assert md.slot_mapping_scale[:n].tolist() == [
        int(expected_scale[0]),
        -1,
        int(expected_scale[2]),
    ]
    assert md.slot_mapping_fp8[n:].tolist() == [-7] * (8 - n)

    # Verify groups: host rows are placeholders that on_update_kv_lens() rewrites.
    md._helix_spec_tokens_valid = True
    md.num_tokens = 6
    md.num_generations = md.num_seqs = 1
    Indexer._helix_host_gen_slot_mappings(md, params)
    assert md.slot_mapping_fp8[:6].tolist() == [-1] * 6


def _helix_rows_metadata(seq_lens, kv_lens, num_contexts=0, topk=2048):
    from types import MethodType, SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )

    num_seqs = len(seq_lens)
    rows_cap = 64
    md = SimpleNamespace(
        num_contexts=num_contexts,
        num_seqs=num_seqs,
        num_generations=num_seqs - num_contexts,
        is_ragged_verify=False,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_kv=torch.tensor(seq_lens, dtype=torch.int32),
        prompt_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32),
        num_sparse_topk=topk,
        kv_cache_block_offsets=torch.arange(num_seqs, dtype=torch.int32)
        .view(1, num_seqs, 1, 1)
        .expand(1, num_seqs, 2, 3)
        .contiguous(),
        helix_row_kv_lens_cuda=torch.zeros(rows_cap, dtype=torch.int32),
        helix_row_kv_lens_host=torch.zeros(rows_cap, dtype=torch.int32),
        helix_row_req_idx_cuda=torch.zeros(rows_cap, dtype=torch.long),
        helix_row_req_idx_host=torch.zeros(rows_cap, dtype=torch.long),
        helix_row_prompt_lens_cuda=torch.zeros(rows_cap, dtype=torch.int32),
        helix_row_prompt_lens_cpu=torch.zeros(rows_cap, dtype=torch.int32),
        helix_row_request_types_host=torch.ones(rows_cap, dtype=torch.int32),
        helix_row_block_offsets=torch.full((1, rows_cap, 2, 3), -1, dtype=torch.int32),
        _helix_num_rows=0,
        _helix_rows_set=False,
    )
    for name in (
        "_prepare_helix_sparse_rows",
        "set_helix_sparse_row_lens",
        "helix_sparse_empty_rows",
        "token_major_gen_view",
    ):
        setattr(md, name, MethodType(getattr(DSAtrtllmAttentionMetadata, name), md))
    md._prepare_helix_sparse_rows(torch.tensor(kv_lens, dtype=torch.int32))
    return md


def test_helix_sparse_rows_present_one_row_per_generation_token() -> None:
    # Two verify groups of 3 tokens with 5000 and 900 local KV tokens.
    md = _helix_rows_metadata(seq_lens=[3, 3], kv_lens=[5000, 900])

    assert md._helix_num_rows == 6
    assert md.helix_row_req_idx_host[:6].tolist() == [0, 0, 0, 1, 1, 1]
    # Host lengths are upper bounds: top-k, or the local KV after this step.
    assert md.helix_row_kv_lens_host[:6].tolist() == [2048] * 3 + [903] * 3
    assert md.helix_row_request_types_host[:6].tolist() == [1] * 6
    assert md.helix_row_block_offsets[0, :6, 0, 0].tolist() == [0, 0, 0, 1, 1, 1]

    view = md.token_major_gen_view(token_sparse=True)
    assert view.num_rows == 6
    assert view.sequence_length.data_ptr() == md.helix_row_kv_lens_cuda.data_ptr()
    # Dense MLA and the RoPE append keep the request-major presentation.
    md._attn_num_rows = 0
    assert md.token_major_gen_view() is None


def test_helix_sparse_rows_take_each_rows_kept_count() -> None:
    md = _helix_rows_metadata(seq_lens=[1, 1, 1], kv_lens=[9000, 9000, 9000])
    rows = torch.full((3, 2048), -1, dtype=torch.int32)
    rows[0, :128] = torch.arange(128)
    rows[2, :7] = torch.arange(7)

    assert md.helix_sparse_empty_rows(3) is None  # no layer has set them yet
    md.set_helix_sparse_row_lens(rows)

    assert md.helix_row_kv_lens_cuda[:3].tolist() == [128, 0, 7]
    # The row that kept nothing must not enter the Helix combine.
    assert md.helix_sparse_empty_rows(3).tolist() == [False, True, False]


def test_helix_sparse_rows_cover_every_verify_token() -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )

    md = SimpleNamespace(
        enable_helix=True,
        kv_cache_manager=SimpleNamespace(num_pools=1, max_blocks_per_seq=4),
        max_num_sequences=8,
        _draft_sizing_cap=5,
        cuda_graph_buffers=None,
        get_empty=lambda buffers, shape, cache_name, dtype, capture_graph: torch.empty(
            shape, dtype=dtype
        ),
    )
    DSAtrtllmAttentionMetadata._create_helix_sparse_row_buffers(md)

    # Eight verify groups of 1 + 5 tokens, plus as many context rows.
    assert md.helix_row_kv_lens_cuda.shape[0] >= 8 * (1 + 5) + 8
    assert md.helix_row_block_offsets.shape == (1, md.helix_row_kv_lens_cuda.shape[0], 2, 4)
