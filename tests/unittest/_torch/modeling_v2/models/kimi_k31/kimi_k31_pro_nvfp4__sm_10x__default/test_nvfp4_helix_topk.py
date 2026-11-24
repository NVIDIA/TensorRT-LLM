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

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.indexer import (  # noqa: E501
    merge_helix_topk,
)


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
