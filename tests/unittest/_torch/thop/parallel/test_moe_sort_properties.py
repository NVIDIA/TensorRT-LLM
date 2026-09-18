# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural properties of trtllm::moe_sort (post-topK routing sort) that
every kernel variant (single block, dynamic block, cluster, cooperative) has to
satisfy, checked across the token counts that pick different variants."""
import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TILE = 128


def _check(num_tokens, num_experts, top_k, local_offset, local_num_experts, seed):
    torch.manual_seed(seed)
    ids = torch.stack([torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(num_tokens)]).to(torch.int32)
    scales = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
    outs = torch.ops.trtllm.moe_sort(ids, scales, num_experts, top_k, local_offset, local_num_experts, TILE)
    tile_to_expert, tile_limit, exp_to_perm, perm_to_exp, total_padded, num_tiles = [o.cpu() for o in outs]
    ids = ids.cpu()
    num_tiles = int(num_tiles.item())
    total_padded = int(total_padded.item())

    local = (ids >= local_offset) & (ids < local_offset + local_num_experts)
    counts = torch.bincount((ids[local] - local_offset).long(), minlength=local_num_experts)
    expected_tiles = int(((counts + TILE - 1) // TILE).sum().item())
    assert num_tiles == expected_tiles
    assert total_padded == num_tiles * TILE
    assert num_tiles <= tile_to_expert.numel() and total_padded <= perm_to_exp.numel()

    # Non-local selections are marked -1; local ones point into the padded range.
    assert torch.all(exp_to_perm[~local] == -1)
    perm = exp_to_perm[local].long()
    assert torch.all(perm >= 0) and torch.all(perm < total_padded)
    assert perm.unique().numel() == perm.numel()  # every local pair owns one row

    # Rows are grouped by tile: [tile*TILE, tile_limit[tile]) holds that tile's
    # expert only, the inverse map agrees, and the padded tail is not addressed.
    seen_rows = 0
    prev_expert = -1
    for tile in range(num_tiles):
        lo, hi = tile * TILE, int(tile_limit[tile].item())
        assert lo < hi <= lo + TILE, (tile, lo, hi)
        expert = int(tile_to_expert[tile].item())
        assert 0 <= expert < local_num_experts
        assert expert >= prev_expert  # experts appear in ascending order
        prev_expert = expert
        for row in range(lo, hi):
            expanded = int(perm_to_exp[row].item())
            token, k = divmod(expanded, top_k)
            assert 0 <= token < num_tokens and 0 <= k < top_k
            assert int(ids[token, k].item()) - local_offset == expert, (tile, row)
            assert int(exp_to_perm[token, k].item()) == row
        seen_rows += hi - lo
    assert seen_rows == int(local.sum().item())
    # Per-expert tile counts match the histogram.
    tiles_per_expert = torch.bincount(tile_to_expert[:num_tiles].long(), minlength=local_num_experts)
    assert torch.equal(tiles_per_expert, (counts + TILE - 1) // TILE)


TOKENS = [1, 3, 4, 5, 8, 12, 16, 17, 31, 32, 33, 64, 200, 1025]
CASES = ([(t, 512, 10, 8) for t in TOKENS] + [(t, 256, 8, 1) for t in TOKENS] + [(t, 64, 6, 2) for t in TOKENS]
         + [(700, 8, 2, 1)])  # few experts, many tokens: experts own several tiles each


@pytest.mark.parametrize("num_tokens,num_experts,top_k,ep_size", CASES)
def test_moe_sort_properties(num_tokens, num_experts, top_k, ep_size):
    local_num_experts = num_experts // ep_size
    for rank in {0, ep_size - 1}:
        _check(num_tokens, num_experts, top_k, rank * local_num_experts, local_num_experts, seed=num_tokens * 7 + rank)
