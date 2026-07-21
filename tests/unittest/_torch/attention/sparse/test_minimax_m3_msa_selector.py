# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fused MiniMax-M3 MSA block selector."""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import _INIT_SCORE, _LOCAL_SCORE
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
    select_blocks_from_maxscore,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_select_blocks(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    num_kv_heads, n_blocks, total_q = max_score_kv.shape
    device = max_score_kv.device
    scores = max_score_kv.permute(2, 0, 1).to(torch.float32).clone()
    block_ids = torch.arange(n_blocks, device=device, dtype=torch.long)
    nvb = n_valid_blocks.to(device=device, dtype=torch.long)

    if init_blocks > 0:
        init_mask = block_ids.view(1, 1, -1) < init_blocks
        scores = torch.where(init_mask, torch.full_like(scores, _INIT_SCORE), scores)
    if local_blocks > 0:
        local_start = (nvb - local_blocks).clamp_min(0)
        local_mask = (block_ids.view(1, -1) >= local_start.view(-1, 1)) & (
            block_ids.view(1, -1) < nvb.view(-1, 1)
        )
        scores = torch.where(local_mask.unsqueeze(1), torch.full_like(scores, _LOCAL_SCORE), scores)
    block_valid = block_ids.view(1, -1) < nvb.view(-1, 1)
    scores = scores.masked_fill(~block_valid.unsqueeze(1), float("-inf"))

    k = min(topk, n_blocks)
    vals, idx = scores.topk(k=k, dim=-1)
    idx = torch.where(vals != float("-inf"), idx, torch.full_like(idx, -1))
    sort_key = torch.where(idx < 0, torch.full_like(idx, n_blocks), idx)
    sort_key, _ = torch.sort(sort_key, dim=-1)
    idx = torch.where(sort_key >= n_blocks, torch.full_like(sort_key, -1), sort_key)
    if k < topk:
        pad = torch.full(
            (total_q, num_kv_heads, topk - k),
            -1,
            dtype=idx.dtype,
            device=device,
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx.to(torch.int32)


@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_blocks", [1, 8, 16, 17, 127, 1024, 1537])
def test_fused_selector_matches_reference_random(num_kv_heads, num_blocks):
    total_q = 19
    generator = torch.Generator(device="cuda").manual_seed(num_blocks)
    scores = torch.randn(
        num_kv_heads,
        num_blocks,
        total_q,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    n_valid_blocks = torch.randint(
        0,
        num_blocks + 1,
        (total_q,),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )

    assert actual.dtype == torch.int32
    assert actual.shape == (total_q, num_kv_heads, 16)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("init_blocks", "local_blocks"),
    [(0, 0), (0, 1), (2, 3), (16, 1), (20, 0), (0, 20)],
)
def test_fused_selector_matches_reference_forced_and_padded(init_blocks, local_blocks):
    scores = (
        torch.tensor(
            [
                [
                    float("-inf"),
                    4.0,
                    3.0,
                    2.0,
                    1.0,
                    0.0,
                    -1.0,
                    -2.0,
                    -3.0,
                    -4.0,
                    -5.0,
                    -6.0,
                    -7.0,
                    -8.0,
                    -9.0,
                    -10.0,
                    -11.0,
                    -12.0,
                    -13.0,
                    -14.0,
                ]
            ],
            device="cuda",
            dtype=torch.float32,
        )
        .unsqueeze(-1)
        .expand(-1, -1, 5)
    )
    n_valid_blocks = torch.tensor([0, 1, 7, 16, 20], dtype=torch.int32)

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("fill_value", [0.0, 1.0e30, 1.0e29])
def test_fused_selector_matches_reference_equal_score_ties(fill_value):
    scores = torch.full((2, 64, 3), fill_value, device="cuda", dtype=torch.float32)
    n_valid_blocks = torch.tensor([15, 32, 64], dtype=torch.int32)

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=20,
        local_blocks=0,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=20,
        local_blocks=0,
    )

    assert torch.equal(actual, expected)


def test_fused_selector_matches_reference_nonfinite_and_validity_bounds():
    scores = (
        torch.tensor(
            [
                float("nan"),
                float("inf"),
                float("-inf"),
                -1.0,
                0.0,
                1.0,
                float("nan"),
                float("-inf"),
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            device="cuda",
            dtype=torch.float32,
        )
        .view(1, 20, 1)
        .expand(-1, -1, 4)
    )
    n_valid_blocks = torch.tensor([-3, 0, 17, 25], dtype=torch.int32)

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )

    assert torch.equal(actual, expected)


def test_fused_selector_supports_strided_scores_and_cuda_validity():
    generator = torch.Generator(device="cuda").manual_seed(7)
    backing = torch.randn(2, 73, 22, generator=generator, device="cuda")
    scores = backing[:, 1:72:2, ::2]
    assert not scores.is_contiguous()
    n_valid_blocks = torch.tensor(
        [0, 1, 3, 8, 15, 16, 17, 20, 30, 35, 36], device="cuda", dtype=torch.int32
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )

    assert torch.equal(actual, expected)


def test_fused_selector_cuda_graph_replay_updates_inputs():
    scores = torch.randn(1, 64, 4, device="cuda")
    n_valid_blocks = torch.tensor([16, 32, 48, 64], device="cuda", dtype=torch.int32)

    for _ in range(3):
        output = select_blocks_from_maxscore(
            scores,
            topk=16,
            n_valid_blocks=n_valid_blocks,
            init_blocks=0,
            local_blocks=1,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = select_blocks_from_maxscore(
            scores,
            topk=16,
            n_valid_blocks=n_valid_blocks,
            init_blocks=0,
            local_blocks=1,
        )

    scores.copy_(torch.arange(64, device="cuda", dtype=torch.float32).view(1, 64, 1))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )
    assert torch.equal(output, expected)
