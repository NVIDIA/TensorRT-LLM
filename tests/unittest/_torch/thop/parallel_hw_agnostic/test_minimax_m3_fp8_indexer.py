# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def _reference(qk, num_heads_q, q_weight, k_weight, position_ids):
    reference = qk.clone()
    torch.ops.trtllm.fused_qk_norm_rope(
        reference,
        num_heads_q,
        1,
        0,
        128,
        64,
        1e-5,
        q_weight,
        k_weight,
        10000.0,
        True,  # is_neox
        position_ids,
        1.0,
        0.0,
        0.0,
        1.0,
        True,  # is_qk_norm
        True,  # use_gemma
        False,  # use_mrope
        0,
        0,
    )
    q, k = reference.split([num_heads_q * 128, 128], dim=-1)
    return q.view(q.shape[0], num_heads_q, 128).to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)


def _strided_cache(num_pages, page_size=128, stride_scale=7):
    backing = torch.zeros(
        num_pages * stride_scale,
        1,
        page_size,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return backing[::stride_scale]


def _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q=4):
    return torch.ops.trtllm.minimax_m3_fp8_indexer_qk_norm_rope(
        qk,
        cache,
        slots,
        num_heads_q,
        128,
        64,
        1e-5,
        q_weight,
        k_weight,
        10000.0,
        position_ids,
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 129])
def test_minimax_m3_fp8_indexer_matches_bf16_then_cast(num_tokens):
    torch.manual_seed(1234)
    num_heads_q = 4
    page_size = 128
    qk = torch.randn(
        num_tokens,
        (num_heads_q + 1) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 8192
    within = torch.arange(num_tokens, dtype=torch.int32, device="cuda") % page_size
    pages = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    slots = pages * page_size + within
    cache = _strided_cache(num_tokens)

    q_out = _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)
    k_out = cache[pages.long(), 0, within.long()]

    assert torch.equal(q_out.view(torch.uint8), q_ref.contiguous().view(torch.uint8))
    assert torch.equal(k_out.view(torch.uint8), k_ref.contiguous().view(torch.uint8))


def test_minimax_m3_fp8_indexer_cuda_graph_replay_updates_outputs():
    torch.manual_seed(5678)
    num_tokens = 16
    num_heads_q = 4
    qk = torch.randn(
        num_tokens,
        (num_heads_q + 1) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 4096
    pages = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    within = (pages * 11) % 128
    slots = pages * 128 + within
    cache = _strided_cache(num_tokens)

    for _ in range(3):
        _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q_out = _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)

    first_q = q_out.clone()
    qk.copy_(torch.randn_like(qk))
    graph.replay()
    torch.cuda.synchronize()
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)
    k_out = cache[pages.long(), 0, within.long()]

    assert not torch.equal(q_out.view(torch.uint8), first_q.view(torch.uint8))
    assert torch.equal(q_out.view(torch.uint8), q_ref.contiguous().view(torch.uint8))
    assert torch.equal(k_out.view(torch.uint8), k_ref.contiguous().view(torch.uint8))
