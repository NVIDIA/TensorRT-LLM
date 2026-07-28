# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rope_cache(max_positions, rotary_dim=64, base=5_000_000.0):
    positions = torch.arange(max_positions, dtype=torch.float32, device="cuda")
    inverse_frequency = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    frequency = torch.outer(positions, inverse_frequency)
    return torch.stack((frequency.cos(), frequency.sin()), dim=1).contiguous()


def _main_cache(num_pages, num_kv_heads, stride_scale=3):
    backing = torch.zeros(
        num_pages * stride_scale,
        2,
        num_kv_heads,
        128,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return backing[::stride_scale]


def _index_cache(num_pages, stride_scale=5):
    backing = torch.zeros(
        num_pages * stride_scale,
        1,
        128,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return backing[::stride_scale]


@pytest.mark.parametrize("num_tokens", [1, 16, 129])
def test_minimax_m3_horizontal_producer_matches_separate_producers(num_tokens):
    torch.manual_seed(1234)
    num_heads_q = 8
    num_kv_heads = 2
    num_index_heads = num_kv_heads
    num_pages = max(4, (num_tokens + 127) // 128 + 2)
    total_heads = num_heads_q + 2 * num_kv_heads + num_index_heads + 1
    packed = torch.randn(
        num_tokens,
        total_heads * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    index_q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    index_k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    slots = (torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 37) % (
        (num_pages - 1) * 128
    )
    rope_cache = _rope_cache(max(256, num_tokens))

    main_width = (num_heads_q + 2 * num_kv_heads) * 128
    main_input = packed[:, :main_width].contiguous()
    index_input = packed[:, main_width:].contiguous()
    reference_main_cache = _main_cache(num_pages, num_kv_heads)
    reference_index_cache = _index_cache(num_pages)
    q_reference = torch.ops.trtllm.minimax_m3_fp8_qk_norm_rope_kv_insert(
        main_input,
        reference_main_cache,
        slots,
        num_heads_q,
        num_kv_heads,
        num_kv_heads,
        128,
        64,
        1e-5,
        q_weight,
        k_weight,
        5_000_000.0,
        True,
        position_ids,
    )
    index_q_reference = torch.ops.trtllm.minimax_m3_fp8_indexer_qk_norm_rope(
        index_input,
        reference_index_cache,
        slots,
        num_index_heads,
        128,
        64,
        1e-5,
        index_q_weight,
        index_k_weight,
        5_000_000.0,
        position_ids,
    )

    main_cache = _main_cache(num_pages, num_kv_heads)
    index_cache = _index_cache(num_pages)
    q, index_q = torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
        packed,
        main_cache,
        index_cache,
        slots,
        num_heads_q,
        num_kv_heads,
        num_index_heads,
        128,
        64,
        1e-5,
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
        rope_cache,
        position_ids,
    )

    pages = slots.long() // 128
    within = slots.long() % 128
    assert torch.equal(q.view(torch.uint8), q_reference.view(torch.uint8))
    # The horizontal producer follows vLLM's CUDA contract and converts its
    # normalized/RoPE FP32 registers directly to E4M3. The existing separate
    # TRT-LLM index producer first materializes BF16, so compare within one FP8
    # ULP rather than requiring byte identity across the different rounding
    # orders. Main Q/K/V retain byte-exact parity above and below.
    torch.testing.assert_close(
        index_q.float(),
        index_q_reference.float(),
        rtol=0.13,
        atol=0.05,
    )
    assert torch.equal(
        main_cache[pages, :, :, within, :].view(torch.uint8),
        reference_main_cache[pages, :, :, within, :].view(torch.uint8),
    )
    torch.testing.assert_close(
        index_cache[pages, :, within, :].float(),
        reference_index_cache[pages, :, within, :].float(),
        rtol=0.13,
        atol=0.05,
    )
