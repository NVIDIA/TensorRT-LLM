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


def _assert_fp8_within_one_ulp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Allow adjacent finite E4M3 values, including subnormals and signed zero."""
    assert actual.dtype == expected.dtype == torch.float8_e4m3fn
    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(expected.float()).all()
    # E4M3 encodes sign/magnitude. Map both signs to monotonically ordered
    # integers, collapsing +0 and -0, so atol=1 means exactly one FP8 step.
    actual_bits = actual.view(torch.uint8).to(torch.int16)
    expected_bits = expected.view(torch.uint8).to(torch.int16)
    actual_ordered = torch.where(actual_bits < 128, actual_bits, 128 - actual_bits)
    expected_ordered = torch.where(expected_bits < 128, expected_bits, 128 - expected_bits)
    torch.testing.assert_close(actual_ordered, expected_ordered, rtol=0, atol=1)


def test_minimax_m3_fp8_one_ulp_comparison() -> None:
    # Cover every adjacent finite pair, including exponent boundaries and
    # subnormals, for both signs. Two-step errors and NaNs must still fail.
    for sign in (0, 128):
        values = (torch.arange(127, dtype=torch.uint8) + sign).view(torch.float8_e4m3fn)
        _assert_fp8_within_one_ulp(values[:-1], values[1:])
        with pytest.raises(AssertionError):
            _assert_fp8_within_one_ulp(values[:-2], values[2:])
    _assert_fp8_within_one_ulp(
        torch.tensor([0, 128, 1, 129], dtype=torch.uint8).view(torch.float8_e4m3fn),
        torch.tensor([128, 0, 128, 0], dtype=torch.uint8).view(torch.float8_e4m3fn),
    )
    with pytest.raises(AssertionError):
        _assert_fp8_within_one_ulp(
            torch.tensor([1], dtype=torch.uint8).view(torch.float8_e4m3fn),
            torch.tensor([129], dtype=torch.uint8).view(torch.float8_e4m3fn),
        )
    for nan_bits in (127, 255):
        nan = torch.tensor([nan_bits], dtype=torch.uint8).view(torch.float8_e4m3fn)
        with pytest.raises(AssertionError):
            _assert_fp8_within_one_ulp(nan, nan)


@pytest.mark.parametrize("num_tokens", [1, 16, 129])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", [(4, 4), (2, 2), (1, 1)])
def test_minimax_m3_horizontal_producer_matches_separate_producers(
    num_tokens, num_kv_heads, num_index_heads
):
    torch.manual_seed(1234)
    num_heads_q = 8
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
    # Keep the parity reference slots valid: the legacy separate main-K/V
    # producer does not support negative slots. Negative-slot handling is
    # exercised below using horizontal eager execution versus graph replay.
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

    valid = slots >= 0
    pages = slots[valid].long() // 128
    within = slots[valid].long() % 128
    # The horizontal producer reads precomputed FP32 RoPE coefficients; the
    # separate main producer computes powf/__sincosf per head. Near an FP8
    # midpoint, their small FP32 differences can round to adjacent E4M3 bins.
    # Bound Q/K differences to one actual FP8 step, not a broad atol/rtol.
    _assert_fp8_within_one_ulp(q, q_reference)
    # The horizontal producer follows vLLM's CUDA contract and converts its
    # normalized/RoPE FP32 registers directly to E4M3. The existing separate
    # TRT-LLM index producer first materializes BF16, so retain its numerical
    # tolerance across the different rounding orders. V is copy-cast only and
    # retains byte-exact parity below.
    torch.testing.assert_close(
        index_q.float(),
        index_q_reference.float(),
        rtol=0.13,
        atol=0.05,
    )
    _assert_fp8_within_one_ulp(
        main_cache[pages, 0, :, within, :],
        reference_main_cache[pages, 0, :, within, :],
    )
    assert torch.equal(
        main_cache[pages, 1, :, within, :].view(torch.uint8),
        reference_main_cache[pages, 1, :, within, :].view(torch.uint8),
    )
    torch.testing.assert_close(
        index_cache[pages, :, within, :].float(),
        reference_index_cache[pages, :, within, :].float(),
        rtol=0.13,
        atol=0.05,
    )

    # Aggregate decode captures this producer in a CUDA graph. The operator
    # allocates compact Q/index-Q outputs while writing graph-stable paged
    # caches through a graph-stable slot mapping, so exercise both capture and
    # replay for decode-sized (1) and larger mixed/prefill token counts.
    graph_packed = packed.clone()
    graph_positions = position_ids.clone()
    graph_slots = slots.clone()
    graph_main_cache = _main_cache(num_pages, num_kv_heads)
    graph_index_cache = _index_cache(num_pages)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_q, graph_index_q = torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
            graph_packed,
            graph_main_cache,
            graph_index_cache,
            graph_slots,
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
            graph_positions,
        )

    # Replay with different projection values, nonuniform positions, and new
    # cache destinations. This proves replay reads the refreshed graph buffers
    # rather than retaining capture-time values or slots.
    replay_packed = torch.randn_like(packed)
    replay_positions = (
        torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 7 + 3
    ) % rope_cache.shape[0]
    replay_slots = (torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 53 + 11) % (
        (num_pages - 1) * 128
    )
    if num_tokens > 1:
        replay_slots[-1] = -1
    graph_packed.copy_(replay_packed)
    graph_positions.copy_(replay_positions)
    graph_slots.copy_(replay_slots)
    graph_main_cache.zero_()
    graph_index_cache.zero_()

    replay_main_cache = _main_cache(num_pages, num_kv_heads)
    replay_index_cache = _index_cache(num_pages)
    replay_q, replay_index_q = torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
        replay_packed,
        replay_main_cache,
        replay_index_cache,
        replay_slots,
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
        replay_positions,
    )
    graph.replay()
    torch.cuda.synchronize()

    replay_valid = replay_slots >= 0
    replay_pages = replay_slots[replay_valid].long() // 128
    replay_within = replay_slots[replay_valid].long() % 128
    assert torch.equal(graph_q.view(torch.uint8), replay_q.view(torch.uint8))
    torch.testing.assert_close(
        graph_index_q.float(),
        replay_index_q.float(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        graph_main_cache[replay_pages, :, :, replay_within, :].view(torch.uint8),
        replay_main_cache[replay_pages, :, :, replay_within, :].view(torch.uint8),
    )
    torch.testing.assert_close(
        graph_index_cache[replay_pages, :, replay_within, :].float(),
        replay_index_cache[replay_pages, :, replay_within, :].float(),
        rtol=0.0,
        atol=0.0,
    )


def test_minimax_m3_horizontal_producer_ignores_out_of_range_cache_slot():
    num_heads_q = 8
    num_kv_heads = 2
    total_heads = num_heads_q + 3 * num_kv_heads + 1
    packed = torch.randn(1, total_heads * 128, dtype=torch.bfloat16, device="cuda")
    weights = [torch.randn(128, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    positions = torch.zeros(1, dtype=torch.int32, device="cuda")
    slots = torch.tensor([2 * 128], dtype=torch.int32, device="cuda")
    main_cache = _main_cache(2, num_kv_heads)
    index_cache = _index_cache(2)
    main_before = main_cache.clone()
    index_before = index_cache.clone()

    q, index_q = torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
        packed,
        main_cache,
        index_cache,
        slots,
        num_heads_q,
        num_kv_heads,
        num_kv_heads,
        128,
        64,
        1e-5,
        *weights,
        _rope_cache(1),
        positions,
    )

    assert q.shape == (1, num_heads_q, 128)
    assert index_q.shape == (1, num_kv_heads, 128)
    assert torch.equal(main_cache.view(torch.uint8), main_before.view(torch.uint8))
    assert torch.equal(index_cache.view(torch.uint8), index_before.view(torch.uint8))


@pytest.mark.parametrize("invalid_weight_index", range(4))
def test_minimax_m3_horizontal_producer_rejects_non_vector_norm_weight(
    invalid_weight_index: int,
) -> None:
    num_heads_q = 8
    num_kv_heads = 2
    num_index_heads = num_kv_heads
    total_heads = num_heads_q + 2 * num_kv_heads + num_index_heads + 1
    packed = torch.randn(1, total_heads * 128, dtype=torch.bfloat16, device="cuda")
    weights = [torch.randn(128, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    weights[invalid_weight_index] = weights[invalid_weight_index].reshape(1, 128)
    positions = torch.zeros(1, dtype=torch.int32, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="norm weights must be one-dimensional"):
        torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
            packed,
            _main_cache(1, num_kv_heads),
            _index_cache(1),
            slots,
            num_heads_q,
            num_kv_heads,
            num_index_heads,
            128,
            64,
            1e-5,
            *weights,
            _rope_cache(1),
            positions,
        )
