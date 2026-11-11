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


def _nvfp4_caches(num_pages, num_kv_heads):
    data_backing = torch.zeros(
        num_pages * 3,
        2,
        num_kv_heads,
        128,
        64,
        dtype=torch.uint8,
        device="cuda",
    )
    scale_backing = torch.zeros(
        num_pages * 5,
        2,
        num_kv_heads,
        128,
        8,
        dtype=torch.uint8,
        device="cuda",
    )
    index_backing = torch.zeros(
        num_pages * 7,
        1,
        128,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return data_backing[::3], scale_backing[::5], index_backing[::7]


def _fp8_caches(num_pages, num_kv_heads):
    main_backing = torch.zeros(
        num_pages * 3,
        2,
        num_kv_heads,
        128,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    index_backing = torch.zeros(
        num_pages * 7,
        1,
        128,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return main_backing[::3], index_backing[::7]


def _run_nvfp4(
    packed,
    data_cache,
    scale_cache,
    index_cache,
    slots,
    inv_scales,
    num_heads_q,
    num_kv_heads,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    rope_cache,
    position_ids,
):
    return torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
        packed,
        data_cache,
        scale_cache,
        index_cache,
        slots,
        inv_scales,
        num_heads_q,
        num_kv_heads,
        num_kv_heads,
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


def _assert_nvfp4_cache_equal(
    data_cache,
    scale_cache,
    slots,
    expected_k_data,
    expected_v_data,
    expected_k_scale,
    expected_v_scale,
):
    expected_data_pool = torch.zeros_like(data_cache)
    expected_scale_pool = torch.zeros_like(scale_cache)
    valid_rows = torch.nonzero(
        (slots >= 0) & (slots < data_cache.shape[0] * 128), as_tuple=False
    ).flatten()
    for row in valid_rows.tolist():
        page, within = divmod(int(slots[row].item()), 128)
        expected_data_pool[page, 0, :, within] = expected_k_data[row]
        expected_data_pool[page, 1, :, within] = expected_v_data[row]
        expected_scale_pool[page, 0, :, within] = expected_k_scale[row]
        for head in range(data_cache.shape[2]):
            v_region = expected_scale_pool[page, 1, head].view(-1)
            offsets = (
                (within // 4) * 32
                + torch.arange(8, device="cuda", dtype=torch.long) * 4
                + within % 4
            )
            v_region[offsets] = expected_v_scale[row, head]
    assert torch.equal(data_cache, expected_data_pool)
    assert torch.equal(scale_cache, expected_scale_pool)


@pytest.mark.parametrize(
    "num_tokens,num_heads_q,num_kv_heads",
    [
        pytest.param(1, 8, 1, id="tp8-replicated-kv-hq8-hkv1-single-token"),
        pytest.param(1, 16, 1, id="tp4-hq16-hkv1-single-token"),
        pytest.param(16, 32, 2, id="tp2-hq32-hkv2"),
        pytest.param(16, 64, 4, id="attention-dp-hq64-hkv4"),
        pytest.param(129, 16, 1, id="multi-page-hq16-hkv1"),
    ],
)
def test_minimax_m3_nvfp4_horizontal_producer_matches_production_quantize(
    num_tokens, num_heads_q, num_kv_heads
):
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        pytest.skip("NVFP4 quantization requires Blackwell")

    torch.manual_seed(1234)
    num_pages = max(5, (num_tokens + 127) // 128 + 3)
    total_heads = num_heads_q + 3 * num_kv_heads + 1
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
    inv_scales = torch.tensor([1.0, 1.75, 0.625], dtype=torch.float32, device="cuda")
    # Position zero makes RoPE the identity. This isolates exact RMSNorm,
    # BF16-rounding, E2M1 packing, and E4M3 scale parity from the generic
    # producer's different (powf versus precomputed-table) RoPE evaluation.
    position_ids = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    slots = (torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 53 + 11) % (
        (num_pages - 1) * 128
    )
    if num_tokens > 1:
        slots[-1] = -1
    if num_tokens > 2:
        slots[-2] = num_pages * 128
    rope_cache = _rope_cache(512)

    main_width = (num_heads_q + 2 * num_kv_heads) * 128
    # A size-one leading dimension can make this narrower view report as
    # contiguous even though it aliases ``packed``.  The reference op is
    # in-place, so force distinct storage before normalizing Q/K.
    materialized_main = packed[:, :main_width].clone(memory_format=torch.contiguous_format)
    torch.ops.trtllm.fused_qk_norm_rope(
        materialized_main,
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
        1.0,
        0.0,
        0.0,
        1.0,
        True,
        True,
        False,
        0,
        0,
    )
    _, k_materialized, v_materialized = materialized_main.view(
        num_tokens, num_heads_q + 2 * num_kv_heads, 128
    ).split([num_heads_q, num_kv_heads, num_kv_heads], dim=1)
    expected_k_data, expected_k_scale = torch.ops.trtllm.fp4_quantize(
        k_materialized.contiguous(), inv_scales[1:2], 16, False, False
    )
    expected_v_data, expected_v_scale = torch.ops.trtllm.fp4_quantize(
        v_materialized.contiguous(), inv_scales[2:3], 16, False, False
    )
    expected_k_data = expected_k_data.view(torch.uint8)
    expected_v_data = expected_v_data.view(torch.uint8)
    expected_k_scale = expected_k_scale.view(num_tokens, num_kv_heads, 8).view(torch.uint8)
    expected_v_scale = expected_v_scale.view(num_tokens, num_kv_heads, 8).view(torch.uint8)

    fp8_main_cache, fp8_index_cache = _fp8_caches(num_pages, num_kv_heads)
    q_reference, index_q_reference = (
        torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
            packed,
            fp8_main_cache,
            fp8_index_cache,
            slots,
            num_heads_q,
            num_kv_heads,
            num_kv_heads,
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
    )
    # Keep guard pages around the exposed pools so rejected slots cannot
    # silently corrupt an adjacent allocation.
    data_pool, scale_pool, index_pool = _nvfp4_caches(num_pages + 2, num_kv_heads)
    data_cache, scale_cache, index_cache = (
        pool[1:-1] for pool in (data_pool, scale_pool, index_pool)
    )
    # Singleton-head strides do not participate in index-K addressing.
    index_cache = index_cache.as_strided(index_cache.shape, (index_cache.stride(0), 1, 128, 1))
    q, index_q = _run_nvfp4(
        packed,
        data_cache,
        scale_cache,
        index_cache,
        slots,
        inv_scales,
        num_heads_q,
        num_kv_heads,
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
        rope_cache,
        position_ids,
    )

    assert torch.equal(q.view(torch.uint8), q_reference.view(torch.uint8))
    assert torch.equal(index_q.view(torch.uint8), index_q_reference.view(torch.uint8))
    valid = (slots >= 0) & (slots < num_pages * 128)
    pages = slots[valid].long() // 128
    within = slots[valid].long() % 128
    assert torch.equal(
        index_cache[pages, :, within].view(torch.uint8),
        fp8_index_cache[pages, :, within].view(torch.uint8),
    )
    assert torch.equal(index_cache.view(torch.uint8), fp8_index_cache.view(torch.uint8))
    for pool in (data_pool, scale_pool, index_pool.view(torch.uint8)):
        assert torch.count_nonzero(pool[0]).item() == 0
        assert torch.count_nonzero(pool[-1]).item() == 0
    _assert_nvfp4_cache_equal(
        data_cache,
        scale_cache,
        slots,
        expected_k_data,
        expected_v_data,
        expected_k_scale,
        expected_v_scale,
    )


@pytest.mark.parametrize(
    "invalid_input,error",
    [
        ("page_count", "same number of pages"),
        ("index_page_overlap", "pages must not overlap"),
        ("index_page_alignment", "32-bit FP8 store alignment"),
        ("packed_alignment", "8-byte-aligned"),
        ("data_alignment", "aligned addresses"),
        ("index_alignment", "aligned addresses"),
        ("launch_geometry", "launch geometry exceeds int32"),
    ],
)
def test_minimax_m3_nvfp4_horizontal_producer_rejects_invalid_inputs(
    invalid_input: str, error: str
) -> None:
    num_tokens, num_heads_q, num_kv_heads, num_pages = 2, 16, 1, 4
    total_heads = num_heads_q + 3 * num_kv_heads + 1
    packed = torch.randn(
        num_tokens,
        total_heads * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    data_cache, scale_cache, index_cache = _nvfp4_caches(num_pages, num_kv_heads)
    slots = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    inv_scales = torch.ones(3, dtype=torch.float32, device="cuda")
    weights = [torch.ones(128, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    rope_cache = _rope_cache(8)
    position_ids = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")

    if invalid_input == "page_count":
        index_cache = index_cache[:-1]
    elif invalid_input.startswith("index_page_"):
        page_stride = 128 * 128 + (1 if invalid_input.endswith("alignment") else -4)
        index_cache = index_cache.as_strided(index_cache.shape, (page_stride, 1, 128, 1))
    elif invalid_input == "packed_alignment":
        packed = torch.empty(packed.numel() + 1, dtype=packed.dtype, device="cuda")[1:].view_as(
            packed
        )
    elif invalid_input == "data_alignment":
        data_cache = torch.empty(data_cache.numel() + 1, dtype=data_cache.dtype, device="cuda")[
            1:
        ].view_as(data_cache)
    elif invalid_input == "index_alignment":
        index_cache = torch.empty(index_cache.numel() + 1, dtype=index_cache.dtype, device="cuda")[
            1:
        ].view_as(index_cache)
    elif invalid_input == "launch_geometry":
        # Rejected before checking the packed width or allocating outputs.
        num_heads_q = 1 << 30

    with pytest.raises(RuntimeError, match=error):
        _run_nvfp4(
            packed,
            data_cache,
            scale_cache,
            index_cache,
            slots,
            inv_scales,
            num_heads_q,
            num_kv_heads,
            *weights,
            rope_cache,
            position_ids,
        )


def test_minimax_m3_nvfp4_horizontal_producer_cuda_graph_replay() -> None:
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        pytest.skip("NVFP4 quantization requires Blackwell")

    torch.manual_seed(4321)
    num_tokens, num_heads_q, num_kv_heads, num_pages = 16, 16, 1, 8
    total_heads = num_heads_q + 3 * num_kv_heads + 1
    packed = torch.randn(
        num_tokens,
        total_heads * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    slots = torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 17
    inv_scales = torch.tensor([1.0, 1.25, 0.75], dtype=torch.float32, device="cuda")
    weights = [torch.randn(128, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    rope_cache = _rope_cache(512)
    data_cache, scale_cache, index_cache = _nvfp4_caches(num_pages, num_kv_heads)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_q, graph_index_q = _run_nvfp4(
            packed,
            data_cache,
            scale_cache,
            index_cache,
            slots,
            inv_scales,
            num_heads_q,
            num_kv_heads,
            *weights,
            rope_cache,
            position_ids,
        )

    replay_packed = torch.randn_like(packed)
    replay_positions = torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 7 + 3
    replay_slots = (torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 29 + 5) % (
        (num_pages - 1) * 128
    )
    replay_slots[-1] = -1
    replay_scales = torch.tensor([1.0, 0.875, 1.5], dtype=torch.float32, device="cuda")
    packed.copy_(replay_packed)
    position_ids.copy_(replay_positions)
    slots.copy_(replay_slots)
    inv_scales.copy_(replay_scales)
    data_cache.zero_()
    scale_cache.zero_()
    index_cache.zero_()

    reference_data, reference_scale, reference_index = _nvfp4_caches(num_pages, num_kv_heads)
    reference_q, reference_index_q = _run_nvfp4(
        replay_packed,
        reference_data,
        reference_scale,
        reference_index,
        replay_slots,
        replay_scales,
        num_heads_q,
        num_kv_heads,
        *weights,
        rope_cache,
        replay_positions,
    )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(graph_q.view(torch.uint8), reference_q.view(torch.uint8))
    assert torch.equal(graph_index_q.view(torch.uint8), reference_index_q.view(torch.uint8))
    assert torch.equal(data_cache, reference_data)
    assert torch.equal(scale_cache, reference_scale)
    assert torch.equal(index_cache.view(torch.uint8), reference_index.view(torch.uint8))
