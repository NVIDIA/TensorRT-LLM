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
    valid_rows = torch.nonzero(slots >= 0, as_tuple=False).flatten()
    for row in valid_rows.tolist():
        page, within = divmod(int(slots[row].item()), 128)
        assert torch.equal(data_cache[page, 0, :, within], expected_k_data[row])
        assert torch.equal(data_cache[page, 1, :, within], expected_v_data[row])
        assert torch.equal(scale_cache[page, 0, :, within], expected_k_scale[row])
        for head in range(data_cache.shape[2]):
            v_region = scale_cache[page, 1, head].view(-1)
            offsets = (
                (within // 4) * 32
                + torch.arange(8, device="cuda", dtype=torch.long) * 4
                + within % 4
            )
            assert torch.equal(v_region[offsets], expected_v_scale[row, head])


@pytest.mark.parametrize(
    "num_tokens,num_heads_q,num_kv_heads",
    [
        pytest.param(1, 16, 1, id="tp4-hq16-hkv1-single-token"),
        pytest.param(16, 32, 2, id="tp2-hq32-hkv2"),
        pytest.param(16, 64, 4, id="replicated-hq64-hkv4"),
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
    data_cache, scale_cache, index_cache = _nvfp4_caches(num_pages, num_kv_heads)
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
    valid = slots >= 0
    pages = slots[valid].long() // 128
    within = slots[valid].long() % 128
    assert torch.equal(
        index_cache[pages, :, within].view(torch.uint8),
        fp8_index_cache[pages, :, within].view(torch.uint8),
    )
    _assert_nvfp4_cache_equal(
        data_cache,
        scale_cache,
        slots,
        expected_k_data,
        expected_v_data,
        expected_k_scale,
        expected_v_scale,
    )


def test_minimax_m3_nvfp4_horizontal_producer_rejects_non_m3_head_ratio() -> None:
    num_tokens, num_heads_q, num_kv_heads, num_pages = 1, 8, 2, 4
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

    with pytest.raises(RuntimeError, match="16:1 Q-to-KV head ratio"):
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


def test_minimax_m3_nvfp4_horizontal_producer_rejects_mismatched_page_counts() -> None:
    num_tokens, num_heads_q, num_kv_heads, num_pages = 1, 16, 1, 4
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

    with pytest.raises(RuntimeError, match="same number of pages"):
        _run_nvfp4(
            packed,
            data_cache,
            scale_cache,
            index_cache[:-1],
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
