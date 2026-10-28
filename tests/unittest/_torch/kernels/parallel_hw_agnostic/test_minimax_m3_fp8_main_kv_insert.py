# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(qkv, num_heads_q, num_kv_heads, q_weight, k_weight, position_ids):
    output = torch.ops.trtllm.fused_qk_norm_rope_to_fp8(
        qkv,
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
    return output.view(qkv.shape[0], num_heads_q + 2 * num_kv_heads, 128).split(
        [num_heads_q, num_kv_heads, num_kv_heads], dim=1
    )


def _strided_kv_cache(num_pages, num_kv_heads, page_size=128, stride_scale=3):
    backing = torch.zeros(
        num_pages * stride_scale,
        2,
        num_kv_heads,
        page_size,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return backing[::stride_scale]


def _inputs(num_tokens, num_heads_q, num_kv_heads):
    qkv = torch.randn(
        num_tokens,
        (num_heads_q + 2 * num_kv_heads) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 8192
    return qkv, q_weight, k_weight, position_ids


def _run(qkv, kv_cache, slots, q_weight, k_weight, position_ids, num_heads_q, num_kv_heads):
    return torch.ops.trtllm.minimax_m3_fp8_qk_norm_rope_kv_insert(
        qkv,
        kv_cache,
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


@pytest.mark.parametrize(("num_heads_q", "num_kv_heads"), [(8, 1), (8, 8), (64, 4)])
@pytest.mark.parametrize("num_tokens", [1, 16, 129])
def test_minimax_m3_fp8_main_kv_insert_matches_materialize_then_scatter(
    num_tokens, num_heads_q, num_kv_heads
):
    torch.manual_seed(1234)
    page_size = 128
    num_pages = max(4, (num_tokens + page_size - 1) // page_size + 2)
    qkv, q_weight, k_weight, position_ids = _inputs(num_tokens, num_heads_q, num_kv_heads)
    slots = (torch.arange(num_tokens, dtype=torch.int32, device="cuda") * 37) % (
        (num_pages - 1) * page_size
    )
    kv_cache = _strided_kv_cache(num_pages, num_kv_heads, page_size)
    guard_page = kv_cache[-1].clone()

    q_out = _run(
        qkv,
        kv_cache,
        slots,
        q_weight,
        k_weight,
        position_ids,
        num_heads_q,
        num_kv_heads,
    )
    q_ref, k_ref, v_ref = _reference(
        qkv, num_heads_q, num_kv_heads, q_weight, k_weight, position_ids
    )
    pages = slots.long() // page_size
    within = slots.long() % page_size

    # The specialized kernel uses powf while fused_qk_norm_rope_to_fp8 uses
    # the exp2f/log2f equivalent, so values at an FP8 boundary can round to
    # adjacent E4M3 values.
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=0.13, atol=0.05)
    torch.testing.assert_close(
        kv_cache[:, 0][pages, :, within, :].float(),
        k_ref.float(),
        rtol=0.13,
        atol=0.05,
    )
    assert torch.equal(
        kv_cache[:, 1][pages, :, within, :].view(torch.uint8),
        v_ref.contiguous().view(torch.uint8),
    )
    assert torch.equal(kv_cache[-1].view(torch.uint8), guard_page.view(torch.uint8))


def test_minimax_m3_fp8_main_kv_insert_uses_64bit_cache_offsets():
    """Exercise a real paged-cache address beyond INT32_MAX elements.

    The smallest contiguous HND pool whose page-65536 K row starts at
    2**31 FP8 elements is about 2 GiB. The former implicit int conversion in
    the store helper wrapped this address negative; this test writes and reads
    that real allocation so arithmetic-only tests cannot mask the bug.
    """
    required_bytes = (65537 * 2 * 128 * 128) + (1 << 30)
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes:
        pytest.skip("64-bit cache-offset test requires about 3 GiB free GPU memory")

    torch.manual_seed(4321)
    page = 65536
    kv_cache = torch.empty(
        page + 1,
        2,
        1,
        128,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    qkv, q_weight, k_weight, position_ids = _inputs(1, 8, 1)
    slots = torch.tensor([page * 128], dtype=torch.int32, device="cuda")

    q_out = _run(qkv, kv_cache, slots, q_weight, k_weight, position_ids, 8, 1)
    q_ref, k_ref, v_ref = _reference(qkv, 8, 1, q_weight, k_weight, position_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=0.13, atol=0.05)
    torch.testing.assert_close(
        kv_cache[page, 0, :, 0, :].float(),
        k_ref[0].float(),
        rtol=0.13,
        atol=0.05,
    )
    assert torch.equal(
        kv_cache[page, 1, :, 0, :].view(torch.uint8),
        v_ref[0].contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("invalid_slot", [-1, 2 * 128])
def test_minimax_m3_fp8_main_kv_insert_ignores_invalid_slot(invalid_slot):
    torch.manual_seed(5678)
    qkv, q_weight, k_weight, position_ids = _inputs(1, 8, 1)
    kv_cache = _strided_kv_cache(2, 1)
    kv_cache.fill_(1.0)
    before = kv_cache.clone()
    slots = torch.tensor([invalid_slot], dtype=torch.int32, device="cuda")

    q_out = _run(qkv, kv_cache, slots, q_weight, k_weight, position_ids, 8, 1)
    q_ref, _, _ = _reference(qkv, 8, 1, q_weight, k_weight, position_ids)

    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=0.13, atol=0.05)
    assert torch.equal(kv_cache.view(torch.uint8), before.view(torch.uint8))
