# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (SM 10.7): the TP2 dense layers' FP8 q/k/v (minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8) carries the bits
of the stock path, the BF16 fused_qk_norm_rope output cast to E4M3, for every norm / RoPE variant: Q and K are rounded
to BF16 before E4M3 (K is stored in the FP8 cache), V is cast."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

assert torch.cuda.is_available(), "the dense FP8 q/k/v op requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the tp2 target is certified on SM 10.7 only", allow_module_level=True)

HQ, HK, HV, D, ROT = 32, 2, 2, 128, 64


@pytest.mark.parametrize("tokens", [1, 4, 8, 4096])
@pytest.mark.parametrize("use_gemma", [False, True], ids=["rmsnorm", "gemma"])
@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "interleaved"])
def test_the_dense_fp8_qkv_is_the_bf16_kernel_cast(tokens, use_gemma, is_neox) -> None:
    if not hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8"):
        pytest.skip("this build does not carry minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8")
    g = torch.Generator(device="cuda").manual_seed(tokens * 4 + int(use_gemma) * 2 + int(is_neox))
    qkv = (torch.randn(tokens, (HQ + HK + HV) * D, device="cuda", generator=g) * 2).to(
        torch.bfloat16
    )
    pos = torch.randint(0, 1 << 20, (tokens,), device="cuda", generator=g, dtype=torch.int32)
    qw = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    kw = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    ours = torch.ops.trtllm.minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8(
        qkv, HQ, HK, HV, ROT, 1e-6, qw, kw, 5e6, is_neox, pos, use_gemma
    )
    stock = qkv.clone()
    torch.ops.trtllm.fused_qk_norm_rope(
        stock, HQ, HK, HV, D, ROT, 1e-6, qw, kw, 5e6, is_neox, pos,
        1.0, 0.0, 0.0, 1.0, True, use_gemma, False, 0, 0,
    )  # fmt: skip
    assert torch.equal(ours.view(torch.uint8), stock.to(torch.float8_e4m3fn).view(torch.uint8))


@pytest.mark.parametrize("tokens,live", [(1, 1), (4, 4), (8, 5), (64, 64)])
@pytest.mark.parametrize("tokens_per_block", [32, 128])
@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "interleaved"])
def test_the_fused_kv_insert_stores_the_bytes_of_write_kv_slots_and_the_compact_q(
    tokens, live, tokens_per_block, is_neox
) -> None:
    if not hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv"):
        pytest.skip("this build does not carry minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv")
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
        write_kv_slots,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens * 1000 + tokens_per_block + int(is_neox))
    qkv = (torch.randn(tokens, (HQ + HK + HV) * D, device="cuda", generator=g) * 2).to(
        torch.bfloat16
    )
    pos = torch.randint(0, 1 << 20, (tokens,), device="cuda", generator=g, dtype=torch.int32)
    qw = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    kw = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    pages, layers, layer = 16, 3, 1
    # The layer's HND view of a multi-layer pool, strided as get_buffers returns it.
    raw = torch.randint(
        0,
        256,
        (pages, layers, 2, HK * tokens_per_block * D),
        device="cuda",
        generator=g,
        dtype=torch.uint8,
    )
    slots = torch.randperm(pages * tokens_per_block, device="cuda", generator=g)[:tokens].to(
        torch.int32
    )
    slots[live:] = -1  # padded rows own no slot

    def view(t):
        return t.view(torch.float8_e4m3fn)[:, layer].view(pages, 2, HK, tokens_per_block, D)

    ours_raw = raw.clone()
    args = (qkv, HQ, HK, HV, ROT, 1e-6, qw, kw, 5e6, is_neox, pos, False)
    q_compact = torch.empty(tokens, HQ, D, device="cuda", dtype=torch.float8_e4m3fn)
    ours = torch.ops.trtllm.minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv(
        *args, view(ours_raw), slots, live, q_compact
    )
    plain = torch.ops.trtllm.minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8(*args)
    assert torch.equal(ours.view(torch.uint8), plain.view(torch.uint8))
    assert torch.equal(
        q_compact.view(tokens, HQ * D).view(torch.uint8), plain[:, : HQ * D].view(torch.uint8)
    )

    ref_raw = raw.clone()
    k = plain[:, HQ * D : (HQ + HK) * D].reshape(tokens, HK, D)
    v = plain[:, (HQ + HK) * D :].reshape(tokens, HV, D)
    ref = view(ref_raw)
    write_kv_slots(ref[:, 0], slots, k, live, layout="HND")
    write_kv_slots(ref[:, 1], slots, v, live, layout="HND")
    assert torch.equal(ours_raw, ref_raw)


def test_the_fused_kv_insert_skips_a_negative_slot() -> None:
    """A live row whose slot is the -1 sentinel stores nothing (no write before the pool row)."""
    if not hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv"):
        pytest.skip("this build does not carry minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv")
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
        write_kv_slots,
    )

    tokens, tokens_per_block, pages = 4, 32, 8
    g = torch.Generator(device="cuda").manual_seed(7)
    qkv = (torch.randn(tokens, (HQ + HK + HV) * D, device="cuda", generator=g) * 2).to(
        torch.bfloat16
    )
    pos = torch.randint(0, 1 << 20, (tokens,), device="cuda", generator=g, dtype=torch.int32)
    qw = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    kw = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    raw = torch.randint(
        0,
        256,
        (pages, 3, 2, HK * tokens_per_block * D),
        device="cuda",
        generator=g,
        dtype=torch.uint8,
    )
    slots = torch.tensor([5, -1, 40, 77], device="cuda", dtype=torch.int32)

    def view(t):
        return t.view(torch.float8_e4m3fn)[:, 1].view(pages, 2, HK, tokens_per_block, D)

    ours_raw = raw.clone()
    args = (qkv, HQ, HK, HV, ROT, 1e-6, qw, kw, 5e6, True, pos, False)
    torch.ops.trtllm.minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv(
        *args, view(ours_raw), slots, tokens
    )
    plain = torch.ops.trtllm.minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8(*args)
    keep = torch.tensor([0, 2, 3], device="cuda")
    ref_raw = raw.clone()
    ref = view(ref_raw)
    k = plain[:, HQ * D : (HQ + HK) * D].reshape(tokens, HK, D)[keep]
    v = plain[:, (HQ + HK) * D :].reshape(tokens, HV, D)[keep]
    write_kv_slots(ref[:, 0], slots[keep], k, 3, layout="HND")
    write_kv_slots(ref[:, 1], slots[keep], v, 3, layout="HND")
    assert torch.equal(ours_raw, ref_raw)
