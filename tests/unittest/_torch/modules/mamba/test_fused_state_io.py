# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-level tests for the fused state I/O Triton kernels.

Covers ``gather_cast_transpose_kv_to_fp32_vk`` and
``transpose_cast_scatter_fp32_vk_to_kv`` from
``tensorrt_llm._torch.modules.fla.fused_state_io``, comparing each against
the equivalent PyTorch chain (``[indices].to(fp32).transpose(-1,-2).contiguous()``
and its inverse). Both kernels are pure layout-transform + dtype cast, so
the expected tolerance is **bit-exact** when input precision is preserved
through the chain.
"""

import pytest
import torch


def _supported_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    # SM90 (Hopper) or SM100 (Blackwell)
    return major in (9, 10)


skip_unsupported = pytest.mark.skipif(
    not _supported_arch(),
    reason="Fused state I/O kernels target SM90 (Hopper) or SM100 (Blackwell)",
)


# Reference (un-fused) implementations -------------------------------------


def _ref_gather_cast_transpose(initial_state, indices):
    """PyTorch reference: ``init[idx].to(fp32).transpose(-1, -2).contiguous()``."""
    if indices is not None:
        gathered = initial_state[indices].to(torch.float32)
    else:
        gathered = initial_state.to(torch.float32)
    return gathered.transpose(-1, -2).contiguous()


def _ref_transpose_cast_scatter(src_vk, dst, scatter_indices):
    """PyTorch reference: ``src.transpose(-1, -2).contiguous().to(dst.dtype)`` + scatter.

    Mutates ``dst`` in place. Returns nothing (matches the fused kernel API).
    """
    out_kv = src_vk.transpose(-1, -2).contiguous().to(dst.dtype, copy=False)
    if scatter_indices is not None:
        dst[scatter_indices] = out_kv
    else:
        dst.copy_(out_kv)


# Forward kernel: gather + cast + transpose --------------------------------


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape,num_seqs",
    [
        ((16, 4, 128, 128), 2),
        ((1, 4, 128, 128), 1),
        ((32, 16, 128, 128), 8),
        ((8, 4, 64, 128), 3),  # K != V
        ((8, 4, 128, 64), 3),  # K != V
    ],
)
def test_gather_cast_transpose_matches_torch_ref(dtype, shape, num_seqs):
    """Indexed gather path — output must be bit-exact vs the PyTorch chain."""
    from tensorrt_llm._torch.modules.fla.fused_state_io import gather_cast_transpose_kv_to_fp32_vk

    N_pool, H, K, V = shape
    torch.manual_seed(0)
    src = (torch.randn(N_pool, H, K, V, device="cuda") * 0.1).to(dtype)
    indices = torch.randperm(N_pool, device="cuda", dtype=torch.int32)[:num_seqs]

    ref = _ref_gather_cast_transpose(src, indices)
    out = gather_cast_transpose_kv_to_fp32_vk(src, indices)

    assert out.shape == (num_seqs, H, V, K)
    assert out.dtype == torch.float32
    assert out.is_contiguous()
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_cast_transpose_no_indices_full_pool(dtype):
    """``indices=None`` must process the entire pool, output shape == input shape (with last two transposed)."""
    from tensorrt_llm._torch.modules.fla.fused_state_io import gather_cast_transpose_kv_to_fp32_vk

    N_pool, H, K, V = 12, 4, 128, 128
    torch.manual_seed(1)
    src = (torch.randn(N_pool, H, K, V, device="cuda") * 0.1).to(dtype)

    ref = _ref_gather_cast_transpose(src, None)
    out = gather_cast_transpose_kv_to_fp32_vk(src, None)

    assert out.shape == (N_pool, H, V, K)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


# Reverse kernel: transpose + cast + scatter -------------------------------


@skip_unsupported
@pytest.mark.parametrize("dst_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "N_pool,num_seqs,H,K,V",
    [
        (16, 2, 4, 128, 128),
        (32, 8, 16, 128, 128),
        (8, 3, 4, 64, 128),  # K != V
    ],
)
def test_transpose_cast_scatter_inplace_preserves_untouched(dst_dtype, N_pool, num_seqs, H, K, V):
    """Scattered slots match torch ref; untouched slots are bit-exact unchanged."""
    from tensorrt_llm._torch.modules.fla.fused_state_io import transpose_cast_scatter_fp32_vk_to_kv

    torch.manual_seed(2)
    src_vk = torch.randn(num_seqs, H, V, K, device="cuda", dtype=torch.float32) * 0.1
    pool_init = (torch.randn(N_pool, H, K, V, device="cuda") * 0.1).to(dst_dtype)
    indices = torch.randperm(N_pool, device="cuda", dtype=torch.int32)[:num_seqs]

    pool_ref = pool_init.clone()
    _ref_transpose_cast_scatter(src_vk, pool_ref, indices)

    pool_out = pool_init.clone()
    transpose_cast_scatter_fp32_vk_to_kv(src_vk, pool_out, indices)

    # The touched slots must match the torch ref bit-exact (RNE cast is deterministic).
    torch.testing.assert_close(pool_out[indices], pool_ref[indices], atol=0.0, rtol=0.0)
    # Untouched slots must remain bit-exact to the initial pool.
    untouched = [i for i in range(N_pool) if i not in indices.tolist()]
    if untouched:
        torch.testing.assert_close(pool_out[untouched], pool_init[untouched], atol=0.0, rtol=0.0)


@skip_unsupported
@pytest.mark.parametrize("dst_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_transpose_cast_scatter_full_write(dst_dtype):
    """``scatter_indices=None`` must write every row; output equals torch ref."""
    from tensorrt_llm._torch.modules.fla.fused_state_io import transpose_cast_scatter_fp32_vk_to_kv

    num_seqs, H, K, V = 5, 4, 128, 128
    torch.manual_seed(3)
    src_vk = torch.randn(num_seqs, H, V, K, device="cuda", dtype=torch.float32) * 0.1
    dst_init = torch.randn(num_seqs, H, K, V, device="cuda").to(dst_dtype)

    dst_ref = dst_init.clone()
    _ref_transpose_cast_scatter(src_vk, dst_ref, None)

    dst_out = dst_init.clone()
    transpose_cast_scatter_fp32_vk_to_kv(src_vk, dst_out, None)

    torch.testing.assert_close(dst_out, dst_ref, atol=0.0, rtol=0.0)


# Roundtrip ----------------------------------------------------------------


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_roundtrip_gather_then_scatter_bitexact(dtype):
    """``gather(src, idx) → scatter back to same idx`` must restore src bit-exact.

    Justification: bf16/fp16 → fp32 is lossless (precision strictly extends);
    fp32 → bf16/fp16 via RNE is deterministic and recovers the original value
    when the fp32 representation came from an unmodified bf16/fp16 source.
    """
    from tensorrt_llm._torch.modules.fla.fused_state_io import (
        gather_cast_transpose_kv_to_fp32_vk,
        transpose_cast_scatter_fp32_vk_to_kv,
    )

    N_pool, H, K, V = 16, 4, 128, 128
    num_seqs = 5
    torch.manual_seed(4)
    src = (torch.randn(N_pool, H, K, V, device="cuda") * 0.1).to(dtype)
    indices = torch.randperm(N_pool, device="cuda", dtype=torch.int32)[:num_seqs]

    # gather → vk fp32
    vk = gather_cast_transpose_kv_to_fp32_vk(src, indices)
    # scatter back to a fresh pool, same indices
    restored = src.clone()
    transpose_cast_scatter_fp32_vk_to_kv(vk, restored, indices)

    # The selected slots should be identical to the original (RNE roundtrip is lossless
    # when no math was performed in fp32).
    torch.testing.assert_close(restored[indices], src[indices], atol=0.0, rtol=0.0)
    # Other slots must be unchanged (we passed in restored = src.clone()).
    untouched = [i for i in range(N_pool) if i not in indices.tolist()]
    if untouched:
        torch.testing.assert_close(restored[untouched], src[untouched], atol=0.0, rtol=0.0)


# Import smoke (no GPU required) ------------------------------------------


def test_module_importable():
    from tensorrt_llm._torch.modules.fla.fused_state_io import (  # noqa: F401
        gather_cast_transpose_kv_to_fp32_vk,
        transpose_cast_scatter_fp32_vk_to_kv,
    )
