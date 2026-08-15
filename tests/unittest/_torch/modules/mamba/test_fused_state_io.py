# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-level tests for the fused state I/O Triton kernels.

Covers ``gather_cast_vk_to_fp32_vk`` and ``cast_scatter_fp32_vk_to_vk`` from
``tensorrt_llm._torch.modules.fla.fused_state_io``, comparing each against the
equivalent PyTorch chain (``pool[indices].to(fp32)`` and its inverse
``.to(dtype)`` + indexed scatter). Both kernels are pure gather/scatter + dtype
cast in the shared ``[slots, HV, V, K]`` layout (no transpose), so the expected
tolerance is **bit-exact** when input precision is preserved through the chain.
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


def _ref_gather_cast(initial_state, indices):
    if indices is not None:
        return initial_state[indices].to(torch.float32).contiguous()
    return initial_state.to(torch.float32).contiguous()


def _ref_cast_scatter(src_vk, dst, scatter_indices):
    out = src_vk.to(dst.dtype, copy=False)
    if scatter_indices is not None:
        dst[scatter_indices] = out
    else:
        dst.copy_(out)


# Forward kernel: gather + cast (vk -> fp32 vk) ----------------------------


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape,num_seqs",
    [
        ((16, 4, 128, 128), 2),
        ((8, 4, 64, 128), 3),
        ((8, 4, 128, 64), 3),
    ],
)
def test_gather_cast_vk_matches_torch_ref(dtype, shape, num_seqs):
    from tensorrt_llm._torch.modules.fla.fused_state_io import gather_cast_vk_to_fp32_vk

    N_pool, H, V, K = shape
    torch.manual_seed(5)
    src = (torch.randn(N_pool, H, V, K, device="cuda") * 0.1).to(dtype)
    indices = torch.randperm(N_pool, device="cuda", dtype=torch.int32)[:num_seqs]

    ref = _ref_gather_cast(src, indices)
    out = gather_cast_vk_to_fp32_vk(src, indices)

    assert out.shape == (num_seqs, H, V, K)
    assert out.dtype == torch.float32
    assert out.is_contiguous()
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


# Reverse kernel: cast + scatter (fp32 vk -> vk) ---------------------------


@skip_unsupported
@pytest.mark.parametrize("dst_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cast_scatter_vk_preserves_untouched(dst_dtype):
    from tensorrt_llm._torch.modules.fla.fused_state_io import cast_scatter_fp32_vk_to_vk

    N_pool, num_seqs, H, V, K = 16, 4, 4, 128, 128
    torch.manual_seed(6)
    src_vk = torch.randn(num_seqs, H, V, K, device="cuda", dtype=torch.float32) * 0.1
    pool_init = (torch.randn(N_pool, H, V, K, device="cuda") * 0.1).to(dst_dtype)
    indices = torch.randperm(N_pool, device="cuda", dtype=torch.int32)[:num_seqs]

    pool_ref = pool_init.clone()
    _ref_cast_scatter(src_vk, pool_ref, indices)

    pool_out = pool_init.clone()
    cast_scatter_fp32_vk_to_vk(src_vk, pool_out, indices)

    torch.testing.assert_close(pool_out[indices], pool_ref[indices], atol=0.0, rtol=0.0)
    untouched = [i for i in range(N_pool) if i not in indices.tolist()]
    if untouched:
        torch.testing.assert_close(pool_out[untouched], pool_init[untouched], atol=0.0, rtol=0.0)


# Roundtrip ----------------------------------------------------------------


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_roundtrip_gather_then_scatter_bitexact(dtype):
    """``gather(src, idx) -> scatter back to same idx`` must restore src bit-exact.

    bf16/fp16 -> fp32 is lossless; fp32 -> bf16/fp16 via RNE is deterministic
    and recovers the original when no math was performed in fp32.
    """
    from tensorrt_llm._torch.modules.fla.fused_state_io import (
        cast_scatter_fp32_vk_to_vk,
        gather_cast_vk_to_fp32_vk,
    )

    N_pool, H, V, K = 16, 4, 128, 128
    num_seqs = 5
    torch.manual_seed(4)
    src = (torch.randn(N_pool, H, V, K, device="cuda") * 0.1).to(dtype)
    indices = torch.randperm(N_pool, device="cuda", dtype=torch.int32)[:num_seqs]

    vk = gather_cast_vk_to_fp32_vk(src, indices)
    restored = src.clone()
    cast_scatter_fp32_vk_to_vk(vk, restored, indices)

    torch.testing.assert_close(restored[indices], src[indices], atol=0.0, rtol=0.0)
    untouched = [i for i in range(N_pool) if i not in indices.tolist()]
    if untouched:
        torch.testing.assert_close(restored[untouched], src[untouched], atol=0.0, rtol=0.0)


# Import smoke (no GPU required) ------------------------------------------


def test_module_importable():
    from tensorrt_llm._torch.modules.fla.fused_state_io import (  # noqa: F401
        cast_scatter_fp32_vk_to_vk,
        gather_cast_vk_to_fp32_vk,
    )
