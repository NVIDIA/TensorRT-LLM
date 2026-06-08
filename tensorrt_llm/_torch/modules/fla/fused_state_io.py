# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused state I/O kernels for the FlashInfer GDN prefill adapter.

The SSM state pool and FlashInfer share the ``[slots, HV, V, K]`` layout
(K innermost), so no transpose is needed. These kernels fuse the gather +
dtype cast (and the reverse cast + scatter) that bridge the bf16 pool and
FlashInfer's fp32 state, avoiding the naive multi-launch PyTorch chain
(``pool[indices].to(fp32)`` / ``.to(bf16)`` + indexed scatter) and its large
intermediate buffers:

- ``gather_cast_vk_to_fp32_vk``: gather pool slots by index + cast bf16->fp32.
- ``cast_scatter_fp32_vk_to_vk``: cast fp32->bf16 + scatter back to pool slots.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_cast_vk_to_fp32_vk_kernel(
    src_ptr,
    dst_ptr,
    indices_ptr,
    HAS_INDICES: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    src_stride_n,
    src_stride_h,
    src_stride_v,
    src_stride_k,
    dst_stride_n,
    dst_stride_h,
    dst_stride_v,
    dst_stride_k,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_vk = tl.program_id(2)

    num_k_blocks: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    pid_vb = pid_vk // num_k_blocks
    pid_kb = pid_vk % num_k_blocks

    if HAS_INDICES:
        src_seq = tl.load(indices_ptr + pid_seq).to(tl.int64)
    else:
        src_seq = pid_seq.to(tl.int64)

    v_offs = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    v_mask = v_offs < V
    k_mask = k_offs < K
    mask = v_mask[:, None] & k_mask[None, :]

    src_addrs = (
        src_ptr
        + src_seq * src_stride_n
        + pid_h * src_stride_h
        + v_offs[:, None] * src_stride_v
        + k_offs[None, :] * src_stride_k
    )
    dst_addrs = (
        dst_ptr
        + pid_seq * dst_stride_n
        + pid_h * dst_stride_h
        + v_offs[:, None] * dst_stride_v
        + k_offs[None, :] * dst_stride_k
    )
    tl.store(dst_addrs, tl.load(src_addrs, mask=mask, other=0.0).to(tl.float32), mask=mask)


def gather_cast_vk_to_fp32_vk(
    initial_state: torch.Tensor,
    initial_state_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused ``initial_state[indices].to(fp32).contiguous()`` for ``[N, H, V, K]`` state."""
    assert initial_state.dim() == 4, f"initial_state must be 4D, got {initial_state.shape}"
    n_pool, h, v, k = initial_state.shape
    if initial_state_indices is not None:
        num_seqs = initial_state_indices.shape[0]
        has_indices = True
    else:
        num_seqs = n_pool
        has_indices = False

    # K and V are typically 128 in GDN; one (BLOCK_K, BLOCK_V) tile covers the full K and V dimensions.
    # entire (K, V) plane per (seq, head). Larger tiles save grid overhead;
    # smaller tiles improve occupancy at small num_seqs * H.
    output = torch.empty(num_seqs, h, v, k, dtype=torch.float32, device=initial_state.device)
    block_v = min(v, 128)
    block_k = min(k, 128)
    num_v_blocks = triton.cdiv(v, block_v)
    num_k_blocks = triton.cdiv(k, block_k)
    grid = (num_seqs, h, num_v_blocks * num_k_blocks)

    _gather_cast_vk_to_fp32_vk_kernel[grid](
        initial_state,
        output,
        initial_state_indices,
        HAS_INDICES=has_indices,
        H=h,
        V=v,
        K=k,
        src_stride_n=initial_state.stride(0),
        src_stride_h=initial_state.stride(1),
        src_stride_v=initial_state.stride(2),
        src_stride_k=initial_state.stride(3),
        dst_stride_n=output.stride(0),
        dst_stride_h=output.stride(1),
        dst_stride_v=output.stride(2),
        dst_stride_k=output.stride(3),
        BLOCK_V=block_v,
        BLOCK_K=block_k,
    )
    return output


@triton.jit
def _cast_scatter_fp32_vk_to_vk_kernel(
    src_ptr,
    dst_ptr,
    indices_ptr,
    HAS_INDICES: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    src_stride_n,
    src_stride_h,
    src_stride_v,
    src_stride_k,
    dst_stride_n,
    dst_stride_h,
    dst_stride_v,
    dst_stride_k,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_vk = tl.program_id(2)

    num_k_blocks: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    pid_vb = pid_vk // num_k_blocks
    pid_kb = pid_vk % num_k_blocks

    if HAS_INDICES:
        dst_seq = tl.load(indices_ptr + pid_seq).to(tl.int64)
    else:
        dst_seq = pid_seq.to(tl.int64)

    v_offs = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    v_mask = v_offs < V
    k_mask = k_offs < K
    mask = v_mask[:, None] & k_mask[None, :]

    src_addrs = (
        src_ptr
        + pid_seq * src_stride_n
        + pid_h * src_stride_h
        + v_offs[:, None] * src_stride_v
        + k_offs[None, :] * src_stride_k
    )
    dst_addrs = (
        dst_ptr
        + dst_seq * dst_stride_n
        + pid_h * dst_stride_h
        + v_offs[:, None] * dst_stride_v
        + k_offs[None, :] * dst_stride_k
    )
    src_data = tl.load(src_addrs, mask=mask, other=0.0)
    tl.store(dst_addrs, src_data.to(dst_ptr.dtype.element_ty), mask=mask)


def cast_scatter_fp32_vk_to_vk(
    src_vk: torch.Tensor,
    dst: torch.Tensor,
    scatter_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused fp32-to-dst cast plus optional indexed scatter for ``[N, H, V, K]`` state."""
    assert src_vk.dim() == 4, f"src_vk must be 4D, got {src_vk.shape}"
    assert dst.dim() == 4, f"dst must be 4D, got {dst.shape}"
    num_seqs, h, v, k = src_vk.shape
    assert dst.shape[1:] == (
        h,
        v,
        k,
    ), f"dst shape {tuple(dst.shape)} incompatible with src {tuple(src_vk.shape)}"
    if scatter_indices is not None:
        assert scatter_indices.shape == (num_seqs,), (
            f"scatter_indices shape {tuple(scatter_indices.shape)} must match num_seqs={num_seqs}"
        )

    has_indices = scatter_indices is not None
    block_v = min(v, 128)
    block_k = min(k, 128)
    num_v_blocks = triton.cdiv(v, block_v)
    num_k_blocks = triton.cdiv(k, block_k)
    grid = (num_seqs, h, num_v_blocks * num_k_blocks)

    _cast_scatter_fp32_vk_to_vk_kernel[grid](
        src_vk,
        dst,
        scatter_indices,
        HAS_INDICES=has_indices,
        H=h,
        V=v,
        K=k,
        src_stride_n=src_vk.stride(0),
        src_stride_h=src_vk.stride(1),
        src_stride_v=src_vk.stride(2),
        src_stride_k=src_vk.stride(3),
        dst_stride_n=dst.stride(0),
        dst_stride_h=dst.stride(1),
        dst_stride_v=dst.stride(2),
        dst_stride_k=dst.stride(3),
        BLOCK_V=block_v,
        BLOCK_K=block_k,
    )
