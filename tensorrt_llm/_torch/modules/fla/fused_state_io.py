# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused state I/O kernels for the FlashInfer GDN prefill adapter.

The FlashInfer prefill kernel consumes the SSM state in fp32 with the last
two dims transposed (V, K) relative to TRT-LLM's native (K, V) layout, and
optionally requires gathering a subset of the SSM pool by index. The naive
PyTorch chain is

    gathered_init = initial_state[indices].to(torch.float32)        # gather + cast
    gathered_init = gathered_init.transpose(-1, -2).contiguous()     # transpose copy

which produces three separate kernel launches and two large intermediate
fp32 buffers (~``num_seqs * H * K * V * 4`` bytes each) per FlashInfer
call. ``gather_cast_transpose_kv_to_fp32_vk`` fuses all three steps into a
single Triton kernel.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_cast_transpose_kv_to_fp32_vk_kernel(
    src_ptr,
    dst_ptr,
    indices_ptr,
    HAS_INDICES: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    src_stride_n,
    src_stride_h,
    src_stride_k,
    src_stride_v,
    dst_stride_n,
    dst_stride_h,
    dst_stride_v,
    dst_stride_k,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Read src[indices[seq], h, k, v] (any dtype) → write dst[seq, h, v, k] (fp32).

    Each program instance handles one (seq, head, K-block, V-block) tile.
    """
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kv = tl.program_id(2)

    num_v_blocks: tl.constexpr = (V + BLOCK_V - 1) // BLOCK_V
    pid_kb = pid_kv // num_v_blocks
    pid_vb = pid_kv % num_v_blocks

    if HAS_INDICES:
        src_seq = tl.load(indices_ptr + pid_seq).to(tl.int64)
    else:
        src_seq = pid_seq.to(tl.int64)

    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offs = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    mask = k_mask[:, None] & v_mask[None, :]

    src_addrs = (
        src_ptr
        + src_seq * src_stride_n
        + pid_h * src_stride_h
        + k_offs[:, None] * src_stride_k
        + v_offs[None, :] * src_stride_v
    )
    src_data = tl.load(src_addrs, mask=mask, other=0.0)
    src_data_fp32 = src_data.to(tl.float32)

    dst_addrs = (
        dst_ptr
        + pid_seq * dst_stride_n
        + pid_h * dst_stride_h
        + v_offs[None, :] * dst_stride_v
        + k_offs[:, None] * dst_stride_k
    )
    tl.store(dst_addrs, src_data_fp32, mask=mask)


def gather_cast_transpose_kv_to_fp32_vk(
    initial_state: torch.Tensor,
    initial_state_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused ``initial_state[indices].to(fp32).transpose(-1, -2).contiguous()``.

    Args:
        initial_state: ``[N_pool, H, K, V]`` of any float dtype.
        initial_state_indices: optional ``[num_seqs]`` int tensor. ``None`` means
            "use the whole pool" (equivalent to identity gather).

    Returns:
        ``[num_seqs, H, V, K]`` fp32 contiguous, where ``num_seqs == len(indices)``
        when ``indices`` is provided, otherwise ``num_seqs == N_pool``.
    """
    assert initial_state.dim() == 4, f"initial_state must be 4D, got {initial_state.shape}"
    N_pool, H, K, V = initial_state.shape

    if initial_state_indices is not None:
        num_seqs = initial_state_indices.shape[0]
        has_indices = True
    else:
        num_seqs = N_pool
        has_indices = False

    output = torch.empty(num_seqs, H, V, K, dtype=torch.float32, device=initial_state.device)

    # K and V are typically 128 in GDN; one (BLOCK_K, BLOCK_V) tile covers the
    # entire (K, V) plane per (seq, head). Larger tiles save grid overhead;
    # smaller tiles improve occupancy at small num_seqs * H.
    BLOCK_K = min(K, 128)
    BLOCK_V = min(V, 128)
    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V

    grid = (num_seqs, H, num_k_blocks * num_v_blocks)

    _gather_cast_transpose_kv_to_fp32_vk_kernel[grid](
        initial_state,
        output,
        initial_state_indices,
        HAS_INDICES=has_indices,
        H=H,
        K=K,
        V=V,
        src_stride_n=initial_state.stride(0),
        src_stride_h=initial_state.stride(1),
        src_stride_k=initial_state.stride(2),
        src_stride_v=initial_state.stride(3),
        dst_stride_n=output.stride(0),
        dst_stride_h=output.stride(1),
        dst_stride_v=output.stride(2),
        dst_stride_k=output.stride(3),
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
    )
    return output


@triton.jit
def _transpose_cast_scatter_fp32_vk_to_kv_kernel(
    src_ptr,
    dst_ptr,
    indices_ptr,
    HAS_INDICES: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    src_stride_n,
    src_stride_h,
    src_stride_v,
    src_stride_k,
    dst_stride_n,
    dst_stride_h,
    dst_stride_k,
    dst_stride_v,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Read src[seq, h, v, k] (fp32 V,K layout) → write dst[dst_seq, h, k, v] (dst dtype, K,V layout).

    ``dst_seq = indices[seq] if HAS_INDICES else seq``. Casts to ``dst_ptr``'s element dtype.
    """
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kv = tl.program_id(2)

    num_v_blocks: tl.constexpr = (V + BLOCK_V - 1) // BLOCK_V
    pid_kb = pid_kv // num_v_blocks
    pid_vb = pid_kv % num_v_blocks

    if HAS_INDICES:
        dst_seq = tl.load(indices_ptr + pid_seq).to(tl.int64)
    else:
        dst_seq = pid_seq.to(tl.int64)

    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offs = pid_vb * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V
    mask = k_mask[:, None] & v_mask[None, :]

    src_addrs = (
        src_ptr
        + pid_seq * src_stride_n
        + pid_h * src_stride_h
        + v_offs[None, :] * src_stride_v
        + k_offs[:, None] * src_stride_k
    )
    src_data = tl.load(src_addrs, mask=mask, other=0.0)
    src_data_cast = src_data.to(dst_ptr.dtype.element_ty)

    dst_addrs = (
        dst_ptr
        + dst_seq * dst_stride_n
        + pid_h * dst_stride_h
        + k_offs[:, None] * dst_stride_k
        + v_offs[None, :] * dst_stride_v
    )
    tl.store(dst_addrs, src_data_cast, mask=mask)


def transpose_cast_scatter_fp32_vk_to_kv(
    src_vk: torch.Tensor,
    dst: torch.Tensor,
    scatter_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused ``src.transpose(-1, -2).contiguous().to(dst.dtype)`` + optional indexed scatter.

    Reverse of :func:`gather_cast_transpose_kv_to_fp32_vk`. Writes ``src_vk`` into
    ``dst`` in TRT-LLM's (K, V) last-two-dims layout, casting to ``dst.dtype``,
    in a single Triton pass.

    Args:
        src_vk: ``[num_seqs, H, V, K]`` fp32 (FlashInfer output_state layout).
        dst: target tensor with last two dims ``(K, V)``. Two cases:

            * ``scatter_indices is None``: ``dst.shape == [num_seqs, H, K, V]`` and
              every row is written.
            * ``scatter_indices is not None``: ``dst.shape == [N_pool, H, K, V]``
              and ``dst[scatter_indices[i]]`` gets row ``i`` of ``src_vk``;
              other rows of ``dst`` are untouched.

        scatter_indices: optional ``[num_seqs]`` int tensor.

    Returns:
        ``None``; writes happen in-place into ``dst``.
    """
    assert src_vk.dim() == 4, f"src_vk must be 4D, got {src_vk.shape}"
    assert dst.dim() == 4, f"dst must be 4D, got {dst.shape}"
    num_seqs, H, V, K = src_vk.shape
    assert dst.shape[1:] == (
        H,
        K,
        V,
    ), f"dst shape {tuple(dst.shape)} incompatible with src {tuple(src_vk.shape)}"
    if scatter_indices is not None:
        assert scatter_indices.shape == (num_seqs,), (
            f"scatter_indices shape {tuple(scatter_indices.shape)} must match num_seqs={num_seqs}"
        )

    has_indices = scatter_indices is not None
    BLOCK_K = min(K, 128)
    BLOCK_V = min(V, 128)
    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    num_v_blocks = (V + BLOCK_V - 1) // BLOCK_V

    grid = (num_seqs, H, num_k_blocks * num_v_blocks)

    _transpose_cast_scatter_fp32_vk_to_kv_kernel[grid](
        src_vk,
        dst,
        scatter_indices,
        HAS_INDICES=has_indices,
        H=H,
        K=K,
        V=V,
        src_stride_n=src_vk.stride(0),
        src_stride_h=src_vk.stride(1),
        src_stride_v=src_vk.stride(2),
        src_stride_k=src_vk.stride(3),
        dst_stride_n=dst.stride(0),
        dst_stride_h=dst.stride(1),
        dst_stride_k=dst.stride(2),
        dst_stride_v=dst.stride(3),
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
    )
