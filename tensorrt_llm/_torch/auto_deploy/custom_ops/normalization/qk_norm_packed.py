# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Packed QK RMSNorm pre/post AllReduce Triton kernels.

Used as the prefill fallback for lamport_sharded_qk_rmsnorm when n_tokens exceeds
the lamport barrier-flag capacity (256). Algorithm mirrors the lamport kernel's
internal structure but replaces the NVLink barrier with a small NCCL AllReduce:

    packed = qk_rms_sum_sq_pack(q, k)            # 1 Triton kernel: pre-reduce
    dist.all_reduce(packed, op=SUM)              # 1 NCCL kernel:  small F32 payload
    q_out, k_out = qk_rms_norm_from_packed(
        q, k, packed, w_q, w_k, eps, ws_q, ws_k) # 1 Triton kernel: post-reduce

Total 3 kernels per Q/K-norm call regardless of batch size, vs ~18 if
sharded_rmsnorm is invoked twice.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


# ---------------------------------------------------------------------------
# Pre-reduce: compute per-token packed (sum(Q^2), sum(K^2)) in fp32.
# ---------------------------------------------------------------------------
@triton.jit
def _qk_rms_sum_sq_pack_kernel(
    q_ptr,
    k_ptr,
    packed_ptr,
    d_q,
    d_k,
    q_row_stride,
    k_row_stride,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per token; accumulates sum-of-squares for Q row and K row."""
    row = tl.program_id(0)

    # Q sum of squares.
    offs_q = tl.arange(0, BLOCK_Q)
    mask_q = offs_q < d_q
    q = tl.load(q_ptr + row * q_row_stride + offs_q, mask=mask_q, other=0.0).to(tl.float32)
    q_sumsq = tl.sum(q * q, axis=0)

    # K sum of squares.
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < d_k
    k = tl.load(k_ptr + row * k_row_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    k_sumsq = tl.sum(k * k, axis=0)

    # Write packed [row, 0] = q_sumsq, [row, 1] = k_sumsq (stride 2 along row dim).
    tl.store(packed_ptr + row * 2 + 0, q_sumsq)
    tl.store(packed_ptr + row * 2 + 1, k_sumsq)


def _next_pow2(x: int) -> int:
    # Triton BLOCK_SIZE must be a power of two.
    return 1 << (x - 1).bit_length()


def _qk_rms_sum_sq_pack_impl(q: Tensor, k: Tensor) -> Tensor:
    assert q.shape[:-1] == k.shape[:-1], "q and k must share leading dims"
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "q/k must be last-dim-contiguous"

    d_q = q.shape[-1]
    d_k = k.shape[-1]
    n_tokens = q.numel() // d_q
    q_row_stride = q.stride(-2) if q.dim() >= 2 else d_q
    k_row_stride = k.stride(-2) if k.dim() >= 2 else d_k

    packed = torch.empty(n_tokens, 2, dtype=torch.float32, device=q.device)
    BLOCK_Q = _next_pow2(d_q)
    BLOCK_K = _next_pow2(d_k)
    _qk_rms_sum_sq_pack_kernel[(n_tokens,)](
        q,
        k,
        packed,
        d_q,
        d_k,
        q_row_stride,
        k_row_stride,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )
    return packed


@torch.library.custom_op("auto_deploy::qk_rms_sum_sq_pack", mutates_args=())
def qk_rms_sum_sq_pack(q: Tensor, k: Tensor) -> Tensor:
    """Per-token packed sum-of-squares for Q and K.

    Returns packed[n_tokens, 2] float32 where:
        packed[i, 0] = sum(q[i, :] ** 2)
        packed[i, 1] = sum(k[i, :] ** 2)

    Call dist.all_reduce(packed) before feeding to qk_rms_norm_from_packed.
    """
    return _qk_rms_sum_sq_pack_impl(q, k)


@qk_rms_sum_sq_pack.register_fake
def _qk_rms_sum_sq_pack_fake(q: Tensor, k: Tensor) -> Tensor:
    n_tokens = q.numel() // q.shape[-1]
    return torch.empty(n_tokens, 2, dtype=torch.float32, device=q.device)


# ---------------------------------------------------------------------------
# Post-reduce: normalize Q and K using the AllReduced packed variance.
# ---------------------------------------------------------------------------
@triton.jit
def _qk_rms_norm_from_packed_kernel(
    q_ptr,
    k_ptr,
    packed_ptr,
    w_q_ptr,
    w_k_ptr,
    q_out_ptr,
    k_out_ptr,
    d_q,
    d_k,
    q_row_stride,
    k_row_stride,
    q_out_row_stride,
    k_out_row_stride,
    eps,
    inv_global_count_q,  # 1 / (d_q * world_size)
    inv_global_count_k,  # 1 / (d_k * world_size)
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)

    q_sumsq = tl.load(packed_ptr + row * 2 + 0)
    k_sumsq = tl.load(packed_ptr + row * 2 + 1)
    rstd_q = tl.rsqrt(q_sumsq * inv_global_count_q + eps)
    rstd_k = tl.rsqrt(k_sumsq * inv_global_count_k + eps)

    # Normalize and scale Q.
    offs_q = tl.arange(0, BLOCK_Q)
    mask_q = offs_q < d_q
    q = tl.load(q_ptr + row * q_row_stride + offs_q, mask=mask_q, other=0.0).to(tl.float32)
    w_q = tl.load(w_q_ptr + offs_q, mask=mask_q, other=0.0).to(tl.float32)
    q_out = q * rstd_q * w_q
    tl.store(
        q_out_ptr + row * q_out_row_stride + offs_q,
        q_out.to(q_out_ptr.dtype.element_ty),
        mask=mask_q,
    )

    # Normalize and scale K.
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < d_k
    k = tl.load(k_ptr + row * k_row_stride + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    w_k = tl.load(w_k_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    k_out = k * rstd_k * w_k
    tl.store(
        k_out_ptr + row * k_out_row_stride + offs_k,
        k_out.to(k_out_ptr.dtype.element_ty),
        mask=mask_k,
    )


def _qk_rms_norm_from_packed_impl(
    q: Tensor,
    k: Tensor,
    packed: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    eps: float,
    world_size: int,
) -> Tuple[Tensor, Tensor]:
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "q/k must be last-dim-contiguous"
    assert packed.dtype == torch.float32, "packed must be float32"
    assert w_q.dtype == torch.float32 and w_k.dtype == torch.float32, "weights must be fp32"

    d_q = q.shape[-1]
    d_k = k.shape[-1]
    n_tokens = q.numel() // d_q
    q_row_stride = q.stride(-2) if q.dim() >= 2 else d_q
    k_row_stride = k.stride(-2) if k.dim() >= 2 else d_k

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    q_out_row_stride = q_out.stride(-2) if q_out.dim() >= 2 else d_q
    k_out_row_stride = k_out.stride(-2) if k_out.dim() >= 2 else d_k

    BLOCK_Q = _next_pow2(d_q)
    BLOCK_K = _next_pow2(d_k)
    inv_count_q = 1.0 / (d_q * world_size)
    inv_count_k = 1.0 / (d_k * world_size)
    _qk_rms_norm_from_packed_kernel[(n_tokens,)](
        q,
        k,
        packed,
        w_q,
        w_k,
        q_out,
        k_out,
        d_q,
        d_k,
        q_row_stride,
        k_row_stride,
        q_out_row_stride,
        k_out_row_stride,
        float(eps),
        float(inv_count_q),
        float(inv_count_k),
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )
    return q_out, k_out


@torch.library.custom_op("auto_deploy::qk_rms_norm_from_packed", mutates_args=())
def qk_rms_norm_from_packed(
    q: Tensor,
    k: Tensor,
    packed: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    eps: float,
    world_size: int,
) -> Tuple[Tensor, Tensor]:
    """Normalize Q and K using AllReduced packed variance.

    Args:
        q, k: local shards, last-dim-contiguous (bf16 or fp16).
        packed: [n_tokens, 2] fp32 AllReduced sum-of-squares.
        w_q, w_k: RMSNorm weights, fp32, local shard.
        eps: RMSNorm epsilon.
        world_size: tensor-parallel world size (for global element count).

    Returns (q_out, k_out) packed contiguous tensors, same dtype as inputs.
    """
    return _qk_rms_norm_from_packed_impl(q, k, packed, w_q, w_k, eps, world_size)


@qk_rms_norm_from_packed.register_fake
def _qk_rms_norm_from_packed_fake(
    q: Tensor,
    k: Tensor,
    packed: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    eps: float,
    world_size: int,
) -> Tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k)
