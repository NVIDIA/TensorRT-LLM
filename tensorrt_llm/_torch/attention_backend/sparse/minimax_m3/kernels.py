# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2025 XunhaoLai. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from SGLang's MiniMax-M3 sparse-attention Triton kernels under
# ``python/sglang/srt/layers/attention/minimax_sparse_ops/`` (the
# ``prefill/topk_sparse.py``, ``decode/topk_sparse.py``, and
# ``prefill/flash_with_topk_idx.py`` modules in the SGLang repository,
# https://github.com/sgl-project/sglang, Apache-2.0). The per-block max
# score reduction + masked GQA softmax follow the same algorithm SGLang
# uses for M3 sparse attention; the kernels here are restructured to
# decouple the reductions from the attention matmul so PyTorch can run
# the (G)QA-broadcast matmul while Triton owns the
# mask + softmax + numerical-stability path.
"""Triton kernels for MiniMax-M3 sparse attention.

Two OpenAI Triton kernels implement the reductions on the hot path:

  1. ``_block_max_score_kernel`` reduces the index-attention QK tensor of
     shape ``[total_q, num_idx_heads, max_k]`` to a per-block score
     tensor of shape ``[total_q, num_idx_heads, n_blocks]`` via per-block
     max over the ``block_size`` positions inside each block.

  2. ``_sparse_softmax_kernel`` applies the block-selection mask plus
     the position-validity mask to a QK tensor of shape
     ``[total_q, num_kv_heads, g, max_k]`` and returns the row-softmax
     probabilities, in fp32, with the all-(-inf) row fix folded in (so
     the captured graph never produces NaN). Decoupling softmax from
     the matmul lets the algorithm reuse PyTorch matmuls (which
     reliably handle the GQA broadcast) while still putting the hot
     mask + softmax + numerical-stability path in Triton.

Both kernels are CUDA-graph safe: they take only static dtype/shape
information through ``tl.constexpr`` and dynamic scalars through plain
tensor arguments. No host-side sync runs inside the kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_max_score_kernel(
    qk_ptr,
    score_ptr,
    total_q,
    num_heads,
    max_k,
    n_blocks,
    stride_qk_q,
    stride_qk_h,
    stride_qk_k,
    stride_score_q,
    stride_score_h,
    stride_score_b,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-block max-score reduction.

    Grid: ``(total_q, num_heads, n_blocks)``. Each program reads one
    ``BLOCK_SIZE``-tile of the ``qk`` tensor (one (q, head) row's
    slice of one block) and writes its max to ``score``.
    """
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_block = tl.program_id(2)

    off_n = tl.arange(0, BLOCK_SIZE)
    pos = pid_block * BLOCK_SIZE + off_n
    pos_mask = pos < max_k

    qk_off = pid_q * stride_qk_q + pid_h * stride_qk_h + pos * stride_qk_k
    qk = tl.load(
        qk_ptr + qk_off,
        mask=pos_mask,
        other=float("-inf"),
    ).to(tl.float32)

    block_max = tl.max(qk, axis=0)

    score_off = pid_q * stride_score_q + pid_h * stride_score_h + pid_block * stride_score_b
    tl.store(score_ptr + score_off, block_max)


@triton.jit
def _sparse_softmax_kernel(
    qk_ptr,
    attended_ptr,
    attn_ptr,
    total_q,
    num_kv_heads,
    g,
    max_k,
    stride_qk_q,
    stride_qk_h,
    stride_qk_g,
    stride_qk_k,
    stride_att_q,
    stride_att_h,
    stride_att_k,
    stride_o_q,
    stride_o_h,
    stride_o_g,
    stride_o_k,
    BLOCK_K: tl.constexpr,
):
    """Masked softmax for sparse GQA.

    Grid: ``(total_q, num_kv_heads, g)``. Reads ``qk`` (one (q, kv,
    group) row of length ``max_k``), applies the per-position
    ``attended`` mask, computes softmax in fp32, and writes the
    row to ``attn``. Rows whose mask is all-False are written as
    all-zeros (the all-(-inf) NaN guard).
    """
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_g = tl.program_id(2)

    # Two-pass softmax to keep memory traffic predictable.
    # Pass 1: find row max for numerical stability.
    row_max = float("-inf")
    for k_start in range(0, max_k, BLOCK_K):
        off_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = off_k < max_k
        qk_off = (
            pid_q * stride_qk_q
            + pid_h * stride_qk_h
            + pid_g * stride_qk_g
            + off_k * stride_qk_k
        )
        att_off = pid_q * stride_att_q + pid_h * stride_att_h + off_k * stride_att_k
        qk = tl.load(qk_ptr + qk_off, mask=k_mask, other=float("-inf")).to(tl.float32)
        att = tl.load(attended_ptr + att_off, mask=k_mask, other=0).to(tl.int1)
        qk = tl.where(att, qk, float("-inf"))
        row_max = tl.maximum(row_max, tl.max(qk, axis=0))

    # row_max may be -inf if no positions are attended. We treat that
    # as the "zero out the row" case (no softmax denominator).
    row_max_safe = tl.where(row_max == float("-inf"), 0.0, row_max)

    # Pass 2: compute exp(qk - row_max), accumulate denom, write into attn.
    denom = 0.0
    for k_start in range(0, max_k, BLOCK_K):
        off_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = off_k < max_k
        qk_off = (
            pid_q * stride_qk_q
            + pid_h * stride_qk_h
            + pid_g * stride_qk_g
            + off_k * stride_qk_k
        )
        att_off = pid_q * stride_att_q + pid_h * stride_att_h + off_k * stride_att_k
        qk = tl.load(qk_ptr + qk_off, mask=k_mask, other=float("-inf")).to(tl.float32)
        att = tl.load(attended_ptr + att_off, mask=k_mask, other=0).to(tl.int1)
        qk = tl.where(att, qk, float("-inf"))
        ex = tl.exp(qk - row_max_safe)
        ex = tl.where(att, ex, 0.0)
        denom += tl.sum(ex, axis=0)

    # Pass 3: divide by denom and write.
    denom_safe = tl.where(denom == 0.0, 1.0, denom)
    for k_start in range(0, max_k, BLOCK_K):
        off_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = off_k < max_k
        qk_off = (
            pid_q * stride_qk_q
            + pid_h * stride_qk_h
            + pid_g * stride_qk_g
            + off_k * stride_qk_k
        )
        att_off = pid_q * stride_att_q + pid_h * stride_att_h + off_k * stride_att_k
        o_off = (
            pid_q * stride_o_q + pid_h * stride_o_h + pid_g * stride_o_g + off_k * stride_o_k
        )
        qk = tl.load(qk_ptr + qk_off, mask=k_mask, other=float("-inf")).to(tl.float32)
        att = tl.load(attended_ptr + att_off, mask=k_mask, other=0).to(tl.int1)
        qk = tl.where(att, qk, float("-inf"))
        ex = tl.exp(qk - row_max_safe)
        ex = tl.where(att, ex, 0.0)
        # If denom == 0 (no attended positions) we already replaced it
        # with 1.0 above, and ex itself is 0, so attn = 0. Good.
        attn = ex / denom_safe
        tl.store(attn_ptr + o_off, attn.to(attn_ptr.dtype.element_ty), mask=k_mask)


def triton_block_max_score(
    qk: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Wrap ``_block_max_score_kernel`` with a torch-friendly signature.

    Inputs:
        qk: ``[total_q, num_heads, max_k]`` float32. ``-inf`` entries are
            already masked (invalid positions, beyond seq_lens, etc.).
        block_size: per-block tile width.

    Returns ``[total_q, num_heads, n_blocks]`` float32 per-block max
    scores. ``n_blocks = ceil(max_k / block_size)``; the final block is
    partial when ``max_k % block_size != 0`` and the kernel reads
    invalid positions as ``-inf`` so they cannot win the max.
    """
    if qk.dim() != 3:
        raise ValueError(f"qk must be [total_q, num_heads, max_k]; got {tuple(qk.shape)}")
    total_q, num_heads, max_k = qk.shape
    n_blocks = (max_k + block_size - 1) // block_size
    qk_fp32 = qk.to(torch.float32, copy=False).contiguous()
    score = torch.empty(
        (total_q, num_heads, n_blocks),
        dtype=torch.float32,
        device=qk.device,
    )
    if total_q == 0 or num_heads == 0 or n_blocks == 0:
        return score
    grid = (total_q, num_heads, n_blocks)
    _block_max_score_kernel[grid](
        qk_fp32,
        score,
        total_q,
        num_heads,
        max_k,
        n_blocks,
        qk_fp32.stride(0),
        qk_fp32.stride(1),
        qk_fp32.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE=block_size,
    )
    return score


def triton_sparse_softmax(
    qk: torch.Tensor,
    attended: torch.Tensor,
) -> torch.Tensor:
    """Wrap ``_sparse_softmax_kernel`` with a torch-friendly signature.

    Inputs:
        qk:       ``[total_q, num_kv_heads, g, max_k]`` float32.
        attended: ``[total_q, num_kv_heads, max_k]`` bool — shared across
                  the GQA group dimension ``g``.

    Returns the per-row softmax of ``qk`` masked by ``attended``, in
    float32 with the all-False-row fix folded in (rows with no attended
    position write zero).
    """
    if qk.dim() != 4:
        raise ValueError(f"qk must be [total_q, num_kv_heads, g, max_k]; got {tuple(qk.shape)}")
    total_q, num_kv_heads, g, max_k = qk.shape
    if attended.shape != (total_q, num_kv_heads, max_k):
        raise ValueError(
            f"attended must be [total_q, num_kv_heads, max_k]; got {tuple(attended.shape)}"
        )
    qk_fp32 = qk.to(torch.float32, copy=False).contiguous()
    attended_u8 = attended.to(torch.uint8).contiguous()
    attn = torch.zeros_like(qk_fp32)
    if total_q == 0 or num_kv_heads == 0 or g == 0 or max_k == 0:
        return attn
    # Pick a BLOCK_K that is a power of two and small enough for typical
    # max_k. The kernel handles the partial tail itself.
    BLOCK_K = 128
    while BLOCK_K > max_k and BLOCK_K > 16:
        BLOCK_K //= 2
    BLOCK_K = max(BLOCK_K, 16)
    grid = (total_q, num_kv_heads, g)
    _sparse_softmax_kernel[grid](
        qk_fp32,
        attended_u8,
        attn,
        total_q,
        num_kv_heads,
        g,
        max_k,
        qk_fp32.stride(0),
        qk_fp32.stride(1),
        qk_fp32.stride(2),
        qk_fp32.stride(3),
        attended_u8.stride(0),
        attended_u8.stride(1),
        attended_u8.stride(2),
        attn.stride(0),
        attn.stride(1),
        attn.stride(2),
        attn.stride(3),
        BLOCK_K=BLOCK_K,
    )
    return attn


__all__ = [
    "triton_block_max_score",
    "triton_sparse_softmax",
]
