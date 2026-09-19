# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-bound kernels of the two-stage SOL predictor.

The predictor summarises ``[batch, tokens, heads, head_dim]`` activations per token block, derives per
channel key statistics, and thresholds centroid scores into packed exact-block words. CUDA tensors run
Triton kernels; other tensors use PyTorch implementations of the same rule. Launch shapes are derived
from tensor shapes, so no autotuning happens at call time and every launch is CUDA Graph safe. Every
helper writes into caller-owned storage so a plan can keep graph-stable outputs.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import triton
import triton.language as tl

_POOL_MAX_WIDTH = 1024
_POOL_TOKENS_PER_LOAD = 8
_STATS_WIDTH = 128
_STATS_ROWS_PER_LOAD = 32
_SELECT_Q_BLOCKS = 64
_WORD_BITS = 32
_LOG2_E = math.log2(math.e)
_THRESHOLD_EPSILON = 1.0e-6
_LOCAL_RADIUS = 1

_Reduce = Literal["mean", "sum"]


def _column_launch(row_width: int, max_width: int) -> tuple[int, int]:
    """Columns per program and number of column chunks for a row of ``row_width`` channels."""
    width = min(max_width, triton.next_power_of_2(row_width))
    return width, triton.cdiv(row_width, width)


def num_blocks(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


def num_words(num_kv_blocks: int) -> int:
    return (num_kv_blocks + _WORD_BITS - 1) // _WORD_BITS


# --------------------------------------------------------------------------- block pooling
@triton.jit
def _block_pool_kernel(
    x_ptr,
    out_ptr,
    seq_len,
    num_blocks,
    num_chunks,
    row_width,
    stride_x_batch,
    stride_x_token,
    stride_out_batch,
    stride_out_block,
    MEAN: tl.constexpr,
    BLOCK: tl.constexpr,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """One program per (batch, block, column chunk): fp32 sum over the block's valid tokens."""
    pid = tl.program_id(0).to(tl.int64)
    chunk = pid % num_chunks
    batch_block = pid // num_chunks
    block = batch_block % num_blocks
    batch = batch_block // num_blocks
    columns = chunk * WIDTH + tl.arange(0, WIDTH)
    in_row = columns < row_width
    first_token = block * BLOCK
    total = tl.zeros([WIDTH], dtype=tl.float32)
    for start in range(0, BLOCK, TOKENS):
        tokens = first_token + start + tl.arange(0, TOKENS)
        values = tl.load(
            x_ptr + batch * stride_x_batch + tokens[:, None] * stride_x_token + columns[None, :],
            mask=(tokens < seq_len)[:, None] & in_row[None, :],
            other=0.0,
        )
        total += tl.sum(values.to(tl.float32), axis=0)
    if MEAN:
        total = total / tl.minimum(seq_len - first_token, BLOCK).to(tl.float32)
    tl.store(
        out_ptr + batch * stride_out_batch + block * stride_out_block + columns,
        total.to(out_ptr.dtype.element_ty),
        mask=in_row,
    )


def _block_pool_torch(
    x: torch.Tensor, out: torch.Tensor, *, block_size: int, reduce: _Reduce
) -> None:
    batch_size, seq_len, num_heads, head_dim = x.shape
    blocks = num_blocks(seq_len, block_size)
    padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, blocks * block_size - seq_len))
    total = padded.view(batch_size, blocks, block_size, num_heads, head_dim).sum(
        dim=2, dtype=torch.float32
    )
    if reduce == "mean":
        valid = torch.clamp(
            seq_len - torch.arange(blocks, device=x.device) * block_size, max=block_size
        )
        total = total / valid.to(torch.float32).view(1, -1, 1, 1)
    out.copy_(total)


def block_pool(x: torch.Tensor, out: torch.Tensor, *, block_size: int, reduce: _Reduce) -> None:
    """Reduce every run of ``block_size`` tokens of ``x`` into ``out`` with an fp32 accumulator.

    Args:
        x: ``[batch, seq_len, heads, head_dim]`` activations; heads and head_dim must be contiguous,
            the batch and token strides are arbitrary.
        out: Contiguous ``[batch, ceil(seq_len / block_size), heads, head_dim]`` buffer of any float
            dtype; it receives the rounded fp32 result.
        block_size: Tokens per block; the final block may be shorter.
        reduce: ``"mean"`` divides by the number of valid tokens of the block, ``"sum"`` does not.
    """
    if reduce not in ("mean", "sum"):
        raise ValueError(f"reduce must be 'mean' or 'sum'; got {reduce!r}")
    if x.ndim != 4 or x.stride(3) != 1 or x.stride(2) != x.shape[3]:
        raise ValueError(
            "x must be [batch, seq_len, heads, head_dim] with contiguous heads and head_dim"
        )
    batch_size, seq_len, num_heads, head_dim = x.shape
    blocks = num_blocks(seq_len, block_size)
    expected = (batch_size, blocks, num_heads, head_dim)
    if tuple(out.shape) != expected or not out.is_contiguous():
        raise ValueError(
            f"out must be a contiguous tensor of shape {expected}; got {tuple(out.shape)}"
        )
    if x.device.type != "cuda":
        _block_pool_torch(x, out, block_size=block_size, reduce=reduce)
        return
    row_width = num_heads * head_dim
    width, chunks = _column_launch(row_width, _POOL_MAX_WIDTH)
    _block_pool_kernel[(batch_size * blocks * chunks,)](
        x,
        out,
        seq_len,
        blocks,
        chunks,
        row_width,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        MEAN=reduce == "mean",
        BLOCK=block_size,
        TOKENS=min(_POOL_TOKENS_PER_LOAD, block_size),
        WIDTH=width,
        num_warps=4,
    )


# --------------------------------------------------------------------------- block statistics
@triton.jit
def _block_statistics_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    num_blocks,
    num_chunks,
    row_width,
    stride_x_batch,
    stride_x_block,
    stride_out_batch,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """One program per (batch, column chunk): mean and clamped variance over the block axis."""
    pid = tl.program_id(0).to(tl.int64)
    chunk = pid % num_chunks
    batch = pid // num_chunks
    columns = chunk * WIDTH + tl.arange(0, WIDTH)
    in_row = columns < row_width
    total = tl.zeros([WIDTH], dtype=tl.float32)
    total_sq = tl.zeros([WIDTH], dtype=tl.float32)
    for start in range(0, num_blocks, ROWS):
        rows = start + tl.arange(0, ROWS)
        values = tl.load(
            x_ptr + batch * stride_x_batch + rows[:, None] * stride_x_block + columns[None, :],
            mask=(rows < num_blocks)[:, None] & in_row[None, :],
            other=0.0,
        ).to(tl.float32)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
    count = num_blocks.to(tl.float32)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    tl.store(mean_ptr + batch * stride_out_batch + columns, mean, mask=in_row)
    tl.store(var_ptr + batch * stride_out_batch + columns, variance, mask=in_row)


def _block_statistics_torch(x: torch.Tensor, out_mean: torch.Tensor, out_var: torch.Tensor) -> None:
    values = x.to(torch.float32)
    mean = values.mean(dim=1)
    out_mean.copy_(mean)
    out_var.copy_(torch.clamp(values.square().mean(dim=1) - mean.square(), min=0.0))


def block_statistics(x: torch.Tensor, out_mean: torch.Tensor, out_var: torch.Tensor) -> None:
    """Per-channel mean and biased variance of ``x`` over its block axis.

    Args:
        x: Contiguous ``[batch, num_blocks, heads, head_dim]`` block summaries.
        out_mean: Contiguous fp32 ``[batch, heads, head_dim]`` buffer.
        out_var: Contiguous fp32 ``[batch, heads, head_dim]`` buffer; negative rounding is clamped to zero.
    """
    batch_size, blocks, num_heads, head_dim = x.shape
    expected = (batch_size, num_heads, head_dim)
    for name, tensor in (("out_mean", out_mean), ("out_var", out_var)):
        if (
            tuple(tensor.shape) != expected
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous fp32 tensor of shape {expected}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.device.type != "cuda":
        _block_statistics_torch(x, out_mean, out_var)
        return
    row_width = num_heads * head_dim
    width, chunks = _column_launch(row_width, _STATS_WIDTH)
    _block_statistics_kernel[(batch_size * chunks,)](
        x,
        out_mean,
        out_var,
        blocks,
        chunks,
        row_width,
        x.stride(0),
        x.stride(1),
        out_mean.stride(0),
        ROWS=_STATS_ROWS_PER_LOAD,
        WIDTH=width,
        num_warps=4,
    )


# --------------------------------------------------------------------------- exact-block selection
@triton.jit
def _select_exact_blocks_kernel(
    centroid_ptr,
    keys_ptr,
    mean_ptr,
    var_ptr,
    bits_ptr,
    num_q_blocks,
    num_kv_blocks,
    num_words,
    num_heads,
    local_radius,
    tau,
    log2_scale,
    epsilon,
    stride_c_batch,
    stride_c_block,
    stride_c_head,
    stride_k_batch,
    stride_k_block,
    stride_k_head,
    stride_s_batch,
    stride_s_head,
    stride_b_batch,
    stride_b_head,
    stride_b_block,
    Q_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program per (batch, head, tile of Q_BLOCKS query blocks); emits every word of the tile.

    Scores are ``log2_scale * <centroid, key summary>``. The fp32 centroid is split into three terms
    of the key dtype so every tensor-core product is exact and only the fp32 accumulation rounds.
    """
    tile = tl.program_id(0)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    q_blocks = tile * Q_BLOCKS + tl.arange(0, Q_BLOCKS)
    q_valid = q_blocks < num_q_blocks
    dims = tl.arange(0, HEAD_DIM)

    centroid = tl.load(
        centroid_ptr
        + batch * stride_c_batch
        + q_blocks[:, None] * stride_c_block
        + head * stride_c_head
        + dims[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    key_mean = tl.load(mean_ptr + batch * stride_s_batch + head * stride_s_head + dims)
    key_var = tl.load(var_ptr + batch * stride_s_batch + head * stride_s_head + dims)
    projected_mean = tl.sum(centroid * key_mean[None, :], axis=1) * log2_scale
    projected_var = tl.sum(centroid * centroid * key_var[None, :], axis=1) * log2_scale * log2_scale
    threshold = projected_mean + tau * tl.sqrt(tl.maximum(projected_var, 0.0) + epsilon)

    high = centroid.to(keys_ptr.dtype.element_ty)
    rest = centroid - high.to(tl.float32)
    mid = rest.to(keys_ptr.dtype.element_ty)
    low = (rest - mid.to(tl.float32)).to(keys_ptr.dtype.element_ty)

    lanes = tl.arange(0, 32)
    lane_bits = 1 << lanes.to(tl.int64)
    for word in range(num_words):
        kv_blocks = word * 32 + lanes
        kv_valid = kv_blocks < num_kv_blocks
        keys = tl.load(
            keys_ptr
            + batch * stride_k_batch
            + kv_blocks[:, None] * stride_k_block
            + head * stride_k_head
            + dims[None, :],
            mask=kv_valid[:, None],
            other=0.0,
        )
        keys_t = tl.trans(keys)
        scores = (tl.dot(high, keys_t) + tl.dot(mid, keys_t) + tl.dot(low, keys_t)) * log2_scale
        distance = q_blocks[:, None] - kv_blocks[None, :]
        is_local = (distance >= -local_radius) & (distance <= local_radius)
        exact = kv_valid[None, :] & ((scores > threshold[:, None]) | is_local)
        packed = tl.sum(tl.where(exact, lane_bits[None, :], 0), axis=1)
        tl.store(
            bits_ptr
            + batch * stride_b_batch
            + head * stride_b_head
            + q_blocks * stride_b_block
            + word,
            packed.to(tl.int32),
            mask=q_valid,
        )


def _select_exact_blocks_torch(
    centroid: torch.Tensor,
    k_summary: torch.Tensor,
    k_mean: torch.Tensor,
    k_var: torch.Tensor,
    exact_block_bits: torch.Tensor,
    *,
    tau: float,
    sm_scale: float,
) -> None:
    log2_scale = float(sm_scale) * _LOG2_E
    q = centroid.to(torch.float64)
    k = k_summary.to(torch.float64)
    projected_mean = torch.einsum("bqhd,bhd->bhq", q, k_mean.to(torch.float64)) * log2_scale
    projected_var = (
        torch.einsum("bqhd,bhd->bhq", q.square(), k_var.to(torch.float64)) * log2_scale * log2_scale
    )
    threshold = projected_mean + float(tau) * torch.sqrt(
        torch.clamp(projected_var, min=0.0) + _THRESHOLD_EPSILON
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * log2_scale
    exact = scores > threshold.unsqueeze(-1)
    num_kv_blocks = k_summary.shape[1]
    ids = torch.arange(num_kv_blocks, device=centroid.device)
    exact |= ((ids[:, None] - ids[None, :]).abs() <= _LOCAL_RADIUS)[None, None]
    words = num_words(num_kv_blocks)
    padded = torch.nn.functional.pad(exact, (0, words * _WORD_BITS - num_kv_blocks))
    weights = 1 << torch.arange(_WORD_BITS, dtype=torch.int64, device=centroid.device)
    packed = (padded.view(*exact.shape[:-1], words, _WORD_BITS).to(torch.int64) * weights).sum(
        dim=-1
    )
    exact_block_bits.copy_(packed.to(torch.uint32))


def select_exact_blocks(
    centroid: torch.Tensor,
    k_summary: torch.Tensor,
    k_mean: torch.Tensor,
    k_var: torch.Tensor,
    exact_block_bits: torch.Tensor,
    *,
    tau: float,
    sm_scale: float,
) -> None:
    """Pack the SOL exact-block decision of every (query block, key block) pair into ``exact_block_bits``.

    A key block is exact when ``sm_scale * log2(e) * <centroid, k_summary>`` exceeds the row threshold
    ``mean + tau * sqrt(var + 1e-6)`` projected from the key statistics, or when it lies within one
    block of the query block. Bit ``r`` of word ``w`` selects key block ``32 * w + r``; padding bits of
    the final word are zero.

    Args:
        centroid: Contiguous fp32 ``[batch, num_q_blocks, heads, head_dim]`` query block means.
        k_summary: Contiguous ``[batch, num_kv_blocks, heads, head_dim]`` key block means (bf16 or fp16).
        k_mean: fp32 ``[batch, heads, head_dim]`` mean of ``k_summary`` over its block axis.
        k_var: fp32 ``[batch, heads, head_dim]`` variance of ``k_summary`` over its block axis.
        exact_block_bits: Contiguous uint32 ``[batch, heads, num_q_blocks, ceil(num_kv_blocks / 32)]``.
        tau: Threshold slope in standard deviations.
        sm_scale: Softmax scale of the attention call.
    """
    batch_size, q_blocks, num_heads, head_dim = centroid.shape
    kv_blocks = k_summary.shape[1]
    expected_bits = (batch_size, num_heads, q_blocks, num_words(kv_blocks))
    if tuple(exact_block_bits.shape) != expected_bits or exact_block_bits.dtype != torch.uint32:
        raise ValueError(f"exact_block_bits must be uint32 of shape {expected_bits}")
    if tuple(k_summary.shape) != (batch_size, kv_blocks, num_heads, head_dim):
        raise ValueError("k_summary must match centroid in batch, heads, and head_dim")
    if (
        centroid.dtype != torch.float32
        or not centroid.is_contiguous()
        or not k_summary.is_contiguous()
    ):
        raise ValueError("centroid must be contiguous fp32 and k_summary contiguous")
    if not exact_block_bits.is_contiguous():
        raise ValueError("exact_block_bits must be contiguous")
    if centroid.device.type != "cuda":
        _select_exact_blocks_torch(
            centroid, k_summary, k_mean, k_var, exact_block_bits, tau=tau, sm_scale=sm_scale
        )
        return
    bits = exact_block_bits.view(torch.int32)
    grid = (triton.cdiv(q_blocks, _SELECT_Q_BLOCKS), batch_size * num_heads)
    _select_exact_blocks_kernel[grid](
        centroid,
        k_summary,
        k_mean,
        k_var,
        bits,
        q_blocks,
        kv_blocks,
        num_words(kv_blocks),
        num_heads,
        _LOCAL_RADIUS,
        float(tau),
        float(sm_scale) * _LOG2_E,
        _THRESHOLD_EPSILON,
        centroid.stride(0),
        centroid.stride(1),
        centroid.stride(2),
        k_summary.stride(0),
        k_summary.stride(1),
        k_summary.stride(2),
        k_mean.stride(0),
        k_mean.stride(1),
        bits.stride(0),
        bits.stride(1),
        bits.stride(2),
        Q_BLOCKS=_SELECT_Q_BLOCKS,
        HEAD_DIM=head_dim,
        num_warps=4,
    )


# --------------------------------------------------------------------------- graph-visible operator
@torch.library.custom_op(
    "trtllm::visual_gen_sol_predictor",
    mutates_args=(
        "exact_block_bits",
        "k_summary",
        "v_summary",
        "k_mean",
        "k_var_diag",
        "q_centroid",
    ),
    device_types="cuda",
)
def visual_gen_sol_predictor(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_block_bits: torch.Tensor,
    k_summary: torch.Tensor,
    v_summary: torch.Tensor,
    k_mean: torch.Tensor,
    k_var_diag: torch.Tensor,
    q_centroid: torch.Tensor,
    block_size: int,
    tau: float,
    sm_scale: float,
) -> None:
    """Update caller-owned SOL route and proxy tensors in place."""

    block_pool(q, q_centroid, block_size=block_size, reduce="mean")
    block_pool(k, k_summary, block_size=block_size, reduce="mean")
    block_pool(v, v_summary, block_size=block_size, reduce="sum")
    block_statistics(k_summary, k_mean, k_var_diag)
    select_exact_blocks(
        q_centroid, k_summary, k_mean, k_var_diag, exact_block_bits, tau=tau, sm_scale=sm_scale
    )


@torch.library.register_fake("trtllm::visual_gen_sol_predictor")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_block_bits: torch.Tensor,
    k_summary: torch.Tensor,
    v_summary: torch.Tensor,
    k_mean: torch.Tensor,
    k_var_diag: torch.Tensor,
    q_centroid: torch.Tensor,
    block_size: int,
    tau: float,
    sm_scale: float,
) -> None:
    return None


__all__ = ["block_pool", "block_statistics", "num_blocks", "num_words", "select_exact_blocks"]
