"""Pointer preprocessing for Triton Sol-Attn when TMA is unavailable."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 64
HEAD_DIM = 128
THRESHOLD_GROUP_SIZE = 64
SUMMARY_PAD = 64


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2)
    ],
    key=["T"],
)
@triton.jit
def _reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    T,
    TP,
    NPAD,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = (
        ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D
        + dims[None, :]
    )
    k_values = tl.load(k + offsets, mask=valid[:, None], other=0.0)
    v_values = tl.load(v + offsets, mask=valid[:, None], other=0.0)
    block_len = tl.minimum(BLOCK, T - block * BLOCK).to(tl.float32)
    summary_offsets = (
        ((batch * NPAD + block) * H + head) * D + dims
    )
    tl.store(kc + summary_offsets, tl.sum(k_values, axis=0) / block_len)
    tl.store(vc + summary_offsets, tl.sum(v_values, axis=0))


@triton.jit
def _reduce_kc_stats_kernel(
    kc,
    kc_mean,
    kc_var_diag,
    NPAD,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch, head = batch_head // H, batch_head % H
    blocks = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    dims = tl.arange(0, D)
    total = tl.zeros((D,), dtype=tl.float32)
    total_sq = tl.zeros((D,), dtype=tl.float32)
    count = tl.full((), 0.0, dtype=tl.float32)
    for start in range(0, N, GROUP):
        block_indices = start + blocks
        valid = block_indices < N
        offsets = (
            ((batch * NPAD + block_indices[:, None]) * H + head) * D
            + dims[None, :]
        )
        values = tl.load(
            kc + offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
        count += tl.sum(valid.to(tl.float32), axis=0)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    tl.store(kc_mean + batch_head * D + dims, mean)
    tl.store(kc_var_diag + batch_head * D + dims, variance)


@triton.jit
def _diag_threshold_kernel(
    q,
    kc_mean,
    kc_var_diag,
    threshold,
    scale,
    T,
    TP,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TAU: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = q_block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = (
        ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D
        + dims[None, :]
    )
    q_values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(kc_mean + batch_head * D + dims)
    var_kc = tl.load(kc_var_diag + batch_head * D + dims)
    log2_scale = scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(
        q_centroid * q_centroid * var_kc,
        axis=0,
    ) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(
        threshold + (batch * N + q_block) * H + head,
        mean + TAU * std,
    )


@triton.jit
def _pool_query_kernel(
    q,
    q_bar,
    T,
    TP,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = q_block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = (
        ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D
        + dims[None, :]
    )
    values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    centroid = tl.sum(values.to(tl.float32), axis=0) / q_len
    tl.store(q_bar + (batch_head * N + q_block) * D + dims, centroid)


@triton.jit
def _exact_fused_threshold_kernel(
    q_bar,
    kc_mean,
    kc_second_moment,
    threshold,
    scale,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TAU: tl.constexpr,
):
    row_tile, batch_head = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    valid_rows = rows < N
    q_centroid = tl.load(
        q_bar + (batch_head * N + rows[:, None]) * D + dims[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    )
    mean_kc = tl.load(kc_mean + batch_head * D + dims)
    second_moment = tl.load(
        kc_second_moment
        + batch_head * D * D
        + dims[:, None] * D
        + dims[None, :]
    )
    raw_mean = tl.sum(q_centroid.to(tl.float32) * mean_kc[None, :], axis=1)
    projected = tl.dot(q_centroid, second_moment, out_dtype=tl.float32)
    raw_second_moment = tl.sum(
        projected * q_centroid.to(tl.float32),
        axis=1,
    )
    log2_scale = scale * 1.4426950408889634
    mean = raw_mean * log2_scale
    variance = tl.maximum(
        raw_second_moment - raw_mean * raw_mean,
        0.0,
    ) * (log2_scale * log2_scale)
    result = mean + TAU * tl.sqrt(variance + 1.0e-6)
    batch, head = batch_head // H, batch_head % H
    tl.store(
        threshold + (batch * N + rows) * H + head,
        result,
        mask=valid_rows,
    )


def _reduce_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, padded_tokens, heads, head_dim = k.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    padded_blocks = triton.cdiv(blocks, SUMMARY_PAD) * SUMMARY_PAD
    kc = torch.zeros(
        (batch, padded_blocks, heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.zeros_like(kc)
    _reduce_kv_kernel[(blocks, batch * heads)](
        k,
        v,
        kc,
        vc,
        tokens,
        padded_tokens,
        padded_blocks,
        heads,
        head_dim,
        BLOCK_SIZE,
    )
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> torch.Tensor:
    batch, padded_tokens, heads, head_dim = q.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    batch_heads = batch * heads
    kc_mean = torch.empty(
        (batch_heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    kc_var_diag = torch.empty_like(kc_mean)
    threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _reduce_kc_stats_kernel[(batch_heads,)](
        kc,
        kc_mean,
        kc_var_diag,
        kc.shape[1],
        heads,
        blocks,
        head_dim,
        THRESHOLD_GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )
    _diag_threshold_kernel[(blocks, batch_heads)](
        q,
        kc_mean,
        kc_var_diag,
        threshold,
        scale,
        tokens,
        padded_tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tau,
        num_warps=4,
        num_stages=2,
    )
    return threshold


def _compute_exact_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> torch.Tensor:
    batch, padded_tokens, heads, head_dim = q.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    batch_heads = batch * heads
    kc_bh = kc[:, :blocks].permute(0, 2, 1, 3)
    kc_mean = kc_bh.mean(dim=2, dtype=torch.float32)
    kc_second_moment = torch.matmul(
        kc_bh.transpose(-1, -2),
        kc_bh,
    )
    kc_second_moment.div_(blocks)
    q_bar = torch.empty(
        (batch_heads, blocks, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _pool_query_kernel[(blocks, batch_heads)](
        q,
        q_bar,
        tokens,
        padded_tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    block_m = 64
    _exact_fused_threshold_kernel[
        (triton.cdiv(blocks, block_m), batch_heads)
    ](
        q_bar,
        kc_mean,
        kc_second_moment,
        threshold,
        scale,
        heads,
        blocks,
        head_dim,
        block_m,
        tau,
        num_warps=4,
        num_stages=1,
    )
    return threshold


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    thresh_type: str = "diag",
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kc, vc = _reduce_kv(k, v, tokens=tokens)
    if thresh_type == "exact":
        threshold = _compute_exact_threshold(
            q,
            kc,
            tau=tau,
            scale=scale,
            tokens=tokens,
        )
    else:
        threshold = _compute_diag_threshold(
            q,
            kc,
            tau=tau,
            scale=scale,
            tokens=tokens,
        )
    return kc, vc, threshold


__all__ = ["prepare"]
