# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels that stage the operands of the DFlash2 drafter's exact FP32 window decode.

The decode (``sparse_mla_rows_eager`` in ``ops/attention/dflash2_window``) attends each block
query row to the pool rows of its window in FP32: two TF32 einsums and a softmax over FP32
operands. Its operands are the window's pool rows and the query, both widened to FP32; its scores
are scaled, and the columns past a row's live length (``-1`` in the index list) are ``-inf``.
Two launches stage them here instead of nine elementwise and gather kernels:

- ``_window_operands_kernel`` writes the FP32 window rows and the FP32 query. Entry ``i`` of the
  index list reads pool row ``max(rows[i], 0)``: a masked column reads row 0, as the decode's clamp
  does. The row is also clamped to the pool's last row, so no entry reads outside the pool.
- ``_scale_mask_kernel`` rewrites the scores in place: ``scores * scale`` where the column's index
  is live, ``-inf`` where it is ``-1``.

FP8 E4M3 and BF16 widen to FP32 exactly, and ``x * scale`` is one FP32 product here as in torch,
so with the einsums and the softmax called as the decode calls them, :func:`window_fp32` returns
the decode's output bit for bit.

Bounds: no data-dependent loop (one static loop over a row's column chunks); every load and store
is masked by its row and column bounds; gathered indices are clamped into the pool; addresses
are 64-bit; no kernel waits on another. Compile hygiene: row counts and strides are runtime
values (``do_not_specialize``), so a new batch never compiles; the row width, the tiles and
whether a dequant scale applies are ``constexpr``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Window or query rows per program of the operand kernel.
BLOCK_ROWS = 16
#: Columns per chunk of a row: the 576-wide latent row is 9 chunks.
BLOCK_COLS = 64
#: Score elements per program of the scale-and-mask kernel.
BLOCK_SCORES = 1024
#: Warps per program of either kernel.
NUM_WARPS = 4


@triton.jit(
    do_not_specialize=[
        "stride_pool",
        "last_row",
        "num_kv",
        "stride_qr",
        "stride_qh",
        "num_q",
        "heads",
    ]
)
def _window_operands_kernel(
    pool_ptr,
    stride_pool,
    last_row,
    rows_ptr,
    num_kv,
    kv_ptr,
    query_ptr,
    stride_qr,
    stride_qh,
    num_q,
    heads,
    q32_ptr,
    kv_scale,
    DIM: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Programs ``[0, cdiv(num_kv, BLOCK_ROWS))`` gather and widen window rows; the rest widen
    query rows (``(row, head)`` pairs, row-major)."""
    pid = tl.program_id(0)
    kv_programs = tl.cdiv(num_kv, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    # No name is bound in both branches: Triton merges such a name into one value of one type
    # after a runtime ``if``, and the pool's element type (FP8 or BF16) need not be the query's.
    if pid < kv_programs:
        kv_i = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        kv_live = kv_i < num_kv
        kv_src = tl.load(rows_ptr + kv_i, mask=kv_live, other=0)
        kv_src = tl.minimum(tl.maximum(kv_src, 0), last_row).to(tl.int64)
        pool_row = pool_ptr + kv_src * stride_pool
        kv_row = kv_ptr + kv_i.to(tl.int64) * DIM
        for kv_c0 in tl.static_range(0, DIM, BLOCK_COLS):
            kv_c = kv_c0 + cols
            kv_m = kv_live[:, None] & (kv_c < DIM)[None, :]
            kv_x = tl.load(pool_row[:, None] + kv_c[None, :], mask=kv_m, other=0.0).to(tl.float32)
            if HAS_SCALE:
                kv_x = kv_x * kv_scale
            tl.store(kv_row[:, None] + kv_c[None, :], kv_x, mask=kv_m)
    else:
        q_i = (pid - kv_programs) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        q_live = q_i < num_q
        q_r = (q_i // heads).to(tl.int64)
        q_h = (q_i % heads).to(tl.int64)
        query_row = query_ptr + q_r * stride_qr + q_h * stride_qh
        q32_row = q32_ptr + q_i.to(tl.int64) * DIM
        for q_c0 in tl.static_range(0, DIM, BLOCK_COLS):
            q_c = q_c0 + cols
            q_m = q_live[:, None] & (q_c < DIM)[None, :]
            q_x = tl.load(query_row[:, None] + q_c[None, :], mask=q_m, other=0.0).to(tl.float32)
            tl.store(q32_row[:, None] + q_c[None, :], q_x, mask=q_m)


@triton.jit(
    do_not_specialize=["stride_sr", "stride_sh", "stride_rows", "num_scores", "heads", "width"]
)
def _scale_mask_kernel(
    scores_ptr,
    stride_sr,
    stride_sh,
    rows_ptr,
    stride_rows,
    num_scores,
    heads,
    width,
    scale,
    BLOCK: tl.constexpr,
):
    """``scores[r, h, k] = scores[r, h, k] * scale`` if ``rows[r, k] >= 0`` else ``-inf``."""
    e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = e < num_scores
    k = e % width
    rh = e // width
    h = rh % heads
    r = rh // heads
    valid = tl.load(rows_ptr + r * stride_rows + k, mask=live, other=-1) >= 0
    ptr = scores_ptr + r * stride_sr + h * stride_sh + k
    x = tl.load(ptr, mask=live, other=0.0)
    tl.store(ptr, tl.where(valid, x * scale, float("-inf")), mask=live)


def launch_window_operands(
    query: torch.Tensor,
    pool_rows: torch.Tensor,
    rows: torch.Tensor,
    kv_scale: float,
    kv_out: torch.Tensor,
    q_out: torch.Tensor,
) -> None:
    """``kv_out [R, K, C]`` = the window rows of ``pool_rows`` (``max(rows, 0)``) widened to FP32
    (times ``kv_scale`` when it is not 1); ``q_out [R, H, C]`` = ``query`` widened to FP32."""
    num_rows, heads, dim = query.shape
    num_kv = rows.numel()
    num_q = num_rows * heads
    programs = triton.cdiv(num_kv, BLOCK_ROWS) + triton.cdiv(num_q, BLOCK_ROWS)
    if programs == 0:
        return
    # The index list is walked as one flat run, a row's columns and the query's last dim directly.
    assert rows.is_contiguous() and pool_rows.stride(1) == 1 and query.stride(2) == 1
    assert kv_out.is_contiguous() and q_out.is_contiguous()
    _window_operands_kernel[(programs,)](
        pool_rows,
        pool_rows.stride(0),
        pool_rows.shape[0] - 1,
        rows,
        num_kv,
        kv_out,
        query,
        query.stride(0),
        query.stride(1),
        num_q,
        heads,
        q_out,
        float(kv_scale),
        DIM=dim,
        HAS_SCALE=kv_scale != 1.0,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=NUM_WARPS,
    )


def launch_scale_mask(scores: torch.Tensor, rows: torch.Tensor, scale: float) -> None:
    """In place: ``scores [R, H, K]`` times ``scale`` where ``rows [R, K]`` is live, else
    ``-inf``."""
    num_rows, heads, width = scores.shape
    num_scores = scores.numel()
    if num_scores == 0:
        return
    # Unit inner strides: the score columns and the index list's columns are walked directly.
    assert scores.stride(2) == 1 and rows.stride(1) == 1, "unit-stride score and index columns"
    _scale_mask_kernel[(triton.cdiv(num_scores, BLOCK_SCORES),)](
        scores,
        scores.stride(0),
        scores.stride(1),
        rows,
        rows.stride(0),
        num_scores,
        heads,
        width,
        float(scale),
        BLOCK=BLOCK_SCORES,
        num_warps=NUM_WARPS,
    )


def window_fp32(
    query: torch.Tensor,
    pool_rows: torch.Tensor,
    rows: torch.Tensor,
    softmax_scale: float,
    latent_dim: int,
    kv_scale: float = 1.0,
) -> torch.Tensor:
    """The exact FP32 window decode with its operands staged in two launches.

    ``query`` ``[R, H, C]`` (contiguous BF16), ``pool_rows`` ``[N, C]`` (FP8 E4M3 or BF16, unit
    inner stride), ``rows`` ``[R, K]`` (contiguous int32, ``-1`` masked). Returns
    ``[R, H, latent_dim]`` in the query's dtype: ``sparse_mla_rows_eager``'s output bit for bit.
    """
    num_rows, heads, dim = query.shape
    kv = torch.empty((num_rows, rows.shape[1], dim), dtype=torch.float32, device=query.device)
    q32 = torch.empty((num_rows, heads, dim), dtype=torch.float32, device=query.device)
    launch_window_operands(query, pool_rows, rows, kv_scale, kv, q32)
    scores = torch.einsum("rhc,rkc->rhk", q32, kv)
    launch_scale_mask(scores, rows, softmax_scale)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("rhk,rkc->rhc", probs, kv[..., :latent_dim])
    return out.to(query.dtype)


__all__ = ["launch_scale_mask", "launch_window_operands", "window_fp32"]
