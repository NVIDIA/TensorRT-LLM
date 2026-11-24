# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 rows of the cep KDA Ulysses all-to-alls: E4M3 codes and one FP32 scale per row and 128 columns.

A byte row of a send / receive buffer holds groups of 128 E4M3 codes at ``code_off + j * 128`` and
their FP32 scales at ``scale_off + j * 4`` (``j`` the group's slot in the row). The pack splits a
BF16 row's groups over destination blocks of ``gd`` slots each; the unpack is its inverse over
source blocks.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_pack_kernel(
    x_ptr,
    x_row_stride,
    u8_ptr,
    f32_ptr,
    dest_stride,
    row_stride,
    num_rows,
    code_off,
    scale_off,
    GD: tl.constexpr,
    BR: tl.constexpr,
):
    """Group g of BF16 rows ``[pid0 * BR, +BR)`` to destination block ``g // GD``, slot ``g % GD``."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    g = tl.program_id(1)
    cols = tl.arange(0, 128)
    mask = rows < num_rows
    x = tl.load(
        x_ptr + rows[:, None].to(tl.int64) * x_row_stride + g * 128 + cols[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    # The reference's rounding: PyTorch divides by the scalar 448 as a product with its FP32
    # reciprocal, and divides by the scale correctly rounded (Triton's FP32 `/` is approximate).
    scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12) * (1.0 / 448.0)
    q = tl.clamp(tl.math.div_rn(x, scale[:, None]), -448.0, 448.0)
    q = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    j = g % GD
    base = (g // GD).to(tl.int64) * dest_stride + rows.to(tl.int64) * row_stride
    tl.store(u8_ptr + base[:, None] + code_off + j * 128 + cols[None, :], q, mask=mask[:, None])
    tl.store(f32_ptr + (base + scale_off + j * 4) // 4, scale, mask=mask)


@triton.jit
def _fp8_unpack_kernel(
    u8_ptr,
    f32_ptr,
    src_stride,
    row_stride,
    code_off,
    scale_off,
    out_ptr,
    out_row_stride,
    out_col_off,
    num_rows,
    GD: tl.constexpr,
    BR: tl.constexpr,
):
    """Slot j of source block pid2's rows to output columns ``out_col_off + (pid2 * GD + j) * 128``."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    j = tl.program_id(1)
    src = tl.program_id(2)
    cols = tl.arange(0, 128)
    mask = rows < num_rows
    base = src.to(tl.int64) * src_stride + rows.to(tl.int64) * row_stride
    q = tl.load(
        u8_ptr + base[:, None] + code_off + j * 128 + cols[None, :], mask=mask[:, None], other=0
    )
    scale = tl.load(f32_ptr + (base + scale_off + j * 4) // 4, mask=mask, other=0.0)
    y = (q.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale[:, None]).to(tl.bfloat16)
    out_cols = (out_col_off + (src * GD + j) * 128 + cols).to(tl.int64)
    tl.store(
        out_ptr + rows[:, None].to(tl.int64) * out_row_stride + out_cols[None, :],
        y,
        mask=mask[:, None],
    )


def fp8_pack_rows(
    x: torch.Tensor,
    buf: torch.Tensor,
    dest_stride: int,
    row_stride: int,
    gd: int,
    code_off: int,
    scale_off: int,
) -> None:
    """BF16 ``x`` ``[R, G * 128]`` (unit column stride) into the uint8 ``buf`` (strides in bytes)."""
    rows, groups = x.shape[0], x.shape[1] // 128
    if rows:
        _fp8_pack_kernel[(triton.cdiv(rows, 32), groups)](
            x,
            x.stride(0),
            buf,
            buf.view(torch.float32),
            dest_stride,
            row_stride,
            rows,
            code_off,
            scale_off,
            GD=gd,
            BR=32,
            num_warps=4,
        )


def fp8_unpack_rows(
    buf: torch.Tensor,
    out: torch.Tensor,
    rows: int,
    num_src: int,
    src_stride: int,
    row_stride: int,
    gd: int,
    code_off: int,
    scale_off: int,
    out_col_off: int,
) -> None:
    """Rows ``[0, rows)`` of each of the ``num_src`` byte blocks of ``buf`` into BF16 ``out``."""
    if rows:
        _fp8_unpack_kernel[(triton.cdiv(rows, 64), gd, num_src)](
            buf,
            buf.view(torch.float32),
            src_stride,
            row_stride,
            code_off,
            scale_off,
            out,
            out.stride(0),
            out_col_off,
            rows,
            GD=gd,
            BR=64,
            num_warps=4,
        )
