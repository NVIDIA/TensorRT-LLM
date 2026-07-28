# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slot-indexed conv-window movement for the KDA decode step.

Every generated token needs two things from a KDA layer's short-convolution
pool: the ``W - 1`` historical columns of each admitted request's window,
repacked into the fused decode kernel's batch-row-dense per-section layout,
and the pool rolled forward by that token. Written in ATen that is four
passes over the same windows — an indexed gather, a strided repack, a
concatenation and an indexed scatter — three of which exist only to move
bytes the first pass already read.

The kernel below does both in one indexed pass, so the decode step touches
each window twice (read, write) instead of eight times.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# 512 window rows per program keeps the [BLOCK, W] pool tile a contiguous
# 4 KB chunk at bf16 / W = 4, and still fills a Blackwell-class GPU at the
# smallest per-rank decode batches (B = 4 gives 288 programs at D = 12288).
_BLOCK = 512


@triton.jit
def _kda_conv_state_decode_step_kernel(
    pool_ptr,
    x_ptr,
    slot_ptr,
    stage_ptr,
    stride_pool_slot,
    stride_pool_dim,
    stride_pool_w,
    stride_x_row,
    stride_x_dim,
    stride_stage_section,
    stride_stage_row,
    stride_stage_dim,
    stride_stage_w,
    dim,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    section = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    offs_w = tl.arange(0, W_PAD)

    mask_d = offs_d < dim
    is_history = offs_w < W - 1
    mask_hist = mask_d[:, None] & is_history[None, :]

    slot = tl.load(slot_ptr + row).to(tl.int64)
    pool = (
        pool_ptr
        + slot * stride_pool_slot
        + (section * dim + offs_d)[:, None] * stride_pool_dim
        + offs_w[None, :] * stride_pool_w
    )

    # Columns 1..W-1 are the history the decode kernel convolves against the
    # incoming token; column 0 ages out of every future convolution.
    history = tl.load(pool + stride_pool_w, mask=mask_hist, other=0)

    stage = (
        stage_ptr
        + section * stride_stage_section
        + row * stride_stage_row
        + offs_d[:, None] * stride_stage_dim
        + offs_w[None, :] * stride_stage_w
    )
    tl.store(stage, history, mask=mask_hist)

    x_new = tl.load(
        x_ptr + row * stride_x_row + (section * dim + offs_d) * stride_x_dim, mask=mask_d, other=0
    )
    # One store for the rolled window: the history shifted down a column and
    # the incoming token appended. Keeping the roll to a single store is what
    # makes it safe in place — every byte a program writes it has already read
    # into registers, and programs partition the pool by (slot, section, dim),
    # so no two of them touch the same element.
    tl.store(
        pool,
        tl.where(is_history[None, :], history, x_new[:, None]),
        mask=mask_d[:, None] & (offs_w < W)[None, :],
    )


def kda_conv_state_decode_step(
    conv_pool: torch.Tensor,
    slot_indices: torch.Tensor,
    x_new: torch.Tensor,
    staging: torch.Tensor,
) -> None:
    """Stage each request's conv history and roll the pool by one token.

    Parameters
    ----------
    conv_pool : ``[slots, sections * dim, W]``
        The layer's short-convolution pool, updated in place.
    slot_indices : ``[B]``
        Pool row owned by each admitted request. Must be distinct — the roll
        is in place, so two requests sharing a row would race.
    x_new : ``[B, sections * dim]``
        This step's raw conv inputs, in the pool's section order. Any row/
        column strides are supported (it is normally a column slice of the
        fused in-projection output).
    staging : ``[sections, B, dim, W - 1]``
        Destination for the history columns, in the decode kernel's dense
        per-section layout.

    Equivalent to, and validated against::

        cs = conv_pool.index_select(0, slot_indices)
        staging.copy_(cs.view(B, sections, dim, W)[:, :, :, 1:].permute(1, 0, 2, 3))
        conv_pool.index_copy_(
            0, slot_indices, torch.cat([cs[:, :, 1:], x_new.unsqueeze(-1)], dim=-1)
        )
    """
    sections, batch, dim, history = staging.shape
    width = conv_pool.shape[-1]
    if history != width - 1:
        raise ValueError(f"staging holds {history} history columns, pool width is {width}")
    if conv_pool.shape[1] != sections * dim:
        raise ValueError(f"pool dim {conv_pool.shape[1]} != {sections} x {dim}")
    if tuple(x_new.shape) != (batch, sections * dim):
        raise ValueError(f"x_new {tuple(x_new.shape)} does not match [{batch}, {sections * dim}]")
    if slot_indices.shape != (batch,):
        raise ValueError(f"slot_indices {tuple(slot_indices.shape)} does not match [{batch}]")

    _kda_conv_state_decode_step_kernel[(batch, sections, triton.cdiv(dim, _BLOCK))](
        conv_pool,
        x_new,
        slot_indices,
        staging,
        conv_pool.stride(0),
        conv_pool.stride(1),
        conv_pool.stride(2),
        x_new.stride(0),
        x_new.stride(1),
        staging.stride(0),
        staging.stride(1),
        staging.stride(2),
        staging.stride(3),
        dim,
        W=width,
        W_PAD=triton.next_power_of_2(width),
        BLOCK=_BLOCK,
        num_warps=4,
    )
