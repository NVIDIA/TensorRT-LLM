# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the Rubin KDA prefill and convolution entries in the target's runtime layouts.

Not a test file: the entry tests import it by name from their own directory. The builders take a
device, so the CPU reference test uses them too.
"""

from __future__ import annotations

import math

import torch

__extra_import_path__ = ["."]
from _kda_ops_util import (  # noqa: E402 — needs the path declared above
    CONV_WIDTH,
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    padded_slots,
    stacked_columns,
    stacked_projection,
)

_SLOT_PAD_BYTES = 64  # between pool slots: a part-page slot stride (an 8-element multiple)


def padded_pool(slots: int, row_shape: tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
    """`[slots, *row_shape]`, contiguous within a slot, padded between slots."""
    return padded_slots(slots, row_shape, dtype, _SLOT_PAD_BYTES, device)


def coalesced_pool(
    slots: int, row_shape: tuple[int, ...], dtype: torch.dtype, device
) -> torch.Tensor:
    """`[slots, *row_shape]` as the V2 manager lays a recurrent pool out: a slot holds the same-size
    pages of two layers, and this is the second layer's page of each slot."""
    row = math.prod(row_shape)
    storage = torch.zeros(slots, 2 * row, dtype=dtype, device=device)
    return storage[:, row:].view(slots, *row_shape)


def cu_seqlens_of(seq_lens, device, dtype=torch.int64) -> torch.Tensor:
    starts = [0]
    for length in seq_lens:
        starts.append(starts[-1] + length)
    return torch.tensor(starts, dtype=dtype, device=device)


def prefill_args(spec: dict, seed: int, device="cuda") -> dict:
    """Keyword arguments of `rubin_kda_prefill` for a cell spec.

    Fresh rows are zeroed, continuation rows hold a random state; the pool also holds the snapshot
    rows. With snapshots the pool has the V2 layout, since the kernel also takes it as its compact
    state checkpoints; otherwise its slots are padded by part of a page. Inputs are the shapes the
    fused convolution and the gate projection produce; with `stacked`, beta is the b columns of
    `stacked_projection`.
    """
    heads, seq_lens = spec["heads"], spec["seq_lens"]
    state_dtype = getattr(torch, spec["state_dtype"])
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = sum(seq_lens)
    shape = (1, tokens, heads, HEAD_DIM)

    def rnd(*size, scale=1.0):
        return torch.randn(*size, generator=gen, device=device) * scale

    snapshot_rows = spec.get("snapshot_rows")
    rows = list(spec["slots"]) + [r for r in (snapshot_rows or ()) if r >= 0]
    make_pool = padded_pool if snapshot_rows is None else coalesced_pool
    pool = make_pool(max(rows) + 2, (heads, HEAD_DIM, HEAD_DIM), state_dtype, device)
    pool.copy_(rnd(*pool.shape, scale=0.05))
    for slot, fresh in zip(spec["slots"], spec["fresh"]):
        if fresh:
            pool[slot].zero_()
    width = spec.get("beta_row_width")
    if spec.get("stacked"):
        b0 = stacked_columns(heads)["b"]
        beta = stacked_projection(tokens, heads, gen, device)[:, b0 : b0 + heads].unsqueeze(0)
    elif width is None:
        beta = rnd(1, tokens, heads).to(torch.bfloat16)
    else:
        merged = rnd(tokens, width).to(torch.bfloat16)
        beta = merged[:, HEAD_DIM : HEAD_DIM + heads].unsqueeze(0)
    args = dict(
        q=rnd(*shape).to(torch.bfloat16),
        k=rnd(*shape).to(torch.bfloat16),
        v=rnd(*shape, scale=0.5).to(torch.bfloat16),
        g=rnd(*shape).to(torch.bfloat16),
        beta=beta,
        state_pool=pool,
        state_indices=torch.tensor(spec["slots"], dtype=torch.int32, device=device),
        cu_seqlens=cu_seqlens_of(seq_lens, device),
        A_log=rnd(heads, scale=0.5),
        dt_bias=rnd(heads * HEAD_DIM, scale=0.5),
        scale=SCALE,
        lower_bound=LOWER_BOUND,
        seq_order=None,
        cu_chunks=None,
        snapshot_rows=None,
        snapshot_chunks=None,
    )
    if snapshot_rows is not None:
        args["snapshot_rows"] = torch.tensor(snapshot_rows, dtype=torch.int32, device=device)
        args["snapshot_chunks"] = torch.tensor(
            spec["snapshot_chunks"], dtype=torch.int32, device=device
        )
    return args


def conv_args(spec: dict, seed: int, device="cuda") -> dict:
    """Keyword arguments of `rubin_kda_prefill_conv` for a cell spec: src is the first 3D columns of
    a `[T, src_width_of * D]` projection, or with `stacked` the whole `stacked_projection` (what the
    specialized targets pass); the pool a padded-slot `[slots, 3D, 3]` bf16 pool."""
    heads, seq_lens = spec["heads"], spec["seq_lens"]
    dim = heads * HEAD_DIM
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = sum(seq_lens)
    if spec.get("stacked"):
        src = stacked_projection(tokens, heads, gen, device)
    else:
        proj = torch.randn(tokens, spec.get("src_width_of", 4) * dim, generator=gen, device=device)
        src = proj.to(torch.bfloat16)[:, : 3 * dim]
    pool = padded_pool(spec["pool_slots"], (3 * dim, CONV_WIDTH - 1), torch.bfloat16, device)
    pool.copy_(torch.randn(pool.shape, generator=gen, device=device) * 0.5)
    slots = torch.randperm(spec["pool_slots"], generator=gen, device=device)[: len(seq_lens)]
    return dict(
        src=src,
        weight=(torch.randn(3 * dim, CONV_WIDTH, generator=gen, device=device) * 0.3).to(
            torch.bfloat16
        ),
        conv_state=pool,
        starts=cu_seqlens_of(seq_lens, device, torch.int32),
        slots=slots.to(torch.int32),
        has_init=torch.tensor(spec["has_init"], dtype=torch.bool, device=device),
        max_seqlen=max(seq_lens),
    )
