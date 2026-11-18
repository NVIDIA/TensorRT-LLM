# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""s316 X1: ask the L2 for the first own-block experts' FC1 weights as soon as the router's top-8 is known.

On a TEP8 decode MoE call of 9..128 rows the router push leaves every rank the step's global top-8 ids;
``ring_buddy_assign`` then places the experts (about 2-4 us) and only after that does FC1 launch and start
streaming its weights from HBM, which is idle meanwhile. This kernel, launched on a side stream right after
the push, finds the selected experts of the rank's own block (global ``[32 rank, 32 rank + 32)``: the ones
the placement keeps on this rank but the few highest-indexed it moves) and issues bulk L2 prefetches
(``cp.async.bulk.prefetch.L2``, ``dense_gemm._bulk_prefetch_l2``) for the FC1 weight and block scales of
the first ``EXPERTS`` of them in ascending local index (the order the expert call's tables list them),
so FC1's first tiles read the L2.

It reads only the ids and writes nothing anyone reads (a sink store that never happens): the step's
values are unchanged bit for bit. A prefetch is a hint; the L2 may drop or evict lines.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ..gemm.dense_gemm import _bulk_prefetch_l2

#: Own-block experts prefetched per call (``MUSESPARK_OPEN_S316_EXPERT_L2`` = that count; 0: off).
EXPERTS = int(os.environ.get("MUSESPARK_OPEN_S316_EXPERT_L2", "0") or 0)
#: Bytes per bulk prefetch request and requests per program.
CHUNK = 32768
LANES = 32
#: Programs per prefetched expert (each covers ``LANES * CHUNK`` bytes per pass).
PER_EXPERT = 16
BLOCK = 32

_sinks = {}


@triton.jit
def _expert_l2_prefetch_kernel(
    sel_ptr,
    n_slots,
    rank,
    w_base,
    w_stride,
    w_bytes,
    s_base,
    s_stride,
    s_bytes,
    sink_ptr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    PER_EXPERT: tl.constexpr,
    CHUNK: tl.constexpr,
    LANES: tl.constexpr,
):
    j = tl.program_id(0) // PER_EXPERT  # the j-th selected own-block expert (ascending local index)
    part = tl.program_id(0) % PER_EXPERT
    offs = tl.arange(0, SLOTS)
    g = tl.load(sel_ptr + offs, mask=offs < n_slots, other=-1)
    local = g - BLOCK * rank
    valid = (local >= 0) & (local < BLOCK)
    hist = tl.histogram(tl.where(valid, local, BLOCK), 2 * BLOCK)  # bin BLOCK takes the others
    idx = tl.arange(0, 2 * BLOCK)
    present = (hist > 0) & (idx < BLOCK)
    order = tl.cumsum(present.to(tl.int32), 0) - 1  # rank of each present expert
    hit = present & (order == j)
    e = tl.sum(tl.where(hit, idx, 0), 0)
    found = tl.sum(hit.to(tl.int32), 0)
    if found > 0:
        lane = tl.arange(0, LANES).to(tl.int64)
        base_w = w_base + e.to(tl.int64) * w_stride
        for c0 in range(part * LANES * CHUNK, w_bytes, PER_EXPERT * LANES * CHUNK):
            start = c0 + lane * CHUNK
            size = tl.minimum(w_bytes - start, CHUNK)
            size = tl.where(size > 0, size, 0).to(tl.int32)
            issued = _bulk_prefetch_l2(base_w + start, size)
            tl.store(sink_ptr + lane, issued, mask=issued != 0)
        base_s = s_base + e.to(tl.int64) * s_stride
        for c0 in range(part * LANES * CHUNK, s_bytes, PER_EXPERT * LANES * CHUNK):
            start = c0 + lane * CHUNK
            size = tl.minimum(s_bytes - start, CHUNK)
            size = tl.where(size > 0, size, 0).to(tl.int32)
            issued = _bulk_prefetch_l2(base_s + start, size)
            tl.store(sink_ptr + lane, issued, mask=issued != 0)


def prefetch(
    selected: torch.Tensor, rank: int, w13: torch.Tensor, sf13: torch.Tensor, experts: int = 0
) -> None:
    """Launch the prefetch of the first ``experts`` (default ``EXPERTS``) selected own-block experts' FC1 weight and
    scales on the current stream. ``selected``: int32 [rows, 8] global ids (rows * 8 <= 1024); ``w13`` / ``sf13``:
    the stack's FC1 weight and scales, own block first."""
    k = experts or EXPERTS
    if k <= 0:
        return
    dev = selected.device
    sink = _sinks.get(dev)
    if sink is None:
        sink = torch.zeros(LANES, dtype=torch.int32, device=dev)
        _sinks[dev] = sink
    n = selected.numel()
    w_bytes = w13[0].numel() * w13.element_size()
    s_bytes = sf13[0].numel() * sf13.element_size()
    _expert_l2_prefetch_kernel[(k * PER_EXPERT,)](
        selected,
        n,
        int(rank),
        w13.data_ptr(),
        w13.stride(0) * w13.element_size(),
        w_bytes,
        sf13.data_ptr(),
        sf13.stride(0) * sf13.element_size(),
        s_bytes,
        sink,
        SLOTS=triton.next_power_of_2(max(n, 16)),
        BLOCK=BLOCK,
        PER_EXPERT=PER_EXPERT,
        CHUNK=CHUNK,
        LANES=LANES,
        num_warps=1,
    )
