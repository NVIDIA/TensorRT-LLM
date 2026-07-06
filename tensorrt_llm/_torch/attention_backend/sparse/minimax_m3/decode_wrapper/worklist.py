# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static persistent-CTA worklists for the M3 sparse decode kernels.

The MSA C++ SM100 FMHA kernel (``csrc/include/fmha_tile_scheduler.hpp``,
``HostPrecomputedTileScheduler``) walks a device worklist:

* ``packed_work_range[cta] = end << 32 | start`` — the slice of
  ``packed_work_info`` CTA ``cta`` owns;
* ``packed_work_info[i] = qo_tile << 32 | (head & 0xFFFF) << 16 |
  (batch & 0xFFFF)`` — one (batch, packed-head, qo-tile) work item.

MSA computes these with a device planner kernel
(``csrc/include/plan.cuh``) parameterised by per-step KV lengths for
load balancing.  For **decode** (``qo_len == 1`` per request,
``num_kv_splits == 1``) the *set* of work items is a pure function of
``(batch_size, num_packed_heads)`` — KV lengths only affect which CTA
runs which item, never whether an item exists, and each item reads its
own KV bounds from device tensors at execution time.  So the worklist
is a per-batch-size constant: build it once on the host, keep it in a
persistent device buffer, and CUDA graph replays stay correct for any
KV lengths.

Item order is batch-major then head; CTA assignment is contiguous
chunks (equal cost per item is the decode regime).  Per-item outputs
are independent, so assignment differences vs. MSA's greedy planner
cannot change results — only load balance.
"""

from __future__ import annotations

from typing import Tuple

import torch


def build_decode_worklist(
    *,
    batch_size: int,
    num_packed_heads: int,
    num_ctas: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``(packed_work_range, packed_work_info)`` for one decode shape.

    Parameters
    ----------
    batch_size : number of requests (== work-item batch dim; for the
        per-token-expanded sparse pass this equals ``total_q``).
    num_packed_heads : ``num_qo_heads // pack_factor`` — the head count
        the kernel iterates after pack-GQA folding.
    num_ctas : persistent grid width (SM count); also the length of
        ``packed_work_range``.
    device : CUDA device for the returned buffers.

    Returns
    -------
    (packed_work_range ``[num_ctas]`` int64, packed_work_info ``[n]`` int64)
    with ``n = batch_size * num_packed_heads`` (every item has
    ``qo_tile == 0`` because decode packs at most 128 q rows per tile).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size > 0xFFFF:
        raise ValueError(f"batch_size {batch_size} exceeds the 16-bit work-item field")
    if num_packed_heads <= 0 or num_packed_heads > 0xFFFF:
        raise ValueError(f"num_packed_heads out of range: {num_packed_heads}")

    n_items = batch_size * num_packed_heads

    # Batch-major enumeration: item = b * H + h  ->  head h, batch b.
    batch_idx = torch.arange(n_items, dtype=torch.int64) // num_packed_heads
    head_idx = torch.arange(n_items, dtype=torch.int64) % num_packed_heads
    work_info = (head_idx << 16) | batch_idx  # qo_tile == 0

    # Contiguous, maximally even split of [0, n_items) across CTAs.
    # CTA c owns [c * n // C rounded, ...) via cumulative fair shares.
    bounds = (torch.arange(num_ctas + 1, dtype=torch.int64) * n_items) // num_ctas
    starts = bounds[:-1]
    ends = bounds[1:]
    work_range = (ends << 32) | starts

    return work_range.to(device), work_info.to(device)


__all__ = ["build_decode_worklist"]
