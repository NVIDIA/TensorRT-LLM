# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The (token, slot) -> row contract between an unfinalized expert call and its finalize.

An expert call made with ``do_finalize=False`` returns ``(rows, weights,
index)``. The finalize (the fused finalize all-reduce, or the unfused
``post_expert_norm_combine``) reads, for token ``t`` and slot ``k`` of its
top-``TOP_K`` selection::

    row = index[t * TOP_K + k]            # int32, contiguous [T * TOP_K]
    the slot contributes  iff  0 <= row < rows.shape[0]
    its expert output is  rows[row, :HIDDEN]

A slot outside that range contributes nothing, and its row must not be read:
the producer does not write it, so it may hold anything, NaN included. The
routing weight comes from the routing method, never from ``weights``.

Producers of ``index``:

* TRTLLM-gen (``fp4_block_scale_moe_runner``): ``expanded_idx_to_permuted_idx``,
  the slot's position in the expert-sorted, tile-padded buffer the routing-indices
  kernel lays out; ``-1`` when the slot's expert is not in the window.
* ``persistent_experts``: ``slot_row_map`` -- one row per (token,
  slot), ``row = t * TOP_K + k`` when the slot's expert is in the window, else
  ``-1``; ``rows`` has ``T * TOP_K`` rows.

A token whose slots name the same in-window expert more than once is routed
to it once: only the first such slot is valid, the later ones are ``-1``.
TRTLLM-gen's routing-indices kernel lays out one row per (token, expert), and
``slot_row_map`` keeps the first slot. (The served routing -- a top-8 over
distinct experts, remapped by ``ring_buddy_assign`` to distinct window ids --
never repeats one.)

Both are the same contract, so a finalize that follows it composes with either.
The selection rule is in one place for each form a finalize takes:
``routed_row_valid`` and ``routed_row_offset`` for a kernel (CuTe DSL or any
traced integer type), ``gather_routed_rows`` for torch. The persistent experts' map makes
``row`` a function of ``(t, k)`` alone; a finalize may only rely on that
(drop the ``index`` load) behind an explicit check of which producer ran.
"""

from __future__ import annotations

from typing import Tuple

import torch

#: Slots per token.
TOP_K = 8


def first_local_slots(topk_ids: torch.Tensor, window: int) -> torch.Tensor:
    """``[T, TOP_K]`` bool: the slot's id is in ``[0, window)`` and no earlier slot of its token holds it."""
    ids = topk_ids.reshape(-1, TOP_K)
    earlier = torch.ones(TOP_K, TOP_K, dtype=torch.bool, device=ids.device).tril(-1)
    repeat = ((ids[:, :, None] == ids[:, None, :]) & earlier).any(dim=-1)
    return (ids >= 0) & (ids < window) & ~repeat


def slot_row_map(topk_ids: torch.Tensor, window: int) -> torch.Tensor:
    """The persistent experts' row map: ``t * TOP_K + k`` for a slot ``first_local_slots`` keeps, else ``-1``.

    ``topk_ids`` is ``[T, TOP_K]`` (or ``T * TOP_K`` flat) int32 in the window's
    id space; returns ``[T, TOP_K]`` int32.
    """
    ids = topk_ids.reshape(-1, TOP_K)
    rows = torch.arange(ids.numel(), dtype=torch.int32, device=ids.device).reshape(ids.shape)
    return torch.where(first_local_slots(ids, window), rows, torch.full_like(rows, -1))


def routed_row_valid(row, num_rows):
    """Whether the slot whose map entry is ``row`` contributes: ``0 <= row < num_rows``.

    Plain comparisons, so it traces for CuTe DSL integers inside a kernel as
    well as for Python ints and torch tensors.
    """
    return (row >= 0) & (row < num_rows)


def routed_row_offset(row, row_stride, column):
    """Element offset of ``rows[row, column]``; only meaningful when ``routed_row_valid``.

    Pass 64-bit values from a kernel (``Int64(row)``), as the product can
    exceed 2^31 for large buffers.
    """
    return row * row_stride + column


def gather_routed_rows(
    rows: torch.Tensor, index: torch.Tensor, num_tokens: int, hidden: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(expert rows [T, TOP_K, hidden] FP32, valid [T, TOP_K])`` under the contract.

    Invalid slots are selected out (zeros), never read through a multiply by
    zero.
    """
    flat = index.reshape(-1)[: num_tokens * TOP_K]
    valid = routed_row_valid(flat, rows.shape[0])
    safe = torch.where(valid, flat, torch.zeros_like(flat))
    gathered = rows.index_select(0, safe.int())[:, :hidden].float()
    gathered = torch.where(valid[:, None], gathered, torch.zeros_like(gathered))
    return gathered.reshape(num_tokens, TOP_K, hidden), valid.reshape(num_tokens, TOP_K)
