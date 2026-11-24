# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A per-slot ring of the DFlash2 MLA drafter's context latents, bounded by its sliding window.

The drafter's query at position ``p`` reads the latents at positions ``[max(0, p - W + 1), p]``
only (``W`` the sliding window, :mod:`..attention.dflash2_window`), and the latent of a position depends
on that position's target hidden state alone. So a slot of the private paged arena holds ``R``
rows instead of one row per servable position, and position ``p`` lives at slot row ``p mod R``
(page ``table[slot, (p mod R) // page_size]``, offset ``(p mod R) % page_size``).

Rows are written per committed position as with a dense arena: the prompt at prefill, then after
each verify the ``V`` rows ``[c, c + V)`` (``c`` the committed length, ``V`` the verify width; the
rejected tail is a placeholder the next block overwrites), then the drafted block's ``B`` rows
``[c', c' + B)`` (``c' = c + accepted``, ``accepted >= 1``) before it is read. A read at ``c'``
needs ``[c' - W + 1, c' + B)``, and the writes since that range was committed reach at most
``max(V - 2, B - 1) + W - 1`` positions past its oldest entry, so ``R >= W + max(V, B) - 1`` keeps
every row a window reads from being overwritten by a later position. :func:`dflash2_ring_rows`
sizes for one verify and one block write past ``c`` (``W + V + B`` rows, whole pages), so a step
whose length update is rolled back (warmup, a failed forward) has not overwritten a row its
retry reads either.

A prefill chunk longer than ``R`` wraps onto itself: only its last ``R`` positions, the ones a later
window can read, are stored (:func:`dflash2_ring_chunk_start`), which also keeps one store free of
duplicate destinations. Positions are slot-local and start at 0 for every request on this arena
(no reused prefix is addressable on it), so the ring never lacks a position a window reads.

Everything here is fixed-shape device math or host integers: CUDA-graph capturable.
"""

from typing import Optional

import torch

#: Tokens per page of the DFlash worker's private paged arena (its ``_ctx_page_size``).
DFLASH2_CTX_PAGE_SIZE = 32


def dflash2_ring_rows(window: int, verify_width: int, block_width: int, page_size: int) -> int:
    """Rows per slot: the window plus one verify write and one block write, in whole pages.

    Args:
        window: positions a query attends, itself included (``sliding_window``).
        verify_width: positions a verify writes from the committed length (``max_draft_len + 1``).
        block_width: positions of the drafted block (the drafter's block width).
        page_size: tokens per arena page.
    """
    if min(window, verify_width, block_width, page_size) <= 0:
        raise ValueError(
            f"ring needs a positive window/verify width/block width/page size, got {window}/"
            f"{verify_width}/{block_width}/{page_size}"
        )
    need = window + verify_width + block_width
    return -(-need // page_size) * page_size


def dflash2_ring_positions(positions: torch.Tensor, ring_rows: Optional[int]) -> torch.Tensor:
    """Slot rows of ``positions`` (``positions mod ring_rows``); unchanged without a ring."""
    if not ring_rows:
        return positions
    return torch.remainder(positions, ring_rows)


def dflash2_ring_chunk_start(length: int, ring_rows: Optional[int]) -> int:
    """First row of a ``length``-position write that the ring keeps (its last ``ring_rows``)."""
    if not ring_rows:
        return 0
    return max(0, int(length) - int(ring_rows))


def dflash2_ctx_rows_per_slot(
    window: Optional[int],
    max_ctx: int,
    verify_width: int,
    block_width: int,
    page_size: int = DFLASH2_CTX_PAGE_SIZE,
) -> int:
    """Rows one slot of the worker's private arena holds, as ``_lazy_init_ctx_buffers`` sizes it.

    The ring (:func:`dflash2_ring_rows`) when the drafter reads a window and the ring is smaller
    than the dense capacity ``max_ctx + block_width``; that capacity in whole pages otherwise.
    """
    capacity = max_ctx + block_width
    if window:
        ring_rows = dflash2_ring_rows(window, verify_width, block_width, page_size)
        if ring_rows < capacity:
            return ring_rows
    return -(-capacity // page_size) * page_size


def dflash2_ctx_bytes_per_slot(
    draft_config: dict, max_seq_len: Optional[int], max_draft_len: int
) -> Optional[int]:
    """Bytes one request slot of the DFlash2 MLA drafter's context arena takes on a GPU.

    One latent (``kv_lora_rank + qk_rope_head_dim``) per position and drafter layer, unsharded,
    in :func:`dflash2_ctx_rows_per_slot` rows; the window is ``sliding_window`` when every layer is
    a ``sliding_attention`` layer. ``max_ctx`` is ``min(max_seq_len, max_position_embeddings)`` and
    the verify and block widths are ``max_draft_len + 1``, as the worker resolves them. None when
    ``draft_config`` has no MLA geometry.
    """
    layers = draft_config.get("num_hidden_layers")
    latent = (draft_config.get("kv_lora_rank") or 0) + (draft_config.get("qk_rope_head_dim") or 0)
    if not layers or not latent:
        return None
    layer_types = draft_config.get("layer_types") or []
    sliding = bool(layer_types) and all(t == "sliding_attention" for t in layer_types)
    window = draft_config.get("sliding_window") if sliding else None
    limits = [c for c in (max_seq_len, draft_config.get("max_position_embeddings")) if c]
    max_ctx = min(limits) if limits else 8192
    width = max_draft_len + 1
    rows = dflash2_ctx_rows_per_slot(window, max_ctx, width, width)
    dtype = draft_config.get("dtype") or draft_config.get("torch_dtype")
    itemsize = 4 if dtype in ("float32", "float") else 2
    return layers * rows * latent * itemsize


__all__ = [
    "DFLASH2_CTX_PAGE_SIZE",
    "dflash2_ctx_bytes_per_slot",
    "dflash2_ctx_rows_per_slot",
    "dflash2_ring_chunk_start",
    "dflash2_ring_positions",
    "dflash2_ring_rows",
]
