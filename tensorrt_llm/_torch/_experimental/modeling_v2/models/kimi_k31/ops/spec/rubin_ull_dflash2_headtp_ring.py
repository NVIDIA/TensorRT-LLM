# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's per-rank replicated window ring for head-parallel attention (helix16 target).

Under the helix layout the drafter's FP8 history is page-sharded: global page ``g`` of a request
lives on CP rank ``g % cp`` (local page ``g // cp``), and every rank attends all heads over its own
keys, then the ranks exchange partials. Head-parallel attention needs every key of the window on
every rank instead. The history latent of a position depends on that position's target hidden
state alone, and every rank already computes the step's latents (``precompute_context_kv`` runs
replicated), so each rank keeps the whole window in its own ring and no step exchanges anything.

Layout: per drafter layer a page-shaped pool ``[slots * pages, 1, 1, page, latent]`` (the shape the
engine's history store writes, so the ring is written by the same kernel and quantization and
holds the same bytes); slot ``s`` owns pages ``[s * pages, (s + 1) * pages)``, and position ``p``
lives at ring row ``p mod R`` (``R = pages * page``): page ``s * pages + (p mod R) // page``,
offset ``p mod page``.

Writes, all fixed-shape (CUDA-graph capturable):
  * every step, the verify's ``width`` latents at ``[c, c + width)`` (``c`` the committed length
    before the step), on every rank. The served store keeps them all (``ring.keep``): the rejected
    ones lie in ``[n, c + width)`` inside the block's ``[n, n + block)``, which the block's own
    latents overwrite before anything reads them (the history store's accepted/rejected mask,
    :func:`ring_store_mask`, leaves the same bytes at every position a block reads);
  * the drafted block's ``block`` latents at ``[n, n + block)`` (``n = c + accepted``), before the
    block attends; the next step's verify write overwrites the rejected ones, as it overwrites the
    positions in the pool. :func:`ring_step` computes both target lists of a step in one launch;
  * once per request, when it first enters generation, the window of its context: every rank packs
    the positions it owns in ``[L - W, L)`` (:func:`fill_pack`), the CP group all-gathers them and
    every rank scatters them into its ring (:func:`fill_scatter`); :func:`ring_fill` does the three
    for one request. :func:`ring_fill_batch` does them for every request of a step in one pack
    launch, one all-gather and one scatter launch, with no host synchronization: which positions a
    rank owns follows from the lengths, so only the latents travel.

A block at ``n`` reads positions ``[n - W + 1, n + block)`` (row ``j`` the ``W`` positions ending at
``n + j``). Every write of the step falls in ``[c, n + block)`` inside that range, so with
``R >= W + block - 1`` no write lands on a row another position of the range uses, and no earlier
write survives at a row of the range unless it is that position's own. :func:`ring_rows` sizes
``W + width + block`` in whole pages (as :func:`..spec.dflash2_ring.dflash2_ring_rows`), so a step
whose length update is rolled back (warmup, a failed forward) has not overwritten a row its retry
reads either.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import torch

from ...kernels.spec.rubin_ull_dflash2_headtp_ring import fill_pack as _fill_pack_kernel
from ...kernels.spec.rubin_ull_dflash2_headtp_ring import fill_scatter as _fill_scatter_kernel
from ...kernels.spec.rubin_ull_dflash2_headtp_ring import step_targets
from ...kernels.spec.rubin_ull_spec_tail_kernels import store_context_latents
from .dflash2_ring import dflash2_ring_rows


class HeadTpRing(NamedTuple):
    """One rank's ring: ``pools[l]`` the page-shaped view of layer ``l``, ``ptrs`` their bases."""

    data: torch.Tensor  # [layers, slots * pages, 1, 1, page, latent]
    pools: tuple  # per layer: data[l]
    ptrs: torch.Tensor  # [layers] int64
    pages: int  # pages per slot
    page: int  # positions per page
    window: int  # W
    width: int  # verify latents per request and step
    block: int  # drafted positions per request and step
    keep: torch.Tensor  # [slots * width] int8 ones: the verify store's mask (keep every row)

    @property
    def rows(self) -> int:
        return self.pages * self.page


def ring_rows(window: int, width: int, block: int, page: int) -> int:
    """Ring rows per slot: ``W + width + block`` in whole pages."""
    return dflash2_ring_rows(window, width, block, page)


def build_ring(
    layers: int,
    slots: int,
    window: int,
    width: int,
    block: int,
    page: int,
    latent: int,
    dtype: torch.dtype,
    device,
) -> HeadTpRing:
    """A zeroed ring for ``slots`` request slots."""
    pages = ring_rows(window, width, block, page) // page
    data = torch.zeros((layers, slots * pages, 1, 1, page, latent), dtype=dtype, device=device)
    pools = tuple(data[i] for i in range(layers))
    ptrs = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device=device)
    keep = torch.ones(slots * width, dtype=torch.int8, device=device)
    return HeadTpRing(data, pools, ptrs, pages, page, window, width, block, keep)


def ring_targets(
    ring: HeadTpRing, slots: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ring ``(pages, offsets)`` of ``positions`` ``[G, k]`` for slots ``[G]`` (int64, flattened)."""
    pos = positions.long()
    rows = torch.remainder(pos, ring.rows)
    pages = slots.long().view(-1, 1) * ring.pages + rows // ring.page
    return pages.reshape(-1), (rows % ring.page).reshape(-1)


def ring_store_targets(
    ring: HeadTpRing, slots: torch.Tensor, committed: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the verify's ``width`` latents ``[c, c + width)`` of each request go (``committed`` the
    lengths ``c`` before the step); store them with the history store's mask, owner filter dropped
    (``mask.clamp(min=0)`` of a helix store mask)."""
    steps = torch.arange(width, device=committed.device, dtype=torch.int64)
    return ring_targets(ring, slots, committed.long().view(-1, 1) + steps)


def ring_store_mask(num_accepted: torch.Tensor, width: int) -> torch.Tensor:
    """``[G * width]`` int8: 1 for the accepted latents, 0 (stored zeroed) for the rejected tail; the
    history store's mask without its owner filter."""
    steps = torch.arange(width, device=num_accepted.device, dtype=torch.int64)
    return (steps < num_accepted.long().view(-1, 1)).to(torch.int8).reshape(-1)


def ring_block_targets(
    ring: HeadTpRing, slots: torch.Tensor, num_ctx: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the block's ``block`` latents ``[n, n + block)`` go (``num_ctx`` the lengths ``n``)."""
    steps = torch.arange(block, device=num_ctx.device, dtype=torch.int64)
    return ring_targets(ring, slots, num_ctx.long().view(-1, 1) + steps)


class RingStep(NamedTuple):
    """One generation step's ring targets (:func:`ring_step`), int64."""

    slots: torch.Tensor  # [G]: each request's context slot
    verify_pages: torch.Tensor  # [G * width]
    verify_offsets: torch.Tensor  # [G * width]
    block_pages: torch.Tensor  # [G * block]
    block_offsets: torch.Tensor  # [G * block]


def ring_step(
    ring: HeadTpRing,
    slot_table: torch.Tensor,
    rows: torch.Tensor,
    ctx_positions: torch.Tensor,
    num_ctx: torch.Tensor,
) -> RingStep:
    """The step's targets in one launch: request ``g`` is batch row ``rows[g]`` of context slot
    ``slot_table[rows[g]]``; its verify latents go to ``ctx_positions[g * width : (g + 1) *
    width]`` (the store's positions ``[c, c + width)``), its block to ``[num_ctx[g], num_ctx[g] +
    block)``. Equal to :func:`ring_store_targets` and :func:`ring_block_targets` on those slots."""
    groups = rows.shape[0]
    i64 = dict(dtype=torch.int64, device=rows.device)
    out = RingStep(
        torch.empty(groups, **i64),
        torch.empty(groups * ring.width, **i64),
        torch.empty(groups * ring.width, **i64),
        torch.empty(groups * ring.block, **i64),
        torch.empty(groups * ring.block, **i64),
    )
    assert ctx_positions.numel() == groups * ring.width and num_ctx.shape[0] == groups
    step_targets(slot_table, rows, ctx_positions, num_ctx, *out, ring.pages, ring.page)
    return out


def ring_store(
    ring: HeadTpRing,
    latents: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """``latents [N, layers, latent]`` (BF16) into every layer's ring at ``(pages, offsets)``, by
    the history store's kernel (the same FP8 bytes); ``mask`` (default ``ring.keep``) its mask."""
    keep = ring.keep[: latents.shape[0]] if mask is None else mask
    store_context_latents(latents, ring.pools, ring.ptrs, pages, offsets, keep)


def window_positions(num_ctx: torch.Tensor, window: int, block: int) -> torch.Tensor:
    """``[G, W + block - 1]``: the positions a block at ``n`` reads, ``n - W + 1 .. n + block - 1``
    (negative ones are absent)."""
    steps = torch.arange(window + block - 1, device=num_ctx.device, dtype=torch.int64)
    return num_ctx.long().view(-1, 1) - (window - 1) + steps


# ---- first-GEN fill ----------------------------------------------------------------------------


def fill_capacity(window: int, page: int, cp_size: int) -> int:
    """Most positions of ``[L - W, L)`` one rank owns: whole pages, one more for a partial first."""
    return (-(-window // (page * cp_size)) + 1) * page


def fill_positions(
    length: int, window: int, page: int, cp_size: int, cp_rank: int, capacity: int
) -> torch.Tensor:
    """``[capacity]`` int64 on the host: the positions of ``[max(0, L - W), L)`` CP rank
    ``cp_rank`` owns (global page ``g`` on rank ``g % cp_size``), in order, then ``-1``."""
    first = max(0, int(length) - window)
    pos = torch.arange(first, int(length), dtype=torch.int64)
    pos = pos[(pos // page) % cp_size == cp_rank]
    assert pos.numel() <= capacity, (pos.numel(), capacity)
    out = torch.full((capacity,), -1, dtype=torch.int64)
    out[: pos.numel()] = pos
    return out


def fill_pack(
    pools: Sequence[torch.Tensor],
    table_row: torch.Tensor,
    length: int,
    window: int,
    page: int,
    cp_size: int,
    cp_rank: int,
    capacity: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's share of a request's context window ``[max(0, L - W), L)``.

    ``pools`` the history's per-layer ``[pages, 1, 1, page, latent]`` views, ``table_row`` the
    request's local page table (``table_row[g // cp]`` holds global page ``g``; a host table
    keeps the indexing off the device). Returns ``(rows [layers, capacity, latent], positions
    [capacity] int64 on the host)``: the positions this rank owns (:func:`fill_positions`), their
    rows, zero past them."""
    capacity = capacity or fill_capacity(window, page, cp_size)
    device = pools[0].device
    positions = fill_positions(length, window, page, cp_size, cp_rank, capacity)
    pos = positions[positions >= 0]
    local = table_row.cpu().long()[(pos // page) // cp_size].to(device)
    offsets = (pos % page).to(device)
    latent = pools[0].shape[-1]
    rows = torch.zeros((len(pools), capacity, latent), dtype=pools[0].dtype, device=device)
    # FP8 indexing varies by build: bytes.
    raw = rows.view(torch.uint8) if rows.element_size() == 1 else rows
    for i, pool in enumerate(pools):
        src = pool.view(torch.uint8) if pool.element_size() == 1 else pool
        raw[i, : pos.numel()] = src[local, 0, 0, offsets]
    return rows, positions


def fill_scatter(ring: HeadTpRing, slot: int, rows: torch.Tensor, positions: torch.Tensor) -> None:
    """Write the gathered shares ``rows [cp, layers, capacity, latent]`` at ``positions [cp,
    capacity]`` (host or device) into ``slot``'s ring rows (``-1`` positions skipped)."""
    rows = rows.transpose(0, 1).reshape(rows.shape[1], -1, rows.shape[-1])
    positions = positions.cpu().reshape(-1)
    live = torch.nonzero(positions >= 0).view(-1)
    pos = positions[live]
    pages, offsets = ring_targets(ring, torch.tensor([slot]), pos.view(1, -1))
    device = rows.device
    live, pages, offsets = live.to(device), pages.to(device), offsets.to(device)
    raw = rows.view(torch.uint8) if rows.element_size() == 1 else rows
    for i, pool in enumerate(ring.pools):
        dst = pool.view(torch.uint8) if pool.element_size() == 1 else pool
        dst[pages, 0, 0, offsets] = raw[i].index_select(0, live)


def _byte_weights(positions: torch.Tensor, latent: int) -> torch.Tensor:
    """``[n, latent]`` int64 weights of a position-sensitive byte checksum (additive over positions)."""
    cols = torch.arange(latent, device=positions.device, dtype=torch.int64)
    return (positions.view(-1, 1) * latent + cols) % 65521 + 1


def window_checksums(
    ring: HeadTpRing,
    slot: int,
    pools: Sequence[torch.Tensor],
    table_row: torch.Tensor,
    length: int,
    page: int,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """``[2]`` int64: the checksum of this rank's owner bytes of ``[max(0, L - W), L)`` in the history
    pools, and of the whole window in ``slot``'s ring (every layer). After a fill, the sum of the
    first over the CP ranks equals the second on every rank."""
    capacity = fill_capacity(ring.window, page, cp_size)
    rows, positions = fill_pack(
        pools, table_row, length, ring.window, page, cp_size, cp_rank, capacity
    )
    device = rows.device
    latent = rows.shape[-1]
    live = positions >= 0
    own_pos = positions[live].to(device)
    own = rows.view(torch.uint8)[:, : own_pos.numel()].long() if rows.element_size() == 1 else None
    assert own is not None, "checksums read FP8 bytes"
    own_sum = (own * _byte_weights(own_pos, latent)).sum()
    pos = torch.arange(max(0, int(length) - ring.window), int(length), device=device)
    pages, offsets = ring_targets(ring, torch.tensor([slot], device=device), pos.view(1, -1))
    got = ring.data.view(torch.uint8)[:, pages, 0, 0, offsets].long()
    ring_sum = (got * _byte_weights(pos, latent)).sum()
    return torch.stack((own_sum, ring_sum))


def ring_fill(
    ring: HeadTpRing,
    slot: int,
    pools: Sequence[torch.Tensor],
    table_row: torch.Tensor,
    length: int,
    page: int,
    cp_size: int,
    cp_rank: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """A request's first generation step: its whole context window into ``slot``'s ring on every
    CP rank. ``all_gather(x)`` returns ``[cp_size, *x.shape]``, the CP ranks' ``x`` in CP rank
    order (one collective; the FP8 rows travel as BF16 pairs of bytes, unchanged). Collective:
    every rank of the group calls it for the same requests in the same order, outside capture."""
    capacity = fill_capacity(ring.window, page, cp_size)
    rows, _ = fill_pack(pools, table_row, length, ring.window, page, cp_size, cp_rank, capacity)
    wire = rows.view(torch.uint8).view(torch.bfloat16) if rows.element_size() == 1 else rows
    gathered = all_gather(wire).view(rows.dtype).view(cp_size, *rows.shape)
    positions = torch.stack(
        [fill_positions(length, ring.window, page, cp_size, r, capacity) for r in range(cp_size)]
    )
    fill_scatter(ring, slot, gathered, positions)


def ring_fill_batch(
    ring: HeadTpRing,
    meta: torch.Tensor,
    pools: Sequence[torch.Tensor],
    pool_ptrs: torch.Tensor,
    page: int,
    cp_size: int,
    cp_rank: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """:func:`ring_fill` of ``G`` requests at once: ``meta [G, 2 + table]`` int32 on the device,
    row ``g`` the request's ring slot, its history length ``L`` and its local page table row
    (``table[k]`` holds global page ``k * cp_size + cp_rank``). Every ring row of each request's
    window ``[max(0, L - W), L)`` ends up with the bytes :func:`ring_fill` writes there; nothing
    else changes. One pack launch, ``all_gather`` once (the ``[G, layers, capacity, row]`` shares as
    BF16 pairs of bytes), one scatter launch; no host read of the device. Collective: every rank of
    the group calls it with the same requests in the same order, outside capture."""
    groups = meta.shape[0]
    if groups == 0:
        return
    pool0 = pools[0]
    assert all(p.stride() == pool0.stride() and p.dtype == pool0.dtype for p in pools)
    assert pool0.stride(-1) == 1 and ring.data.dtype == pool0.dtype and ring.data.is_contiguous()
    size = pool0.element_size()
    row_bytes = pool0.shape[-1] * size
    capacity = fill_capacity(ring.window, page, cp_size)
    wire = torch.empty(
        (groups, len(pools), capacity, row_bytes), dtype=torch.uint8, device=pool0.device
    )
    _fill_pack_kernel(
        meta,
        pool_ptrs,
        pool0.stride(0) * size,
        pool0.stride(3) * size,
        wire,
        cp_rank,
        ring.window,
        page,
        cp_size,
    )
    gathered = all_gather(wire.view(torch.bfloat16)).view(torch.uint8)
    gathered = gathered.view(cp_size, groups, len(pools), capacity, row_bytes)
    _fill_scatter_kernel(
        meta, gathered, ring.data.view(torch.uint8), ring.pages, ring.window, page, cp_size
    )


__all__ = [
    "HeadTpRing",
    "RingStep",
    "build_ring",
    "fill_capacity",
    "fill_pack",
    "fill_positions",
    "fill_scatter",
    "ring_block_targets",
    "ring_fill",
    "ring_fill_batch",
    "ring_rows",
    "ring_step",
    "ring_store",
    "ring_store_mask",
    "ring_store_targets",
    "ring_targets",
    "window_checksums",
    "window_positions",
]
