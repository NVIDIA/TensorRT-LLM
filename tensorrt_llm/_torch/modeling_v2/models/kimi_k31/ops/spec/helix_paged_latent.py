# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash worker's generation-step preparation and context-latent store over one Helix CP rank's pages."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.speculative.spec_tail_kernels import PagedLatentPrepare

from ...kernels.spec.helix_paged_latent import (
    helix_paged_latent_prepare as _helix_paged_latent_prepare,
)
from ...kernels.spec.helix_paged_latent import (
    helix_store_context_latents as _helix_store_context_latents,
)


class HelixPagedLatentPrepare(OpWrapper):
    """``dflash_paged_latent_prepare`` (``speculative/spec_tail_kernels.py``) on CP rank ``cp_rank`` of ``cp_size``.

    Global page ``g`` lives on rank ``g % cp_size`` as local page ``g // cp_size``: the context latents of this step
    go to rank-local columns, the store mask is -1 on other ranks' positions, and ``num_ctx`` is capped at
    ``capacity * cp_size - block_size``. ``ctx_len`` and ``seq_lens_cuda`` are updated in place as there.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="CP2 rank 1: two requests, one crossing a page, one on the dummy slot",
            spec=dict(cp_size=2, cp_rank=1, gens=2, page_size=64, block_size=5, width=5),
        ),
        Cell(
            why="CP8 rank 7: eight requests, context past the cap of the rank's pages",
            spec=dict(cp_size=8, cp_rank=7, gens=8, page_size=64, block_size=5, width=5),
        ),
    )
    note = """
    The K3.1 DFlash worker calls it once per generation step under Helix context parallelism
    (``UnifiedDraftCacheWorkerMixin._prepare_helix_paged_latent_gen_inputs``), on every CP rank with the rank's own
    block tables; without Helix the worker runs the engine's kernel. One Triton program for the whole batch,
    integer arithmetic only, CUDA-graph capturable; it compiles on first use per (page, block, width, CP) shape.
    """

    def __call__(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        base_pages: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        first_row: int,
        dummy_slot: int,
        max_ctx: int,
        page_size: int,
        block_size: int,
        width: int,
        cp_size: int,
        cp_rank: int,
    ) -> PagedLatentPrepare:
        return _helix_paged_latent_prepare(
            num_accepted_tokens,
            accepted_tokens,
            slots,
            ctx_len,
            base_pages,
            block_counts,
            block_tables,
            seq_lens_cuda,
            first_row,
            dummy_slot,
            max_ctx,
            page_size,
            block_size,
            width,
            cp_size,
            cp_rank,
        )

    def reference(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        base_pages: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        first_row: int,
        dummy_slot: int,
        max_ctx: int,
        page_size: int,
        block_size: int,
        width: int,
        cp_size: int,
        cp_rank: int,
    ) -> PagedLatentPrepare:
        """The same outputs and in-place updates with torch ops."""
        rows = torch.arange(first_row, first_row + slots.shape[0], device=slots.device)
        nacc = num_accepted_tokens[rows].long()
        slot = slots.long()
        old = ctx_len[slot]
        capacity = block_counts[rows].long() * page_size
        table = block_tables[rows].long()
        last = block_tables.shape[1] - 1
        bonus = accepted_tokens[rows, (nacc - 1).clamp_min(0)].long()
        now = (old + nacc).clamp_max(max_ctx)
        j = torch.arange(block_size, device=slots.device)
        pos = old[:, None] + torch.arange(width, device=slots.device)
        gpage = pos // page_size
        col = (
            ((gpage // cp_size) * page_size + pos % page_size)
            .minimum(capacity[:, None] - 1)
            .clamp_min(0)
        )
        flag = (torch.arange(width, device=slots.device) < nacc[:, None]).to(torch.int8)
        flag = torch.where(gpage % cp_size == cp_rank, flag, -1).to(torch.int8)
        new_len = torch.where(slot == dummy_slot, 0, now)
        ctx_len[slot] = new_len
        ctx_len[dummy_slot] = 0
        num_ctx = torch.minimum(new_len, (capacity * cp_size - block_size).clamp_min(0))
        base = base_pages[slot].long()
        rebase = torch.minimum(base, num_ctx // page_size)
        num_ctx = (num_ctx - base * page_size).clamp_min(0)
        seq_lens_cuda[rows] = width
        bpos = num_ctx[:, None] + j
        blk_seq_lens = (num_ctx + block_size).int()
        return PagedLatentPrepare(
            bonus=bonus,
            query_positions=now[:, None] + j,
            ctx_positions=pos.reshape(-1),
            store_pages=table.gather(1, (col // page_size).clamp_max(last)).reshape(-1),
            store_offsets=(col % page_size).reshape(-1),
            store_mask=flag.reshape(-1),
            num_ctx=num_ctx,
            blk_pages=table.gather(1, (rebase[:, None] + bpos // page_size).clamp_max(last)),
            blk_offsets=bpos % page_size,
            blk_seq_lens=blk_seq_lens,
            blk_kv_bounds=blk_seq_lens.repeat_interleave(block_size),
            rebase_pages=rebase,
        )


class HelixStoreContextLatents(OpWrapper):
    """``store_context_latents`` (``speculative/spec_tail_kernels.py``) with Helix store flags: for every token ``t``
    and layer ``l``, ``pools[l][pages[t], 0, 0, offsets[t]] = latent[t, l] * (mask[t] > 0)`` where ``mask[t] >= 0``;
    a -1 slot (another CP rank's position) is left untouched. ``latent`` is ``[N, L, D]`` with a unit inner stride,
    ``pools`` the ``L`` ``[pages, 1, 1, page_size, D]`` views with equal strides and ``pool_ptrs`` their addresses.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="CP2: 2 x 5 tokens, 61 layers of 512-wide BF16 latents, flags 1 / 0 / -1",
            spec=dict(cp_size=2, tokens=10, layers=61, head_dim=512, dtype=torch.bfloat16),
        ),
        Cell(
            why="CP8: 8 x 5 tokens, mostly -1 flags, FP8 E4M3 latents",
            spec=dict(cp_size=8, tokens=40, layers=4, head_dim=512, dtype=torch.float8_e4m3fn),
        ),
    )
    note = """
    Called right after ``helix_paged_latent_prepare`` with its ``store_*`` outputs. One Triton program per token and
    layer; a stored value is the latent times its 0 / 1 flag in the latent dtype, so it matches torch exactly.
    """

    def __call__(
        self,
        latent: torch.Tensor,
        pools: list[torch.Tensor],
        pool_ptrs: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        _helix_store_context_latents(latent, pools, pool_ptrs, pages, offsets, mask)

    def reference(
        self,
        latent: torch.Tensor,
        pools: list[torch.Tensor],
        pool_ptrs: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """The same stores with torch indexing."""
        own = mask >= 0
        keep = (mask[own] > 0).float()[:, None]
        for layer, pool in enumerate(pools):
            vals = (latent[own, layer].float() * keep).to(latent.dtype)
            pool[pages[own], 0, 0, offsets[own]] = vals


helix_paged_latent_prepare = HelixPagedLatentPrepare()
helix_store_context_latents = HelixStoreContextLatents()
