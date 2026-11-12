# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash paged-context bookkeeping of one generation step, Helix-sharded or not, in one
Triton launch (one program)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import dflash_gen_prepare


def gen_prepare_reference(
    num_accepted_tokens: torch.Tensor,
    accepted_tokens: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    block_counts: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    first_row: int,
    dummy_slot: int,
    max_ctx: int,
    page_size: int,
    block_size: int,
    width: int,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """``(outputs, ctx_len after, seq_lens_cuda after)`` of the step preparation for the
    ``len(slots)`` generation requests at batch rows ``first_row + g``; nothing is written."""
    device = ctx_len.device
    num_gens = slots.shape[0]
    g_rows = first_row + torch.arange(num_gens, device=device)
    nacc = num_accepted_tokens[g_rows].to(torch.int64)
    slot = slots.to(torch.int64)
    old = ctx_len[slot].to(torch.int64)
    counts = block_counts[g_rows].to(torch.int64)
    table = block_tables[g_rows].to(torch.int64)
    max_blocks = block_tables.shape[1]

    def page_of(local_pos):
        return table.gather(1, (local_pos // page_size).clamp(max=max_blocks - 1))

    bonus = accepted_tokens[g_rows].gather(1, (nacc - 1).clamp(min=0)[:, None]).squeeze(1)
    bonus = bonus.to(torch.int64)
    now = (old + nacc).clamp(max=max_ctx)
    j_block = torch.arange(block_size, device=device)
    j_width = torch.arange(width, device=device)
    pos = old[:, None] + j_width[None, :]
    capacity = counts * page_size
    gpage = pos // page_size
    col = (gpage // cp_size) * page_size + pos % page_size
    col = torch.minimum(col, capacity[:, None] - 1).clamp(min=0)
    store = (j_width[None, :] < nacc[:, None]).to(torch.int8)
    store = torch.where(gpage % cp_size == cp_rank, store, torch.full_like(store, -1))
    new_len = torch.where(slot == dummy_slot, torch.zeros_like(now), now)
    num_ctx = torch.minimum(new_len, (capacity * cp_size - block_size).clamp(min=0))
    bpos = num_ctx[:, None] + j_block[None, :]
    bgpage = bpos // page_size
    bslot = (bgpage // cp_size) * page_size + bpos % page_size
    total = num_ctx + block_size
    out = dict(
        bonus=bonus,
        query_positions=now[:, None] + j_block[None, :],
        ctx_positions=pos.reshape(-1),
        store_pages=page_of(col).reshape(-1),
        store_offsets=(col % page_size).reshape(-1),
        store_mask=store.reshape(-1),
        num_ctx=num_ctx,
        blk_pages=page_of(bslot),
        blk_offsets=bslot % page_size,
    )
    if cp_size > 1:
        out["blk_owned"] = (bgpage % cp_size == cp_rank).to(torch.int8)
        span = page_size * cp_size
        rem = total % span - cp_rank * page_size
        total = (total // span) * page_size + rem.clamp(min=0, max=page_size)
    out["blk_seq_lens"] = total.to(torch.int32)
    out["blk_kv_bounds"] = total.to(torch.int32).repeat_interleave(block_size)
    ctx_out = ctx_len.clone()
    ctx_out[slot] = new_len.to(ctx_len.dtype)
    ctx_out[dummy_slot] = 0
    seq_out = seq_lens_cuda.clone()
    seq_out[g_rows] = width
    return out, ctx_out, seq_out


class K3DflashGenPrepare(OpWrapper):
    """The per-request bookkeeping of a DFlash generation step whose drafter context is the
    paged single-latent cache, as a dict of new tensors, and its two side effects: ``ctx_len``
    advanced in place (the dummy slot zeroed) and ``seq_lens_cuda`` of the generation rows set
    to ``width``.

    Generation request ``g`` is batch row ``num_contexts + g`` and context slot ``slots[g]``. The
    dict holds the bonus token, ``embed_ids`` (the bonus tokens then ``mask_token_id``), the block
    query positions, the RoPE positions / destination page / offset / write mask of the ``width``
    target latents, the advertised context length and the block-decode page / offset / bound
    operands. With ``cp_size > 1`` the pages are Helix-sharded: the store mask is -1 on other
    ranks' positions, the block operands are rank-local and ``blk_owned`` marks this rank's.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="dep16 (c383): 16 requests, DSpark 5 (width and block 6), one rank's pages",
            spec=dict(num_gens=16, draft_len=5, cp_size=1, cp_rank=0, dummies=0),
        ),
        Cell(
            why="helix8 c240: 20 requests of DSpark 5, pages over 8 ranks (rank 3)",
            spec=dict(num_gens=20, draft_len=5, cp_size=8, cp_rank=3, dummies=0),
        ),
        Cell(
            why="helix8 c75/c174: 8 requests of DSpark 7 on the last of 8 ranks",
            spec=dict(num_gens=8, draft_len=7, cp_size=8, cp_rank=7, dummies=0),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4): 4 requests of DSpark 7 over 16 ranks (rank 5), two "
            "padding rows on the dummy slot",
            spec=dict(num_gens=4, draft_len=7, cp_size=16, cp_rank=5, dummies=2),
        ),
    )

    note = """
    ``K3DFlashWorker.prepare_1st_drafter_inputs`` and ``_build_mla_block_fixup`` for that cache, in
    one program of 4 warps; every lane reads its old context length before any lane writes one.
    The advertised context is capped so the block fits the allocation. ``accepted_tokens`` and
    ``block_tables`` need unit inner strides.
    """

    def __call__(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        num_contexts: int,
        num_gens: int,
        dummy_slot: int,
        max_ctx: int,
        mask_token_id: int,
        page_size: int,
        block_size: int,
        width: int,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> dict:
        return dflash_gen_prepare(
            num_accepted_tokens,
            accepted_tokens,
            slots,
            ctx_len,
            block_counts,
            block_tables,
            seq_lens_cuda,
            num_contexts,
            num_gens,
            dummy_slot,
            max_ctx,
            mask_token_id,
            page_size,
            block_size,
            width,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

    def reference(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        num_contexts: int,
        num_gens: int,
        dummy_slot: int,
        max_ctx: int,
        mask_token_id: int,
        page_size: int,
        block_size: int,
        width: int,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """``(the op's dict, ctx_len after, seq_lens_cuda after)``; nothing is written."""
        out, ctx_out, seq_out = gen_prepare_reference(
            num_accepted_tokens,
            accepted_tokens,
            slots[:num_gens],
            ctx_len,
            block_counts,
            block_tables,
            seq_lens_cuda,
            num_contexts,
            dummy_slot,
            max_ctx,
            page_size,
            block_size,
            width,
            cp_size,
            cp_rank,
        )
        mask = torch.full((1,), int(mask_token_id), dtype=torch.int64, device=ctx_len.device)
        out["embed_ids"] = torch.cat((out["bonus"], mask))
        return out, ctx_out, seq_out

    def is_valid(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
        num_contexts: int,
        num_gens: int,
        dummy_slot: int,
        max_ctx: int,
        mask_token_id: int,
        page_size: int,
        block_size: int,
        width: int,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> None:
        # ctx_len is stored through an int64 pointer; one program covers every request.
        assert ctx_len.dtype == torch.int64, "ctx_len must be int64"
        assert slots.shape[0] >= num_gens, "one slot per generation request"
        assert 0 <= cp_rank < cp_size, "cp_rank out of range"
        live = slots[:num_gens][slots[:num_gens] != dummy_slot]
        # Two live requests on one slot race on the advanced length.
        assert live.unique().numel() == live.numel(), "generation requests share a slot"


k3_dflash_gen_prepare = K3DflashGenPrepare()
