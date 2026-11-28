# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 worker's per-step drafter input bookkeeping for the generation requests, in one launch."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_gen_prepare import VEC, gen_prepare


class GenPrepared(NamedTuple):
    """The outputs of ``k31_dflash2_gen_prepare`` (see the op)."""

    noise: torch.Tensor
    bonus: torch.Tensor
    query_positions: torch.Tensor
    ctx_positions: torch.Tensor
    store_rows: torch.Tensor
    store_mask: torch.Tensor
    num_ctx: torch.Tensor
    page_rows: torch.Tensor
    block_pages: torch.Tensor
    block_offsets: torch.Tensor


def _spec(n, width, block, *, ring=True, slots=33, pages=33, special="none") -> dict:
    return dict(n=n, width=width, block=block, ring=ring, slots=slots, pages=pages, special=special)


class K31DFlash2GenPrepare(OpWrapper):
    """The generation requests' drafter inputs of ``K31DFlashWorker.prepare_1st_drafter_inputs`` over a
    private paged context arena (no manager block tables, no dense buffer, no helix) and the head of the
    MLA drafter's ``dflash_forward``, with their side effects.

    For request ``r`` (slot ``s = slots[r]``, ``na = num_accepted[r]``, ``c = ctx_len[s]`` before the call):

    * ``bonus[r] = accepted_tokens[r, max(na - 1, 0)]``;
    * ``query_positions[r, j] = min(c + na, max_ctx) + j`` (``j < block``);
    * ``ctx_positions[r * width + j] = c + j`` and the accepted rows' pool row (``page * page_size +
      offset``) of position ``min(c + j, max_ctx - 1)`` (modulo ``ring_rows`` with a ring) in
      ``page_table[s]``, kept where ``j < na`` (``store_mask``);
    * ``ctx_len[s] += na``, then every slot clamped to ``max_ctx`` and the dummy slot emptied;
      ``num_ctx[r] = ctx_len[s]`` after; ``page_rows[r] = page_table[s]`` (int64);
    * the block's store address of ``num_ctx[r] + j`` (``j < block``, modulo ``ring_rows``);
    * ``seq_lens[r] = width``; ``noise[r] = [embed(bonus[r]) | mask_row x (block - 1)]``, an id outside
      the embedding's vocabulary reading a zero row.

    Args:
        num_accepted: ``[n]`` int32. accepted_tokens: ``[n, width]`` int32. slots: ``[n]`` int64.
        ctx_len: ``[slots]`` int64 (in place). page_table: ``[slots, pages]`` int32.
        embed_weight: ``[vocab, hidden]`` BF16. mask_row: ``[hidden]`` BF16.
        seq_lens: ``[n]`` int32 (in place). block, max_ctx, dummy_slot, ring_rows (0: none), page_size.
    Returns:
        :class:`GenPrepared`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, verify 8, block 8, ring: the ll step", spec=_spec(1, 8, 8)),
        Cell(why="2 requests, verify 5, block 5, ring: the mid step", spec=_spec(2, 5, 5)),
        Cell(why="12 requests, verify 5, block 5, ring: the M2 step", spec=_spec(12, 5, 5)),
        Cell(why="32 requests, verify 8, block 8, ring", spec=_spec(32, 8, 8)),
        Cell(
            why="8 requests, no ring (dense positions)", spec=_spec(8, 5, 5, ring=False, pages=40)
        ),
        Cell(
            why="padding rows on the dummy slot, a context at max_ctx, a -1 anchor",
            spec=_spec(6, 8, 8, special="edges"),
        ),
    )

    note = """
    Kernel: ``kimi_k31_pro_mxfp4__sm_103__tep/kernels/spec/dflash2_gen_prepare.py``: one launch (a bookkeeping CTA and a
    CTA per request for the noise block) where the torch path takes about forty. Bitwise: integer arithmetic and copies.
    The dummy slot may repeat (padding rows); its advance is emptied, as in the torch path.
    """

    def __call__(
        self,
        num_accepted: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        page_table: torch.Tensor,
        embed_weight: torch.Tensor,
        mask_row: torch.Tensor,
        seq_lens: torch.Tensor,
        block: int,
        max_ctx: int,
        dummy_slot: int,
        ring_rows: int,
        page_size: int,
    ) -> GenPrepared:
        out = gen_prepare(
            num_accepted,
            accepted_tokens,
            slots,
            ctx_len,
            page_table,
            embed_weight,
            mask_row,
            seq_lens,
            block=block,
            max_ctx=max_ctx,
            dummy_slot=dummy_slot,
            ring_rows=ring_rows,
            page_size=page_size,
        )
        return GenPrepared(**out)

    def reference(
        self,
        num_accepted: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        page_table: torch.Tensor,
        embed_weight: torch.Tensor,
        mask_row: torch.Tensor,
        seq_lens: torch.Tensor,
        block: int,
        max_ctx: int,
        dummy_slot: int,
        ring_rows: int,
        page_size: int,
    ) -> GenPrepared:
        """The torch path: ``prepare_1st_drafter_inputs``' generation branch, ``_store_context_kv_paged``'s
        addresses, ``_advance_ctx_len`` and the head of ``dflash_forward``, statement by statement."""
        n, width = accepted_tokens.shape
        device = accepted_tokens.device
        bonus_idx = (num_accepted - 1).clamp_min(0).long().unsqueeze(1)
        bonus = accepted_tokens.gather(1, bonus_idx).squeeze(1).long()
        ctx_len_gen = ctx_len[slots]
        j_block = torch.arange(block, dtype=torch.long, device=device)
        offsets = torch.arange(width, dtype=torch.long, device=device)
        query_position = (ctx_len_gen + num_accepted.long()).clamp(max=max_ctx)
        query_positions = query_position.unsqueeze(1) + j_block.unsqueeze(0)
        ctx_positions = (ctx_len_gen.unsqueeze(1) + offsets.unsqueeze(0)).long().reshape(-1)
        # dflash_noise_block_embedding over an unsharded embedding (ids outside it read zeros).
        ids = torch.cat([bonus, torch.zeros(1, dtype=torch.long, device=device)])
        valid = (ids >= 0) & (ids < embed_weight.shape[0])
        rows = F.embedding(torch.where(valid, ids, 0), embed_weight)
        rows.masked_fill_(~valid.unsqueeze(-1), 0)
        noise = mask_row.expand(n, block, -1).clone()
        noise[:, 0, :] = rows[:n]
        col_idx = ctx_len[slots].unsqueeze(1) + offsets.unsqueeze(0)
        store_mask = (offsets.unsqueeze(0) < num_accepted.long().unsqueeze(1)).reshape(-1)
        col_idx = col_idx.clamp(max=max_ctx - 1)
        positions = col_idx.reshape(-1).long()
        if ring_rows:
            positions = torch.remainder(positions, ring_rows)
        row_slots = slots.unsqueeze(1).expand(-1, width).reshape(-1).long()
        store_pages = page_table[row_slots, positions // page_size].long()
        store_offsets = positions % page_size
        store_rows = store_pages * page_size + store_offsets
        ctx_len[slots] += num_accepted.long()
        ctx_len.clamp_(max=max_ctx)
        ctx_len[dummy_slot].zero_()
        num_ctx = ctx_len[slots]
        seq_lens[:] = width
        page_rows = page_table.index_select(0, slots.long()).long()
        block_pos = num_ctx.long().view(-1, 1) + torch.arange(block, device=device)
        if ring_rows:
            block_pos = torch.remainder(block_pos, ring_rows)
        block_pages = page_rows.gather(1, block_pos // page_size).view(-1)
        block_offsets = (block_pos % page_size).view(-1)
        return GenPrepared(
            noise,
            bonus,
            query_positions,
            ctx_positions,
            store_rows,
            store_mask.to(torch.uint8),
            num_ctx,
            page_rows,
            block_pages,
            block_offsets,
        )

    def is_valid(
        self,
        num_accepted: torch.Tensor,
        accepted_tokens: torch.Tensor,
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        page_table: torch.Tensor,
        embed_weight: torch.Tensor,
        mask_row: torch.Tensor,
        seq_lens: torch.Tensor,
        block: int,
        max_ctx: int,
        dummy_slot: int,
        ring_rows: int,
        page_size: int,
    ) -> None:
        assert num_accepted.dtype == torch.int32 and accepted_tokens.dtype == torch.int32
        assert slots.dtype == torch.int64 and ctx_len.dtype == torch.int64
        assert page_table.dtype == torch.int32 and seq_lens.dtype == torch.int32
        assert embed_weight.dtype == mask_row.dtype == torch.bfloat16
        for t in (
            num_accepted,
            accepted_tokens,
            slots,
            ctx_len,
            page_table,
            embed_weight,
            mask_row,
            seq_lens,
        ):
            assert t.is_contiguous(), "contiguous operands"
        assert embed_weight.shape[1] % VEC == 0 and mask_row.shape == embed_weight.shape[1:]
        assert int(num_accepted.min()) >= 0 and int(num_accepted.max()) <= accepted_tokens.shape[1]
        assert int(slots.min()) >= 0 and int(slots.max()) < ctx_len.shape[0]
        assert 0 <= dummy_slot < ctx_len.shape[0]
        # Every position read lies inside the slot's pages.
        reach = (ring_rows or (max_ctx + max(block, accepted_tokens.shape[1]))) - 1
        assert reach // page_size < page_table.shape[1], "positions past the slot's pages"


k31_dflash2_gen_prepare = K31DFlash2GenPrepare()

__all__ = ["GenPrepared", "K31DFlash2GenPrepare", "k31_dflash2_gen_prepare"]
