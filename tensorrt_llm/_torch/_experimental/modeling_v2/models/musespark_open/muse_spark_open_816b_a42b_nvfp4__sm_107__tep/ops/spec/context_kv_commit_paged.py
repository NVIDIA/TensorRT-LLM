# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash commit over a paged drafter context: the context lengths and what the drafter reads, in one launch.

With the drafter's context K/V in pages (``TRTLLM`` drafter attention: the
target's KV cache manager, or the worker's private paged pool) the context-K/V
fork's commit (``context_kv_fork/fork.py``) runs the worker's ``_commit_context``
(``_torch/speculative/dflash.py``), about fifteen launches on a few integers:

* ``_advance_ctx_len``: ``ctx_len[slots] += num_accepted``,
  ``ctx_len.clamp_(max=max_ctx)``, ``ctx_len[dummy_slot] = 0`` (and ``ctx_base[dummy_slot] = 0``
  when the worker keeps a context base: ``TRTLLM_DFLASH_WINDOWED_CTX_ARENA``, which a paged
  context never compacts);
* ``_drafter_context_lengths``: ``ctx_len[slots]`` (less ``ctx_base[slots]``); with the manager's block
  tables, the allocation limit ``(block_counts * page_size - block_size).clamp(min=0)[rows]``,
  the unified cache's check that no length passes it (``_assert_async``) and
  ``minimum`` with it;

then copies the lengths into the commit's output.

This entry writes every element of ``ctx_len`` exactly once -- the dummy slot 0,
a batch slot ``min(old + n, max_ctx)``, any other slot ``min(old, max_ctx)`` -- and
each request's length from the value it computed for that request's slot, so
nothing is read after it is written. A length past its limit under the check
traps the program, which ends the CUDA context as the built-in's device assert
does. Integer arithmetic: bitwise the built-in.

``context_kv_commit_paged_preamble`` (``drafter_preamble_from_commit``) is the same launch
writing the drafter forward's integer preamble too: the int32 values ``windowed_block_decode``
derives from the lengths, the batch index and the query positions, and the prepare's
``_seq_lens_cuda`` fill.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Generation requests one call takes (the fork's ``MAX_REQUESTS``).
MAX_REQUESTS = 64
#: Elements of ``ctx_len`` per loop iteration.
SLOT_BLOCK = 128


@triton.jit
def _trap_where(bad):
    """Trap on every lane whose ``bad`` is nonzero; 0 elsewhere (fold the result into a live value)."""
    return tl.inline_asm_elementwise(
        asm="{ .reg .pred p; setp.ne.s32 p, $1, 0; @p trap; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[bad],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _context_kv_commit_paged_kernel(
    ctx_len_ptr,
    ctx_base_ptr,
    num_slots,
    slots_ptr,
    counts_ptr,
    rows_ptr,
    block_counts_ptr,
    out_ptr,
    num_gens,
    max_ctx,
    dummy_slot,
    page_size,
    block_size,
    num_ctx_i32_ptr,
    seq_after_i32_ptr,
    batch_idx_ptr,
    batch_idx_i32_ptr,
    qpos_ptr,
    qpos_i32_ptr,
    seq_lens_ptr,
    seq_lens_value,
    GENS: tl.constexpr,
    SLOT_BLOCK: tl.constexpr,
    LIMIT: tl.constexpr,
    CHECK: tl.constexpr,
    BASE: tl.constexpr,
    PREAMBLE: tl.constexpr,
    DRAFT_BLOCK: tl.constexpr,
    QBLOCK: tl.constexpr,
):
    g = tl.arange(0, GENS)
    live = g < num_gens
    slots = tl.load(slots_ptr + g, mask=live, other=-1)
    counts = tl.load(counts_ptr + g, mask=live, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slots, mask=live, other=0)
    base = tl.zeros([GENS], dtype=tl.int64)
    if BASE:
        base = tl.load(ctx_base_ptr + slots, mask=live, other=0)
    # Every read of ctx_len and ctx_base above precedes every write below.
    tl.debug_barrier()
    # _advance_ctx_len's result for each request's slot: the add, the clamp, the dummy's zero;
    # then the physical length the drafter reads (the dummy's base is zeroed too).
    dummy = slots == dummy_slot
    length = tl.where(dummy, 0, tl.minimum(old + counts, max_ctx)) - tl.where(dummy, 0, base)
    if BASE:
        tl.store(ctx_base_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))
    for start in range(0, num_slots, SLOT_BLOCK):
        idx = start + tl.arange(0, SLOT_BLOCK)
        keep = idx < num_slots
        cur = tl.load(ctx_len_ptr + idx, mask=keep, other=0)
        hit = (idx[:, None] == slots[None, :]) & live[None, :]
        in_batch = tl.max(hit.to(tl.int32), axis=1) > 0
        added = tl.sum(tl.where(hit, counts[None, :], 0), axis=1)
        value = tl.minimum(tl.where(in_batch, cur + added, cur), max_ctx)
        value = tl.where(idx == dummy_slot, 0, value)
        tl.store(ctx_len_ptr + idx, value, mask=keep)
    if LIMIT:
        rows = tl.load(rows_ptr + g, mask=live, other=0)
        held = tl.load(block_counts_ptr + rows, mask=live, other=0)
        allocated = tl.maximum(held * page_size - block_size, 0)
        if CHECK:
            bad = (live & (length > allocated)).to(tl.int32)
            length = length + _trap_where(bad).to(tl.int64)
        length = tl.minimum(length, allocated)
    tl.store(out_ptr + g, length, mask=live)
    if PREAMBLE:
        # The drafter forward's preamble (windowed_block_decode): num_ctx_per_req.to(int32),
        # that + block_size, ctx_cache_batch_idx.to(int32); and the prepare's
        # _seq_lens_cuda[0:gens] = K + 1.
        length32 = length.to(tl.int32)
        tl.store(num_ctx_i32_ptr + g, length32, mask=live)
        tl.store(seq_after_i32_ptr + g, length32 + DRAFT_BLOCK, mask=live)
        batch_idx = tl.load(batch_idx_ptr + g, mask=live, other=0)
        tl.store(batch_idx_i32_ptr + g, batch_idx.to(tl.int32), mask=live)
        tl.store(seq_lens_ptr + g, tl.zeros([GENS], dtype=tl.int64) + seq_lens_value, mask=live)
        # query_positions.reshape(-1).to(int32), written by the verify fold before this launch.
        q = tl.arange(0, QBLOCK)
        qm = q < num_gens * DRAFT_BLOCK
        tl.store(qpos_i32_ptr + q, tl.load(qpos_ptr + q, mask=qm).to(tl.int32), mask=qm)


def _launch(
    ctx_len,
    ctx_base,
    slots,
    num_accepted,
    rows,
    block_counts,
    page_size,
    block_size,
    max_ctx,
    dummy_slot,
    check,
    out,
    preamble=None,
) -> torch.Tensor:
    """The one launch of both entries; ``preamble`` is ``(batch_idx, query_positions,
    seq_lens_cuda, seq_lens_value, draft_block, i32s)`` or None."""
    num_gens = slots.shape[0]
    limit = block_counts is not None
    if preamble is None:
        # Unread placeholders for the preamble's operands.
        batch_idx, qpos, seq_lens, seq_lens_value, draft_block = slots, slots, out, 0, 1
        i32s = (out, out, out, out)
    else:
        batch_idx, qpos, seq_lens, seq_lens_value, draft_block, i32s = preamble
    num_ctx_i32, seq_after, batch_idx_i32, qpos_i32 = i32s
    _context_kv_commit_paged_kernel[(1,)](
        ctx_len,
        ctx_len if ctx_base is None else ctx_base,
        ctx_len.numel(),
        slots,
        num_accepted,
        rows if limit else slots,
        block_counts if limit else ctx_len,
        out,
        num_gens,
        max_ctx,
        dummy_slot,
        page_size,
        block_size,
        num_ctx_i32,
        seq_after,
        batch_idx,
        batch_idx_i32,
        qpos,
        qpos_i32,
        seq_lens,
        seq_lens_value,
        GENS=max(triton.next_power_of_2(num_gens), 2),
        SLOT_BLOCK=SLOT_BLOCK,
        LIMIT=limit,
        CHECK=bool(limit and check),
        BASE=ctx_base is not None,
        PREAMBLE=preamble is not None,
        DRAFT_BLOCK=draft_block,
        QBLOCK=1 if preamble is None else triton.next_power_of_2(max(1, num_gens * draft_block)),
        num_warps=4,
    )
    return out


class ContextKvCommitPaged(OpWrapper):
    """``DFlashWorker._commit_context`` over a paged drafter context, into ``out``.

        ctx_len        [slots] int64, the worker's ``_ctx_len`` (updated in place)
        ctx_base       the worker's ``_ctx_base`` ([slots] int64; its dummy slot zeroed in
                       place), or None
        slots          [g] int64, the requests' slots
        num_accepted   [g] int32 or int64
        rows           [g] int64 batch rows (read only with ``block_counts``)
        block_counts   the worker's ``_ctx_block_counts`` (int64), or None without the
                       manager's block tables (no allocation limit)
        page_size, block_size, max_ctx, dummy_slot
                       the worker's ``_ctx_page_size``, ``_compute_block_size``,
                       ``_max_ctx``, ``_dummy_slot``
        check          the unified cache's allocation check (``_ctx_unified_cache``)
        out            [g] int64, the lengths the drafter reads

    The ``_advance_ctx_len`` part needs a dummy slot no request owns: two
    requests never share a slot other than it.
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: the worker's ``_ctx_len`` (read-modify-write) and
    ``_ctx_block_counts`` (read). One Triton program (grid 1, 4 warps), JIT-compiled
    per (request block, limit, check); no programmatic-dependent-launch attributes.

    Call site: ``ContextKvFork.commit`` over a paged drafter context, under the
    ``paged_context_commit`` fast path, in place of ``_commit_context``.

    Numerics: integer results, bitwise the built-in method (the reference).

    Hard limits
      g <= MAX_REQUESTS.
    """

    def __call__(
        self,
        ctx_len: torch.Tensor,
        ctx_base: Optional[torch.Tensor],
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        rows: torch.Tensor,
        block_counts: Optional[torch.Tensor],
        page_size: int,
        block_size: int,
        max_ctx: int,
        dummy_slot: int,
        check: bool,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return _launch(
            ctx_len,
            ctx_base,
            slots,
            num_accepted,
            rows,
            block_counts,
            page_size,
            block_size,
            max_ctx,
            dummy_slot,
            check,
            out,
        )

    def reference(
        self,
        ctx_len: torch.Tensor,
        ctx_base: Optional[torch.Tensor],
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        rows: torch.Tensor,
        block_counts: Optional[torch.Tensor],
        page_size: int,
        block_size: int,
        max_ctx: int,
        dummy_slot: int,
        check: bool,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """``_commit_context``'s statements over a paged context (which never compacts), into ``out``."""
        from tensorrt_llm._torch.speculative.dflash import dflash_allocated_ctx_limit

        ctx_len[slots] += num_accepted.long()
        ctx_len.clamp_(max=max_ctx)
        ctx_len[dummy_slot].zero_()
        if ctx_base is not None:
            ctx_base[dummy_slot].zero_()
            num_ctx = ctx_len[slots] - ctx_base[slots]
        else:
            num_ctx = ctx_len[slots]
        if block_counts is not None:
            allocated = dflash_allocated_ctx_limit(block_counts, page_size, block_size)[rows]
            if check:
                torch._assert_async((num_ctx <= allocated).all(), "allocation check")
            num_ctx = torch.minimum(num_ctx, allocated)
        out.copy_(num_ctx)
        return out

    def is_valid(
        self,
        ctx_len: torch.Tensor,
        ctx_base: Optional[torch.Tensor],
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        rows: torch.Tensor,
        block_counts: Optional[torch.Tensor],
        page_size: int,
        block_size: int,
        max_ctx: int,
        dummy_slot: int,
        check: bool,
        out: torch.Tensor,
    ) -> None:
        num_gens = slots.shape[0]
        if not 1 <= num_gens <= MAX_REQUESTS:
            raise ValueError(f"1..{MAX_REQUESTS} requests, got {num_gens}")
        if ctx_len.dtype != torch.int64 or ctx_len.dim() != 1 or not ctx_len.is_contiguous():
            raise ValueError("ctx_len must be contiguous int64 [slots]")
        if slots.dtype != torch.int64 or not slots.is_contiguous():
            raise ValueError("slots must be contiguous int64")
        if num_accepted.dtype not in (torch.int32, torch.int64) or num_accepted.numel() < num_gens:
            raise ValueError("num_accepted must be int32 or int64 [g]")
        if not num_accepted.is_contiguous():
            raise ValueError("num_accepted must be contiguous")
        if out.dtype != torch.int64 or out.numel() < num_gens or not out.is_contiguous():
            raise ValueError("out must be contiguous int64 [g]")
        if ctx_base is not None and (
            ctx_base.dtype != torch.int64
            or ctx_base.shape != ctx_len.shape
            or not ctx_base.is_contiguous()
        ):
            raise ValueError("ctx_base must be contiguous int64 shaped as ctx_len")
        if not 0 <= dummy_slot < ctx_len.numel():
            raise ValueError("the dummy slot must be a slot of ctx_len")
        if block_counts is not None:
            if block_counts.dtype != torch.int64 or not block_counts.is_contiguous():
                raise ValueError("block_counts must be contiguous int64")
            if rows.dtype != torch.int64 or rows.numel() < num_gens or not rows.is_contiguous():
                raise ValueError("rows must be contiguous int64 [g]")


context_kv_commit_paged = ContextKvCommitPaged()


def context_kv_commit_paged_preamble(
    ctx_len: torch.Tensor,
    ctx_base: Optional[torch.Tensor],
    slots: torch.Tensor,
    num_accepted: torch.Tensor,
    rows: torch.Tensor,
    block_counts: Optional[torch.Tensor],
    page_size: int,
    block_size: int,
    max_ctx: int,
    dummy_slot: int,
    check: bool,
    out: torch.Tensor,
    batch_idx: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    seq_lens_value: int,
    draft_block: int,
    i32s: tuple,
) -> torch.Tensor:
    """``context_kv_commit_paged``, writing the drafter preamble too (``drafter_preamble_from_commit``).

    ``i32s = (num_ctx_i32 [g], seq_lens_after_i32 [g], batch_idx_i32 [g],
    query_positions_i32 [g * draft_block])`` (int32, allocated by the caller) get the values
    ``windowed_block_decode`` derives from ``out``, ``batch_idx`` (the forward's
    ``ctx_cache_batch_idx``: the batch rows over the manager's block tables, the slots over the
    worker's own pages; integer, contiguous) and ``query_positions`` (int64
    ``[g, draft_block]`` contiguous, written by the verify fold before this launch); also sets
    ``seq_lens_cuda[:g] = seq_lens_value``. Integer copies and one add: the same values.
    """
    return _launch(
        ctx_len,
        ctx_base,
        slots,
        num_accepted,
        rows,
        block_counts,
        page_size,
        block_size,
        max_ctx,
        dummy_slot,
        check,
        out,
        (batch_idx, query_positions, seq_lens_cuda, seq_lens_value, draft_block, i32s),
    )
