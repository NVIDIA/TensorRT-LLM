# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash verify: acceptance, the forced override and the step's bookkeeping in one launch.

``dflash_accept`` (fast path ``fused_accept``) already turns the sampled target tokens
into the step's accepted tokens and counts in one launch. Behind it the
built-in DFlash worker spends 18 more launches on integers derived from those
counts and from its own buffers (``speculative/dflash.py``):

* ``_prepare_attn_metadata_for_dflash``: the saved copy of the attention
  metadata's ``_seq_lens_cuda`` (``prepare_for_spec_dec``'s ``clone``);
* ``_prepare_kv_for_draft_forward``: ``1 - num_accepted`` (the KV-length
  rewind applied after the draft) and ``kv_lens_cuda += 1`` in place;
* ``prepare_1st_drafter_inputs``: the generation rows' ``arange``, the bonus
  token (the last accepted token), each request's context length
  ``_ctx_len[slot]``, the drafter's query positions and the context positions
  of this step's accepted rows, the ``[K + 1]`` offsets, the ``int64`` counts,
  and the ids ``[bonus..., mask]`` the drafter embeds.

This entry computes the counts exactly as ``dflash_accept`` does and then, in
the same single program and after each count exists, writes all of the above.
Nothing reads a count before it is computed; the work only leaves the launch
count. Integer arithmetic plus ``dflash_accept``'s one fp32 comparison, so
every output is bitwise the built-in one.

Generation-only steps (no context rows), which is every decode step the target
captures; a step with context rows takes the built-in path.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .dflash_accept import (
    COUNTER_STRIDE,
    FORCE_FRACTIONAL,
    LOOP_REQUESTS,
    SLOT_STRIDE,
    dflash_accept,
)

#: The requests one launch takes: its single program loops over them, as
#: ``dflash_accept``'s per-request loop does up to ``LOOP_REQUESTS``.
MAX_REQUESTS = LOOP_REQUESTS
#: Elements of ``_seq_lens_cuda`` copied per loop iteration.
SEQ_BLOCK = 128


class VerifyFold(NamedTuple):
    """What one fold launch writes (shapes for ``g`` generation requests, ``K`` drafts, block ``B``)."""

    #: ``[g, K + 1]`` int32, the built-in ``accepted_tokens``.
    accepted: torch.Tensor
    #: ``[g]`` int32, the built-in ``num_accepted_tokens``.
    num_accepted: torch.Tensor
    #: ``[g]`` int32, ``1 - num_accepted`` (``_kv_rewind_amount``).
    rewind: torch.Tensor
    #: ``[g]`` int64, ``arange(g)`` (``gen_rows_out``).
    gen_rows: torch.Tensor
    #: ``[g]`` int64, the last accepted token (``bonus``, ``first_prev_tokens``).
    bonus: torch.Tensor
    #: ``[g]`` int64, ``num_accepted.long()``.
    num_accepted_long: torch.Tensor
    #: ``[g, B]`` int64, ``min(_ctx_len[slot] + num_accepted, max_ctx) + j`` (``query_positions``).
    query_positions: torch.Tensor
    #: ``[g, K + 1]`` int64, ``_ctx_len[slot] + c`` (``ctx_position_ids``).
    ctx_positions: torch.Tensor
    #: ``[K + 1]`` int64, ``arange(K + 1)`` (``offsets_kp1``).
    offsets: torch.Tensor
    #: ``[g + 1]`` int64, ``cat([bonus, mask_token_id])``, the drafter's embedding input.
    embed_ids: torch.Tensor
    #: ``[g, B]`` int64, the noise block's ids: the bonus token, then ``B - 1`` mask ids.
    noise_ids: torch.Tensor
    #: ``_seq_lens_cuda``'s saved copy (``prepare_for_spec_dec``), or None if not asked for.
    seq_lens_copy: Optional[torch.Tensor]


@triton.jit
def _dflash_verify_fold_kernel(
    target_ptr,
    draft_ptr,
    draft_row_stride,
    accepted_ptr,
    num_accepted_ptr,
    pool_ptr,
    counter_ptr,
    num_gens,
    base_total,
    max_total,
    frac,
    slots_ptr,
    ctx_len_ptr,
    max_ctx,
    kv_lens_ptr,
    seq_lens_ptr,
    seq_lens_copy_ptr,
    seq_lens_numel,
    rewind_ptr,
    gen_rows_ptr,
    bonus_ptr,
    num_accepted_long_ptr,
    query_positions_ptr,
    ctx_positions_ptr,
    offsets_ptr,
    embed_ids_ptr,
    noise_ids_ptr,
    mask_token_id,
    block_size,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    QBLOCK: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    BUMP_KV_LENS: tl.constexpr,
    COPY_SEQ_LENS: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    qcols = tl.arange(0, QBLOCK)
    row_width = K + 1
    in_row = cols < row_width
    in_block = qcols < block_size
    mask_id = tl.zeros([], dtype=tl.int64) + mask_token_id
    # prepare_for_spec_dec's clone of _seq_lens_cuda: reads a graph input only.
    if COPY_SEQ_LENS:
        for start in range(0, seq_lens_numel, SEQ_BLOCK):
            idx = start + tl.arange(0, SEQ_BLOCK)
            keep = idx < seq_lens_numel
            tl.store(seq_lens_copy_ptr + idx, tl.load(seq_lens_ptr + idx, mask=keep), mask=keep)
    tl.store(offsets_ptr + cols, cols.to(tl.int64), mask=in_row)
    counter = tl.zeros([], dtype=tl.int64)
    if FORCE == 2:
        # One bump per call, before any request reads it (as dflash_accept).
        counter = tl.load(counter_ptr).to(tl.int64) + 1
        tl.store(counter_ptr, counter)
    for g in range(num_gens):
        # --- acceptance: dflash_accept's statements for a generation row ---
        tokens = tl.load(target_ptr + g * row_width + cols, mask=in_row, other=0)
        tokens32 = tokens.to(tl.int32)
        tl.store(accepted_ptr + g * row_width + cols, tokens32, mask=in_row)
        if FORCE == 0:
            drafts = tl.load(draft_ptr + g * draft_row_stride + cols, mask=cols < K, other=0)
            mismatch = (cols < K) & (drafts.to(tl.int64) != tokens.to(tl.int64))
            accepted = tl.min(tl.where(mismatch, cols, K), axis=0) + 1
        elif FORCE == 1:
            accepted = tl.zeros([], dtype=tl.int32) + base_total
        else:
            index = (counter * COUNTER_STRIDE + g * SLOT_STRIDE) & POOL_MASK
            draw = tl.load(pool_ptr + index)
            accepted = tl.minimum(base_total + (draw < frac).to(tl.int32), max_total)
        accepted = accepted.to(tl.int32)
        tl.store(num_accepted_ptr + g, accepted)
        # --- the bookkeeping, each value written after the count it uses ---
        tl.store(rewind_ptr + g, 1 - accepted)
        if BUMP_KV_LENS:
            tl.store(kv_lens_ptr + g, tl.load(kv_lens_ptr + g) + 1)
        tl.store(gen_rows_ptr + g, g.to(tl.int64))
        bonus_index = tl.maximum(accepted - 1, 0)
        bonus = tl.sum(tl.where(in_row & (cols == bonus_index), tokens32, 0), axis=0).to(tl.int64)
        tl.store(bonus_ptr + g, bonus)
        accepted64 = accepted.to(tl.int64)
        tl.store(num_accepted_long_ptr + g, accepted64)
        slot = tl.load(slots_ptr + g)
        ctx_len = tl.load(ctx_len_ptr + slot).to(tl.int64)
        ctx_now = tl.minimum(ctx_len + accepted64, max_ctx)
        tl.store(
            query_positions_ptr + g * block_size + qcols,
            ctx_now + qcols.to(tl.int64),
            mask=in_block,
        )
        tl.store(ctx_positions_ptr + g * row_width + cols, ctx_len + cols.to(tl.int64), mask=in_row)
        tl.store(embed_ids_ptr + g, bonus)
        tl.store(
            noise_ids_ptr + g * block_size + qcols,
            tl.where(qcols == 0, bonus, mask_id),
            mask=in_block,
        )
    tl.store(embed_ids_ptr + num_gens, mask_id)


class DflashVerifyFold(OpWrapper):
    """``dflash_accept`` for a generation-only verify block, plus the DFlash worker's step bookkeeping.

    Inputs: the ``g * (K + 1)`` sampled target tokens, the ``[g, K]`` drafts,
    the forced-acceptance plan (as ``dflash_accept``), the worker's
    ``_batch_to_slot[:g]`` (``slots``), ``_ctx_len`` and ``_max_ctx``, the
    drafter block width ``block_size`` and ``mask_token_id``; optionally the
    attention metadata's ``kv_lens_cuda`` (``+= 1`` on its first ``g``
    entries, in place) and ``_seq_lens_cuda`` (copied). Returns a
    ``VerifyFold``; bumps ``counter`` in place under ``FORCE_FRACTIONAL``.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: the worker's ``_batch_to_slot`` / ``_ctx_len`` (read),
    the attention metadata's ``kv_lens_cuda`` (read-modify-write) and
    ``_seq_lens_cuda`` (read). One Triton program (grid 1, 1 warp), JIT-compiled
    per (K, block, force mode, buffer flags); loops over the requests
    (``MAX_REQUESTS``). No programmatic-dependent-launch attributes.

    Call site: the DFlash worker's ``_sample_and_accept_draft_tokens_base``
    under the ``verify_fold`` fast path (``verify_fold/fold.py``), on generation-only uniform
    verify blocks outside the eager CUDA-graph warmup, in place of
    ``dflash_accept``. The later worker methods take its outputs instead of
    recomputing them.

    Numerics: integer results, bitwise the built-in chain (the reference),
    including the fractional override's fp32 ``pool[i] < frac``.

    Hard limits
      K <= 63; block_size <= 64; the pool length is a power of two.

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        force_mode: int,
        base_total: int,
        max_total: int,
        frac: float,
        pool: Optional[torch.Tensor],
        counter: Optional[torch.Tensor],
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        max_ctx: int,
        block_size: int,
        mask_token_id: int,
        kv_lens: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
    ) -> VerifyFold:
        num_gens, k = draft_tokens.shape
        device = target_tokens.device
        i32, i64 = torch.int32, torch.int64
        out = VerifyFold(
            accepted=torch.empty((num_gens, k + 1), dtype=i32, device=device),
            num_accepted=torch.empty((num_gens,), dtype=i32, device=device),
            rewind=torch.empty((num_gens,), dtype=i32, device=device),
            gen_rows=torch.empty((num_gens,), dtype=i64, device=device),
            bonus=torch.empty((num_gens,), dtype=i64, device=device),
            num_accepted_long=torch.empty((num_gens,), dtype=i64, device=device),
            query_positions=torch.empty((num_gens, block_size), dtype=i64, device=device),
            ctx_positions=torch.empty((num_gens, k + 1), dtype=i64, device=device),
            offsets=torch.empty((k + 1,), dtype=i64, device=device),
            embed_ids=torch.empty((num_gens + 1,), dtype=i64, device=device),
            noise_ids=torch.empty((num_gens, block_size), dtype=i64, device=device),
            seq_lens_copy=None if seq_lens is None else torch.empty_like(seq_lens),
        )
        fractional = force_mode == FORCE_FRACTIONAL
        dummy = out.num_accepted
        _dflash_verify_fold_kernel[(1,)](
            target_tokens,
            draft_tokens,
            draft_tokens.stride(0),
            out.accepted,
            out.num_accepted,
            pool if fractional else dummy,
            counter if fractional else dummy,
            num_gens,
            base_total,
            max_total,
            frac,
            slots,
            ctx_len,
            max_ctx,
            dummy if kv_lens is None else kv_lens,
            dummy if seq_lens is None else seq_lens,
            dummy if seq_lens is None else out.seq_lens_copy,
            0 if seq_lens is None else seq_lens.numel(),
            out.rewind,
            out.gen_rows,
            out.bonus,
            out.num_accepted_long,
            out.query_positions,
            out.ctx_positions,
            out.offsets,
            out.embed_ids,
            out.noise_ids,
            mask_token_id,
            block_size,
            K=k,
            BLOCK=triton.next_power_of_2(k + 1),
            QBLOCK=triton.next_power_of_2(block_size),
            FORCE=force_mode,
            COUNTER_STRIDE=COUNTER_STRIDE,
            SLOT_STRIDE=SLOT_STRIDE,
            POOL_MASK=(pool.numel() - 1) if fractional else 0,
            BUMP_KV_LENS=kv_lens is not None,
            COPY_SEQ_LENS=seq_lens is not None,
            SEQ_BLOCK=SEQ_BLOCK,
            num_warps=1,
        )
        return out

    def reference(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        force_mode: int,
        base_total: int,
        max_total: int,
        frac: float,
        pool: Optional[torch.Tensor],
        counter: Optional[torch.Tensor],
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        max_ctx: int,
        block_size: int,
        mask_token_id: int,
        kv_lens: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
    ) -> VerifyFold:
        """The built-in chain, statement for statement, on a batch with no context rows.

        ``dflash_accept``'s reference, then the lines of
        ``_prepare_attn_metadata_for_dflash`` (the ``_seq_lens_cuda`` clone),
        ``_prepare_kv_for_draft_forward`` and the head of
        ``prepare_1st_drafter_inputs``; bumps ``counter`` and ``kv_lens`` in
        place like the op.
        """
        num_contexts, batch_size = 0, draft_tokens.shape[0]
        num_gens, k = draft_tokens.shape
        accepted_tokens, num_accepted_tokens = dflash_accept.reference(
            target_tokens, draft_tokens, 0, force_mode, base_total, max_total, frac, pool, counter
        )
        seq_lens_copy = None if seq_lens is None else seq_lens.clone()
        rewind = 1 - num_accepted_tokens[num_contexts:batch_size]
        if kv_lens is not None:
            kv_lens[num_contexts:batch_size] += 1
        gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
        gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]
        gen_rows_out = torch.arange(
            num_contexts, num_contexts + num_gens, dtype=torch.long, device=target_tokens.device
        )
        bonus_idx = (gen_num_accepted - 1).clamp_min(0).long().unsqueeze(1)
        bonus = gen_accepted_tokens.gather(1, bonus_idx).squeeze(1).long()
        ctx_len_gen = ctx_len[slots]
        j_block = torch.arange(block_size, dtype=torch.long, device=target_tokens.device)
        offsets_kp1 = torch.arange(k + 1, dtype=torch.long, device=target_tokens.device)
        ctx_len_now = (ctx_len_gen + gen_num_accepted.long()).clamp_(max=max_ctx)
        query_position_ids = ctx_len_now.unsqueeze(1) + j_block.unsqueeze(0)
        ctx_position_ids = ctx_len_gen.unsqueeze(1) + offsets_kp1.unsqueeze(0)
        mask_tok = torch.full(
            (1,), int(mask_token_id), dtype=torch.long, device=target_tokens.device
        )
        embed_ids = torch.cat([bonus, mask_tok], dim=0)
        noise_ids = mask_tok.expand(num_gens, block_size).clone()
        noise_ids[:, 0] = bonus
        return VerifyFold(
            accepted=accepted_tokens,
            num_accepted=num_accepted_tokens,
            rewind=rewind,
            gen_rows=gen_rows_out,
            bonus=bonus,
            num_accepted_long=gen_num_accepted.long(),
            query_positions=query_position_ids.long(),
            ctx_positions=ctx_position_ids,
            offsets=offsets_kp1,
            embed_ids=embed_ids,
            noise_ids=noise_ids,
            seq_lens_copy=seq_lens_copy,
        )

    def is_valid(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        force_mode: int,
        base_total: int,
        max_total: int,
        frac: float,
        pool: Optional[torch.Tensor],
        counter: Optional[torch.Tensor],
        slots: torch.Tensor,
        ctx_len: torch.Tensor,
        max_ctx: int,
        block_size: int,
        mask_token_id: int,
        kv_lens: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
    ) -> None:
        assert draft_tokens.dim() == 2 and draft_tokens.stride(-1) == 1, (
            "draft_tokens must be [gens, K] with unit column stride"
        )
        num_gens, k = draft_tokens.shape
        assert 1 <= num_gens <= MAX_REQUESTS, f"1..{MAX_REQUESTS} requests, got {num_gens}"
        assert 1 <= k <= 63, f"K must be in [1, 63], got {k}"
        assert 1 <= block_size <= 64, f"block_size must be in [1, 64], got {block_size}"
        assert target_tokens.dim() == 1 and target_tokens.is_contiguous()
        assert target_tokens.numel() == num_gens * (k + 1), (
            "a generation-only block: g * (K + 1) tokens"
        )
        assert (
            not target_tokens.dtype.is_floating_point and not draft_tokens.dtype.is_floating_point
        )
        assert slots.dtype == torch.int64 and slots.numel() >= num_gens and slots.is_contiguous()
        assert ctx_len.dtype == torch.int64 and ctx_len.is_contiguous()
        if kv_lens is not None:
            assert (
                kv_lens.dtype == torch.int32
                and kv_lens.numel() >= num_gens
                and kv_lens.is_contiguous()
            )
        if seq_lens is not None:
            assert seq_lens.is_contiguous(), "seq_lens is copied as a flat buffer"
        if force_mode == FORCE_FRACTIONAL:
            assert pool is not None and pool.dtype == torch.float32 and pool.dim() == 1
            assert pool.numel() & (pool.numel() - 1) == 0, "the pool length must be a power of two"
            assert counter is not None and counter.dtype == torch.int64 and counter.numel() == 1


dflash_verify_fold = DflashVerifyFold()
