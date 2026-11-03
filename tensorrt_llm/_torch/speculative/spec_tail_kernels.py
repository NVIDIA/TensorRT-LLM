# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton kernels for the bookkeeping of a one-model speculative-decoding step.

Each kernel replaces a chain of small eager PyTorch ops with one launch and
produces the same values:

* :func:`strict_accept` -- strict (token-equality) acceptance of the previous
  draft against the sampled target tokens.
* :func:`decode_draft_block_table` -- the DFlash draft-KV block-table decode
  (blocks per request, clamped and divided block offsets).
* :func:`dflash_paged_latent_prepare` -- the per-request bookkeeping of a
  DFlash generation step whose drafter context is a paged single-latent cache.
* :func:`store_context_latents` -- the masked store of the context latents of
  every drafter layer into its page pool.

Launch shapes depend only on host batch sizes, so all of them are CUDA-graph
capturable.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from ..pyexecutor.input_commit import k31_draft_tp_engine, k31_tp_generation_server
from .force_replay import ForceReplay, replay_lookup
from .vocab_parallel import _INT32_MAX as _VP_INT32_MAX

# Largest number of generation requests the one-program fused tail kernels
# (selection + acceptance, the Markov steps) take; larger batches keep the
# unfused, row-parallel kernels. Measured in the decode graph at 1-4 requests.
TAIL_V2_MAX_GENS = 16

_BLOCK_TABLE_BLOCK = 2048
# Columns per program of the parallel draft block-table decode.
BLOCK_TABLE_CHUNK = _BLOCK_TABLE_BLOCK


def spec_tail_fusion_enabled() -> bool:
    """Whether the MLA drafter's V un-absorb is written by ``bmm_out`` and its block
    latents stored by one fused launch (Kimi K3.1 generation servers without attention
    DP; drafter rounding may differ from the ``einsum`` path)."""
    return k31_tp_generation_server()


def tail_v2_enabled() -> bool:
    """Whether the second set of fused tail kernels runs (drafting Kimi K3.1 engines
    without attention DP). They write bitwise the values of the paths they replace: the
    MLA drafter's context and block RoPE kernels with in-kernel table lookups and the
    block latents stored straight into the paged pool, the column-parallel draft
    block-table decode, the fused target selection + acceptance, and shape-only operands
    built once instead of every step."""
    return k31_draft_tp_engine()


def ctx_side_stream_enabled() -> bool:
    """Whether the drafter's context latents are computed on a side stream (with its own
    MNNVL all-reduce workspace), concurrently with the target-token selection and
    acceptance (Kimi K3.1 generation servers without attention DP)."""
    return k31_tp_generation_server()


#: Multiplier of the tie count in the state-keyed draw index (see :class:`ForceAcceptKey`).
FORCE_ACCEPT_TIE_STRIDE = 4093
_TIE_STRIDE = tl.constexpr(FORCE_ACCEPT_TIE_STRIDE)
#: Columns per pass of the tie count (rows x this many earlier rows are compared at once).
FORCE_ACCEPT_TIE_CHUNK = 16


class ForceAcceptKey(NamedTuple):
    """Per-request key of the state-keyed synthetic acceptance draw.

    With ``TLLM_SPEC_DECODE_FORCE_ACCEPT_KEY=state`` the fractional draw of
    generation request ``g`` reads the fixed uniform pool at
    ``(p * a + t * b + k * FORCE_ACCEPT_TIE_STRIDE) & (pool size - 1)``, with
    ``p = positions[g * stride]`` and ``t = tokens[g * stride]`` the global
    position and the id of the request's newest committed token (the first
    token of its verify group in the target step's inputs) and ``k`` the number
    of earlier generation rows with the same ``(p, t)``. The draw then depends
    only on the request's own state, so the same request replays the same
    acceptance whatever its batch row, its admission step or the number of
    earlier replays, and every rank (all see the same input ids and global
    positions) draws the same value. Requests in the same state (the same
    position and newest token, e.g. two copies of a prompt) are told apart by
    ``k``; which of them gets which draw depends on their rows, but swapping
    two requests in the same state leaves the batch's content unchanged.
    """

    positions: torch.Tensor
    tokens: torch.Tensor
    stride: int


def force_launch_args(force: Optional[tuple], dummy: torch.Tensor) -> tuple:
    """``(FORCE, base_total, frac, max_total, pool, counter, key positions, key tokens, key
    stride, stride a, stride b, pool mask)`` of a ``force`` override: None (FORCE 0),
    ``(base_total,)`` (1), or ``(base_total, frac, max_total, pool, source, stride a, stride
    b)`` whose draw is keyed by the device counter ``source`` (bumped in place, 2) or by a
    :class:`ForceAcceptKey` ``source`` (3). ``dummy`` fills the unused pointers."""
    if force is None:
        return 0, 0, 0.0, 0, dummy, dummy, dummy, dummy, 0, 1, 1, 0
    if len(force) == 1:
        return 1, int(force[0]), 0.0, 0, dummy, dummy, dummy, dummy, 0, 1, 1, 0
    base_total, frac, max_total, pool, source, stride_a, stride_b = force
    pool_mask = pool.numel() - 1
    assert pool.numel() & pool_mask == 0 and pool.dtype == torch.float32
    if isinstance(source, ForceAcceptKey):
        assert source.positions.stride(-1) == 1 and source.tokens.stride(-1) == 1
        return (
            3,
            int(base_total),
            float(frac),
            int(max_total),
            pool,
            dummy,
            source.positions,
            source.tokens,
            int(source.stride),
            int(stride_a),
            int(stride_b),
            pool_mask,
        )
    assert source.dtype == torch.int64 and source.numel() == 1
    return (
        2,
        int(base_total),
        float(frac),
        int(max_total),
        pool,
        source,
        dummy,
        dummy,
        0,
        int(stride_a),
        int(stride_b),
        pool_mask,
    )


@triton.jit
def state_key(key_pos_ptr, key_tok_ptr, key_stride, num_rows, BLOCK: tl.constexpr):
    """ForceAcceptKey: each request's newest committed (global position, token)."""
    rows = tl.arange(0, BLOCK)
    mask = rows < num_rows
    pos = tl.load(key_pos_ptr + rows.to(tl.int64) * key_stride, mask=mask, other=0).to(tl.int64)
    tok = tl.load(key_tok_ptr + rows.to(tl.int64) * key_stride, mask=mask, other=0).to(tl.int64)
    return pos, tok


@triton.jit
def state_keyed_draw(
    pool_ptr,
    key_pos_ptr,
    key_tok_ptr,
    key_stride,
    num_rows,
    pos,
    tok,
    POS_STRIDE: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """The pool entry of each request's key (``pos``, ``tok`` from :func:`state_key`) and its
    tie count; rows at or past ``num_rows`` draw 1.0 (no extra token)."""
    rows = tl.arange(0, BLOCK)
    mask = rows < num_rows
    ties = tl.zeros([BLOCK], dtype=tl.int32)
    for c0 in tl.static_range(0, BLOCK, CHUNK):
        cols = c0 + tl.arange(0, CHUNK)
        cmask = cols < num_rows
        cpos = tl.load(key_pos_ptr + cols.to(tl.int64) * key_stride, mask=cmask, other=0)
        ctok = tl.load(key_tok_ptr + cols.to(tl.int64) * key_stride, mask=cmask, other=0)
        same = (pos[:, None] == cpos.to(tl.int64)[None, :]) & (
            tok[:, None] == ctok.to(tl.int64)[None, :]
        )
        earlier = cmask[None, :] & (cols[None, :] < rows[:, None])
        ties += tl.sum((same & earlier).to(tl.int32), axis=1)
    idx = (pos * POS_STRIDE + tok * TOK_STRIDE + ties.to(tl.int64) * _TIE_STRIDE) & POOL_MASK
    return tl.load(pool_ptr + idx, mask=mask, other=1.0)


@triton.jit
def _strict_accept_kernel(
    target_ptr,
    draft_ptr,
    stride_draft_row,
    stride_draft_col,
    accepted_ptr,
    num_accepted_ptr,
    num_ctx,
    num_gen,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    width = DRAFT_LEN + 1
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)

    # Context requests: accept the sampled token only.
    ctx_mask = rows < num_ctx
    t_ctx = tl.load(target_ptr + rows, mask=ctx_mask, other=0)
    tl.store(accepted_ptr + rows * width, t_ctx.to(tl.int32), mask=ctx_mask)
    tl.store(num_accepted_ptr + rows, tl.full([BLOCK_B], 1, tl.int32), mask=ctx_mask)

    # Generation requests: the accepted row is the K+1 target tokens, and the
    # count is 1 + the number of leading draft tokens equal to them.
    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    t_gen = tl.load(
        target_ptr + num_ctx + rows[:, None] * width + cols[None, :], mask=tile_mask, other=0
    ).to(tl.int32)
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    draft_mask = gen_mask[:, None] & (cols[None, :] < DRAFT_LEN)
    draft = tl.load(
        draft_ptr + rows[:, None] * stride_draft_row + cols[None, :] * stride_draft_col,
        mask=draft_mask,
        other=0,
    ).to(tl.int32)
    mismatch = draft_mask & (draft != t_gen)
    first_mismatch = tl.min(tl.where(mismatch, cols[None, :], DRAFT_LEN), axis=1)
    tl.store(num_accepted_ptr + num_ctx + rows, (1 + first_mismatch).to(tl.int32), mask=gen_mask)


def strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict acceptance of a uniform-width draft in one launch.

    Args:
        target_tokens: ``[num_contexts + num_gens * (K + 1)]`` sampled target
            tokens (any integer dtype).
        draft_tokens: ``[num_gens, K]`` previous draft tokens.
        num_contexts, batch_size: request counts.
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` and ``num_accepted_tokens
        [batch_size]``, both int32. Context rows define only column 0.
    """
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = target_tokens.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    if batch_size == 0:
        return accepted, num_accepted
    block_b = min(triton.next_power_of_2(max(num_contexts, num_gens, 1)), 128)
    grid = (triton.cdiv(max(num_contexts, num_gens, 1), block_b),)
    _strict_accept_kernel[grid](
        target_tokens,
        draft_tokens,
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        DRAFT_LEN=draft_len,
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        num_warps=4,
    )
    return accepted, num_accepted


@triton.jit
def _pair_argmax(vals, idx, axis: tl.constexpr):
    # The index the vocab-parallel _argmax_combine reduction selects, from
    # native reductions: NaN beats every number and the lowest index wins among
    # NaNs; otherwise the largest value wins and the lowest index among equal
    # values (+0.0 == -0.0). The selected index is unique, so it does not depend
    # on the reduction order.
    is_nan = vals != vals
    any_nan = tl.max(is_nan.to(tl.int32), axis=axis) > 0
    vmax = tl.max(tl.where(is_nan, float("-inf"), vals), axis=axis)
    cand = tl.where(
        tl.expand_dims(any_nan, axis), is_nan, (vals == tl.expand_dims(vmax, axis)) & ~is_nan
    )
    return tl.min(tl.where(cand, idx, _VP_INT32_MAX), axis=axis)


@triton.jit
def _select_accept_kernel(
    pairs_ptr,
    stride_row,
    n_slots,
    draft_ptr,
    stride_draft,
    accepted_ptr,
    num_accepted_ptr,
    num_ctx,
    num_gen,
    pool_ptr,
    counter_ptr,
    base_total,
    frac,
    max_total,
    key_pos_ptr,
    key_tok_ptr,
    key_stride,
    ref_base_ptr,
    ref_len_ptr,
    ref_tok_ptr,
    ref_stride,
    ref_start_ptr,
    num_refs,
    replay_drafts_ptr,
    replay_flags_ptr,
    replay_stride,
    prev_ptr,
    stride_prev,
    state_idx_ptr,
    stride_state_idx,
    dummy_ptr,
    stride_dummy,
    kv_lens_ptr,
    rewind_ptr,
    DRAFT_LEN: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    KDA_REPLAY: tl.constexpr,
    HAS_DUMMY: tl.constexpr,
    KV_REWIND: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    TIE_CHUNK: tl.constexpr,
    REPLAY: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # FORCE: 0 none, 1 constant, 2 counter-keyed draw (COUNTER_STRIDE x counter + SLOT_STRIDE x
    # row), 3 state-keyed draw (the same multipliers on the ForceAcceptKey position and token).
    # Pairs are (value, index) float32 couples, slot s of a row at columns
    # [2 s, 2 s + 1]; a row is loaded as one contiguous run and split.
    width = DRAFT_LEN + 1
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)
    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    draft_mask = gen_mask[:, None] & (cols[None, :] < DRAFT_LEN)

    # Everything that does not depend on the selection is read first.
    d = tl.load(
        draft_ptr + rows[:, None] * stride_draft + cols[None, :], mask=draft_mask, other=0
    ).to(tl.int32)
    if KDA_REPLAY:
        slot = tl.load(
            state_idx_ptr + (num_ctx + rows) * stride_state_idx, mask=gen_mask, other=0
        ).to(tl.int64)
        if HAS_DUMMY:
            dummy = (
                tl.load(dummy_ptr + (num_ctx + rows) * stride_dummy, mask=gen_mask, other=0) != 0
            )
        else:
            dummy = rows < 0
        current = tl.load(prev_ptr + slot * stride_prev, mask=gen_mask, other=0)
    if KV_REWIND:
        kv = tl.load(kv_lens_ptr + num_ctx + rows, mask=gen_mask)
    if FORCE == 2:
        counter = tl.load(counter_ptr) + 1
        idx = (counter * COUNTER_STRIDE + rows.to(tl.int64) * SLOT_STRIDE) & POOL_MASK
        rand = tl.load(pool_ptr + idx, mask=gen_mask, other=1.0)
        # Every warp reads the old value before any warp writes the new one.
        tl.debug_barrier()
        tl.store(counter_ptr, counter)
    elif FORCE == 3:
        kpos, ktok = state_key(key_pos_ptr, key_tok_ptr, key_stride, num_gen, BLOCK_B)
        rand = state_keyed_draw(
            pool_ptr,
            key_pos_ptr,
            key_tok_ptr,
            key_stride,
            num_gen,
            kpos,
            ktok,
            COUNTER_STRIDE,
            SLOT_STRIDE,
            POOL_MASK,
            BLOCK_B,
            TIE_CHUNK,
        )
        if REPLAY:
            # force_replay: a row that follows a recorded stream commits the stream's next
            # tokens instead of the selected ones.
            rk, rj, rlen = replay_lookup(
                ref_base_ptr,
                ref_len_ptr,
                ref_tok_ptr,
                ref_stride,
                ref_start_ptr,
                num_refs,
                kpos,
                key_tok_ptr,
                key_stride,
                num_gen,
                DRAFT_LEN + 1,
                BLOCK_B,
                BLOCK_R,
                BLOCK_K,
            )
            rk = tl.where(gen_mask, rk, -1)
            rrow = tl.maximum(rk, 0).to(tl.int64)[:, None] * ref_stride
            commit_idx = rj[:, None] + 1 + cols[None, :]
            commit_mask = tile_mask & (rk[:, None] >= 0) & (commit_idx < rlen[:, None])
            recorded = tl.load(
                ref_tok_ptr + rrow + tl.where(commit_mask, commit_idx, 0), mask=commit_mask, other=0
            )

    lanes = tl.arange(0, 2 * BLOCK_SLOTS)
    lane_ok = lanes < 2 * n_slots

    # Context requests: accept the selected token only.
    c = tl.arange(0, BLOCK_C)
    cmask = c < num_ctx
    cm2 = cmask[:, None] & lane_ok[None, :]
    craw = tl.load(
        pairs_ptr + c[:, None].to(tl.int64) * stride_row + lanes[None, :], mask=cm2, other=0.0
    )
    cvals, cidxf = tl.split(tl.reshape(craw, [BLOCK_C, BLOCK_SLOTS, 2]))
    cok = cmask[:, None] & (tl.arange(0, BLOCK_SLOTS)[None, :] < n_slots)
    t_ctx = _pair_argmax(
        tl.where(cok, cvals, float("-inf")), tl.where(cok, cidxf.to(tl.int32), _VP_INT32_MAX), 1
    )
    tl.store(accepted_ptr + c * width, t_ctx, mask=cmask)
    tl.store(num_accepted_ptr + c, tl.full([BLOCK_C], 1, tl.int32), mask=cmask)

    # Generation requests: the K+1 selected tokens of each request (the global
    # argmax / draw of every row's exchanged pairs, as the vocab-parallel
    # selection reduces them), then strict acceptance as _strict_accept_kernel.
    prow = num_ctx + rows[:, None] * width + cols[None, :]
    m3 = tile_mask[:, :, None] & lane_ok[None, None, :]
    graw = tl.load(
        pairs_ptr + prow[:, :, None].to(tl.int64) * stride_row + lanes[None, None, :],
        mask=m3,
        other=0.0,
    )
    gvals, gidxf = tl.split(tl.reshape(graw, [BLOCK_B, BLOCK_K, BLOCK_SLOTS, 2]))
    gok = tile_mask[:, :, None] & (tl.arange(0, BLOCK_SLOTS)[None, None, :] < n_slots)
    t_gen = _pair_argmax(
        tl.where(gok, gvals, float("-inf")), tl.where(gok, gidxf.to(tl.int32), _VP_INT32_MAX), 2
    )
    if FORCE == 3 and REPLAY:
        t_gen = tl.where(commit_mask, recorded, t_gen)
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    mismatch = draft_mask & (d != t_gen)
    first_mismatch = tl.min(tl.where(mismatch, cols[None, :], DRAFT_LEN), axis=1)
    n = 1 + first_mismatch
    if FORCE == 1:
        n = tl.full([BLOCK_B], 0, tl.int32) + base_total
    elif FORCE >= 2:
        extra = (rand < frac).to(tl.int32)
        n = tl.minimum(base_total + extra, max_total)
    n = n.to(tl.int32)
    tl.store(num_accepted_ptr + num_ctx + rows, n, mask=gen_mask)
    if FORCE == 3 and REPLAY:
        # The recorded continuation after the n committed tokens: the next step's drafts (the
        # next-new-tokens kernel puts them in).
        replay_mask = draft_mask & (rk[:, None] >= 0)
        didx = rj[:, None] + n[:, None] + 1 + cols[None, :]
        nxt = tl.load(
            ref_tok_ptr + rrow + tl.where(replay_mask, didx, 0), mask=replay_mask, other=0
        )
        tl.store(
            replay_drafts_ptr + rows[:, None] * replay_stride + cols[None, :], nxt, mask=replay_mask
        )
        tl.store(replay_flags_ptr + rows, (rk >= 0).to(tl.int32), mask=gen_mask)

    if KDA_REPLAY:
        # The KDA replay record of update_mamba_states (KDAReplayState.record_acceptance):
        # the accepted draft prefix per state slot; dummy requests keep the slot's current
        # value. All slots were read before any is written, as gather-then-put does.
        accepted_drafts = tl.maximum(n - 1, 0).to(current.dtype)
        tl.debug_barrier()
        tl.store(
            prev_ptr + slot * stride_prev, tl.where(dummy, current, accepted_drafts), mask=gen_mask
        )
    if KV_REWIND:
        # DFlashWorker._prepare_kv_for_draft_forward.
        tl.store(rewind_ptr + rows, 1 - n, mask=gen_mask)
        tl.store(kv_lens_ptr + num_ctx + rows, kv + 1, mask=gen_mask)


def fused_select_accept(
    pairs: tuple,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    force: Optional[tuple] = None,
    kda_replay: Optional[tuple] = None,
    kv_lens: Optional[torch.Tensor] = None,
    replay: Optional[ForceReplay] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Vocab-parallel target selection, strict acceptance and the per-step
    bookkeeping that follows it, in one launch.

    Args:
        pairs: ``(pairs, row stride, slot stride, slots)`` from
            ``VocabShardedLogits.select_pairs``: every rank's ``(value, global
            index)`` candidates of the ``num_contexts + num_gens * (K + 1)`` rows.
        draft_tokens: ``[num_gens, K]`` draft tokens being verified.
        num_contexts, batch_size: request counts.
        force: the forced-acceptance override (see :func:`force_launch_args`): None for
            plain strict acceptance, ``(base_total,)``, or the fractional 7-tuple keyed by
            a device counter (bumped in place) or a :class:`ForceAcceptKey`.
        kda_replay: ``(prev_num_accepted_tokens, state_indices, dummy_mask)`` to
            also record the KDA replay acceptance that ``update_mamba_states``
            records (``dspark_glue_kernels.kda_record_acceptance``): 1-D strided
            views; ``dummy_mask`` (bool / uint8 per batch row) may be None.
        kv_lens: ``attn_metadata.kv_lens_cuda`` to also apply the KV-length prep of
            ``DFlashWorker._prepare_kv_for_draft_forward`` (its gen rows += 1), or None.
        replay: with a state-keyed ``force``, the :class:`ForceReplay` tables: matching
            generation rows commit their recorded tokens, and their next drafts and replay
            flags go to ``replay.drafts`` / ``replay.flags`` (see ``force_replay``).
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` int32, ``num_accepted_tokens
        [batch_size]`` int32 (the values of the pair reduce + strict acceptance
        path) and, with ``kv_lens``, the ``[num_gens]`` int32 KV rewind amounts
        ``1 - num_accepted``.
    """
    pair_buf, stride_row, stride_slot, n_slots = pairs
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = pair_buf.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    rewind = (
        torch.empty((num_gens,), dtype=torch.int32, device=device) if kv_lens is not None else None
    )
    if batch_size == 0:
        return accepted, num_accepted, rewind
    assert draft_tokens.dim() == 2 and (num_gens == 0 or draft_tokens.stride(-1) == 1)
    (
        force_mode,
        base_total,
        frac,
        max_total,
        pool,
        counter,
        key_pos,
        key_tok,
        key_stride,
        counter_stride,
        slot_stride,
        pool_mask,
    ) = force_launch_args(force, pair_buf)
    if replay is not None:
        assert force_mode == 3, "force_replay needs the state-keyed forced acceptance"
        assert num_gens <= replay.flags.shape[0] and draft_len <= replay.drafts.shape[1]
        ref_start = (
            replay.ref_start if replay.ref_start is not None else torch.zeros_like(replay.ref_len)
        )
        ref_operands = (
            replay.ref_base,
            replay.ref_len,
            replay.ref_tok,
            replay.ref_tok.stride(0),
            ref_start,
        )
        replay_operands = (replay.num_refs, replay.drafts, replay.flags, replay.drafts.stride(0))
    else:
        ref_operands = (accepted, accepted, accepted, 0, accepted)
        replay_operands = (0, accepted, accepted, 0)
    block_b = triton.next_power_of_2(max(num_gens, 1))
    prev, state_idx, dummy = kda_replay if kda_replay is not None else (accepted, accepted, None)
    if kda_replay is not None:
        assert prev.dim() == 1 and prev.dtype == torch.int32 and state_idx.dim() == 1
    has_dummy = dummy is not None
    if not has_dummy:
        dummy = accepted
    elif dummy.dtype == torch.bool:
        dummy = dummy.view(torch.uint8)
    if kv_lens is not None:
        assert kv_lens.dim() == 1 and kv_lens.stride(0) == 1
    # Interleaved (value, index) slots, as the MNNVL / single-rank exchange lays them out.
    assert stride_slot == 2 and pair_buf.stride(-1) == 1
    _select_accept_kernel[(1,)](
        pair_buf,
        stride_row,
        n_slots,
        draft_tokens,
        draft_tokens.stride(0) if num_gens > 0 else draft_len,
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        pool,
        counter,
        base_total,
        float(frac),
        max_total,
        key_pos,
        key_tok,
        key_stride,
        *ref_operands,
        *replay_operands,
        prev,
        prev.stride(0),
        state_idx,
        state_idx.stride(0),
        dummy,
        dummy.stride(0) if dummy.dim() > 0 else 0,
        kv_lens if kv_lens is not None else accepted,
        rewind if rewind is not None else accepted,
        DRAFT_LEN=draft_len,
        FORCE=force_mode,
        COUNTER_STRIDE=int(counter_stride),
        SLOT_STRIDE=int(slot_stride),
        POOL_MASK=int(pool_mask),
        KDA_REPLAY=kda_replay is not None,
        HAS_DUMMY=has_dummy,
        KV_REWIND=kv_lens is not None,
        BLOCK_C=triton.next_power_of_2(max(num_contexts, 1)),
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        BLOCK_SLOTS=triton.next_power_of_2(n_slots),
        TIE_CHUNK=min(block_b, FORCE_ACCEPT_TIE_CHUNK),
        REPLAY=replay is not None,
        BLOCK_R=triton.next_power_of_2(replay.num_refs) if replay is not None else 1,
        num_warps=4,
    )
    return accepted, num_accepted, rewind


@triton.jit
def _decode_block_table_kernel(
    src_ptr,
    stride_src_row,
    stride_src_col,
    n_blocks,
    divisor,
    table_ptr,
    stride_table,
    counts_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    src_row = src_ptr + row.to(tl.int64) * stride_src_row
    table_row = table_ptr + row.to(tl.int64) * stride_table
    count = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(0, n_blocks, BLOCK):
        cols = start + offs
        mask = cols < n_blocks
        encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
        count += (encoded >= 0).to(tl.int64)
        tl.store(table_row + cols, (tl.maximum(encoded, 0) // divisor).to(tl.int32), mask=mask)
    tl.store(counts_ptr + row, tl.sum(count, axis=0))


def decode_draft_block_table(
    src: torch.Tensor, divisor: int, table: torch.Tensor, counts: torch.Tensor
) -> None:
    """``counts[i] = (src[i] >= 0).sum()``; ``table[i] = src[i].clamp(min=0) // divisor``.

    Args:
        src: ``[rows, blocks]`` encoded block offsets (any strides).
        divisor: encoded offset -> pool block index divisor.
        table: ``[>= rows, blocks]`` int32 output, unit inner stride.
        counts: ``[>= rows]`` int64 output.
    """
    rows, n_blocks = src.shape
    assert table.shape[-1] == n_blocks and table.stride(-1) == 1
    if rows == 0:
        return
    _decode_block_table_kernel[(rows,)](
        src,
        src.stride(0),
        src.stride(1),
        n_blocks,
        int(divisor),
        table,
        table.stride(0),
        counts,
        BLOCK=min(_BLOCK_TABLE_BLOCK, triton.next_power_of_2(n_blocks)),
        num_warps=4,
    )


class DraftBlockPlacement(NamedTuple):
    """Where a paged drafter context places each request's own block.

    Block position ``j`` of request ``g`` goes to page ``pages[g, j]`` at
    ``offsets[g, j]``, right after the ``num_ctx[g]`` context positions; the
    block attends ``seq_lens[g] = num_ctx[g] + block`` keys, repeated per query
    token in ``kv_bounds``.
    """

    pages: torch.Tensor  # [G, block] int64
    offsets: torch.Tensor  # [G, block] int64
    seq_lens: torch.Tensor  # [G] int32
    kv_bounds: torch.Tensor  # [G * block] int32
    # Helix: [G, block, 1] bool, True where this rank's pages hold the block
    # position; pages / offsets and the lengths above are then rank-local.
    owned: Optional[torch.Tensor] = None


class PagedLatentPrepare(NamedTuple):
    """What :func:`dflash_paged_latent_prepare` computes for ``G`` generation
    requests drafting ``block`` positions from ``width = K + 1`` target tokens."""

    bonus: torch.Tensor  # [G] int64, the last accepted token
    query_positions: torch.Tensor  # [G, block] int64
    ctx_positions: torch.Tensor  # [G * width] int64, RoPE positions of the target latents
    store_pages: torch.Tensor  # [G * width] int64, destination page of each latent
    store_offsets: torch.Tensor  # [G * width] int64, offset inside that page
    store_mask: torch.Tensor  # [G * width] int8, 1 for accepted tokens
    num_ctx: torch.Tensor  # [G] int64, context length the block attends
    blk_pages: torch.Tensor  # [G, block] int64, page of each block position
    blk_offsets: torch.Tensor  # [G, block] int64
    blk_seq_lens: torch.Tensor  # [G] int32, num_ctx + block
    blk_kv_bounds: torch.Tensor  # [G * block] int32, blk_seq_lens per query token
    blk_owned: Optional[torch.Tensor] = None  # helix: [G, block] int8, 1 where this rank owns it


@triton.jit
def _decode_block_table_chunk_kernel(
    src_ptr,
    stride_src,
    stride_src_col,
    n_blocks,
    divisor,
    table_ptr,
    stride_table,
    partial_ptr,
    BLOCK: tl.constexpr,
):
    # Program (row, chunk) decodes columns [chunk * BLOCK, (chunk + 1) * BLOCK)
    # and records how many of them are allocated (encoded >= 0).
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_blocks
    src_row = src_ptr + row.to(tl.int64) * stride_src
    encoded = tl.load(src_row + cols * stride_src_col, mask=mask, other=-1).to(tl.int64)
    decoded = tl.maximum(encoded, 0) // divisor
    tl.store(table_ptr + row.to(tl.int64) * stride_table + cols, decoded.to(tl.int32), mask=mask)
    count = tl.sum((encoded >= 0).to(tl.int32), axis=0)
    tl.store(partial_ptr + row * tl.num_programs(1) + chunk, count)


@triton.jit
def _sum_block_counts_kernel(
    partial_ptr, n_chunks, counts_ptr, rows, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr
):
    r = tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    mask = (r[:, None] < rows) & (c[None, :] < n_chunks)
    partial = tl.load(partial_ptr + r[:, None] * n_chunks + c[None, :], mask=mask, other=0)
    tl.store(counts_ptr + r, tl.sum(partial.to(tl.int64), axis=1), mask=r < rows)


def decode_draft_block_table_parallel(
    src: torch.Tensor,
    divisor: int,
    table: torch.Tensor,
    counts: torch.Tensor,
    partial: torch.Tensor,
) -> None:
    """:func:`decode_draft_block_table` with the columns of a row spread over programs.

    Same outputs (integer arithmetic, order-free). ``partial`` is an int32
    scratch of at least ``rows * cdiv(blocks, _BLOCK_TABLE_BLOCK)`` elements.
    """
    rows, n_blocks = src.shape
    assert table.shape[-1] == n_blocks and table.stride(-1) == 1
    if rows == 0:
        return
    n_chunks = triton.cdiv(n_blocks, _BLOCK_TABLE_BLOCK)
    assert partial.dtype == torch.int32 and partial.numel() >= rows * n_chunks
    _decode_block_table_chunk_kernel[(rows, n_chunks)](
        src,
        src.stride(0),
        src.stride(1),
        n_blocks,
        int(divisor),
        table,
        table.stride(0),
        partial,
        BLOCK=_BLOCK_TABLE_BLOCK,
        num_warps=8,
    )
    _sum_block_counts_kernel[(1,)](
        partial,
        n_chunks,
        counts,
        rows,
        BLOCK_R=triton.next_power_of_2(rows),
        BLOCK_C=triton.next_power_of_2(n_chunks),
        num_warps=4,
    )


@triton.jit
def _ctx_positions_kernel(
    ctx_len_ptr, slots_ptr, out_ptr, num_gens, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    mask = i < num_gens * WIDTH
    g = i // WIDTH
    slot = tl.load(slots_ptr + g, mask=mask, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot, mask=mask, other=0).to(tl.int64)
    tl.store(out_ptr + i, old + (i - g * WIDTH), mask=mask)


def dflash_ctx_positions(ctx_len: torch.Tensor, slots: torch.Tensor, width: int) -> torch.Tensor:
    """``ctx_len[slots[g]] + j`` for the ``width`` tokens ``j`` of each request
    ``g``: the RoPE positions :func:`dflash_gen_prepare` assigns the step's
    context latents (it reads ``ctx_len`` before advancing it)."""
    num_gens = slots.shape[0]
    out = torch.empty((num_gens * width,), dtype=torch.long, device=ctx_len.device)
    if num_gens > 0:
        _ctx_positions_kernel[(1,)](
            ctx_len,
            slots,
            out,
            num_gens,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(num_gens * width),
            num_warps=1,
        )
    return out


@triton.jit
def _paged_latent_prepare_kernel(
    num_accepted_ptr,
    accepted_ptr,
    stride_accepted,
    slots_ptr,
    ctx_len_ptr,
    block_counts_ptr,
    table_ptr,
    stride_table,
    max_blocks,
    seq_lens_ptr,
    bonus_ptr,
    query_pos_ptr,
    ctx_pos_ptr,
    store_pages_ptr,
    store_offsets_ptr,
    store_mask_ptr,
    num_ctx_ptr,
    blk_pages_ptr,
    blk_offsets_ptr,
    blk_seq_lens_ptr,
    blk_kv_bounds_ptr,
    blk_owned_ptr,
    position_offset_ptr,
    num_gens,
    first_row,
    dummy_slot,
    max_ctx,
    cp_rank,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    CP_SIZE: tl.constexpr,
    HAS_POSITION_OFFSET: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    g = tl.arange(0, BLOCK_G)
    j = tl.arange(0, BLOCK_W)
    gm = g < num_gens
    row = first_row + g
    nacc = tl.load(num_accepted_ptr + row, mask=gm, other=1).to(tl.int64)
    slot = tl.load(slots_ptr + g, mask=gm, other=0).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot, mask=gm, other=0).to(tl.int64)
    capacity = tl.load(block_counts_ptr + row, mask=gm, other=0).to(tl.int64) * PAGE_SIZE
    table_row = table_ptr + row.to(tl.int64) * stride_table

    bonus = tl.load(
        accepted_ptr + row.to(tl.int64) * stride_accepted + tl.maximum(nacc - 1, 0),
        mask=gm,
        other=0,
    )
    tl.store(bonus_ptr + g, bonus.to(tl.int64), mask=gm)

    # The block continues from the context length advanced by this step.
    now = tl.minimum(old + nacc, max_ctx)
    qmask = gm[:, None] & (j[None, :] < BLOCK_SIZE)
    blk = g[:, None] * BLOCK_SIZE + j[None, :]
    tl.store(query_pos_ptr + blk, now[:, None] + j[None, :], mask=qmask)

    # Context latents of this step's target tokens: RoPE position, destination
    # (column clamped into the request's allocation) and write mask. Under helix
    # (CP_SIZE > 1) global page p lives on rank p % CP_SIZE as its local page
    # p // CP_SIZE; positions on other ranks' pages are skipped (mask -1).
    wmask = gm[:, None] & (j[None, :] < WIDTH)
    flat = g[:, None] * WIDTH + j[None, :]
    pos = old[:, None] + j[None, :]
    tl.store(ctx_pos_ptr + flat, pos, mask=wmask)
    gpage = pos // PAGE_SIZE
    col = (gpage // CP_SIZE) * PAGE_SIZE + pos % PAGE_SIZE
    col = tl.maximum(tl.minimum(col, capacity[:, None] - 1), 0)
    page = tl.load(table_row[:, None] + tl.minimum(col // PAGE_SIZE, max_blocks - 1), mask=wmask)
    tl.store(store_pages_ptr + flat, page.to(tl.int64), mask=wmask)
    tl.store(store_offsets_ptr + flat, col % PAGE_SIZE, mask=wmask)
    store = (j[None, :] < nacc[:, None]).to(tl.int8)
    store = tl.where(gpage % CP_SIZE == cp_rank, store, -1)
    tl.store(store_mask_ptr + flat, store, mask=wmask)

    # Advance the context length; the shared dummy slot stays empty. A length
    # clamped at max_ctx moves the history's position offset instead.
    new_len = tl.where(slot == dummy_slot, 0, now)
    if HAS_POSITION_OFFSET:
        live = gm & (slot != dummy_slot)
        offset = tl.load(position_offset_ptr + slot, mask=live, other=0)
        overflow = tl.maximum(old + nacc - max_ctx, 0)
        tl.store(position_offset_ptr + slot, offset + overflow, mask=live)
    # Every lane has read its old length before any lane writes a new one.
    tl.debug_barrier()
    tl.store(ctx_len_ptr + slot, new_len, mask=gm)
    tl.store(ctx_len_ptr + dummy_slot, tl.zeros([], dtype=tl.int64))

    # Advertised context, capped so the block still fits in the allocation
    # (CP_SIZE local pages cover CP_SIZE times as many global positions).
    num_ctx = tl.minimum(new_len, tl.maximum(capacity * CP_SIZE - BLOCK_SIZE, 0))
    tl.store(num_ctx_ptr + g, num_ctx, mask=gm)
    tl.store(
        seq_lens_ptr + row,
        tl.full([BLOCK_G], WIDTH, tl.int32).to(seq_lens_ptr.dtype.element_ty),
        mask=gm,
    )

    # Block-decode operands: where the block's own latents go, and its bounds
    # (rank-local slots and lengths under helix, as HelixDraftShard computes).
    bpos = num_ctx[:, None] + j[None, :]
    bgpage = bpos // PAGE_SIZE
    bslot = (bgpage // CP_SIZE) * PAGE_SIZE + bpos % PAGE_SIZE
    bpage = tl.load(table_row[:, None] + tl.minimum(bslot // PAGE_SIZE, max_blocks - 1), mask=qmask)
    tl.store(blk_pages_ptr + blk, bpage.to(tl.int64), mask=qmask)
    tl.store(blk_offsets_ptr + blk, bslot % PAGE_SIZE, mask=qmask)
    total = num_ctx + BLOCK_SIZE
    if CP_SIZE > 1:
        tl.store(blk_owned_ptr + blk, (bgpage % CP_SIZE == cp_rank).to(tl.int8), mask=qmask)
        span = PAGE_SIZE * CP_SIZE
        rem = total % span - cp_rank * PAGE_SIZE
        total = (total // span) * PAGE_SIZE + tl.minimum(tl.maximum(rem, 0), PAGE_SIZE)
    seq_len = total.to(tl.int32)
    tl.store(blk_seq_lens_ptr + g, seq_len, mask=gm)
    tl.store(
        blk_kv_bounds_ptr + blk,
        tl.zeros([BLOCK_G, BLOCK_W], tl.int32) + seq_len[:, None],
        mask=qmask,
    )


def dflash_paged_latent_prepare(
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
    position_offset: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> PagedLatentPrepare:
    """The DFlash generation-step bookkeeping for a paged single-latent context.

    For generation request ``g`` (batch row ``first_row + g``, context slot
    ``slots[g]``) computes what ``DFlashWorker.prepare_1st_drafter_inputs`` and
    ``_build_mla_block_fixup`` compute eagerly, and applies the same side
    effects: ``ctx_len`` advanced in place (the dummy slot zeroed) and
    ``seq_lens_cuda`` of the generation rows set to ``width``.

    Args:
        num_accepted_tokens: ``[batch]`` accepted counts.
        accepted_tokens: ``[batch, width]`` accepted tokens, unit inner stride.
        slots: ``[G]`` context slots.
        ctx_len: ``[slots]`` int64 context lengths.
        block_counts: ``[batch]`` pages held per batch row.
        block_tables: ``[batch, max_blocks]`` int32 page tables, unit inner stride.
        seq_lens_cuda: ``[batch]`` attention sequence lengths.
        first_row: batch row of generation request 0.
        dummy_slot: the shared padding slot.
        max_ctx: context length ceiling.
        page_size, block_size, width: page size, drafted block positions and
            target tokens per request.
        position_offset: the unified draft cache's ``[slots]`` history position
            offset; takes what a length clamped at ``max_ctx`` loses, in place.
        cp_size, cp_rank: helix sharding of the context pages
            (``HelixDraftShard``). With ``cp_size > 1`` the store mask is -1 on
            other ranks' positions, the block operands are rank-local and
            ``blk_owned`` marks the block positions this rank holds.
    """
    num_gens = slots.shape[0]
    device = ctx_len.device
    i64 = dict(dtype=torch.int64, device=device)
    out = PagedLatentPrepare(
        bonus=torch.empty((num_gens,), **i64),
        query_positions=torch.empty((num_gens, block_size), **i64),
        ctx_positions=torch.empty((num_gens * width,), **i64),
        store_pages=torch.empty((num_gens * width,), **i64),
        store_offsets=torch.empty((num_gens * width,), **i64),
        store_mask=torch.empty((num_gens * width,), dtype=torch.int8, device=device),
        num_ctx=torch.empty((num_gens,), **i64),
        blk_pages=torch.empty((num_gens, block_size), **i64),
        blk_offsets=torch.empty((num_gens, block_size), **i64),
        blk_seq_lens=torch.empty((num_gens,), dtype=torch.int32, device=device),
        blk_kv_bounds=torch.empty((num_gens * block_size,), dtype=torch.int32, device=device),
        blk_owned=(
            torch.empty((num_gens, block_size), dtype=torch.int8, device=device)
            if cp_size > 1
            else None
        ),
    )
    if num_gens == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and block_tables.stride(-1) == 1
    _paged_latent_prepare_kernel[(1,)](
        num_accepted_tokens,
        accepted_tokens,
        accepted_tokens.stride(0),
        slots,
        ctx_len,
        block_counts,
        block_tables,
        block_tables.stride(0),
        block_tables.shape[1],
        seq_lens_cuda,
        *out[:-1],
        out.store_mask if out.blk_owned is None else out.blk_owned,
        ctx_len if position_offset is None else position_offset,
        num_gens,
        first_row,
        dummy_slot,
        max_ctx,
        cp_rank,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
        WIDTH=width,
        CP_SIZE=cp_size,
        HAS_POSITION_OFFSET=position_offset is not None,
        BLOCK_G=triton.next_power_of_2(num_gens),
        BLOCK_W=triton.next_power_of_2(max(width, block_size)),
        num_warps=4,
    )
    return out


@triton.jit
def _store_context_latents_kernel(
    latent_ptr,
    stride_latent_tok,
    stride_latent_layer,
    pool_ptrs_ptr,
    stride_pool_page,
    stride_pool_off,
    pages_ptr,
    offsets_ptr,
    mask_ptr,
    head_dim,
    BLOCK_D: tl.constexpr,
    POOL_FP8: tl.constexpr,
):
    tok = tl.program_id(0)
    layer = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < head_dim
    page = tl.load(pages_ptr + tok)
    off = tl.load(offsets_ptr + tok)
    flag = tl.load(mask_ptr + tok)
    keep = (flag > 0).to(tl.float32)
    src = latent_ptr + tok.to(tl.int64) * stride_latent_tok + layer * stride_latent_layer
    vals = tl.load(src + offs, mask=dmask, other=0.0)
    # The eager path multiplies by the mask in the latent dtype: x * 1 is x and
    # x * 0 a signed zero either way.
    vals = (vals.to(tl.float32) * keep).to(vals.dtype)
    if POOL_FP8:
        # to_draft_fp8: saturated E4M3 at unit scale, round to nearest even.
        vals = tl.minimum(tl.maximum(vals.to(tl.float32), -448.0), 448.0).to(tl.float8e4nv)
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(vals.dtype))
    # A -1 flag leaves the slot untouched (another helix rank owns it).
    tl.store(
        pool + page * stride_pool_page + off * stride_pool_off + offs,
        vals,
        mask=dmask & (flag >= 0),
    )


def store_context_latents(
    latent: torch.Tensor,
    pools: list[torch.Tensor],
    pool_ptrs: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """``pools[l][pages[t], 0, 0, offsets[t]] = latent[t, l] * mask[t]`` for every
    token ``t`` and layer ``l``.

    Args:
        latent: ``[N, L, D]`` per-token, per-layer latents (unit inner stride).
        pools: the ``L`` per-layer ``[pages, 1, 1, page_size, D]`` pool views,
            all with the same strides.
        pool_ptrs: ``[L]`` int64 base addresses of ``pools``.
        pages, offsets: ``[N]`` destination page and offset in the page.
        mask: ``[N]`` int8 write mask; 0 stores a zeroed latent, -1 leaves the
            slot untouched.

    An FP8 E4M3 pool stores ``to_draft_fp8`` of the masked latent.
    """
    num_tokens, num_layers, head_dim = latent.shape
    if num_tokens == 0:
        return
    pool0 = pools[0]
    assert latent.stride(-1) == 1 and pool0.stride(-1) == 1
    assert all(p.stride() == pool0.stride() and p.dtype == pool0.dtype for p in pools)
    pool_fp8 = pool0.dtype == torch.float8_e4m3fn
    assert pool_fp8 or pool0.dtype == latent.dtype
    _store_context_latents_kernel[(num_tokens, num_layers)](
        latent,
        latent.stride(0),
        latent.stride(1),
        pool_ptrs,
        pool0.stride(0),
        pool0.stride(3),
        pages,
        offsets,
        mask,
        head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        POOL_FP8=pool_fp8,
        num_warps=4,
    )


@triton.jit
def _scatter_rows_kernel(
    src_ptr,
    stride_src,
    dst_ptr,
    stride_dst_page,
    stride_dst_off,
    pages_ptr,
    offsets_ptr,
    rows_mask_ptr,
    width,
    BLOCK_D: tl.constexpr,
    POOL_FP8: tl.constexpr,
    HAS_ROWS_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < width
    if HAS_ROWS_MASK:
        mask = mask & (tl.load(rows_mask_ptr + row) != 0)
    page = tl.load(pages_ptr + row)
    off = tl.load(offsets_ptr + row)
    vals = tl.load(src_ptr + row.to(tl.int64) * stride_src + offs, mask=mask)
    if POOL_FP8:
        # to_draft_fp8: saturated E4M3 at unit scale, round to nearest even.
        vals = tl.minimum(tl.maximum(vals.to(tl.float32), -448.0), 448.0).to(tl.float8e4nv)
    tl.store(dst_ptr + page * stride_dst_page + off * stride_dst_off + offs, vals, mask=mask)


def scatter_rows_to_pool(
    src: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    rows_mask: Optional[torch.Tensor] = None,
) -> None:
    """``pool[pages[i], offsets[i]] = src[i]`` for a ``[pages, page_size, D]`` pool view.

    The same writes as the eager advanced-index assignment (a pure copy); an
    FP8 E4M3 pool stores ``to_draft_fp8(src[i])``. With ``rows_mask`` (bool or
    integer ``[rows]``), rows where it is 0 / False write nothing: the slot
    keeps its value, as ``torch.where(rows_mask, src, pool[pages, offsets])``
    written back would leave it.
    """
    rows, width = src.shape
    if rows == 0:
        return
    assert src.stride(-1) == 1 and pool.stride(-1) == 1 and pool.shape[-1] == width
    pool_fp8 = pool.dtype == torch.float8_e4m3fn
    assert pool_fp8 or pool.dtype == src.dtype
    if rows_mask is not None:
        rows_mask = rows_mask.reshape(-1)
        assert rows_mask.numel() == rows and rows_mask.is_contiguous()
        if rows_mask.dtype == torch.bool:
            rows_mask = rows_mask.view(torch.uint8)
    _scatter_rows_kernel[(rows,)](
        src,
        src.stride(0),
        pool,
        pool.stride(0),
        pool.stride(1),
        pages,
        offsets,
        pages if rows_mask is None else rows_mask,
        width,
        BLOCK_D=triton.next_power_of_2(width),
        POOL_FP8=pool_fp8,
        HAS_ROWS_MASK=rows_mask is not None,
        num_warps=4,
    )


@triton.jit
def _next_new_tokens_kernel(
    accepted_ptr,
    stride_accepted,
    num_accepted_ptr,
    draft_ptr,
    stride_draft,
    out_ptr,
    batch_size,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)
    rmask = rows < batch_size
    nacc = tl.load(num_accepted_ptr + rows, mask=rmask, other=1)
    last = tl.load(accepted_ptr + rows.to(tl.int64) * stride_accepted + (nacc - 1), mask=rmask)
    width = DRAFT_LEN + 1
    tl.store(out_ptr + rows * width, last.to(out_ptr.dtype.element_ty), mask=rmask)
    dmask = rmask[:, None] & (cols[None, :] < DRAFT_LEN)
    d = tl.load(draft_ptr + rows[:, None].to(tl.int64) * stride_draft + cols[None, :], mask=dmask)
    tl.store(
        out_ptr + rows[:, None] * width + 1 + cols[None, :],
        d.to(out_ptr.dtype.element_ty),
        mask=dmask,
    )


@triton.jit
def _next_new_tokens_rewind_kernel(
    accepted_ptr,
    stride_accepted,
    num_accepted_ptr,
    draft_ptr,
    stride_draft,
    out_ptr,
    batch_size,
    kv_lens_ptr,
    rewind_ptr,
    num_ctx,
    num_gen,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # _next_new_tokens_kernel for a batch in one program, plus
    # DFlashWorker._apply_kv_rewind_after_draft on the gen rows.
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)
    rmask = rows < batch_size
    nacc = tl.load(num_accepted_ptr + rows, mask=rmask, other=1)
    last = tl.load(accepted_ptr + rows.to(tl.int64) * stride_accepted + (nacc - 1), mask=rmask)
    width = DRAFT_LEN + 1
    tl.store(out_ptr + rows * width, last.to(out_ptr.dtype.element_ty), mask=rmask)
    dmask = rmask[:, None] & (cols[None, :] < DRAFT_LEN)
    d = tl.load(draft_ptr + rows[:, None].to(tl.int64) * stride_draft + cols[None, :], mask=dmask)
    tl.store(
        out_ptr + rows[:, None] * width + 1 + cols[None, :],
        d.to(out_ptr.dtype.element_ty),
        mask=dmask,
    )
    gmask = rows < num_gen
    kv = tl.load(kv_lens_ptr + num_ctx + rows, mask=gmask)
    amount = tl.load(rewind_ptr + rows, mask=gmask)
    tl.store(kv_lens_ptr + num_ctx + rows, tl.maximum(kv - amount, 0), mask=gmask)


def fused_next_new_tokens_rewind(
    accepted_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    num_contexts: int,
) -> torch.Tensor:
    """:func:`fused_next_new_tokens` and ``kv_lens[num_contexts:][:num_gens] =
    (kv_lens - rewind).clamp(min=0)`` in one launch (one program)."""
    batch_size, draft_len = next_draft_tokens.shape
    num_gens = rewind.shape[0]
    dtype = torch.promote_types(accepted_tokens.dtype, next_draft_tokens.dtype)
    out = torch.empty((batch_size, draft_len + 1), dtype=dtype, device=accepted_tokens.device)
    if batch_size == 0 and num_gens == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and next_draft_tokens.stride(-1) == 1
    assert kv_lens.stride(-1) == 1 and rewind.is_contiguous()
    _next_new_tokens_rewind_kernel[(1,)](
        accepted_tokens,
        accepted_tokens.stride(0),
        num_accepted_tokens,
        next_draft_tokens,
        next_draft_tokens.stride(0),
        out,
        batch_size,
        kv_lens,
        rewind,
        num_contexts,
        num_gens,
        DRAFT_LEN=draft_len,
        BLOCK_B=triton.next_power_of_2(max(batch_size, num_gens, 1)),
        BLOCK_K=triton.next_power_of_2(max(draft_len, 1)),
        num_warps=4,
    )
    return out


def fused_next_new_tokens(
    accepted_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """``cat([accepted[b, nacc[b] - 1], next_draft[b]])`` per request, in one launch
    (``SpecWorkerBase._prepare_next_new_tokens`` with identity batch indices)."""
    batch_size, draft_len = next_draft_tokens.shape
    dtype = torch.promote_types(accepted_tokens.dtype, next_draft_tokens.dtype)
    out = torch.empty((batch_size, draft_len + 1), dtype=dtype, device=accepted_tokens.device)
    if batch_size == 0:
        return out
    assert accepted_tokens.stride(-1) == 1 and next_draft_tokens.stride(-1) == 1
    block_b = 128
    _next_new_tokens_kernel[(triton.cdiv(batch_size, block_b),)](
        accepted_tokens,
        accepted_tokens.stride(0),
        num_accepted_tokens,
        next_draft_tokens,
        next_draft_tokens.stride(0),
        out,
        batch_size,
        DRAFT_LEN=draft_len,
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(max(draft_len, 1)),
        num_warps=4,
    )
    return out


@triton.jit
def _vocab_shard_embedding_kernel(
    ids_ptr, weight_ptr, out_ptr, vocab_start, vocab_end, hidden, BLOCK_H: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden
    token = tl.load(ids_ptr + row).to(tl.int64)
    local = (token >= vocab_start) & (token < vocab_end)
    src = tl.where(local, token - vocab_start, 0)
    vals = tl.load(weight_ptr + src * hidden + offs, mask=mask & local, other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * hidden + offs, vals, mask=mask)


def vocab_shard_embedding(embedding, ids: torch.Tensor) -> Optional[torch.Tensor]:
    """``embedding(ids)`` for a COLUMN (vocab-sharded) TP ``Embedding``: this
    rank's rows of the table, zero for ids outside its shard, then the module's
    all-reduce. One lookup kernel instead of the eager mask / lookup / fill
    chain; same values. Returns None for an embedding this does not cover."""
    from ..modules.linear import TensorParallelMode

    weight = getattr(embedding, "weight", None)
    if (
        getattr(embedding, "tp_size", 1) <= 1
        or getattr(embedding, "tp_mode", None) != TensorParallelMode.COLUMN
        or getattr(embedding, "all_reduce", None) is None
        or weight is None
        or not weight.is_contiguous()
        or ids.dim() != 1
    ):
        return None
    hidden = weight.shape[1]
    out = torch.empty((ids.shape[0], hidden), dtype=weight.dtype, device=weight.device)
    if ids.shape[0] > 0:
        block_h = 1024
        _vocab_shard_embedding_kernel[(ids.shape[0], triton.cdiv(hidden, block_h))](
            ids,
            weight,
            out,
            embedding.vocab_start_index,
            embedding.vocab_end_index,
            hidden,
            BLOCK_H=block_h,
        )
    return embedding.all_reduce(out)
