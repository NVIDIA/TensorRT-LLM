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
"""Fused bookkeeping kernels for the one-model DFlash / DSpark generation step.

On drafting Kimi K3.1 engines without attention DP these kernels replace the chains of small PyTorch kernels that
:mod:`spec_tail_kernels` leaves between the target verification forward, the
acceptance, the drafter forward and the next step's inputs with one Triton launch
each:

* :func:`vocab_shards_to_fp32` -- the vocab-parallel target logits gathered for
  sampling: the all-gather's concatenation and the FP32 widening in one pass;
* :func:`strict_accept_forced` -- strict acceptance with the synthetic
  (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``) acceptance override;
* :func:`kda_record_acceptance` -- the KDA replay record of the accepted counts;
* :func:`kv_prepare_for_draft` / :func:`next_new_tokens_and_rewind` -- the
  drafter's KV-length prepare, its rewind and the next step's input tokens;
* :func:`window_block_fixup` -- the MLA drafter's attention window
  (``TLLM_DSPARK_DRAFT_ATTN_WINDOW``) over its block-decode operands: the page
  table slice, the shortened lengths and the per-query key bounds;
* piece ``embed`` -- the drafter's noise-block embedding through the owner-rank
  multicast site (``dspark_ar_sites.draft_embed_site``) instead of the
  vocab-sharded lookup + all-reduce.

Every kernel computes the values of the chain it replaces exactly (integer
bookkeeping, copies and exact BF16 -> FP32 widening), so the accepted tokens, the
draft tokens and every buffer the step writes are bit-identical. Launch shapes
depend only on tensor shapes, nothing synchronizes with the host, every buffer is
indexed with its strides (the KDA replay counts are a strided per-slot view of the
V2 state pool), and every output lands in a buffer the eager chain also writes,
so the kernels are CUDA-graph capturable.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from ..pyexecutor.input_commit import k31_draft_tp_engine
from .spec_tail_kernels import (
    FORCE_ACCEPT_TIE_CHUNK,
    force_launch_args,
    state_key,
    state_keyed_draw,
)

# Comma-separated pieces to keep on the eager path (bisection): logits, accept, kda, kv, next,
# gather, spec_meta, pre_meta, window, embed.
GLUE_FUSION_DISABLE_ENV_VAR = "TLLM_DSPARK_GLUE_FUSION_DISABLE"
GLUE_FUSION_PIECES = (
    "logits",
    "accept",
    "kda",
    "kv",
    "next",
    "gather",
    "spec_meta",
    "pre_meta",
    "window",
    "embed",
)


def glue_fusion_enabled(piece: Optional[str] = None) -> bool:
    """Whether the fused bookkeeping kernels run (drafting Kimi K3.1 engines without
    attention DP; for ``piece``: and it is not listed in ``TLLM_DSPARK_GLUE_FUSION_DISABLE``)."""
    if not k31_draft_tp_engine():
        return False
    if piece is None:
        return True
    disabled = os.environ.get(GLUE_FUSION_DISABLE_ENV_VAR, "")
    return piece not in {name.strip() for name in disabled.split(",")}


# ---------------------------------------------------------------------------
# Vocab-sharded logits: dim-0 all-gather layout -> contiguous FP32 rows
# ---------------------------------------------------------------------------

_SHARD_BLOCK_V = 4096


@triton.jit
def _vocab_shards_to_fp32_kernel(
    src_ptr,
    dst_ptr,
    rows,
    shard_vocab,
    num_shards,
    BLOCK_V: tl.constexpr,
):
    # src is the dim-0 all-gather of the per-rank [rows, shard_vocab] logits,
    # i.e. [num_shards, rows, shard_vocab]; dst is [rows, num_shards * shard_vocab].
    pid_sr = tl.program_id(0)
    pid_v = tl.program_id(1)
    shard = pid_sr // rows
    row = pid_sr - shard * rows
    offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < shard_vocab
    vals = tl.load(src_ptr + pid_sr.to(tl.int64) * shard_vocab + offs, mask=mask)
    dst_row = row.to(tl.int64) * (num_shards * shard_vocab) + shard * shard_vocab
    tl.store(dst_ptr + dst_row + offs, vals.to(tl.float32), mask=mask)


def vocab_shards_to_fp32(gathered: torch.Tensor, num_shards: int) -> torch.Tensor:
    """``torch.cat(gathered.chunk(num_shards), dim=-1).float()`` in one pass.

    Args:
        gathered: ``[num_shards * rows, shard_vocab]`` contiguous logits, the
            per-rank vocab shards stacked along dim 0 (a dim-0 all-gather).
        num_shards: number of vocab shards (the TP size).
    Returns:
        ``[rows, num_shards * shard_vocab]`` contiguous FP32 logits.
    """
    assert gathered.dim() == 2 and gathered.is_contiguous()
    total_rows, shard_vocab = gathered.shape
    assert total_rows % num_shards == 0
    rows = total_rows // num_shards
    out = torch.empty((rows, num_shards * shard_vocab), dtype=torch.float32, device=gathered.device)
    if rows == 0 or shard_vocab == 0:
        return out
    grid = (total_rows, triton.cdiv(shard_vocab, _SHARD_BLOCK_V))
    _vocab_shards_to_fp32_kernel[grid](
        gathered, out, rows, shard_vocab, num_shards, BLOCK_V=_SHARD_BLOCK_V, num_warps=8
    )
    return out


# ---------------------------------------------------------------------------
# Strict acceptance with the synthetic (forced) acceptance override
# ---------------------------------------------------------------------------


@triton.jit
def _strict_accept_forced_kernel(
    target_ptr,
    draft_ptr,
    stride_draft_row,
    stride_draft_col,
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
    DRAFT_LEN: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TIE_CHUNK: tl.constexpr,
):
    # spec_tail_kernels._strict_accept_kernel for the whole batch in one program,
    # then SpecWorkerBase._apply_force_accepted_tokens on the generation counts:
    # a constant count (FORCE 1), or base_total + (pool[idx] < frac) clamped to
    # max_total with idx = (counter * COUNTER_STRIDE + row * SLOT_STRIDE) &
    # POOL_MASK after bumping the device counter (FORCE 2), or the state-keyed
    # draw of spec_tail_kernels.ForceAcceptKey (FORCE 3).
    width = DRAFT_LEN + 1
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)

    ctx_mask = rows < num_ctx
    t_ctx = tl.load(target_ptr + rows, mask=ctx_mask, other=0)
    tl.store(accepted_ptr + rows * width, t_ctx.to(tl.int32), mask=ctx_mask)
    tl.store(num_accepted_ptr + rows, tl.full([BLOCK_B], 1, tl.int32), mask=ctx_mask)

    gen_mask = rows < num_gen
    tile_mask = gen_mask[:, None] & (cols[None, :] < width)
    t_gen = tl.load(
        target_ptr + num_ctx + rows[:, None] * width + cols[None, :], mask=tile_mask, other=0
    ).to(tl.int32)
    tl.store(
        accepted_ptr + (num_ctx + rows[:, None]) * width + cols[None, :], t_gen, mask=tile_mask
    )
    if FORCE == 2:
        counter = tl.load(counter_ptr) + 1
        # Every thread reads the old counter before the new one is stored.
        tl.debug_barrier()
        tl.store(counter_ptr, counter)
        idx = (counter * COUNTER_STRIDE + rows.to(tl.int64) * SLOT_STRIDE) & POOL_MASK
        rand = tl.load(pool_ptr + idx, mask=gen_mask, other=1.0)
        extra = (rand < frac).to(tl.int32)
        n = tl.minimum(base_total + extra, max_total)
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
        extra = (rand < frac).to(tl.int32)
        n = tl.minimum(base_total + extra, max_total)
    else:
        n = tl.full([BLOCK_B], 0, tl.int32) + base_total
    tl.store(num_accepted_ptr + num_ctx + rows, n.to(tl.int32), mask=gen_mask)


def strict_accept_forced(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    force: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``spec_tail_kernels.strict_accept`` followed by
    ``SpecWorkerBase._apply_force_accepted_tokens`` in one launch.

    Args:
        target_tokens: ``[num_contexts + num_gens * (K + 1)]`` sampled target tokens.
        draft_tokens: ``[num_gens, K]`` draft tokens being verified.
        num_contexts, batch_size: request counts.
        force: ``(base_total,)`` (constant forced count) or ``(base_total, frac,
            max_total, pool, source, stride a, stride b)`` (fractional forced count,
            keyed by the device counter ``source``, bumped in place, or by a
            ``spec_tail_kernels.ForceAcceptKey``; see ``force_launch_args``).
    Returns:
        ``accepted_tokens [batch_size, K + 1]`` int32 (context rows define column
        0 only) and ``num_accepted_tokens [batch_size]`` int32.
    """
    num_gens = batch_size - num_contexts
    draft_len = draft_tokens.shape[-1]
    device = target_tokens.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    if batch_size == 0:
        return accepted, num_accepted
    assert draft_tokens.dim() == 2 and target_tokens.dim() == 1 and force is not None
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
    ) = force_launch_args(force, target_tokens)
    block_b = triton.next_power_of_2(max(num_contexts, num_gens, 1))
    _strict_accept_forced_kernel[(1,)](
        target_tokens,
        draft_tokens,
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        accepted,
        num_accepted,
        num_contexts,
        num_gens,
        pool,
        counter,
        base_total,
        frac,
        max_total,
        key_pos,
        key_tok,
        key_stride,
        DRAFT_LEN=draft_len,
        FORCE=force_mode,
        COUNTER_STRIDE=counter_stride,
        SLOT_STRIDE=slot_stride,
        POOL_MASK=pool_mask,
        BLOCK_B=block_b,
        BLOCK_K=triton.next_power_of_2(draft_len + 1),
        TIE_CHUNK=min(block_b, FORCE_ACCEPT_TIE_CHUNK),
        num_warps=4,
    )
    return accepted, num_accepted


# ---------------------------------------------------------------------------
# KDA replay: record this step's accepted draft counts per state slot
# ---------------------------------------------------------------------------


@triton.jit
def _kda_record_acceptance_kernel(
    prev_ptr,
    stride_prev,
    state_idx_ptr,
    stride_state_idx,
    num_accepted_ptr,
    stride_num_accepted,
    dummy_ptr,
    stride_dummy,
    num_ctx,
    num_gen,
    HAS_DUMMY: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # prev is a strided per-slot view (one V2 state page per slot).
    rows = tl.arange(0, BLOCK_B)
    mask = rows < num_gen
    slots = tl.load(state_idx_ptr + (num_ctx + rows) * stride_state_idx, mask=mask, other=0).to(
        tl.int64
    )
    n = tl.load(num_accepted_ptr + (num_ctx + rows) * stride_num_accepted, mask=mask, other=1)
    accepted = tl.maximum((n - 1).to(tl.int32), 0)
    if HAS_DUMMY:
        dummy = tl.load(dummy_ptr + (num_ctx + rows) * stride_dummy, mask=mask, other=0) != 0
        current = tl.load(prev_ptr + slots * stride_prev, mask=mask & dummy, other=0)
        accepted = tl.where(dummy, current, accepted)
    tl.store(prev_ptr + slots * stride_prev, accepted, mask=mask)


def kda_record_acceptance(
    prev_num_accepted: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_contexts: int,
    num_gens: int,
    is_dummy_request: Optional[torch.Tensor],
) -> None:
    """``KDAReplayState.record_acceptance`` as the V2 hybrid manager's
    ``update_mamba_states`` calls it, in one launch: for every generation row
    ``g`` (batch row ``num_contexts + g``), ``prev[state_indices[row]] =
    max(num_accepted_tokens[row] - 1, 0)``, or the slot's current value where
    ``is_dummy_request[row]`` (a bool / uint8 batch-row mask) is set."""
    if num_gens <= 0:
        return
    assert prev_num_accepted.dtype == torch.int32 and prev_num_accepted.dim() == 1
    has_dummy = is_dummy_request is not None
    dummy = is_dummy_request if has_dummy else prev_num_accepted
    if has_dummy and dummy.dtype == torch.bool:
        dummy = dummy.view(torch.uint8)
    _kda_record_acceptance_kernel[(1,)](
        prev_num_accepted,
        prev_num_accepted.stride(0),
        state_indices,
        state_indices.stride(0),
        num_accepted_tokens,
        num_accepted_tokens.stride(0),
        dummy,
        dummy.stride(0),
        num_contexts,
        num_gens,
        HAS_DUMMY=has_dummy,
        BLOCK_B=triton.next_power_of_2(num_gens),
        num_warps=1,
    )


# ---------------------------------------------------------------------------
# Draft-forward KV lengths: prepare (+1, remember the rewind) and rewind
# ---------------------------------------------------------------------------


@triton.jit
def _kv_prepare_kernel(
    kv_lens_ptr,
    stride_kv,
    num_accepted_ptr,
    stride_num_accepted,
    rewind_ptr,
    num_ctx,
    num_gen,
    BLOCK_B: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_B)
    mask = rows < num_gen
    n = tl.load(num_accepted_ptr + (num_ctx + rows) * stride_num_accepted, mask=mask, other=1)
    tl.store(rewind_ptr + rows, 1 - n, mask=mask)
    kv = tl.load(kv_lens_ptr + (num_ctx + rows) * stride_kv, mask=mask, other=0)
    tl.store(kv_lens_ptr + (num_ctx + rows) * stride_kv, kv + 1, mask=mask)


def kv_prepare_for_draft(
    kv_lens: torch.Tensor, num_accepted_tokens: torch.Tensor, num_contexts: int, batch_size: int
) -> torch.Tensor:
    """``rewind = 1 - num_accepted[nc:bs]`` and ``kv_lens[nc:bs] += 1`` in one launch;
    returns ``rewind`` (contiguous, ``num_accepted``'s dtype)."""
    num_gens = batch_size - num_contexts
    rewind = torch.empty((num_gens,), dtype=num_accepted_tokens.dtype, device=kv_lens.device)
    if num_gens > 0:
        _kv_prepare_kernel[(1,)](
            kv_lens,
            kv_lens.stride(0),
            num_accepted_tokens,
            num_accepted_tokens.stride(0),
            rewind,
            num_contexts,
            num_gens,
            BLOCK_B=triton.next_power_of_2(num_gens),
            num_warps=1,
        )
    return rewind


@triton.jit
def _next_new_tokens_kernel(
    accepted_ptr,
    stride_accepted,
    num_accepted_ptr,
    stride_num_accepted,
    batch_idx_ptr,
    stride_batch_idx,
    draft_ptr,
    stride_draft,
    out_ptr,
    kv_lens_ptr,
    stride_kv,
    rewind_ptr,
    stride_rewind,
    num_rows,
    num_ctx,
    num_rewind,
    replay_drafts_ptr,
    replay_flags_ptr,
    replay_stride,
    replay_row0,
    replay_rows,
    REWIND: tl.constexpr,
    DRAFT_LEN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REPLAY: tl.constexpr,
):
    width = DRAFT_LEN + 1
    rows = tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_K)
    mask = rows < num_rows
    n = tl.load(num_accepted_ptr + rows * stride_num_accepted, mask=mask, other=1)
    bidx = tl.load(batch_idx_ptr + rows * stride_batch_idx, mask=mask, other=0).to(tl.int64)
    col = (n - 1).to(tl.int64)
    col = tl.where(col < 0, col + width, col)  # torch negative-index wrap
    last = tl.load(accepted_ptr + bidx * stride_accepted + col, mask=mask, other=0)
    tl.store(out_ptr + rows * width, last, mask=mask)
    dmask = mask[:, None] & (cols[None, :] < DRAFT_LEN)
    d = tl.load(draft_ptr + rows[:, None] * stride_draft + cols[None, :], mask=dmask, other=0)
    if REPLAY:
        # force_replay: the rows the accept kernel replayed take its recorded drafts, also
        # written back into the drafter's output (the next step verifies them).
        g = rows - replay_row0
        gmask = (g >= 0) & (g < replay_rows) & mask
        gsafe = tl.where(gmask, g, 0)
        flag = tl.load(replay_flags_ptr + gsafe, mask=gmask, other=0) != 0
        rmask = dmask & gmask[:, None] & flag[:, None]
        rd = tl.load(
            replay_drafts_ptr + gsafe[:, None].to(tl.int64) * replay_stride + cols[None, :],
            mask=rmask,
            other=0,
        )
        d = tl.where(rmask, rd.to(d.dtype), d)
        tl.store(draft_ptr + rows[:, None] * stride_draft + cols[None, :], d, mask=rmask)
    tl.store(out_ptr + rows[:, None] * width + 1 + cols[None, :], d, mask=dmask)
    if REWIND:
        g = tl.arange(0, BLOCK_B)
        gmask = g < num_rewind
        kv = tl.load(kv_lens_ptr + (num_ctx + g) * stride_kv, mask=gmask, other=0)
        r = tl.load(rewind_ptr + g * stride_rewind, mask=gmask, other=0)
        tl.store(kv_lens_ptr + (num_ctx + g) * stride_kv, tl.maximum(kv - r, 0), mask=gmask)


def next_new_tokens_and_rewind(
    accepted_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    batch_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    batch_size: int,
    kv_lens: Optional[torch.Tensor] = None,
    rewind: Optional[torch.Tensor] = None,
    num_contexts: int = 0,
    replay: Optional[tuple] = None,
) -> torch.Tensor:
    """``SpecWorkerBase._prepare_next_new_tokens`` (the last accepted token of row
    ``batch_indices[r]`` followed by row ``r``'s next draft tokens) and, with
    ``kv_lens``, ``DFlashWorker._apply_kv_rewind_after_draft`` (``kv_lens[nc:nc +
    len(rewind)] = (kv_lens[...] - rewind).clamp(min=0)``) in one launch.

    ``replay``: ``(ForceReplay, first generation row, generation rows)`` of the step's
    replaying accept launch (``force_replay``): its flagged rows take the recorded drafts,
    in the output and in ``next_draft_tokens``."""
    draft_len = next_draft_tokens.shape[1]
    assert accepted_tokens.dim() == 2 and accepted_tokens.stride(-1) == 1
    assert next_draft_tokens.stride(-1) == 1 and next_draft_tokens.dtype == accepted_tokens.dtype
    out = torch.empty(
        (batch_size, draft_len + 1), dtype=accepted_tokens.dtype, device=accepted_tokens.device
    )
    do_rewind = kv_lens is not None and rewind is not None and rewind.numel() > 0
    num_rewind = rewind.numel() if do_rewind else 0
    if batch_size == 0 and num_rewind == 0:
        return out
    kv = kv_lens if do_rewind else out
    rw = rewind if do_rewind else out
    if replay is not None:
        tables, replay_row0, replay_rows = replay
        assert draft_len <= tables.drafts.shape[1] and replay_rows <= tables.flags.shape[0]
        replay_operands = (
            tables.drafts,
            tables.flags,
            tables.drafts.stride(0),
            replay_row0,
            replay_rows,
        )
    else:
        replay_operands = (out, out, 0, 0, 0)
    _next_new_tokens_kernel[(1,)](
        accepted_tokens,
        accepted_tokens.stride(0),
        num_accepted_tokens,
        num_accepted_tokens.stride(0),
        batch_indices,
        batch_indices.stride(0),
        next_draft_tokens,
        next_draft_tokens.stride(0),
        out,
        kv,
        kv.stride(0),
        rw,
        rw.stride(0) if do_rewind else 1,
        batch_size,
        num_contexts,
        num_rewind,
        *replay_operands,
        REWIND=do_rewind,
        DRAFT_LEN=draft_len,
        BLOCK_B=triton.next_power_of_2(max(batch_size, num_rewind, 1)),
        BLOCK_K=triton.next_power_of_2(max(draft_len, 1)),
        REPLAY=replay is not None,
        num_warps=1,
    )
    return out


# ---------------------------------------------------------------------------
# MLA drafter attention window over the block-decode operands
# ---------------------------------------------------------------------------

_WINDOW_BLOCK_COLS = 1024


@triton.jit
def _window_block_fixup_kernel(
    table_ptr,
    stride_table,
    table_out_ptr,
    seq_lens_ptr,
    seq_lens_out_ptr,
    kv_bounds_ptr,
    kv_bounds_out_ptr,
    ctx_len_ptr,
    cols,
    window,
    page_size,
    cp_size,
    cp_rank,
    BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    HAS_CTX_LEN: tl.constexpr,
):
    b = tl.program_id(0)
    chunk = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
    if HAS_CTX_LEN:
        ctx_len = tl.load(ctx_len_ptr + b).to(tl.int64)
    else:
        ctx_len = seq_len - BLOCK
    first = tl.maximum(ctx_len - window, 0) // page_size
    start = (tl.maximum(first - cp_rank, 0) + (cp_size - 1)) // cp_size
    col = chunk * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    in_row = col < cols
    src = tl.minimum(start + col, cols - 1)
    pages = tl.load(table_ptr + b * stride_table + src, mask=in_row)
    tl.store(table_out_ptr + b * cols + col, pages, mask=in_row)
    if chunk == 0:
        skipped = start * page_size
        tl.store(seq_lens_out_ptr + b, (seq_len - skipped).to(tl.int32))
        q = tl.arange(0, BLOCK_Q)
        in_block = q < BLOCK
        bound = tl.load(kv_bounds_ptr + b * BLOCK + q, mask=in_block, other=0).to(tl.int64)
        tl.store(kv_bounds_out_ptr + b * BLOCK + q, (bound - skipped).to(tl.int32), mask=in_block)


def window_block_fixup(
    page_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_bounds: torch.Tensor,
    window: int,
    block_size: int,
    page_size: int,
    ctx_len: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Optional[tuple]:
    """``modeling_dspark._window_mla_block_fixup``'s operands in one launch: the page table rows
    from each request's first window page on (``[B, cols]``, the last column repeated past the
    end, in the table's dtype), the lengths and the per-query key bounds less the skipped positions
    (int32).

    Request ``b`` skips ``start_b = ceil((max(0, ctx_b - window) // page_size - cp_rank)^+ /
    cp_size)`` pages of its (rank-local) table, ``ctx_b`` = ``ctx_len[b]`` or ``seq_lens[b] -
    block_size``. None when the operands are not the layout the decode takes (``[B, cols]`` integer
    rows with unit column stride, ``[B]`` int32 lengths, ``[B * block_size]`` contiguous int32 key
    bounds, batch the slow axis), so the caller keeps the eager chain."""
    if (
        page_tables.dim() != 2
        or page_tables.dtype not in (torch.int32, torch.int64)
        or page_tables.stride(1) != 1
    ):
        return None
    batch, cols = page_tables.shape
    if (
        seq_lens.shape != (batch,)
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
        or kv_bounds.shape != (batch * block_size,)
        or kv_bounds.dtype != torch.int32
        or not kv_bounds.is_contiguous()
        or (ctx_len is not None and (ctx_len.shape != (batch,) or not ctx_len.is_contiguous()))
        or cols == 0
    ):
        return None
    table_out = torch.empty((batch, cols), dtype=page_tables.dtype, device=page_tables.device)
    seq_lens_out = torch.empty_like(seq_lens)
    kv_bounds_out = torch.empty_like(kv_bounds)
    if batch == 0:
        return table_out, seq_lens_out, kv_bounds_out
    grid = (batch, triton.cdiv(cols, _WINDOW_BLOCK_COLS))
    _window_block_fixup_kernel[grid](
        page_tables,
        page_tables.stride(0),
        table_out,
        seq_lens,
        seq_lens_out,
        kv_bounds,
        kv_bounds_out,
        ctx_len if ctx_len is not None else seq_lens,
        cols,
        window,
        page_size,
        cp_size,
        cp_rank,
        BLOCK=block_size,
        BLOCK_Q=triton.next_power_of_2(max(block_size, 1)),
        BLOCK_COLS=_WINDOW_BLOCK_COLS,
        HAS_CTX_LEN=ctx_len is not None,
        num_warps=4,
    )
    return table_out, seq_lens_out, kv_bounds_out
