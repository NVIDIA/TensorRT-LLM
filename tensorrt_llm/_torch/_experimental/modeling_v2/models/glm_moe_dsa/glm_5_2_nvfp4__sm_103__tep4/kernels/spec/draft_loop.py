# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the MTP draft loop's bookkeeping (`spec_worker.py`).

Ported from the glm_moe_dsa draft loop (`models/glm_moe_dsa/shared/kernels/draft_loop/`,
TensorRT-LLM PR #20524 at d8735393c2): the acceptance, the finalize stack and the per-step DSA
metadata update, each one launch in place of the built-in worker's chains of small torch ops.
Every output holds the value the built-in chain writes; the DeepGEMM schedule's split length
is confirmed against the native op once per process (`schedule_split_kv`).

Changes from the source: the metadata update has a third mode, the refresh the loop runs
before draft step 0 (`MODE_REFRESH`: the lengths stay, every derived buffer is rebuilt), and
the row gather and draft embedding kernels are not ported (the torch gather is one launch
too, and this target embeds inside its MTP layer).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The DeepGEMM paged-MQA-logits schedule's KV block (`indexer._DG_SCHEDULE_BLOCK_KV`).
SCHEDULE_BLOCK_KV = 64
_SPLIT_KV_CANDIDATES = (64, 128, 256, 512)
#: The fused domain: requests per batch and draft tokens per request.
MAX_BATCH = 64
MAX_DRAFT_LEN = 8

#: `draft_step` modes: rebuild from unchanged lengths (before draft step 0), the step-0
#: switch to one query token per request, and the +1 advance of steps 1...
MODE_REFRESH = 0
MODE_STEP0 = 1
MODE_ADVANCE = 2


@triton.jit
def _accept_kernel(
    target_ptr,  # int32/int64 [B * (K + 1)] sampled target tokens
    draft_ptr,  # int32 [B * K] previous draft tokens
    batch_idx_ptr,  # int32 [>= B] spec_metadata.batch_indices_cuda
    counter_ptr,  # int64 [1] forced-acceptance draw counter (FORCE_MODE 2)
    pool_ptr,  # float32 [POOL] forced-acceptance uniform pool (FORCE_MODE 2)
    accepted_ptr,  # int32 [B, K + 1] (out)
    num_accepted_ptr,  # int32 [B] (out)
    input_ids_ptr,  # int32 [B * (K + 1)] draft-0 input ids (out)
    gather_ptr,  # int64 [B] step-0 last-accepted rows (out)
    frac,  # float32 fractional forced acceptance
    counter_stride,
    slot_stride,
    pool_mask,
    B: tl.constexpr,
    K: tl.constexpr,
    FORCE_MODE: tl.constexpr,
    BASE_TOTAL: tl.constexpr,
    MAX_TOTAL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    offs_k = tl.arange(0, BLOCK_K)
    mask_b = offs_b < B
    row_mask = mask_b[:, None] & (offs_k[None, :] < K + 1)
    flat = offs_b[:, None] * (K + 1) + offs_k[None, :]
    target = tl.load(target_ptr + flat, mask=row_mask, other=0)
    target32 = target.to(tl.int32)
    tl.store(accepted_ptr + flat, target32, mask=row_mask)
    tl.store(input_ids_ptr + flat, target32, mask=row_mask)

    draft_mask = mask_b[:, None] & (offs_k[None, :] < K)
    draft = tl.load(draft_ptr + offs_b[:, None] * K + offs_k[None, :], mask=draft_mask, other=0)
    mismatch = draft_mask & (draft.to(tl.int64) != target.to(tl.int64))
    # Leading matches == sum(cumprod(draft == target)).
    num_accepted = 1 + tl.min(tl.where(mismatch, offs_k[None, :], K), 1)

    if FORCE_MODE == 1:
        num_accepted = tl.full((BLOCK_B,), BASE_TOTAL, tl.int32)
    elif FORCE_MODE == 2:
        counter = tl.load(counter_ptr) + 1
        tl.store(counter_ptr, counter)
        index = (counter * counter_stride + offs_b.to(tl.int64) * slot_stride) & pool_mask
        draw = tl.load(pool_ptr + index, mask=mask_b, other=1.0)
        extra = (draw < frac).to(tl.int32)
        num_accepted = tl.minimum(BASE_TOTAL + extra, MAX_TOTAL)
    tl.store(num_accepted_ptr + offs_b, num_accepted, mask=mask_b)

    batch_idx = tl.load(batch_idx_ptr + offs_b, mask=mask_b, other=0)
    rows = (batch_idx * (K + 1)).to(tl.int64) + num_accepted.to(tl.int64) - 1
    tl.store(gather_ptr + offs_b, rows, mask=mask_b)


def accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    batch_indices: torch.Tensor,
    counter: torch.Tensor,
    pool: torch.Tensor,
    *,
    batch_size: int,
    draft_len: int,
    force_mode: int,
    base_total: int,
    frac: float,
    counter_stride: int,
    slot_stride: int,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict acceptance with the forced-acceptance override, in one launch.

    Returns `(accepted [B, K+1] int32, num_accepted [B] int32, input_ids [B * (K+1)] int32,
    gather_ids [B] int64)`: the built-in `_sample_and_accept_draft_tokens_base` outputs, the
    draft-0 input ids (the accepted tokens, flattened, in a buffer of their own) and the
    draft loop's step-0 rows `batch_indices * (K + 1) + num_accepted - 1`. `force_mode` 0 is off, 1 the integer
    override, 2 the fractional one drawing from `pool` at `counter` (bumped in place)."""
    device = target_tokens.device
    accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    input_ids = torch.empty((batch_size * (draft_len + 1),), dtype=torch.int32, device=device)
    gather_ids = torch.empty((batch_size,), dtype=torch.int64, device=device)
    _accept_kernel[(1,)](
        target_tokens,
        draft_tokens,
        batch_indices,
        counter,
        pool,
        accepted,
        num_accepted,
        input_ids,
        gather_ids,
        float(frac),
        counter_stride,
        slot_stride,
        pool_size - 1,
        B=batch_size,
        K=draft_len,
        FORCE_MODE=force_mode,
        BASE_TOTAL=base_total,
        MAX_TOTAL=draft_len + 1,
        BLOCK_B=max(2, triton.next_power_of_2(batch_size)),
        BLOCK_K=max(2, triton.next_power_of_2(draft_len + 1)),
        # One warp: every lane reads the draw counter before it is stored.
        num_warps=1,
    )
    return accepted, num_accepted, input_ids, gather_ids


@triton.jit
def _finalize_kernel(
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,  # int32 [B] draft tokens of steps 0..K-1
    accepted_ptr,  # int32 [B, ACC_COLS]
    num_accepted_ptr,  # int32 [B]
    batch_idx_ptr,  # int32 [>= B]
    next_draft_ptr,  # int32 [B, K] (out)
    next_new_ptr,  # int32 [B, K + 1] (out)
    B: tl.constexpr,
    K: tl.constexpr,
    ACC_COLS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask = offs_b < B
    batch_idx = tl.load(batch_idx_ptr + offs_b, mask=mask, other=0).to(tl.int64)
    last = tl.load(num_accepted_ptr + offs_b, mask=mask, other=1).to(tl.int64) - 1
    bonus = tl.load(accepted_ptr + batch_idx * ACC_COLS + last, mask=mask, other=0)
    tl.store(next_new_ptr + offs_b * (K + 1), bonus, mask=mask)
    for k in tl.static_range(K):
        if k == 0:
            tok = tl.load(d0 + offs_b, mask=mask)
        elif k == 1:
            tok = tl.load(d1 + offs_b, mask=mask)
        elif k == 2:
            tok = tl.load(d2 + offs_b, mask=mask)
        elif k == 3:
            tok = tl.load(d3 + offs_b, mask=mask)
        elif k == 4:
            tok = tl.load(d4 + offs_b, mask=mask)
        elif k == 5:
            tok = tl.load(d5 + offs_b, mask=mask)
        elif k == 6:
            tok = tl.load(d6 + offs_b, mask=mask)
        else:
            tok = tl.load(d7 + offs_b, mask=mask)
        tl.store(next_draft_ptr + offs_b * K + k, tok, mask=mask)
        tl.store(next_new_ptr + offs_b * (K + 1) + 1 + k, tok, mask=mask)


def finalize(
    draft_tokens: list[torch.Tensor],
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    batch_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`torch.stack(draft_tokens, 1)` and the built-in `_prepare_next_new_tokens`, in one
    launch: `(next_draft [B, K] int32, next_new [B, K+1] int32)`."""
    batch_size, draft_len = num_accepted.shape[0], len(draft_tokens)
    device = accepted.device
    next_draft = torch.empty((batch_size, draft_len), dtype=torch.int32, device=device)
    next_new = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
    ptrs = list(draft_tokens) + [draft_tokens[0]] * (MAX_DRAFT_LEN - draft_len)
    _finalize_kernel[(1,)](
        *ptrs,
        accepted,
        num_accepted,
        batch_indices,
        next_draft,
        next_new,
        B=batch_size,
        K=draft_len,
        ACC_COLS=accepted.shape[1],
        BLOCK_B=max(2, triton.next_power_of_2(batch_size)),
        num_warps=1,
    )
    return next_draft, next_new


@triton.jit
def _write_schedule(
    ctx,  # int32 [BLOCK_R] context length per logical request
    rmask,  # bool [BLOCK_R] prefix mask r < num_requests
    num_requests,
    sched_ptr,  # int32 [(NUM_SMS + 1) * 2] (out)
    NEXT_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    BLOCK_SM: tl.constexpr,
):
    """DeepGEMM's SM100 paged-MQA-logits metadata (`sm100_paged_mqa_logits.cuh`): request
    work `ceil(ctx / SPLIT_KV) * next_n`; SM `i` starts at work unit `i * q + min(i, rem)`
    and records `(request_q_token_start, work_in_request // next_n)`, with a
    `(num_q_tokens, 0)` sentinel past the last request."""
    splits = (ctx + SPLIT_KV - 1) // SPLIT_KV
    work = tl.where(rmask, splits * NEXT_N, 0)
    prefix = tl.cumsum(work, 0)
    total = tl.sum(work, 0)
    q = total // NUM_SMS
    rem = total % NUM_SMS
    sm = tl.arange(0, BLOCK_SM)
    sm_mask = sm <= NUM_SMS
    w = sm * q + tl.minimum(sm, rem)
    # The first request whose inclusive prefix exceeds w == the count of prefixes <= w
    # (prefix is non-decreasing over the live requests).
    owned = (prefix[None, :] <= w[:, None]) & rmask[None, :]
    request = tl.sum(owned.to(tl.int32), 1)
    work_before = tl.max(tl.where(owned, prefix[None, :], 0), 1)
    in_range = request < num_requests
    q_token = tl.where(in_range, request * NEXT_N, num_requests * NEXT_N)
    kv_split = tl.where(in_range, (w - work_before) // NEXT_N, 0)
    tl.store(sched_ptr + 2 * sm, q_token, mask=sm_mask)
    tl.store(sched_ptr + 2 * sm + 1, kv_split, mask=sm_mask)


@triton.jit
def _draft_step_kernel(
    kv_lens_ptr,  # int32 [>= B] (in/out)
    seq_lens_ptr,  # int32 [>= B] (out in MODE_STEP0, else in)
    num_accepted_ptr,  # int32 [B] (MODE_STEP0)
    pos_in_ptr,  # int32/int64 position ids of this step (HAS_POS)
    pos_out_ptr,  # same dtype [B] (out, HAS_POS)
    gather_ptr,  # int64 [B] rows of pos_in to advance (MODE_STEP0 and HAS_POS)
    kv2d_ptr,  # int32 [max_seqs, cap] (out)
    kv2d_stride,
    sched_ptr,  # int32 [(NUM_SMS + 1) * 2] (out)
    sched_full_ptr,  # int32 [(NUM_SMS + 1) * 2] (out)
    kv_exp_ptr,  # int32 [>= B * EXPAND] (out, EXPAND > 1)
    sched_exp_ptr,  # int32 [(NUM_SMS + 1) * 2] (out, EXPAND > 1)
    block_offsets_ptr,  # int32 [max_seqs, max_blocks]
    block_offsets_stride_0,
    req_idx_ptr,  # int32 [>= num_tokens] (out)
    slot_fp8_ptr,  # int64 [>= num_tokens] (out)
    slot_scale_ptr,  # int64 [>= num_tokens] (out)
    gen_kv_indptr_ptr,  # int64 [>= B + 1] (out)
    gen_cached_indptr_ptr,  # int64 [>= B + 1] (out)
    num_tokens,
    B: tl.constexpr,
    DRAFT_LEN: tl.constexpr,
    MODE: tl.constexpr,
    HAS_POS: tl.constexpr,
    WIDTH: tl.constexpr,
    EXPAND: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    max_blocks: tl.constexpr,
    block_offsets_stride_1: tl.constexpr,
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_SM: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    # ----- KV lengths and query lengths (the loop's in-place updates) -------
    kv = tl.load(kv_lens_ptr + offs_b, mask=mask_b, other=0)
    # The lengths are updated in place, and each warp holds its own copy of them: without
    # this barrier a warp that loads after another warp stored applies the update twice.
    tl.debug_barrier()
    if MODE == 1:
        accepted = tl.load(num_accepted_ptr + offs_b, mask=mask_b, other=0)
        kv = kv - (DRAFT_LEN - accepted)
        seq = tl.full((BLOCK_B,), 1, tl.int32)
        tl.store(seq_lens_ptr + offs_b, seq, mask=mask_b)
        tl.store(kv_lens_ptr + offs_b, kv, mask=mask_b)
    else:
        seq = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=0)
        if MODE == 2:
            kv = kv + 1
            tl.store(kv_lens_ptr + offs_b, kv, mask=mask_b)
    kv = tl.where(mask_b, kv, 0)
    seq = tl.where(mask_b, seq, 0)

    # ----- next step's position ids ------------------------------------------
    if HAS_POS:
        if MODE == 1:
            rows = tl.load(gather_ptr + offs_b, mask=mask_b, other=0)
        else:
            rows = offs_b.to(tl.int64)
        pos = tl.load(pos_in_ptr + rows, mask=mask_b, other=0)
        tl.store(pos_out_ptr + offs_b, pos + 1, mask=mask_b)

    # ----- 2-D KV lengths and the paged-MQA-logits schedules -----------------
    offs_w = tl.arange(0, BLOCK_W)
    kv2d_mask = mask_b[:, None] & (offs_w[None, :] < WIDTH)
    tl.store(
        kv2d_ptr + offs_b[:, None] * kv2d_stride + offs_w[None, :],
        kv[:, None] + tl.zeros((BLOCK_B, BLOCK_W), tl.int32),
        mask=kv2d_mask,
    )
    _write_schedule(kv, mask_b, B, sched_ptr, 1, NUM_SMS, SPLIT_KV, BLOCK_SM)
    _write_schedule(kv, mask_b, B, sched_full_ptr, WIDTH, NUM_SMS, SPLIT_KV, BLOCK_SM)
    if EXPAND > 1:
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < B * EXPAND
        # repeat_interleave from registers.
        owner = (offs_e[:, None] // EXPAND) == offs_b[None, :]
        kv_e = tl.sum(tl.where(owner, kv[None, :], 0), 1)
        tl.store(kv_exp_ptr + offs_e, kv_e, mask=mask_e)
        _write_schedule(kv_e, mask_e, B * EXPAND, sched_exp_ptr, 1, NUM_SMS, SPLIT_KV, BLOCK_SM)

    # ----- fused DSA decode metadata (`dsa/kernels.py`, the same arithmetic) --
    kv64 = kv.to(tl.int64)
    seq64 = seq.to(tl.int64)
    tl.store(gen_kv_indptr_ptr, tl.full((), 0, tl.int64))
    tl.store(gen_cached_indptr_ptr, tl.full((), 0, tl.int64))
    tl.store(gen_kv_indptr_ptr + 1 + offs_b, tl.cumsum(kv64, 0), mask=mask_b)
    tl.store(gen_cached_indptr_ptr + 1 + offs_b, tl.cumsum(kv64 - seq64, 0), mask=mask_b)

    flat_start = tl.cumsum(seq64, 0) - seq64
    start_pos = kv64 - seq64
    offs_t = tl.arange(0, BLOCK_T).to(tl.int64)
    g = flat_start[:, None] + offs_t[None, :]
    mask_t = mask_b[:, None] & (offs_t[None, :] < seq64[:, None]) & (g < num_tokens)
    tl.store(req_idx_ptr + g, offs_b[:, None] + tl.zeros((BLOCK_B, BLOCK_T), tl.int32), mask=mask_t)
    gpos = start_pos[:, None] + offs_t[None, :]
    blk = gpos // tokens_per_block
    pos_in_blk = gpos % tokens_per_block
    pos_in_blk = tl.where(pos_in_blk < 0, pos_in_blk + tokens_per_block, pos_in_blk)
    blk = tl.minimum(tl.maximum(blk, 0), max_blocks - 1)
    bid = tl.load(
        block_offsets_ptr
        + offs_b[:, None].to(tl.int64) * block_offsets_stride_0
        + blk * block_offsets_stride_1,
        mask=mask_t,
        other=0,
    ).to(tl.int64)
    tl.store(slot_fp8_ptr + g, bid * block_stride + pos_in_blk * data_bytes_per_token, mask=mask_t)
    tl.store(
        slot_scale_ptr + g,
        bid * block_stride + scale_base_offset + pos_in_blk * scale_size,
        mask=mask_t,
    )


def draft_step(
    md,
    *,
    mode: int,
    batch_size: int,
    draft_len: int,
    split_kv: int,
    num_accepted: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    gather_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """The device side of one metadata update of the draft loop, in one launch.

    `md` is the loop's `DSAtrtllmAttentionMetadata` in the fused domain
    (`spec_worker.step_ineligibility`); every tensor written is the one the built-in
    `on_update_kv_lens` (with the loop's length edits before it) writes. With
    `position_ids`, also returns the next step's positions (`position_ids[gather_ids] + 1`
    in `MODE_STEP0`, `position_ids[:B] + 1` in `MODE_ADVANCE`)."""
    kv_mgr = md.kv_cache_manager
    index_head_dim = kv_mgr.index_head_dim
    use_fp4 = getattr(kv_mgr, "use_fp4", False)
    data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
    scale_size = index_head_dim // md.indexer_quant_block_size * 4
    tokens_per_block = md._tokens_per_block
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    width = min(md.gen_token_stride, md.kv_lens_cuda_2d.shape[1])
    expand = md.dsl_expand_factor if md.expand_for_dsl else 1
    num_sms = md.num_sms
    block_offsets = md.indexer_k_cache_block_offsets
    has_pos = position_ids is not None
    pos_out = torch.empty_like(position_ids[:batch_size]) if has_pos else None
    dummy = md.kv_lens_cuda
    _draft_step_kernel[(1,)](
        md.kv_lens_cuda,
        md.seq_lens_cuda,
        num_accepted if mode == MODE_STEP0 else dummy,
        position_ids if has_pos else dummy,
        pos_out if has_pos else dummy,
        gather_ids if (mode == MODE_STEP0 and has_pos) else dummy,
        md.kv_lens_cuda_2d,
        md.kv_lens_cuda_2d.stride(0),
        md.scheduler_metadata_buffer,
        md.scheduler_metadata_buffer_full_next_n,
        md.kv_lens_expanded_cuda if expand > 1 else dummy,
        md.scheduler_metadata_buffer_expanded if expand > 1 else dummy,
        block_offsets,
        block_offsets.stride(0),
        md.req_idx_per_token,
        md.slot_mapping_fp8,
        md.slot_mapping_scale,
        md.gen_kv_indptr,
        md.gen_cached_token_indptr,
        md.num_tokens,
        B=batch_size,
        DRAFT_LEN=draft_len,
        MODE=mode,
        HAS_POS=has_pos,
        WIDTH=width,
        EXPAND=expand,
        NUM_SMS=num_sms,
        SPLIT_KV=split_kv,
        max_blocks=block_offsets.shape[1],
        block_offsets_stride_1=block_offsets.stride(1),
        tokens_per_block=tokens_per_block,
        data_bytes_per_token=data_bytes_per_token,
        scale_size=scale_size,
        block_stride=block_stride,
        scale_base_offset=tokens_per_block * data_bytes_per_token,
        BLOCK_B=max(2, triton.next_power_of_2(batch_size)),
        BLOCK_W=max(2, triton.next_power_of_2(width)),
        BLOCK_E=max(2, triton.next_power_of_2(batch_size * expand)),
        BLOCK_T=max(2, triton.next_power_of_2(1 + md.max_draft_tokens)),
        BLOCK_SM=triton.next_power_of_2(num_sms + 1),
        num_warps=4,
    )
    return pos_out


@triton.jit
def _schedule_probe_kernel(
    ctx_ptr,
    sched_ptr,
    num_requests,
    NEXT_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_SM: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_R)
    rmask = offs < num_requests
    ctx = tl.load(ctx_ptr + offs * NEXT_N + NEXT_N - 1, mask=rmask, other=0)
    _write_schedule(ctx, rmask, num_requests, sched_ptr, NEXT_N, NUM_SMS, SPLIT_KV, BLOCK_SM)


def paged_mqa_schedule(context_lens: torch.Tensor, num_sms: int, split_kv: int) -> torch.Tensor:
    """`get_paged_mqa_logits_metadata(context_lens, 64, num_sms)` for int32 contiguous
    `[num_requests, next_n]` lengths equal along a row: int32 `[num_sms + 1, 2]`."""
    num_requests, next_n = context_lens.shape
    out = torch.empty((num_sms + 1, 2), dtype=torch.int32, device=context_lens.device)
    _schedule_probe_kernel[(1,)](
        context_lens,
        out,
        num_requests,
        NEXT_N=next_n,
        NUM_SMS=num_sms,
        SPLIT_KV=split_kv,
        BLOCK_R=max(2, triton.next_power_of_2(num_requests)),
        BLOCK_SM=triton.next_power_of_2(num_sms + 1),
        num_warps=4,
    )
    return out


_SPLIT_KV_BY_SMS: dict[int, int | None] = {}


def schedule_split_kv(num_sms: int, device: torch.device) -> int | None:
    """The DeepGEMM `SPLIT_KV` `paged_mqa_schedule` reproduces exactly for `num_sms`, or
    None (then the draft loop keeps the built-in update).

    Compared with the native `get_paged_mqa_logits_metadata` on fixed probes (1-D and 2-D
    lengths, empty, short, block-boundary and near-2^20 contexts), eagerly, once per process
    and SM count. Under stream capture with no eager result yet, None."""
    if num_sms in _SPLIT_KV_BY_SMS:
        return _SPLIT_KV_BY_SMS[num_sms]
    if torch.cuda.is_current_stream_capturing():
        return None
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    probes = []
    lengths = [0, 1, 63, 64, 65, 2047, 2048, 2049, 32768, 131072, 391061, 491067, (1 << 20) + 14]
    for length in lengths:
        for next_n in (1, 6):
            probes.append(torch.full((1, next_n), length, dtype=torch.int32))
    gen = torch.Generator().manual_seed(8)
    for num_requests in (2, 3, 8, 33, 64):
        for next_n in (1, 2, 4, 6):
            lens = torch.randint(0, 600000, (num_requests, 1), generator=gen, dtype=torch.int32)
            probes.append(lens.expand(-1, next_n).contiguous())
    found = None
    for split_kv in _SPLIT_KV_CANDIDATES:
        if all(
            torch.equal(
                get_paged_mqa_logits_metadata(lens.to(device), SCHEDULE_BLOCK_KV, num_sms),
                paged_mqa_schedule(lens.to(device), num_sms, split_kv),
            )
            for lens in probes
        ):
            found = split_kv
            break
    _SPLIT_KV_BY_SMS[num_sms] = found
    return found
