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
"""Single-launch per-step attention-metadata update for the MTP draft loop.

After each draft step the native one-model loop (``Eagle3OneModelWorker.
_forward_linear_draft_loop``) advances the next step's position ids and the
KV lengths, then calls ``DSAtrtllmAttentionMetadata.update_for_spec_dec``,
which rebuilds the DSA decode metadata: the 2-D KV-length view, three DeepGEMM
paged-MQA-logits schedules (each computed into a fresh tensor and copied into
its persistent buffer), the expanded KV lengths for the CuTe DSL logits path,
and the fused decode metadata (request index, indexer K-cache slot mappings and
generation indptrs). That is about twelve small launches per step.

``draft_step_update`` performs the same update with one Triton launch plus the
native host-side bookkeeping. Every device tensor it writes holds exactly the
value the native chain writes. The schedule is the DeepGEMM SM100 metadata
partition (``deep_gemm/scheduler/sm100_paged_mqa_logits.cuh``): request work
``ceil(ctx / SPLIT_KV) * next_n``, SM ``i`` starts at work unit
``i * q + min(i, rem)`` and records ``(request_q_token_start,
work_in_request // next_n)``, with a ``(num_q_tokens, 0)`` sentinel past the
last request. ``SPLIT_KV`` is confirmed against the native op once per process
(``schedule_self_check``); without a match the native path runs.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

_SCHEDULE_BLOCK_KV = 64
_SPLIT_KV_CANDIDATES = (64, 128, 256, 512)
_MAX_BATCH = 64
_NUM_WARPS = 4


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
    splits = (ctx + SPLIT_KV - 1) // SPLIT_KV
    work = tl.where(rmask, splits * NEXT_N, 0)
    prefix = tl.cumsum(work, 0)
    total = tl.sum(work, 0)
    q = total // NUM_SMS
    rem = total % NUM_SMS
    sm = tl.arange(0, BLOCK_SM)
    sm_mask = sm <= NUM_SMS
    w = sm * q + tl.minimum(sm, rem)
    # First request whose inclusive prefix exceeds w == count of prefixes <= w
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
    seq_lens_ptr,  # int32 [>= B] (out when STEP0, else in)
    num_accepted_ptr,  # int32 [B] (STEP0)
    pos_in_ptr,  # int32/int64 position ids of this step (HAS_POS)
    pos_out_ptr,  # same dtype [B] (out, HAS_POS)
    gather_ptr,  # int64 [B] rows of pos_in to advance (STEP0 and HAS_POS)
    kv2d_ptr,  # int32 [max_seqs, cap] (out)
    kv2d_stride,
    sched_ptr,  # int32 [(NUM_SMS + 1) * 2] (out)
    sched_full_ptr,  # int32 [(NUM_SMS + 1) * 2] (out, HAS_FULL)
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
    STEP0: tl.constexpr,
    HAS_POS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_FULL: tl.constexpr,
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
    kv_old = tl.load(kv_lens_ptr + offs_b, mask=mask_b, other=0)
    if STEP0:
        accepted = tl.load(num_accepted_ptr + offs_b, mask=mask_b, other=0)
        kv = kv_old - (DRAFT_LEN - accepted)
        seq = tl.full((BLOCK_B,), 1, tl.int32)
        tl.store(seq_lens_ptr + offs_b, seq, mask=mask_b)
    else:
        kv = kv_old + 1
        seq = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=0)
    tl.store(kv_lens_ptr + offs_b, kv, mask=mask_b)
    kv = tl.where(mask_b, kv, 0)
    seq = tl.where(mask_b, seq, 0)

    # ----- next step's position ids ------------------------------------------
    if HAS_POS:
        if STEP0:
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
    if HAS_FULL:
        _write_schedule(kv, mask_b, B, sched_full_ptr, WIDTH, NUM_SMS, SPLIT_KV, BLOCK_SM)
    if EXPAND > 1:
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < B * EXPAND
        # repeat_interleave from registers (the updated lengths are not
        # re-read from memory within this launch).
        owner = (offs_e[:, None] // EXPAND) == offs_b[None, :]
        kv_e = tl.sum(tl.where(owner, kv[None, :], 0), 1)
        tl.store(kv_exp_ptr + offs_e, kv_e, mask=mask_e)
        _write_schedule(kv_e, mask_e, B * EXPAND, sched_exp_ptr, 1, NUM_SMS, SPLIT_KV, BLOCK_SM)

    # ----- fused DSA decode metadata (same arithmetic as the native kernel) --
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


def paged_mqa_schedule(
    context_lens: torch.Tensor, num_sms: int, split_kv: int, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """draft_loop reproduction of ``get_paged_mqa_logits_metadata`` (non-varlen).

    Args:
        context_lens: int32 ``[num_requests, next_n]`` contiguous CUDA tensor.
        num_sms: SM count the schedule is balanced over.
        split_kv: KV split length of the DeepGEMM scheduler.
        out: optional int32 ``[num_sms + 1, 2]`` output.

    Returns:
        int32 ``[num_sms + 1, 2]`` per-SM ``(q_token_start, kv_split)`` starts.
    """
    num_requests, next_n = context_lens.shape
    if out is None:
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
        num_warps=_NUM_WARPS,
    )
    return out


_SPLIT_KV_BY_SMS: dict = {}


def schedule_self_check(num_sms: int, device: torch.device) -> Optional[int]:
    """Return the DeepGEMM ``SPLIT_KV`` reproduced exactly for ``num_sms``, or None.

    Compares ``paged_mqa_schedule`` with the native
    ``get_paged_mqa_logits_metadata(lens, 64, num_sms)`` on fixed probes (1-D and
    2-D lengths, empty, short, block-boundary and near-2^20 contexts). Runs
    eagerly once per process and SM count; must not be called during capture.
    """
    if num_sms in _SPLIT_KV_BY_SMS:
        return _SPLIT_KV_BY_SMS[num_sms]
    if torch.cuda.is_current_stream_capturing():
        # The probe synchronizes; without an eager result the native path runs.
        return None
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    probes = []
    lengths = [0, 1, 63, 64, 65, 2047, 2048, 2049, 32768, 131072, 391061, 491067, (1 << 20) + 14]
    for length in lengths:
        for next_n in (1, 6):
            probes.append(torch.full((1, next_n), length, dtype=torch.int32))
    gen = torch.Generator().manual_seed(8)
    for num_requests in (2, 3, 8):
        for next_n in (1, 2, 6):
            lens = torch.randint(0, 600000, (num_requests, 1), generator=gen, dtype=torch.int32)
            probes.append(lens.expand(-1, next_n).contiguous())
    found = None
    for split_kv in _SPLIT_KV_CANDIDATES:
        ok = True
        for lens in probes:
            lens = lens.to(device)
            native = get_paged_mqa_logits_metadata(lens, _SCHEDULE_BLOCK_KV, num_sms)
            if not torch.equal(native, paged_mqa_schedule(lens, num_sms, split_kv)):
                ok = False
                break
        if ok:
            found = split_kv
            break
    _SPLIT_KV_BY_SMS[num_sms] = found
    return found


@dataclass
class DraftStepResult:
    """What the draft_loop step update produced for the draft loop."""

    position_ids: Optional[torch.Tensor]


def step_ineligibility(md, batch_size: int, num_contexts: int) -> Optional[str]:
    """Return why the fused step update cannot replace the native chain, or None.

    Checked on the host before any device work. The supported domain is the
    pure-decode DSA metadata configuration of the GLM-5.2 GEN server: no
    context rows, the fused native decode-metadata path, uncompressed indexer
    lengths, no expanded MTP buffers, no ragged verification, no FP4/FP8-DS MLA
    state and no dense top-k fallback.
    """
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
        _fused_dsa_meta_enabled,
    )

    if not isinstance(md, DSAtrtllmAttentionMetadata):
        return "metadata is not DSAtrtllmAttentionMetadata"
    cls = type(md)
    for name in ("update_for_spec_dec", "on_update_kv_lens", "on_update"):
        if getattr(cls, name) is not getattr(DSAtrtllmAttentionMetadata, name):
            return f"metadata overrides {name}"
    from tensorrt_llm._torch.attention.backends.sparse.dsa import is_dsa_cache_manager

    # With a separate draft KV cache the loop has rebound the manager and the
    # indexer buffers to the draft ones; the update reads them at call time.
    if md.kv_cache_manager is None or not is_dsa_cache_manager(md.kv_cache_manager):
        return "KV cache manager is not a DSA cache manager"
    if num_contexts != 0 or md.num_contexts != 0:
        return "context rows present"
    if batch_size <= 0 or batch_size > _MAX_BATCH or batch_size != md.num_seqs:
        return f"batch size {batch_size} outside the fused domain"
    if not _fused_dsa_meta_enabled():
        return "native fused DSA metadata disabled"
    if md.use_fp8_ds_mla or getattr(md, "fp4_mla_state", None) is not None:
        return "FP8-DS/FP4 MLA metadata state"
    if md.enable_ragged_verification or md.is_ragged_verify:
        return "ragged verification"
    if md._indexer_compress_ratio > 1:
        return "compressed indexer lengths"
    if getattr(md, "use_expanded_buffers_for_mtp", True):
        return "expanded MTP buffers"
    if md.max_draft_tokens <= 0:
        return "no draft tokens"
    if getattr(md, "skip_indexer_for_gen_reqs", True) or getattr(
        md, "skip_indexer_for_ctx_reqs", True
    ):
        return "dense top-k fallback active"
    if md.needs_gvr_prior and batch_size * md.gen_token_stride >= 2 * md.num_sms:
        return "GVR row reorder active"
    if md.kv_lens_cuda.dtype != torch.int32 or md.seq_lens_cuda.dtype != torch.int32:
        return "unexpected length dtype"
    if md.indexer_k_cache_block_offsets.dtype != torch.int32:
        return "unexpected block-offset dtype"
    if schedule_self_check(md.num_sms, md.kv_lens_cuda.device) is None:
        return "DeepGEMM schedule not reproduced"
    return None


def draft_step_update(
    md,
    step: int,
    batch_size: int,
    num_contexts: int,
    runtime_draft_len: int,
    num_accepted_tokens: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    gather_ids: Optional[torch.Tensor],
) -> Optional[DraftStepResult]:
    """Fused replacement of one draft step's metadata update, or None (native).

    Reproduces, for draft step ``step``, the native loop tail: the next step's
    position ids (``position_ids[gather_ids] + 1`` at step 0,
    ``position_ids[:batch_size] + 1`` later; skipped when ``position_ids`` is
    None), the step-0 switch to one query token per request, the KV-length
    update and ``update_for_spec_dec``.

    Args:
        md: the draft loop's attention metadata.
        step: draft step index.
        batch_size: number of requests.
        num_contexts: context requests at the start of the loop.
        runtime_draft_len: draft length K of this iteration.
        num_accepted_tokens: int32 ``[batch_size]`` accepted counts (step 0).
        position_ids: 1-D position ids of this step, or None on the last step.
        gather_ids: int64 ``[batch_size]`` rows of ``position_ids`` (step 0).

    Returns:
        ``DraftStepResult`` when the fused update ran, None when the caller
        must run the native update (no state has been modified then).
    """
    if step_ineligibility(md, batch_size, num_contexts) is not None:
        return None
    if step == 0:
        if (
            num_accepted_tokens is None
            or num_accepted_tokens.dtype != torch.int32
            or not num_accepted_tokens.is_contiguous()
        ):
            return None
        if position_ids is not None and (
            gather_ids is None or gather_ids.dtype != torch.int64 or not gather_ids.is_contiguous()
        ):
            return None
    if position_ids is not None and (position_ids.dim() != 1 or not position_ids.is_contiguous()):
        return None

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    if step == 0:
        # Host side of the native step-0 switch to one query token per request.
        md._seq_lens[:batch_size].fill_(1)
        md.on_update()
        if hasattr(md, "host_request_types"):
            md.host_request_types[: md.num_contexts].fill_(1)
        md.num_contexts = 0

    kv_mgr = md.kv_cache_manager
    index_head_dim = kv_mgr.index_head_dim
    use_fp4 = getattr(kv_mgr, "use_fp4", False)
    data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
    scale_size = index_head_dim // kv_mgr.quant_block_size * 4
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
        num_accepted_tokens if step == 0 else dummy,
        position_ids if has_pos else dummy,
        pos_out if has_pos else dummy,
        gather_ids if (step == 0 and has_pos) else dummy,
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
        DRAFT_LEN=runtime_draft_len,
        STEP0=step == 0,
        HAS_POS=has_pos,
        WIDTH=width,
        HAS_FULL=True,
        EXPAND=expand,
        NUM_SMS=num_sms,
        SPLIT_KV=_SPLIT_KV_BY_SMS[num_sms],
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
        num_warps=_NUM_WARPS,
    )

    # Host side of update_for_spec_dec -> on_update_kv_lens (no device work in
    # the eligible domain, checked above).
    TrtllmAttentionMetadata.update_for_spec_dec(md)
    md.max_ctx_kv_len = 0
    md.num_ctx_cached_tokens = 0
    md.max_gen_seq_len = 1
    TrtllmAttentionMetadata.on_update_kv_lens(md)
    md._invalidate_pool_view_cache()
    md._fused_dsa_meta_armed = True
    md.gen_indexer_kv_lens_cuda_runtime = md.get_indexer_kv_lens(
        md.kv_lens_cuda[md.num_contexts : md.num_seqs]
    )
    md._compute_kv_lens_row_reorder()
    md.prepare_dense_topk_indices(md.kv_lens_cuda, device=True)
    if step == 0:
        md.use_spec_decoding = False
    return DraftStepResult(position_ids=pos_out)
