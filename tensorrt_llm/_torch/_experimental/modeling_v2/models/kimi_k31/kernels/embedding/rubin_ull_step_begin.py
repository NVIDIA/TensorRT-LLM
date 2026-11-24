# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode step's begin in one launch: the engine's speculative-decoding head prep, the Helix
DSA decode metadata and the flat chain's begin.

Unfolded, a Helix generation step starts with three serial launches: the engine's
``_spec_head_prep_kernel`` (``speculative/spec_head_kernels.py``: the overlap scheduler's position
and KV-length correction and the Helix verify-group buffers), the engine's
``_fused_dsa_helix_decode_metadata_kernel`` (``attention/backends/sparse/dsa/kernels.py``: request
indices, index-key slots, indptrs, per-token lengths and, on SM 10.x, both DeepGEMM schedules,
read from what the prep wrote) and this family's ``_flat_begin_kernel``
(``rubin_ull_flat_begin.py``: the embedding rows and the KDA checkpoint indices, which read nothing
the other two write and are read by neither).

:func:`step_begin` runs them in one launch of grid ``(rows, column blocks + 1)``:

* programs ``(r, c)`` with ``c`` below the column blocks are the begin kernel's programs (the begin
  kernel itself, called with the same program ids);
* program ``(0, column blocks)`` runs the prep kernel's single program, then, after a CTA barrier
  (every write the metadata reads is this CTA's), each program of the metadata kernel in turn:
  the per-sequence outputs, the token blocks, then the two schedules;
* programs ``(r > 0, column blocks)`` do nothing.

The prep kernel, the begin kernel and the schedule function are called as they are; the
metadata kernel's per-program bodies are its own code with the program id as an argument. Every
output is the unfolded launches' bit for bit: integer arithmetic and row copies, each in the
order the unfolded kernel does it.
"""

import inspect

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.sparse.dsa.kernels import (
    _DG_SCHEDULE_BLOCK_R,
    _DG_SM100_SPLIT_KV,
    _dg_sm100_schedule,
    fused_dsa_helix_decode_metadata,
)
from tensorrt_llm._torch.speculative.spec_head_kernels import (
    _spec_head_prep_kernel,
    fused_spec_head_prep,
)

from .rubin_ull_flat_begin import _BLOCK_H, _flat_begin_kernel, folds_embedding


@triton.jit
def _meta_sequences(
    seq_lens_ptr,
    kv_lens_ptr,
    gen_kv_indptr_ptr,
    gen_cached_indptr_ptr,
    kv_lens_2d_ptr,
    kv_lens_2d_stride_0,
    num_seqs,
    width,
    BLOCK_S: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # _fused_dsa_helix_decode_metadata_kernel, program 0: per-sequence outputs.
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < num_seqs
    seq = tl.load(seq_lens_ptr + offs_s, mask=mask_s, other=0)
    kv = tl.load(kv_lens_ptr + offs_s, mask=mask_s, other=0)
    tl.store(gen_kv_indptr_ptr + 1 + offs_s, tl.cumsum(kv.to(tl.int64), 0), mask=mask_s)
    tl.store(gen_cached_indptr_ptr + 1 + offs_s, tl.cumsum((kv - seq).to(tl.int64), 0), mask=mask_s)
    offs_w = tl.arange(0, BLOCK_W)
    kv_2d = kv[:, None] + tl.zeros((BLOCK_S, BLOCK_W), dtype=tl.int32)
    tl.store(
        kv_lens_2d_ptr + offs_s[:, None] * kv_lens_2d_stride_0 + offs_w[None, :],
        kv_2d,
        mask=mask_s[:, None] & (offs_w[None, :] < width),
    )


@triton.jit
def _meta_tokens(
    pid,
    seq_lens_ptr,
    block_offsets_ptr,
    helix_local_slots_ptr,
    helix_kv_bounds_ptr,
    req_idx_per_token_ptr,
    slot_fp8_ptr,
    slot_scale_ptr,
    kv_lens_expanded_ptr,
    block_offsets_stride_0,
    block_offsets_stride_1,
    num_seqs,
    num_tokens,
    max_blocks,
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # _fused_dsa_helix_decode_metadata_kernel, programs 1..: BLOCK_T tokens each.
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < num_seqs
    seq = tl.load(seq_lens_ptr + offs_s, mask=mask_s, other=0)
    t = (pid - 1) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t < num_tokens
    ends = tl.cumsum(seq, 0)
    before = (ends[None, :] <= t[:, None]) & mask_s[None, :]
    req = tl.sum(before.to(tl.int32), axis=1)
    tl.store(req_idx_per_token_ptr + t, req, mask=mask_t)
    local = tl.load(helix_local_slots_ptr + t, mask=mask_t, other=0).to(tl.int64)
    owned = local >= 0
    pos = tl.maximum(local, 0)
    blk = tl.minimum(pos // tokens_per_block, max_blocks - 1)
    in_blk = pos % tokens_per_block
    bid = tl.load(
        block_offsets_ptr
        + req.to(tl.int64) * block_offsets_stride_0
        + blk * block_offsets_stride_1,
        mask=mask_t,
        other=0,
    ).to(tl.int64)
    fp8 = bid * block_stride + in_blk * data_bytes_per_token
    scale = bid * block_stride + scale_base_offset + in_blk * scale_size
    tl.store(slot_fp8_ptr + t, tl.where(owned, fp8, -1), mask=mask_t)
    tl.store(slot_scale_ptr + t, tl.where(owned, scale, -1), mask=mask_t)
    bounds = tl.load(helix_kv_bounds_ptr + t, mask=mask_t, other=1)
    tl.store(kv_lens_expanded_ptr + t, tl.maximum(bounds, 1), mask=mask_t)


@triton.jit
def _step_begin_kernel(
    # the begin (_flat_begin_kernel)
    ids_ptr,
    table_ptr,
    embed_ptr,
    vocab,
    hidden,
    state_ptr,
    state_stride,
    count_ptr,
    count_stride,
    slots_ptr,
    accepted_ptr,
    num_requests,
    header_units,
    slot_units,
    state_units,
    # the prep (_spec_head_prep_kernel)
    pos_ids_ptr,
    pos_offset_ptr,
    num_pos,
    kv_lens_ptr,
    kv_offset_ptr,
    kv_start,
    num_kv,
    helix_pos_ptr,
    helix_slots_ptr,
    helix_bounds_ptr,
    num_helix,
    phys,
    cp_size,
    cp_rank,
    kv_bound_start,
    tokens_per_seq,
    # the metadata (_fused_dsa_helix_decode_metadata_kernel)
    m_seq_lens_ptr,
    m_kv_lens_ptr,
    m_block_offsets_ptr,
    m_helix_local_slots_ptr,
    m_helix_kv_bounds_ptr,
    m_req_idx_per_token_ptr,
    m_slot_fp8_ptr,
    m_slot_scale_ptr,
    m_gen_kv_indptr_ptr,
    m_gen_cached_indptr_ptr,
    m_kv_lens_2d_ptr,
    m_kv_lens_expanded_ptr,
    m_schedule_ptr,
    m_schedule_expanded_ptr,
    m_block_offsets_stride_0,
    m_block_offsets_stride_1,
    m_kv_lens_2d_stride_0,
    m_num_seqs,
    m_num_tokens,
    m_width,
    m_max_blocks,
    m_num_slots,
    # the begin's constants
    TOKENS: tl.constexpr,
    TOKENS_POW2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VEC: tl.constexpr,
    COLUMNS: tl.constexpr,
    # the prep's constants
    HAS_OVERLAP: tl.constexpr,
    HAS_HELIX: tl.constexpr,
    PREP_T: tl.constexpr,
    PREP_S: tl.constexpr,
    # the metadata's constants
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    META_S: tl.constexpr,
    META_T: tl.constexpr,
    META_W: tl.constexpr,
    SCHEDULE: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    if tl.program_id(1) < COLUMNS:
        _flat_begin_kernel(
            ids_ptr,
            table_ptr,
            embed_ptr,
            vocab,
            hidden,
            state_ptr,
            state_stride,
            count_ptr,
            count_stride,
            slots_ptr,
            accepted_ptr,
            num_requests,
            header_units,
            slot_units,
            state_units,
            TOKENS=TOKENS,
            TOKENS_POW2=TOKENS_POW2,
            BLOCK_H=BLOCK_H,
            VEC=VEC,
        )
    elif tl.program_id(0) == 0:
        _spec_head_prep_kernel(
            pos_ids_ptr,
            pos_offset_ptr,
            num_pos,
            kv_lens_ptr,
            kv_offset_ptr,
            kv_start,
            num_kv,
            helix_pos_ptr,
            helix_slots_ptr,
            helix_bounds_ptr,
            num_helix,
            phys,
            cp_size,
            cp_rank,
            kv_bound_start,
            tokens_per_seq,
            HAS_OVERLAP=HAS_OVERLAP,
            HAS_HELIX=HAS_HELIX,
            BLOCK_T=PREP_T,
            BLOCK_S=PREP_S,
        )
        # The metadata reads the prep's KV lengths and Helix slots and bounds.
        tl.debug_barrier()
        _meta_sequences(
            m_seq_lens_ptr,
            m_kv_lens_ptr,
            m_gen_kv_indptr_ptr,
            m_gen_cached_indptr_ptr,
            m_kv_lens_2d_ptr,
            m_kv_lens_2d_stride_0,
            m_num_seqs,
            m_width,
            META_S,
            META_W,
        )
        for pid in range(1, 1 + tl.cdiv(m_num_tokens, META_T)):
            _meta_tokens(
                pid,
                m_seq_lens_ptr,
                m_block_offsets_ptr,
                m_helix_local_slots_ptr,
                m_helix_kv_bounds_ptr,
                m_req_idx_per_token_ptr,
                m_slot_fp8_ptr,
                m_slot_scale_ptr,
                m_kv_lens_expanded_ptr,
                m_block_offsets_stride_0,
                m_block_offsets_stride_1,
                m_num_seqs,
                m_num_tokens,
                m_max_blocks,
                tokens_per_block,
                data_bytes_per_token,
                scale_size,
                block_stride,
                scale_base_offset,
                META_S,
                META_T,
            )
        if SCHEDULE:
            _dg_sm100_schedule(
                m_kv_lens_ptr,
                m_schedule_ptr,
                m_num_seqs,
                m_num_slots,
                False,
                SPLIT_KV,
                BLOCK_SM,
                BLOCK_R,
            )
            _dg_sm100_schedule(
                m_helix_kv_bounds_ptr,
                m_schedule_expanded_ptr,
                m_num_tokens,
                m_num_slots,
                True,
                SPLIT_KV,
                BLOCK_SM,
                BLOCK_R,
            )


_PREP = inspect.signature(fused_spec_head_prep)
_META = inspect.signature(fused_dsa_helix_decode_metadata)


def step_begin(
    prep: dict,
    meta_args: tuple,
    meta_kwargs: dict,
    replay,
    state_indices: torch.Tensor,
    ids: torch.Tensor,
    table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fused_spec_head_prep(**prep)``, then ``fused_dsa_helix_decode_metadata(*meta_args,
    **meta_kwargs)``, and ``rubin_ull_flat_begin.begin_fold(replay, state_indices, ids, table)``,
    in one launch; returns the begin's ``(slots, accepted, embedding rows)``.

    The arguments are the three functions' own, with their checks; ``ids`` and ``table`` must
    be ones :func:`rubin_ull_flat_begin.folds_embedding` accepts, and the metadata's step has at
    least one token (its launch has no empty grid).
    """
    p = _PREP.bind(**prep)
    p.apply_defaults()
    p = p.arguments
    m = _META.bind(*meta_args, **meta_kwargs)
    m.apply_defaults()
    m = m.arguments
    # fused_spec_head_prep's checks and launch constants.
    for t in (p["position_ids"], p["pos_offsets"], p["kv_lens"], p["kv_offsets"]):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    has_overlap, num_helix = p["has_overlap"], p["num_helix"]
    has_helix = num_helix > 0
    helix_positions, helix_slots, helix_bounds = (
        p["helix_positions"],
        p["helix_slots"],
        p["helix_bounds"],
    )
    tokens_per_seq = p["tokens_per_seq"]
    if has_helix:
        for t in (helix_positions, helix_slots, helix_bounds):
            assert t.dtype == torch.int32 and t.is_contiguous()
        assert num_helix % tokens_per_seq == 0
    else:
        helix_positions = helix_slots = helix_bounds = p["kv_lens"]
    prep_t = triton.next_power_of_2(max(p["num_pos"] if has_overlap else 0, num_helix, 1))
    prep_s = triton.next_power_of_2(
        max(p["num_kv"] if has_overlap else 0, num_helix // max(tokens_per_seq, 1), 1)
    )
    # fused_dsa_helix_decode_metadata's checks and launch constants.
    seq_lens, kv_lens, block_offsets = m["seq_lens"], m["kv_lens"], m["block_offsets"]
    num_seqs, num_tokens, width = seq_lens.shape[0], m["num_tokens"], m["width"]
    assert num_seqs > 0 and num_tokens > 0
    assert block_offsets.shape[0] >= num_seqs
    schedule, schedule_expanded = m["schedule"], m["schedule_expanded"]
    with_schedule = schedule is not None
    assert with_schedule == (schedule_expanded is not None)
    if with_schedule:
        num_slots = schedule.shape[0] - 1
        for buf in (schedule, schedule_expanded):
            assert buf.dtype == torch.int32 and buf.is_contiguous()
            assert buf.shape == (num_slots + 1, 2)
        assert num_slots > 0
    else:
        num_slots = 1
        schedule = schedule_expanded = m["kv_lens_expanded"]
    index_head_dim = m["index_head_dim"]
    data_bytes_per_token = m["data_bytes_per_token"]
    if data_bytes_per_token is None:
        data_bytes_per_token = index_head_dim
    scale_size = index_head_dim // m["quant_block_size"] * 4
    tokens_per_block = m["tokens_per_block"]
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token
    kv_lens_2d = m["kv_lens_2d"]
    # begin_fold's checks and outputs.
    if not folds_embedding(ids, table):
        raise ValueError("step_begin: ids and table the embedding row copy does not take")
    if not replay.checkpoints or not replay.checkpoint_slot_units:
        raise RuntimeError("KDA checkpoint indices need a bound checkpoint-mode state")
    counts = replay.prev_num_accepted_tokens
    if state_indices.dim() != 1 or counts.dim() != 1:
        raise ValueError("The step begin takes 1-D state indices and accepted counts")
    num_requests = state_indices.shape[0]
    rows = ids.shape[0]
    if num_requests > rows:
        raise ValueError("Every request needs at least one embedding row")
    tokens = 1 + replay.num_speculative_tokens
    device = table.device
    slots = torch.empty((num_requests, tokens), dtype=torch.int32, device=device)
    accepted = torch.empty((num_requests,), dtype=torch.int32, device=device)
    vocab, hidden = table.shape
    embed = torch.empty((rows, hidden), dtype=table.dtype, device=device)
    columns = triton.cdiv(hidden, _BLOCK_H)
    _step_begin_kernel[(rows, columns + 1)](
        ids,
        table,
        embed,
        vocab,
        hidden,
        state_indices,
        state_indices.stride(0),
        counts,
        counts.stride(0),
        slots,
        accepted,
        num_requests,
        replay.checkpoint_header_units,
        replay.checkpoint_slot_units,
        replay.checkpoint_state_units,
        p["position_ids"],
        p["pos_offsets"],
        p["num_pos"],
        p["kv_lens"],
        p["kv_offsets"],
        p["kv_start"],
        p["num_kv"],
        helix_positions,
        helix_slots,
        helix_bounds,
        num_helix,
        p["tokens_per_block"],
        p["cp_size"],
        p["cp_rank"],
        p["kv_bound_start"],
        tokens_per_seq,
        seq_lens,
        kv_lens,
        block_offsets,
        m["helix_local_slots"],
        m["helix_kv_bounds"],
        m["req_idx_per_token"],
        m["slot_mapping_fp8"],
        m["slot_mapping_scale"],
        m["gen_kv_indptr"],
        m["gen_cached_token_indptr"],
        kv_lens_2d,
        m["kv_lens_expanded"],
        schedule,
        schedule_expanded,
        block_offsets.stride(0),
        block_offsets.stride(1),
        kv_lens_2d.stride(0),
        num_seqs,
        num_tokens,
        width,
        block_offsets.shape[1],
        num_slots,
        TOKENS=tokens,
        TOKENS_POW2=triton.next_power_of_2(tokens),
        BLOCK_H=_BLOCK_H,
        VEC=16 // table.element_size(),
        COLUMNS=columns,
        HAS_OVERLAP=has_overlap,
        HAS_HELIX=has_helix,
        PREP_T=prep_t,
        PREP_S=prep_s,
        tokens_per_block=tokens_per_block,
        data_bytes_per_token=data_bytes_per_token,
        scale_size=scale_size,
        block_stride=block_stride,
        scale_base_offset=scale_base_offset,
        META_S=triton.next_power_of_2(num_seqs),
        META_T=min(128, triton.next_power_of_2(num_tokens)),
        META_W=triton.next_power_of_2(max(width, 1)),
        SCHEDULE=with_schedule,
        SPLIT_KV=_DG_SM100_SPLIT_KV,
        BLOCK_SM=triton.next_power_of_2(num_slots + 1),
        BLOCK_R=_DG_SCHEDULE_BLOCK_R,
        num_warps=4,
    )
    return slots, accepted, embed
