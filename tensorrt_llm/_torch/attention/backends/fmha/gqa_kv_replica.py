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
"""KV-replica split generation attention: when TP replicates each KV head over a group of ranks, they split its pages.

Every rank still appends every token, but page p of a request is read only by CP rank (p mod CP) of the KV group:
  1. all-gather q within the KV group (distributed/ll_exchange.py),
  2. partial (O, LSE) of every group head over this rank's pages: fused kernel (gqa_kv_replica_fused.py,
     kv_replica_fused_decode) or the segmented trtllm-gen decode (kv_replica_decode),
  3. one-launch all-to-all + base-2 LSE merge (+ output gate) of each rank's own heads.
"""

import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from ....modules.fla.op import gdc_wait_with_memory_clobber
from .gqa_kv_replica_fused import KvReplicaFusedState, kv_replica_fused_partial
from .gqa_segmented_decode import draft_block, seg_geometry, segment_attention

# Channels of the KV group's exchange: 0 carries the q all-gather, 1 the all-to-all + merge.
EXCHANGE_CHANNELS = 2


@triton.jit
def _kv_replica_prep_kernel(
    q_all,
    q_stride_src,
    q_stride_tok,
    q_stride_head,
    table,
    table_stride_req,
    table_stride_kv,
    seq_lens,
    seg_q,
    seg_table,
    seg_lens,
    rank,
    num_req,
    kv_pool,
    bmm1_ptr,
    bmm1_val,
    bmm2_ptr,
    bmm2_val,
    extra_o,
    extra_lse,
    valid,
    q_loc,
    ql_stride_tok,
    ql_stride_head,
    QL: tl.constexpr,
    HQL: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    PAGE: tl.constexpr,
    P_SEG: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    COPY_Q: tl.constexpr = True,
    BQL: tl.constexpr = 8,
    HAS_BMM1_PTR: tl.constexpr = False,
    HAS_BMM2_PTR: tl.constexpr = False,
    DRAFT_PRE_WAIT: tl.constexpr = False,
):
    """Programs [0, num_req * S): the page-strided segment table / length (and the folded q) of one FMHA request
    (request order s * num_req + r). Programs [num_req * S, + rows * HQL): the causal draft-block partial of one
    (row, own head) into extra_o / extra_lse."""
    pid = tl.program_id(0)
    if pid >= num_req * S:
        _cp_prep_draft(
            pid - num_req * S,
            q_all,
            q_stride_src,
            q_stride_tok,
            q_stride_head,
            table,
            table_stride_req,
            table_stride_kv,
            seq_lens,
            rank,
            kv_pool,
            bmm1_ptr,
            bmm1_val,
            bmm2_ptr,
            bmm2_val,
            extra_o,
            extra_lse,
            valid,
            q_loc,
            ql_stride_tok,
            ql_stride_head,
            num_req,
            QL,
            HQL,
            CP,
            D,
            S,
            PAGE,
            BQL,
            HAS_BMM1_PTR,
            HAS_BMM2_PTR,
            LAUNCH_WITH_PDL,
            DRAFT_PRE_WAIT,
        )
    else:
        _cp_prep_table(
            pid,
            q_all,
            q_stride_src,
            q_stride_tok,
            q_stride_head,
            table,
            table_stride_req,
            table_stride_kv,
            seq_lens,
            seg_q,
            seg_table,
            seg_lens,
            rank,
            num_req,
            QL,
            HQL,
            CP,
            D,
            S,
            PAGE,
            P_SEG,
            BLOCK_P,
            BLOCK_Q,
            LAUNCH_WITH_PDL,
            COPY_Q,
        )


@triton.jit
def _cp_prep_table(
    pid,
    q_all,
    q_stride_src,
    q_stride_tok,
    q_stride_head,
    table,
    table_stride_req,
    table_stride_kv,
    seq_lens,
    seg_q,
    seg_table,
    seg_lens,
    rank,
    num_req,
    QL: tl.constexpr,
    HQL: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    PAGE: tl.constexpr,
    P_SEG: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    COPY_Q: tl.constexpr,
):
    """Segment table / length (and the folded q) of FMHA request pid = s * num_req + r."""
    s = pid // num_req
    r = pid % num_req
    L = tl.load(seq_lens + r)
    j0, npg, n, n_pages = seg_geometry(L, s, QL, S, PAGE, CP, rank)
    valid = n > 0
    kv = tl.arange(0, 2)
    p = tl.arange(0, BLOCK_P)
    gpage = tl.where(valid, (j0 + p) * CP + rank, 0)
    m1 = (p < P_SEG) & (((p < npg) & (gpage < n_pages)) | ((~valid) & (p == 0)))
    m = (kv[:, None] < 2) & m1[None, :]
    # Pre-wait: reads of host-uploaded lengths / tables only. Every write comes after the wait: the outputs are fresh
    # allocations, and the caching allocator may hand them a block that a kernel still running before the wait uses.
    vals = tl.load(
        table + r.to(tl.int64) * table_stride_req + kv[:, None] * table_stride_kv + gpage[None, :],
        mask=m,
        other=0,
    )
    if LAUNCH_WITH_PDL:
        # Kept even without the q copy: the FMHA's wait on this grid must also cover the q producer.
        gdc_wait_with_memory_clobber()
    tl.store(seg_lens + pid, tl.where(valid, n, 1))
    tl.store(seg_table + (pid.to(tl.int64) * 2 + kv[:, None]) * P_SEG + p[None, :], vals, mask=m)
    if COPY_Q:
        # Folded query: head hh = t * (CP * HQL) + src * HQL + h  <-  q_all[src, r * QL + t, h, :]
        HH: tl.constexpr = QL * CP * HQL
        e = tl.arange(0, BLOCK_Q)
        em = e < HH * D
        hh = e // D
        dd = e % D
        t = hh // (CP * HQL)
        gh = hh % (CP * HQL)
        src_rank = gh // HQL
        h = gh % HQL
        src = (
            q_all
            + src_rank.to(tl.int64) * q_stride_src
            + (r * QL + t).to(tl.int64) * q_stride_tok
            + h * q_stride_head
            + dd
        )
        tl.store(seg_q + pid.to(tl.int64) * HH * D + e, tl.load(src, mask=em), mask=em)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _cp_prep_draft(
    dp,
    q_all,
    q_stride_src,
    q_stride_tok,
    q_stride_head,
    table,
    table_stride_req,
    table_stride_kv,
    seq_lens,
    rank,
    kv_pool,
    bmm1_ptr,
    bmm1_val,
    bmm2_ptr,
    bmm2_val,
    extra_o,
    extra_lse,
    valid,
    q_loc,
    ql_stride_tok,
    ql_stride_head,
    num_req,
    QL: tl.constexpr,
    HQL: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    PAGE: tl.constexpr,
    BQL: tl.constexpr,
    HAS_BMM1_PTR: tl.constexpr,
    HAS_BMM2_PTR: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    DRAFT_PRE_WAIT: tl.constexpr,
):
    """Draft-block partial (O, base-2 LSE) of (row, own head h) = divmod(dp, HQL) -> extra_o / extra_lse. The h == 0
    program of each row also writes valid[c, s, row] = (member c's segment s of this row's request is non-empty): an
    empty segment's FMHA request attends to a dummy token and must not be merged."""
    row = dp // HQL
    h = dp % HQL
    r = row // QL
    t = row % QL
    d = tl.arange(0, D)
    # Pre-wait: host-uploaded lengths / block tables (and the DRAFT_PRE_WAIT draft block below).
    L = tl.load(seq_lens + r)
    j = tl.arange(0, BQL)
    jmask = (j <= t) & (j < QL)
    pos = L - QL + j
    tab = table + r.to(tl.int64) * table_stride_req
    blk_k = tl.load(tab + pos // PAGE, mask=jmask, other=0).to(tl.int64)
    blk_v = tl.load(tab + table_stride_kv + pos // PAGE, mask=jmask, other=0).to(tl.int64)
    off = (pos % PAGE).to(tl.int64)
    if DRAFT_PRE_WAIT:
        # Only behind a q all-gather that releases this grid after ITS wait (kv_replica_decode): the QKV preprocess
        # (this rank's q = q_loc, the draft K/V, the scales) has completed, so the draft overlaps the exchange.
        o_suf, lse_suf = draft_block(
            q_loc,
            ql_stride_tok,
            ql_stride_head,
            h,
            kv_pool,
            0,
            blk_k,
            blk_v,
            off,
            jmask,
            row,
            bmm1_ptr,
            bmm1_val,
            bmm2_ptr,
            bmm2_val,
            d,
            1,
            PAGE,
            D,
            HAS_BMM1_PTR,
            HAS_BMM2_PTR,
        )
    if LAUNCH_WITH_PDL:
        gdc_wait_with_memory_clobber()
    # Writes only after the wait (fresh allocations, see _cp_prep_table).
    if h == 0:
        rows = num_req * QL
        for c in tl.static_range(CP):
            for sg in tl.static_range(S):
                _, _, n_seg, _ = seg_geometry(L, sg, QL, S, PAGE, CP, c)
                tl.store(valid + (c * S + sg) * rows + row, (n_seg > 0).to(tl.int32))
    if not DRAFT_PRE_WAIT:
        # This rank's own q: member `rank` of the all-gather.
        o_suf, lse_suf = draft_block(
            q_all + rank * q_stride_src,
            q_stride_tok,
            q_stride_head,
            h,
            kv_pool,
            0,
            blk_k,
            blk_v,
            off,
            jmask,
            row,
            bmm1_ptr,
            bmm1_val,
            bmm2_ptr,
            bmm2_val,
            d,
            1,
            PAGE,
            D,
            HAS_BMM1_PTR,
            HAS_BMM2_PTR,
        )
    tl.store(extra_o + (row.to(tl.int64) * HQL + h) * D + d, o_suf.to(extra_o.dtype.element_ty))
    tl.store(extra_lse + row.to(tl.int64) * HQL + h, lse_suf)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def kv_replica_partial_merge_free(
    q_all: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_kv_len: int,
    q_len: int,
    cp_size: int,
    cp_rank: int,
    num_segments: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    enable_pdl: bool,
    out_dtype: torch.dtype = torch.bfloat16,
    q_local: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """This rank's partials for TPGroupExchange.alltoall_merge with extra_o / extra_lse (no local merge kernel): the
    segmented decode (gqa_segmented_decode.py) of the pages this rank owns, for every head of the KV group.

    Args:
        q_local: [b * q_len, Hq_local, D], this rank's own q, read by the draft programs BEFORE the prep kernel's
            wait. Only valid when the q all-gather releases its dependents after its own wait on the QKV preprocess.

    Returns:
        o [cp, S, b * q_len, Hq_local, D] (group head d * Hq_local + h for destination d) and lse [cp, S, b * q_len,
        Hq_local] (base 2), strided views of the segment-major FMHA (request s * b + r); extra_o [b * q_len, Hq_local,
        D] / extra_lse [b * q_len, Hq_local]: causal draft-block partial of this rank's own heads; valid [cp, S,
        b * q_len] int32: 0 where member c's segment s is empty (dummy one-token attention the merge must skip).
    """
    CP, S = cp_size, num_segments
    _, num_tokens, HQL, D = q_all.shape
    b = num_tokens // q_len
    GH = CP * HQL
    HH = q_len * GH
    page = kv_pool.shape[-2]
    assert kv_pool.is_contiguous() and kv_pool.shape[1] == 1 and kv_pool.shape[-1] == D
    assert block_tables.dim() == 3 and block_tables.shape[1] == 2 and block_tables.stride(-1) == 1
    assert q_all.stride(-1) == 1
    max_pages = math.ceil(max(max_kv_len, 1) / page)
    p_seg = math.ceil(math.ceil(max_pages / CP) / S) + 1
    dev = q_all.device
    fold_free = (
        S == 1 and q_all.stride(0) == HQL * D and q_all.stride(1) == GH * D and q_all.stride(2) == D
    )
    if fold_free:
        seg_q = q_all.as_strided((b, HH, D), (HH * D, D, 1), q_all.storage_offset())
    else:
        seg_q = torch.empty(b * S, HH, D, dtype=q_all.dtype, device=dev)
    seg_table = torch.empty(b * S, 2, p_seg, dtype=torch.int32, device=dev)
    seg_lens = torch.empty(b * S, dtype=torch.int32, device=dev)
    seg_out = torch.empty(b * S, HH, D, dtype=out_dtype, device=dev)
    seg_lse = torch.empty(b * S, HH, dtype=torch.float32, device=dev)
    extra_o = torch.empty(num_tokens, HQL, D, dtype=out_dtype, device=dev)
    extra_lse = torch.empty(num_tokens, HQL, dtype=torch.float32, device=dev)
    valid = torch.empty(CP, S, num_tokens, dtype=torch.int32, device=dev)
    has_b1 = isinstance(bmm1_scale, torch.Tensor)
    has_b2 = isinstance(bmm2_scale, torch.Tensor)
    _kv_replica_prep_kernel[(b * S + num_tokens * HQL,)](
        q_all,
        q_all.stride(0),
        q_all.stride(1),
        q_all.stride(2),
        block_tables,
        block_tables.stride(0),
        block_tables.stride(1),
        seq_lens,
        seg_q,
        seg_table,
        seg_lens,
        cp_rank,
        b,
        kv_pool,
        bmm1_scale if has_b1 else seg_lens,
        0.0 if has_b1 else float(bmm1_scale),
        bmm2_scale if has_b2 else seg_lens,
        0.0 if has_b2 else float(bmm2_scale),
        extra_o,
        extra_lse,
        valid,
        q_local if q_local is not None else q_all,
        q_local.stride(0) if q_local is not None else 0,
        q_local.stride(1) if q_local is not None else 0,
        QL=q_len,
        HQL=HQL,
        CP=CP,
        D=D,
        S=S,
        PAGE=page,
        P_SEG=p_seg,
        BLOCK_P=triton.next_power_of_2(p_seg),
        BLOCK_Q=triton.next_power_of_2(HH * D),
        LAUNCH_WITH_PDL=enable_pdl,
        COPY_Q=not fold_free,
        BQL=max(2, triton.next_power_of_2(q_len)),
        HAS_BMM1_PTR=has_b1,
        HAS_BMM2_PTR=has_b2,
        DRAFT_PRE_WAIT=q_local is not None,
        num_warps=8,
        launch_pdl=enable_pdl,
    )
    segment_attention(
        seg_q,
        kv_pool,
        seg_table,
        seg_lens,
        p_seg * page,
        bmm1_scale,
        bmm2_scale,
        workspace,
        counter,
        seg_out,
        seg_lse,
    )
    # request s * b + r, folded head hh = t * GH + d * HQL + h  ->  [d, s, r * q_len + t, h]
    o = seg_out.view(S, num_tokens, CP, HQL, D).permute(2, 0, 1, 3, 4)
    lse = seg_lse.view(S, num_tokens, CP, HQL).permute(2, 0, 1, 3)
    return o, lse, extra_o, extra_lse, valid


def _allgather_q(exchange, q: torch.Tensor, enable_pdl: bool) -> torch.Tensor:
    """q of every member of the KV group, [cp, b * q_len, Hq_local, D] (channel 0).

    With PDL the all-gather releases its dependents right after its own wait on the QKV preprocessing; before their
    own wait they read only host-uploaded lengths / tables, KV written by earlier steps and the preprocess outputs
    (this rank's q, the step's K/V, the scales).
    """
    rows, hq, d = q.shape
    trigger = "after_wait" if enable_pdl else False
    return (
        exchange.allgather(q, channel=0, trigger=trigger)
        .view(rows, exchange.group_size, hq, d)
        .permute(1, 0, 2, 3)
    )


def kv_replica_fused_decode(
    exchange,
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    out: torch.Tensor,
    fused_state: KvReplicaFusedState,
    enable_pdl: bool,
    gate: Optional[torch.Tensor] = None,
) -> None:
    """KV-replica split generation attention of this rank's heads, partials from the fused kernel
    (gqa_kv_replica_fused.py).

    exchange: TPGroupExchange (distributed/ll_exchange.py) whose groups are the KV groups (member = CP rank);
    channel 0 carries the q all-gather, channel 1 the all-to-all + LSE merge (+ gate) of the per-destination
    (O, LSE) partials. q [b * q_len, Hq_local, D] (FP8), out [b * q_len, Hq_local, D]; gate optional
    (out * sigmoid(gate)). fused_state: the fused kernel's counters and workspace on this device.
    """
    cp, cp_rank = exchange.group_size, exchange.grank
    rows, hq, d = q.shape
    q_all = _allgather_q(exchange, q, enable_pdl)
    o = torch.empty(cp, rows, hq, d, dtype=out.dtype, device=q.device)
    lse = torch.empty(cp, rows, hq, dtype=torch.float32, device=q.device)
    kv_replica_fused_partial(
        q_all,
        kv_pool,
        block_tables,
        seq_lens,
        q_len,
        cp_rank,
        cp,
        bmm1_scale,
        bmm2_scale,
        o,
        lse,
        fused_state,
        enable_pdl=enable_pdl,
    )
    exchange.alltoall_merge(o, lse, out.view(rows, hq, d), gate=gate, channel=1)


def kv_replica_decode(
    exchange,
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_kv_len: int,
    q_len: int,
    num_segments: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: bool,
    gate: Optional[torch.Tensor] = None,
) -> None:
    """KV-replica split generation attention of this rank's heads, partials from the segmented trtllm-gen decode.

    As kv_replica_fused_decode; no local merge kernel: the FMHA partials (num_segments per member) and the own-head
    draft partial go straight into the one-launch all-to-all + merge.
    """
    cp, cp_rank = exchange.group_size, exchange.grank
    rows, hq, d = q.shape
    q_all = _allgather_q(exchange, q, enable_pdl)
    o, lse, extra_o, extra_lse, valid = kv_replica_partial_merge_free(
        q_all,
        kv_pool,
        block_tables,
        seq_lens,
        max_kv_len,
        q_len,
        cp,
        cp_rank,
        num_segments,
        bmm1_scale,
        bmm2_scale,
        workspace,
        counter,
        enable_pdl,
        out_dtype=out.dtype,
        q_local=q if enable_pdl else None,
    )
    exchange.alltoall_merge(
        o,
        lse,
        out.view(rows, hq, d),
        gate=gate,
        channel=1,
        extra_o=extra_o,
        extra_lse=extra_lse,
        valid=valid,
    )
