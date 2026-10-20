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
"""Context-parallel ("CP-lite") generation attention building blocks for Qwen3.5 at TP8 (low latency).

TP8 replicates each of the 2 KV heads on the 4 ranks of its KV group (8 ranks at TP16), and today every rank reads
the full KV of its head.
CP-lite keeps the KV cache as it is (every rank still appends every token) but splits the READ: page p of a request is
read by CP rank (p mod CP) of the group. Each rank computes a partial attention for ALL q heads of its group
(gathered from the CP peers) over its pages, and the partials are combined by their base-2 LSE per head:
  1. gather q of the group (CP peers; collective supplied by the caller),
  2. q35_cp_local_partial(): fold the q_len tokens x group heads into the head dim (q_len 1, no causal effect),
     page-strided segment tables over the owned context-prefix pages, trtllm-gen FMHA + LSE, local merge of the
     segments plus the causal q_len x q_len draft block (each rank for the heads it owns, or CP rank 0 for all)
     -> partial (O, LSE) for every group head,
  3. LSE-weighted combine of the CP partials of each rank's own heads (collective supplied by the caller;
     q35_cp_combine() is the local reference / fallback combine of already exchanged partials).
"""

import functools
import inspect
import math
import os
import sys
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from . import q35_segmented as _seg
from .q35_segmented import _PDL_EARLY_TRIGGER, _gdc_wait_with_memory_clobber

# TRTLLM_Q35_ATTN_CP_A2A=merge (default; when the exchange provides it): one-launch all-to-all + LSE merge (+ gate)
# (TPGroupExchange.alltoall_merge, channel 1); =combine: TPGroupExchange.alltoall (channel 2) +
# q35_cp_combine.
_CP_A2A = os.environ.get("TRTLLM_Q35_ATTN_CP_A2A", "merge")
# TRTLLM_Q35_ATTN_CP_DRAFT_EARLY=1 (merge-free path only): the q all-gather releases its dependents right after its
# own wait on the QKV preprocessing (TPGroupExchange trigger="after_wait"), so the prep kernel's own-head draft programs
# read the local q / draft K/V before their griddepcontrol.wait and overlap the exchange. Relies on the predecessor's
# completion flush being device-wide; default off.
_CP_DRAFT_EARLY = os.environ.get("TRTLLM_Q35_ATTN_CP_DRAFT_EARLY", "0") == "1"
# TRTLLM_Q35_ATTN_CP_FUSED=1 (merge-free exchanges only; default off): one fused CP partial kernel
# (fmha/q35_cp_fused.py: q35_cp_fused_partial) replaces the prep + FMHA + LSE (+ own-head draft) launches of the
# merge-free path. It streams the pages this rank owns (page p belongs to CP rank p % cp) for all group heads and
# writes one (O, LSE) partial per (row, group head) straight into alltoall_merge's input layout. The contract is
# q35_cp_fused_partial_ref below. Without that module, or for an unsupported shape, the merge-free path runs.
_CP_FUSED = os.environ.get("TRTLLM_Q35_ATTN_CP_FUSED", "0") == "1"
# The partial-kernel entry the fused branch calls; None: resolved from fmha/q35_cp_fused.py at first use (tests
# assign q35_cp_fused_partial_ref or the kernel).
_CP_FUSED_FN = None
# Rows of one fused call per KV head (q_len x group heads) the kernel supports (MMA N <= 128).
_CP_FUSED_MAX_ROWS = 128
# TRTLLM_Q35_ATTN_CP_FUSED_CHECK=1 (eager calls only, testing): every fused call also runs the merge-free path on the
# same inputs and logs the difference of the outputs (CP_FUSED_CHECK_STATS: max abs, max relative to the row's largest
# |reference|, max bf16 ulps at the worst element, mean abs; a warning above 5 % of the row max).
_CP_FUSED_CHECK = os.environ.get("TRTLLM_Q35_ATTN_CP_FUSED_CHECK", "0") == "1"
CP_FUSED_CHECK_STATS = {"calls": 0, "max_abs": 0.0}


@triton.jit
def _cp_seg_geometry(
    L, s, QL: tl.constexpr, S: tl.constexpr, PAGE: tl.constexpr, CP: tl.constexpr, RANK
):
    """Pages owned by RANK in the context prefix [0, L - QL), split into S segments; returns (first owned index,
    pages in this segment, tokens in this segment, total prefix pages)."""
    prefix = L - QL
    n_pages = (prefix + PAGE - 1) // PAGE
    n_owned = tl.maximum((n_pages - RANK + CP - 1) // CP, 0)
    seg_pages = (n_owned + S - 1) // S
    j0 = s * seg_pages
    npg = tl.maximum(tl.minimum(seg_pages, n_owned - j0), 0)
    last_tok = (
        prefix - (n_pages - 1) * PAGE
    )  # tokens in the last prefix page (1..PAGE) when n_pages > 0
    last_global = (j0 + npg - 1) * CP + RANK
    n = npg * PAGE - tl.where((npg > 0) & (last_global == n_pages - 1), PAGE - last_tok, 0)
    return j0, npg, n, n_pages


@triton.jit
def _q35_cp_prep_kernel(
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
    SEG_MAJOR: tl.constexpr = False,
    DRAFT_OWN: tl.constexpr = False,
    BQL: tl.constexpr = 8,
    HAS_BMM1_PTR: tl.constexpr = False,
    HAS_BMM2_PTR: tl.constexpr = False,
    DRAFT_PRE_WAIT: tl.constexpr = False,
    DEBUG_SPIN_NS: tl.constexpr = 0,
    DEBUG_ENTRY_TRIGGER: tl.constexpr = False,
):
    """Programs [0, num_req * S): the page-strided segment table / length (and the folded q) of one FMHA request
    (request order r * S + s, or s * num_req + r with SEG_MAJOR). DRAFT_OWN: programs [num_req * S, + rows * HQL)
    compute the causal draft-block partial of one (row, own head) into extra_o / extra_lse.
    DEBUG_SPIN_NS / DEBUG_ENTRY_TRIGGER: PDL hazard probe hooks (q35_segmented._DEBUG_*), tests only."""
    if LAUNCH_WITH_PDL and DEBUG_ENTRY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    if DRAFT_OWN:
        if pid >= num_req * S:
            _cp_prep_draft(
                pid - num_req * S,
                q_all,
                q_stride_src,
                q_stride_tok,
                q_stride_head,
                q_loc,
                ql_stride_tok,
                ql_stride_head,
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
                SEG_MAJOR,
                DEBUG_SPIN_NS,
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
            SEG_MAJOR,
            DEBUG_SPIN_NS,
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
    SEG_MAJOR: tl.constexpr,
    DEBUG_SPIN_NS: tl.constexpr,
):
    """Segment table / length (and the folded q) of FMHA request pid."""
    if SEG_MAJOR:
        s = pid // num_req
        r = pid % num_req
    else:
        r = pid // S
        s = pid % S
    L = tl.load(seq_lens + r)
    j0, npg, n, n_pages = _cp_seg_geometry(L, s, QL, S, PAGE, CP, rank)
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
        _gdc_wait_with_memory_clobber()
    if DEBUG_SPIN_NS > 0:
        t0 = tl.extra.cuda.globaltimer()
        while tl.extra.cuda.globaltimer() - t0 < DEBUG_SPIN_NS:
            pass
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
    q_loc,
    ql_stride_tok,
    ql_stride_head,
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
    L = tl.load(seq_lens + r)
    j = tl.arange(0, BQL)
    jmask = (j <= t) & (j < QL)
    pos = L - QL + j
    tab = table + r.to(tl.int64) * table_stride_req
    blk_k = tl.load(tab + pos // PAGE, mask=jmask, other=0).to(tl.int64)
    blk_v = tl.load(tab + table_stride_kv + pos // PAGE, mask=jmask, other=0).to(tl.int64)
    off = (pos % PAGE).to(tl.int64)
    if DRAFT_PRE_WAIT:
        # Only with a q all-gather that releases this grid after ITS wait (see q35_cp_decode_ll): the QKV preprocess
        # (local q, draft K/V, scales) has then completed, so the draft overlaps the exchange. q_loc = this rank's q.
        o_suf, lse_suf = _cp_draft_block(
            q_loc,
            0,
            ql_stride_tok,
            ql_stride_head,
            kv_pool,
            blk_k,
            blk_v,
            off,
            jmask,
            row,
            h,
            bmm1_ptr,
            bmm1_val,
            bmm2_ptr,
            bmm2_val,
            d,
            HQL,
            PAGE,
            D,
            HAS_BMM1_PTR,
            HAS_BMM2_PTR,
            False,
        )
    if LAUNCH_WITH_PDL:
        _gdc_wait_with_memory_clobber()
    # Writes only after the wait (fresh allocations, see _cp_prep_table).
    if h == 0:
        rows = num_req * QL
        for c in tl.static_range(CP):
            for sg in tl.static_range(S):
                _, _, n_seg, _ = _cp_seg_geometry(L, sg, QL, S, PAGE, CP, c)
                tl.store(valid + (c * S + sg) * rows + row, (n_seg > 0).to(tl.int32))
    if not DRAFT_PRE_WAIT:
        o_suf, lse_suf = _cp_draft_block(
            q_all,
            q_stride_src,
            q_stride_tok,
            q_stride_head,
            kv_pool,
            blk_k,
            blk_v,
            off,
            jmask,
            row,
            rank * HQL + h,
            bmm1_ptr,
            bmm1_val,
            bmm2_ptr,
            bmm2_val,
            d,
            HQL,
            PAGE,
            D,
            HAS_BMM1_PTR,
            HAS_BMM2_PTR,
            False,
        )
    tl.store(extra_o + (row.to(tl.int64) * HQL + h) * D + d, o_suf.to(extra_o.dtype.element_ty))
    tl.store(extra_lse + row.to(tl.int64) * HQL + h, lse_suf)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _cp_draft_block(
    q_all,
    q_stride_src,
    q_stride_tok,
    q_stride_head,
    kv_pool,
    blk_k,
    blk_v,
    off,
    jmask,
    row,
    gh,
    bmm1_ptr,
    bmm1_val,
    bmm2_ptr,
    bmm2_val,
    d,
    HQL: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    HAS_BMM1_PTR: tl.constexpr,
    HAS_BMM2_PTR: tl.constexpr,
    DRAFT_OWNER: tl.constexpr,
):
    """(O, base-2 LSE) of one (row, group head) over the causal draft block; (0, -inf) where jmask is all false."""
    if HAS_BMM1_PTR:
        bmm1 = tl.load(bmm1_ptr).to(tl.float32)
    else:
        bmm1 = bmm1_val
    if HAS_BMM2_PTR:
        bmm2 = tl.load(bmm2_ptr).to(tl.float32)
    else:
        bmm2 = bmm2_val
    src_rank = gh // HQL
    h = gh % HQL
    qv = tl.load(
        q_all
        + src_rank.to(tl.int64) * q_stride_src
        + row.to(tl.int64) * q_stride_tok
        + h * q_stride_head
        + d
    ).to(tl.float32)
    k_t = tl.load(
        kv_pool + (blk_k * PAGE + off)[:, None] * D + d[None, :], mask=jmask[:, None], other=0.0
    ).to(tl.float32)
    v_t = tl.load(
        kv_pool + (blk_v * PAGE + off)[:, None] * D + d[None, :], mask=jmask[:, None], other=0.0
    ).to(tl.float32)
    s_j = tl.where(jmask, tl.sum(k_t * qv[None, :], axis=1) * bmm1, float("-inf"))
    m_suf = tl.max(s_j, axis=0)
    p = tl.where(jmask, tl.exp(s_j - m_suf), 0.0)
    l_suf = tl.sum(p, axis=0)
    if DRAFT_OWNER:
        # Heads owned by another rank carry no draft block here (l_suf == 0).
        has_suf = l_suf > 0
        o_suf = tl.sum(p[:, None] * v_t, axis=0) * tl.where(
            has_suf, bmm2 / tl.where(has_suf, l_suf, 1.0), 0.0
        )
        lse_suf = tl.where(
            has_suf,
            m_suf * 1.4426950408889634 + tl.log2(tl.where(has_suf, l_suf, 1.0)),
            float("-inf"),
        )
    else:
        o_suf = tl.sum(p[:, None] * v_t, axis=0) * (bmm2 / l_suf)
        lse_suf = m_suf * 1.4426950408889634 + tl.log2(l_suf)
    return o_suf, lse_suf


@triton.jit
def _q35_cp_local_merge_kernel(
    seg_out,
    seg_lse,
    q_all,
    q_stride_src,
    q_stride_tok,
    q_stride_head,
    kv_pool,
    table,
    table_stride_req,
    table_stride_kv,
    seq_lens,
    bmm1_ptr,
    bmm1_val,
    bmm2_ptr,
    bmm2_val,
    part_o,
    o_stride_dest,
    o_stride_row,
    part_lse,
    l_stride_dest,
    l_stride_row,
    rank,
    QL: tl.constexpr,
    BQL: tl.constexpr,
    HQL: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    PAGE: tl.constexpr,
    INCLUDE_DRAFT: tl.constexpr,
    HAS_BMM1_PTR: tl.constexpr,
    HAS_BMM2_PTR: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    DRAFT_OWNER: tl.constexpr = False,
    DRAFT_PRE_WAIT: tl.constexpr = False,
):
    """Partial (O, base-2 LSE) of one (token row, group head): this rank's segments (+ the draft block).

    INCLUDE_DRAFT: the causal q_len x q_len draft block is added for every group head (DRAFT_OWNER False: exactly one
    rank of the group sets it) or only for the heads this rank owns (DRAFT_OWNER True: every rank sets it; balances
    the draft work over the group).

    Group head gh goes to destination gh // HQL (the rank owning it): part_o[dest, row, gh % HQL, :] and
    part_lse[dest, row, gh % HQL] with the given destination / row strides (head stride D / 1)."""
    row = tl.program_id(0)  # r * QL + t
    gh = tl.program_id(1)  # src * HQL + h, group head
    r = row // QL
    t = row % QL
    GH: tl.constexpr = CP * HQL
    HH: tl.constexpr = QL * GH
    hh = t * GH + gh
    d = tl.arange(0, D)
    L = tl.load(seq_lens + r)
    prefix = L - QL
    sv = tl.arange(0, BS)
    j0, npg, n_s, n_pages = _cp_seg_geometry(L, sv, QL, S, PAGE, CP, rank)
    smask = (sv < S) & (n_s > 0)
    if INCLUDE_DRAFT:
        j = tl.arange(0, BQL)
        if DRAFT_OWNER:
            jmask = (j <= t) & (j < QL) & ((gh // HQL) == rank)
        else:
            jmask = (j <= t) & (j < QL)
        pos = prefix + j
        tab = table + r.to(tl.int64) * table_stride_req
        blk_k = tl.load(tab + pos // PAGE, mask=jmask, other=0).to(tl.int64)
        blk_v = tl.load(tab + table_stride_kv + pos // PAGE, mask=jmask, other=0).to(tl.int64)
        off = (pos % PAGE).to(tl.int64)
        if DRAFT_PRE_WAIT:
            # q (all-gather output), the draft K/V and the scales were complete before this call's prep kernel
            # released its dependents (see q35_segmented._DRAFT_PRE_WAIT): overlap the draft block with the FMHA.
            o_suf, lse_suf = _cp_draft_block(
                q_all,
                q_stride_src,
                q_stride_tok,
                q_stride_head,
                kv_pool,
                blk_k,
                blk_v,
                off,
                jmask,
                row,
                gh,
                bmm1_ptr,
                bmm1_val,
                bmm2_ptr,
                bmm2_val,
                d,
                HQL,
                PAGE,
                D,
                HAS_BMM1_PTR,
                HAS_BMM2_PTR,
                DRAFT_OWNER,
            )
    if LAUNCH_WITH_PDL:
        _gdc_wait_with_memory_clobber()
    seg_rows = (r * S + sv).to(tl.int64) * HH + hh
    lse_s = tl.load(seg_lse + seg_rows, mask=smask, other=float("-inf"))
    o_s = tl.load(seg_out + seg_rows[:, None] * D + d[None, :], mask=smask[:, None], other=0.0).to(
        tl.float32
    )
    m_all = tl.max(lse_s, axis=0)
    if INCLUDE_DRAFT:
        if not DRAFT_PRE_WAIT:
            o_suf, lse_suf = _cp_draft_block(
                q_all,
                q_stride_src,
                q_stride_tok,
                q_stride_head,
                kv_pool,
                blk_k,
                blk_v,
                off,
                jmask,
                row,
                gh,
                bmm1_ptr,
                bmm1_val,
                bmm2_ptr,
                bmm2_val,
                d,
                HQL,
                PAGE,
                D,
                HAS_BMM1_PTR,
                HAS_BMM2_PTR,
                DRAFT_OWNER,
            )
        m_all = tl.maximum(m_all, lse_suf)
    w_s = tl.where(smask, tl.exp2(lse_s - m_all), 0.0)
    acc = tl.sum(w_s[:, None] * o_s, axis=0)
    wsum = tl.sum(w_s, axis=0)
    if INCLUDE_DRAFT:
        if DRAFT_OWNER:
            w_suf = tl.where(lse_suf > float("-inf"), tl.exp2(lse_suf - m_all), 0.0)
        else:
            w_suf = tl.exp2(lse_suf - m_all)
        acc += w_suf * o_suf
        wsum += w_suf
    has = wsum > 0
    res = tl.where(has, acc / tl.where(has, wsum, 1.0), 0.0)
    lse_out = tl.where(has, m_all + tl.log2(tl.where(has, wsum, 1.0)), float("-inf"))
    dest = (gh // HQL).to(tl.int64)
    hl = gh % HQL
    tl.store(
        part_o + dest * o_stride_dest + row.to(tl.int64) * o_stride_row + hl * D + d,
        res.to(part_o.dtype.element_ty),
    )
    tl.store(part_lse + dest * l_stride_dest + row.to(tl.int64) * l_stride_row + hl, lse_out)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _q35_cp_combine_kernel(
    parts_o,
    o_stride_cp,
    o_stride_row,
    parts_lse,
    l_stride_cp,
    l_stride_row,
    out,
    out_stride_tok,
    out_stride_head,
    gate,
    gate_stride_tok,
    gate_stride_head,
    CP: tl.constexpr,
    BCP: tl.constexpr,
    D: tl.constexpr,
    HAS_GATE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr = False,
):
    """out[row, h] = LSE-weighted average over the CP partials parts_*[c, row, h] (each rank's own heads),
    optionally x sigmoid(gate[row, h]) (same arithmetic as the separate output-gate kernel)."""
    row = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.arange(0, BCP)
    cm = c < CP
    d = tl.arange(0, D)
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    if LAUNCH_WITH_PDL:
        _gdc_wait_with_memory_clobber()
    lse = tl.load(
        parts_lse + c.to(tl.int64) * l_stride_cp + row.to(tl.int64) * l_stride_row + h,
        mask=cm,
        other=float("-inf"),
    )
    o_base = c.to(tl.int64) * o_stride_cp + row.to(tl.int64) * o_stride_row + h * D
    o = tl.load(parts_o + o_base[:, None] + d[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
    m = tl.max(lse, axis=0)
    w = tl.where(lse > float("-inf"), tl.exp2(lse - m), 0.0)
    den = tl.sum(w, axis=0)
    res = tl.where(den > 0, tl.sum(w[:, None] * o, axis=0) / tl.where(den > 0, den, 1.0), 0.0)
    res = res.to(out.dtype.element_ty)
    if HAS_GATE:
        gv = tl.load(gate + row.to(tl.int64) * gate_stride_tok + h * gate_stride_head + d).to(
            tl.float32
        )
        res = (res.to(tl.float32) * tl.sigmoid(gv)).to(out.dtype.element_ty)
    tl.store(out + row.to(tl.int64) * out_stride_tok + h * out_stride_head + d, res)
    if LAUNCH_WITH_PDL and not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


def _cp_fmha_pdl(enable_pdl: bool, fmha_pdl: bool) -> bool:
    """enable_pdl of the trtllm-gen call after the CP prep kernel. The call's own launches (FMHA, split-KV reduction,
    ComputeLSEFromMD) all take it. Off unless TRTLLM_Q35_ATTN_FMHA_PDL=1 (q35_segmented._FMHA_PDL): the FMHA may
    read the prep's seg_lens / seg_table before its dependency wait."""
    return bool(enable_pdl and fmha_pdl and _seg._FMHA_PDL)


def _cp_debug_zero(
    seg_table: torch.Tensor, seg_lens: torch.Tensor, seg_q: Optional[torch.Tensor]
) -> None:
    """PDL hazard probe (tests only, q35_segmented._DEBUG_SPIN_NS > 0): zero the prep kernel's outputs first, so that
    an early read of them by the FMHA sees zeros."""
    if _seg._DEBUG_SPIN_NS > 0:
        seg_table.zero_()
        seg_lens.zero_()
        if seg_q is not None:
            seg_q.zero_()


def q35_cp_local_partial(
    run_func,
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
    bmm1_scale_arg,
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    sm_count: int,
    kv_scale_pool: Optional[torch.Tensor],
    enable_pdl: bool,
    include_draft: bool,
    out_dtype: torch.dtype = torch.bfloat16,
    fmha_pdl: bool = True,
    out_o: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    draft_owner: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This rank's partial attention for every group head.

    q_all: [cp_size, b * q_len, Hq_local, D] gathered queries of the KV group (Hkv_local = 1), kv_pool: [blocks, 1,
    page, D], block_tables: [b, 2, max_pages] (K / V page indices), seq_lens: [b] incl. the q_len new tokens.
    Returns (partial_o [b * q_len, cp_size * Hq_local, D], partial_lse [b * q_len, cp_size * Hq_local], base 2).
    The draft block is added only where include_draft (exactly one rank of the group must set it); with draft_owner
    every rank adds it for the heads it owns (include_draft is then ignored).
    At num_segments == 1 a q_all in the all-gather layout (strided view of a contiguous [b * q_len, cp_size *
    Hq_local * D] buffer) is read by the FMHA in place (no folded copy).
    out_o [cp_size, b * q_len, Hq_local, D] / out_lse [cp_size, b * q_len, Hq_local] (optional, e.g. views of an
    all-to-all send buffer): write the partial of group head src * Hq_local + h to out_*[src, row, h] instead and
    return them.
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
    assert counter.numel() * counter.element_size() >= 4 * max(b * S * HH, sm_count), (
        "multi-CTA counter buffer too small for the CP-lite call"
    )
    max_pages = math.ceil(max(max_kv_len, 1) / page)
    p_seg = math.ceil(math.ceil(max_pages / CP) / S) + 1
    dev = q_all.device
    fold_free = (
        S == 1 and q_all.stride(0) == HQL * D and q_all.stride(1) == GH * D and q_all.stride(2) == D
    )
    if fold_free:
        # [b * q_len, GH, D] row-major == [b, q_len * GH, D] with hh = t * GH + gh: the folded layout itself.
        seg_q = q_all.as_strided((b, HH, D), (HH * D, D, 1), q_all.storage_offset())
    else:
        seg_q = torch.empty(b * S, HH, D, dtype=q_all.dtype, device=dev)
    seg_table = torch.empty(b * S, 2, p_seg, dtype=torch.int32, device=dev)
    seg_lens = torch.empty(b * S, dtype=torch.int32, device=dev)
    seg_out = torch.empty(b * S, HH, D, dtype=out_dtype, device=dev)
    seg_lse = torch.empty(b * S, HH, dtype=torch.float32, device=dev)
    if out_o is None:
        part_o = torch.empty(num_tokens, GH, D, dtype=out_dtype, device=dev)
        part_lse = torch.empty(num_tokens, GH, dtype=torch.float32, device=dev)
        o_view = part_o.view(num_tokens, CP, HQL, D).permute(1, 0, 2, 3)
        l_view = part_lse.view(num_tokens, CP, HQL).permute(1, 0, 2)
    else:
        assert (
            tuple(out_o.shape) == (CP, num_tokens, HQL, D)
            and out_o.stride(3) == 1
            and out_o.stride(2) == D
        )
        assert tuple(out_lse.shape) == (CP, num_tokens, HQL) and out_lse.stride(2) == 1
        assert out_lse.dtype == torch.float32
        part_o, part_lse, o_view, l_view = out_o, out_lse, out_o, out_lse
    _cp_debug_zero(seg_table, seg_lens, None if fold_free else seg_q)
    _q35_cp_prep_kernel[(b * S,)](
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
        seg_lens,
        seg_lens,
        0.0,
        seg_lens,
        0.0,
        seg_lens,
        seg_lens,
        seg_lens,
        seg_lens,
        0,
        0,
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
        DEBUG_SPIN_NS=_seg._DEBUG_SPIN_NS,
        DEBUG_ENTRY_TRIGGER=_seg._DEBUG_ENTRY_TRIGGER,
        num_warps=8,
        launch_pdl=enable_pdl,
    )
    run_func(
        seg_out,
        None,
        seg_q,
        kv_pool,
        kv_pool,
        workspace,
        counter,
        seg_table,
        seg_lens,
        1,
        p_seg * page,
        bmm1_scale_arg,
        bmm2_scale,
        -1.0,
        -1,
        0,
        b * S,
        -1,
        0,
        sm_count,
        _cp_fmha_pdl(enable_pdl, fmha_pdl),
        workspace.numel() * workspace.element_size(),
        None,
        None,
        kv_scale_pool,
        kv_scale_pool,
        None,
        False,
        seg_lse,
        seg_lse.stride(0),
        seg_lse.stride(1),
        False,
    )
    has_b1 = isinstance(bmm1_scale, torch.Tensor)
    has_b2 = isinstance(bmm2_scale, torch.Tensor)
    _q35_cp_local_merge_kernel[(num_tokens, GH)](
        seg_out,
        seg_lse,
        q_all,
        q_all.stride(0),
        q_all.stride(1),
        q_all.stride(2),
        kv_pool,
        block_tables,
        block_tables.stride(0),
        block_tables.stride(1),
        seq_lens,
        bmm1_scale if has_b1 else seg_lens,
        0.0 if has_b1 else float(bmm1_scale),
        bmm2_scale if has_b2 else seg_lens,
        0.0 if has_b2 else float(bmm2_scale),
        o_view,
        o_view.stride(0),
        o_view.stride(1),
        l_view,
        l_view.stride(0),
        l_view.stride(1),
        cp_rank,
        QL=q_len,
        BQL=max(2, triton.next_power_of_2(q_len)),
        HQL=HQL,
        CP=CP,
        D=D,
        S=S,
        BS=max(2, triton.next_power_of_2(S)),
        PAGE=page,
        INCLUDE_DRAFT=include_draft or draft_owner,
        HAS_BMM1_PTR=has_b1,
        HAS_BMM2_PTR=has_b2,
        LAUNCH_WITH_PDL=enable_pdl,
        DRAFT_OWNER=draft_owner,
        DRAFT_PRE_WAIT=_seg._DRAFT_PRE_WAIT,
        num_warps=4,
        launch_pdl=enable_pdl,
    )
    return part_o, part_lse


def q35_cp_partial_merge_free(
    run_func,
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
    bmm1_scale_arg,
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    sm_count: int,
    kv_scale_pool: Optional[torch.Tensor],
    enable_pdl: bool,
    out_dtype: torch.dtype = torch.bfloat16,
    fmha_pdl: bool = True,
    q_local: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """This rank's partials without a local merge kernel, for an all-to-all + merge that takes several partials per
    member plus one local extra partial (TPGroupExchange.alltoall_merge with extra_o / extra_lse).

    The FMHA requests run segment-major (request s * b + r), so its outputs are uniform strided views:
      o   [cp_size, S, b * q_len, Hq_local, D]  (partial of group head d * Hq_local + h for destination d),
      lse [cp_size, S, b * q_len, Hq_local]     (base 2),
    and the prep kernel's extra programs compute the causal draft-block partial of this rank's own heads:
      extra_o [b * q_len, Hq_local, D], extra_lse [b * q_len, Hq_local],
    and valid [cp_size, S, b * q_len] int32 for the partials this rank RECEIVES: member c's segment s of a row's request
    is empty when c owns fewer than s + 1 segments' worth of the context pages (short contexts); such a partial is a
    dummy one-token attention and must be skipped by the merge.
    q_local [b * q_len, Hq_local, D] (optional): this rank's own q, read by the draft programs BEFORE the prep
    kernel's wait. Only valid when the kernel ahead of the prep (the q all-gather) releases its dependents after its
    own wait on the QKV preprocessing (so the preprocess outputs are complete when the prep starts).
    Returns (o, lse, extra_o, extra_lse, valid).
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
    assert counter.numel() * counter.element_size() >= 4 * max(b * S * HH, sm_count), (
        "multi-CTA counter buffer too small for the CP-lite call"
    )
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
    _cp_debug_zero(seg_table, seg_lens, None if fold_free else seg_q)
    _q35_cp_prep_kernel[(b * S + num_tokens * HQL,)](
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
        SEG_MAJOR=True,
        DRAFT_OWN=True,
        BQL=max(2, triton.next_power_of_2(q_len)),
        HAS_BMM1_PTR=has_b1,
        HAS_BMM2_PTR=has_b2,
        DRAFT_PRE_WAIT=q_local is not None,
        DEBUG_SPIN_NS=_seg._DEBUG_SPIN_NS,
        DEBUG_ENTRY_TRIGGER=_seg._DEBUG_ENTRY_TRIGGER,
        num_warps=8,
        launch_pdl=enable_pdl,
    )
    run_func(
        seg_out,
        None,
        seg_q,
        kv_pool,
        kv_pool,
        workspace,
        counter,
        seg_table,
        seg_lens,
        1,
        p_seg * page,
        bmm1_scale_arg,
        bmm2_scale,
        -1.0,
        -1,
        0,
        b * S,
        -1,
        0,
        sm_count,
        _cp_fmha_pdl(enable_pdl, fmha_pdl),
        workspace.numel() * workspace.element_size(),
        None,
        None,
        kv_scale_pool,
        kv_scale_pool,
        None,
        False,
        seg_lse,
        seg_lse.stride(0),
        seg_lse.stride(1),
        False,
    )
    # request s * b + r, folded head hh = t * GH + d * HQL + h  ->  [d, s, r * q_len + t, h]
    o = seg_out.view(S, num_tokens, CP, HQL, D).permute(2, 0, 1, 3, 4)
    lse = seg_lse.view(S, num_tokens, CP, HQL).permute(2, 0, 1, 3)
    return o, lse, extra_o, extra_lse, valid


def q35_cp_combine(
    parts_o: torch.Tensor,
    parts_lse: torch.Tensor,
    out: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> None:
    """LSE-weighted combine of exchanged partials: parts_o [CP, rows, H, D], parts_lse [CP, rows, H] (any CP / row
    strides, e.g. views of an all-to-all receive buffer) -> out [rows, H, D]; gate (optional, [rows, H, D] or
    [rows, H * D]) applies out * sigmoid(gate)."""
    CP, rows, H, D = parts_o.shape
    assert parts_o.stride(3) == 1 and parts_o.stride(2) == D
    assert tuple(parts_lse.shape) == (CP, rows, H) and parts_lse.stride(2) == 1
    assert out.stride(-1) == 1
    if gate is not None:
        gate = gate.view(rows, H, D)
        assert gate.stride(-1) == 1
    _q35_cp_combine_kernel[(rows, H)](
        parts_o,
        parts_o.stride(0),
        parts_o.stride(1),
        parts_lse,
        parts_lse.stride(0),
        parts_lse.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        gate if gate is not None else out,
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        CP=CP,
        BCP=max(2, triton.next_power_of_2(CP)),
        D=D,
        HAS_GATE=gate is not None,
        LAUNCH_WITH_PDL=enable_pdl,
        EARLY_TRIGGER=_PDL_EARLY_TRIGGER,
        num_warps=4,
        launch_pdl=enable_pdl,
    )


@functools.lru_cache(maxsize=None)
def _allgather_after_wait(exchange_cls) -> bool:
    """Whether the exchange's allgather supports trigger="after_wait"."""
    fn = getattr(exchange_cls, "allgather", None)
    mode = getattr(exchange_cls, "_trigger_mode", None)
    if fn is None or mode is None or "trigger" not in inspect.signature(fn).parameters:
        return False
    try:
        return '"after_wait"' in inspect.getsource(mode)
    except (OSError, TypeError):
        return False


@functools.lru_cache(maxsize=None)
def _merge_takes_extra(exchange_cls) -> bool:
    """Whether the exchange's alltoall_merge takes several partials per member plus a local extra partial."""
    fn = getattr(exchange_cls, "alltoall_merge", None)
    return fn is not None and {"extra_o", "valid"} <= set(inspect.signature(fn).parameters)


def _cp_fused_fn():
    """The fused CP partial kernel (fmha/q35_cp_fused.py), or None when it is not available."""
    global _CP_FUSED_FN
    if _CP_FUSED_FN is None:
        try:
            from .q35_cp_fused import q35_cp_fused_partial
        except ImportError:
            q35_cp_fused_partial = False
        _CP_FUSED_FN = q35_cp_fused_partial
    return _CP_FUSED_FN or None


def _cp_fused_shape_ok(
    fn, q: torch.Tensor, kv_pool: torch.Tensor, cp: int, q_len: int, sm_count: int
) -> bool:
    rows, hq, d = q.shape
    if not (
        q.dtype == torch.float8_e4m3fn
        and kv_pool.dtype == torch.float8_e4m3fn
        and d == 256
        and kv_pool.dim() == 4
        and kv_pool.shape[1] == 1
        and kv_pool.shape[2] == 64
        and 1 <= q_len <= 16
        and q_len * cp * hq <= _CP_FUSED_MAX_ROWS
        and rows % q_len == 0
    ):
        return False
    # The kernel module's own limits (batch, combine geometry), when it states them.
    mod = sys.modules.get(getattr(fn, "__module__", None) or "")
    supported = getattr(mod, "q35_cp_fused_supported", None)
    return supported is None or bool(supported(q_len, cp, hq, rows // q_len, sm_count))


def q35_cp_fused_partial_ref(
    q_all: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    cp_rank: int,
    cp_size: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    out_o: torch.Tensor,
    out_lse: torch.Tensor,
    sm_count: int = 0,
    enable_pdl: bool = False,
) -> None:
    """Contract of the fused CP partial kernel (TRTLLM_Q35_ATTN_CP_FUSED); FP32 torch reference for tests.

    Inputs:
      q_all [cp, rows, hq, D] FP8: query rows of every member s of the KV group (the all-gather output viewed
        member-major, any strides, D dense); rows = b * q_len, row r * q_len + t is token t of request r; group head
        g = s * hq + h.
      kv_pool [blocks, 1, PAGE, D] FP8; block_tables [>= b, 2, max_pages] int32 (row 0 K, row 1 V block ids);
      seq_lens [>= b] int32: L_r including the q_len new tokens (already appended to the cache).
      bmm1_scale: float or a [2] tensor (raw, raw * log2e); bmm2_scale: float or a [1] tensor.
    Keys visible to row (r, t): positions j < L_r - q_len + t + 1 (prefix + causal draft). This rank owns position j
    iff (j // PAGE) % cp_size == cp_rank, and reads only its own keys.
    Outputs (alltoall_merge's input layout, P = 1, member d = the owner of group head d * hq + h):
      out_o [cp, rows, hq, D] (BF16): bmm2 * softmax-weighted V over this rank's visible keys of row (r, t), group
        head d * hq + h, normalized by this rank's sum; 0 if this rank sees no key of that row.
      out_lse [cp, rows, hq] FP32: base-2 LSE of the bmm1-scaled scores over those keys (log2e * m + log2(sum));
        -inf if none.
    Kernel rules: no global writes before griddepcontrol.wait; q is read after it; the prefix KV may be read before it
    except the last prefix page (MTP passes write it earlier in the step).
    """
    cp, rows, hq, D = q_all.shape
    assert cp == cp_size and rows % q_len == 0
    b = rows // q_len
    page = kv_pool.shape[2]
    dev = q_all.device
    bmm1 = (
        bmm1_scale.flatten()[0]
        if isinstance(bmm1_scale, torch.Tensor)
        else torch.tensor(float(bmm1_scale), device=dev)
    ).float()
    bmm2 = (
        bmm2_scale.flatten()[0]
        if isinstance(bmm2_scale, torch.Tensor)
        else torch.tensor(float(bmm2_scale), device=dev)
    ).float()
    own = torch.arange(cp_rank, block_tables.shape[2], cp_size, device=dev)
    if own.numel() == 0:  # this rank owns no page of the table (tiny contexts)
        out_o.zero_()
        out_lse.fill_(float("-inf"))
        return
    k_blk = block_tables[:b, 0].index_select(1, own).long()
    v_blk = block_tables[:b, 1].index_select(1, own).long()
    k = kv_pool[k_blk, 0].float()  # [b, n_own, PAGE, D]
    v = kv_pool[v_blk, 0].float()
    pos = own[:, None] * page + torch.arange(page, device=dev)[None, :]  # [n_own, PAGE]
    # queries [b, q_len, G, D], group head g = s * hq + h
    q = (
        q_all.float()
        .reshape(cp, b, q_len, hq, D)
        .permute(1, 2, 0, 3, 4)
        .reshape(b, q_len, cp * hq, D)
    )
    s = torch.einsum("btgd,bnpd->btgnp", q, k) * bmm1
    limit = (
        seq_lens[:b].long()[:, None] - q_len + torch.arange(q_len, device=dev)[None, :] + 1
    )  # [b, q_len]
    vis = pos[None, None, :, :] < limit[:, :, None, None]  # [b, q_len, n_own, PAGE]
    s = s.masked_fill(~vis[:, :, None], float("-inf"))
    m = s.amax(dim=(3, 4), keepdim=True)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(s - m_safe)
    den = p.sum(dim=(3, 4))  # [b, q_len, G]
    o = torch.einsum("btgnp,bnpd->btgd", p, v) / den.clamp_min(1e-30)[..., None] * bmm2
    o = torch.where((den > 0)[..., None], o, torch.zeros_like(o))
    lse = torch.where(
        den > 0,
        m.squeeze(-1).squeeze(-1) * 1.4426950408889634 + torch.log2(den.clamp_min(1e-30)),
        torch.full_like(den, float("-inf")),
    )
    # [b, q_len, cp, hq, ...] -> [cp, rows, hq, ...]
    out_o.copy_(o.reshape(b, q_len, cp, hq, D).permute(2, 0, 1, 3, 4).reshape(cp, rows, hq, D))
    out_lse.copy_(lse.reshape(b, q_len, cp, hq).permute(2, 0, 1, 3).reshape(cp, rows, hq))


def _cp_fused_check(
    exchange,
    run_func,
    q,
    kv_pool,
    block_tables,
    seq_lens,
    max_kv_len,
    q_len,
    num_segments,
    bmm1_scale,
    bmm1_scale_arg,
    bmm2_scale,
    workspace,
    counter,
    sm_count,
    enable_pdl,
    out,
    gate,
    fmha_pdl,
    draft_owner,
) -> None:
    """TRTLLM_Q35_ATTN_CP_FUSED_CHECK: the merge-free path on the same inputs into a separate output."""
    global _CP_FUSED
    ref = torch.empty_like(out)
    _CP_FUSED = False
    try:
        q35_cp_decode_ll(
            exchange,
            run_func,
            q,
            kv_pool,
            block_tables,
            seq_lens,
            max_kv_len,
            q_len,
            num_segments,
            bmm1_scale,
            bmm1_scale_arg,
            bmm2_scale,
            workspace,
            counter,
            sm_count,
            enable_pdl,
            ref,
            gate=gate,
            fmha_pdl=fmha_pdl,
            draft_owner=draft_owner,
        )
    finally:
        _CP_FUSED = True
    o32, r32 = out.float(), ref.float()
    diff = (o32 - r32).abs()
    err = float(diff.max())
    worst = int(diff.argmax())
    ref_at = abs(float(r32.flatten()[worst]))
    # bf16 ulp at the reference value of the worst element; relative error vs that row's largest |reference|
    ulp = 2.0 ** (math.floor(math.log2(max(ref_at, 2.0**-126))) - 7)
    row_max = float(r32.abs().amax(dim=-1).flatten()[worst // r32.shape[-1]])
    rel = err / max(row_max, 1e-6)
    st = CP_FUSED_CHECK_STATS
    st["calls"] += 1
    st["max_abs"] = max(st["max_abs"], err)
    st["max_rel"] = max(st.get("max_rel", 0.0), rel)
    st["max_ulp"] = max(st.get("max_ulp", 0.0), err / ulp)
    st["sum_mean_abs"] = st.get("sum_mean_abs", 0.0) + float(diff.mean())
    from tensorrt_llm.logger import logger

    if not (rel <= 0.05):
        logger.warning(
            f"TRTLLM_Q35_ATTN_CP_FUSED_CHECK: max abs {err:.4f} at |ref| {ref_at:.4f} ({err / ulp:.1f} bf16 "
            f"ulp, {rel:.4f} of the row max {row_max:.4f}), mean abs {float(diff.mean()):.6f} vs merge-free "
            f"(rows {q.shape[0]}, q_len {q_len}, max_kv_len {max_kv_len})"
        )
    elif st["calls"] % 64 == 1:
        logger.info(
            f"TRTLLM_Q35_ATTN_CP_FUSED_CHECK: calls {st['calls']} max_abs {st['max_abs']:.4f} max_rel "
            f"{st['max_rel']:.4f} max_ulp {st['max_ulp']:.1f} mean_abs {st['sum_mean_abs'] / st['calls']:.6f}"
        )


def q35_cp_decode_ll(
    exchange,
    run_func,
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_kv_len: int,
    q_len: int,
    num_segments: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm1_scale_arg,
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    sm_count: int,
    enable_pdl: bool,
    out: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    fmha_pdl: bool = True,
    draft_owner: bool = True,
) -> torch.Tensor:
    """CP-lite generation attention of this rank's heads with one-launch LL exchanges within the KV group.

    exchange: TPGroupExchange (distributed/ll_exchange.py) whose groups are the KV groups (member = CP rank);
    channel 0 carries the q all-gather; the packed per-destination (O, LSE) partials go through the exchange's
    alltoall_merge on channel 1 (all-to-all + LSE merge + gate in one launch) or, with TRTLLM_Q35_ATTN_CP_A2A=combine
    or an exchange without it, through alltoall on channel 2 + q35_cp_combine.
    q [b * q_len, Hq_local, D] (FP8), out [b * q_len, Hq_local, D]; gate optional (out * sigmoid(gate)).
    """
    cp, cp_rank = exchange.group_size, exchange.grank
    rows, hq, d = q.shape
    merge = getattr(exchange, "alltoall_merge", None)
    merge_free = _CP_A2A == "merge" and merge is not None and _merge_takes_extra(type(exchange))
    fused_fn = _cp_fused_fn() if (_CP_FUSED and merge_free) else None
    if fused_fn is not None and _cp_fused_shape_ok(fused_fn, q, kv_pool, cp, q_len, sm_count):
        # TRTLLM_Q35_ATTN_CP_FUSED: all-gather -> one partial kernel (O / LSE of every group head over this rank's
        # pages, into alltoall_merge's layout) -> all-to-all + merge (+ gate). The kernel reads only host data and the
        # prefix KV before its wait, so the all-gather may release it right after its own wait.
        trigger = "after_wait" if (enable_pdl and _allgather_after_wait(type(exchange))) else False
        q_all = (
            exchange.allgather(q, channel=0, trigger=trigger)
            .view(rows, cp, hq, d)
            .permute(1, 0, 2, 3)
        )
        o = torch.empty(cp, rows, hq, d, dtype=out.dtype, device=q.device)
        lse = torch.empty(cp, rows, hq, dtype=torch.float32, device=q.device)
        fused_fn(
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
            sm_count=sm_count,
            enable_pdl=enable_pdl,
        )
        merge(o, lse, out.view(rows, hq, d), gate=gate, channel=1)
        if _CP_FUSED_CHECK and not torch.cuda.is_current_stream_capturing():
            _cp_fused_check(
                exchange,
                run_func,
                q,
                kv_pool,
                block_tables,
                seq_lens,
                max_kv_len,
                q_len,
                num_segments,
                bmm1_scale,
                bmm1_scale_arg,
                bmm2_scale,
                workspace,
                counter,
                sm_count,
                enable_pdl,
                out,
                gate,
                fmha_pdl,
                draft_owner,
            )
        return out
    draft_early = (
        merge_free and enable_pdl and _CP_DRAFT_EARLY and _allgather_after_wait(type(exchange))
    )
    # member-major [rows, cp * hq * d] -> [cp, rows, hq, d] (strided view; group head = src * hq + h)
    if draft_early:
        q_all = exchange.allgather(q, channel=0, trigger="after_wait")
    else:
        q_all = exchange.allgather(q, channel=0)
    q_all = q_all.view(rows, cp, hq, d).permute(1, 0, 2, 3)
    if merge_free:
        # No local merge kernel: the FMHA partials (several per member) and the own-head draft partial go straight
        # into the one-launch all-to-all + merge.
        o, lse, extra_o, extra_lse, valid = q35_cp_partial_merge_free(
            run_func,
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
            bmm1_scale_arg,
            bmm2_scale,
            workspace,
            counter,
            sm_count,
            None,
            enable_pdl,
            out_dtype=out.dtype,
            fmha_pdl=fmha_pdl,
            q_local=q if draft_early else None,
        )
        merge(
            o,
            lse,
            out.view(rows, hq, d),
            gate=gate,
            channel=1,
            extra_o=extra_o,
            extra_lse=extra_lse,
            valid=valid,
        )
        return out
    o_words = rows * hq * d * out.element_size() // 4
    assert o_words + rows * hq <= exchange.max_words, (
        "CP-lite partials exceed the exchange's per-pair payload"
    )
    send = torch.empty(cp, o_words + rows * hq, dtype=torch.int32, device=q.device)
    send_o = send[:, :o_words].view(out.dtype).view(cp, rows, hq, d)
    send_l = send[:, o_words:].view(torch.float32).view(cp, rows, hq)
    q35_cp_local_partial(
        run_func,
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
        bmm1_scale_arg,
        bmm2_scale,
        workspace,
        counter,
        sm_count,
        None,
        enable_pdl,
        include_draft=(cp_rank == 0 and not draft_owner),
        out_dtype=out.dtype,
        fmha_pdl=fmha_pdl,
        out_o=send_o,
        out_lse=send_l,
        draft_owner=draft_owner,
    )
    if _CP_A2A == "merge" and hasattr(exchange, "alltoall_merge"):
        exchange.alltoall_merge(send_o, send_l, out.view(rows, hq, d), gate=gate, channel=1)
        return out
    recv = exchange.alltoall(send, channel=2)
    recv_o = recv[:, :o_words].view(out.dtype).view(cp, rows, hq, d)
    recv_l = recv[:, o_words:].view(torch.float32).view(cp, rows, hq)
    q35_cp_combine(recv_o, recv_l, out.view(rows, hq, d), gate=gate, enable_pdl=enable_pdl)
    return out


def q35_cp_decode_nccl(
    group,
    cp_rank: int,
    run_func,
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_kv_len: int,
    q_len: int,
    num_segments: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm1_scale_arg,
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    sm_count: int,
    enable_pdl: bool,
    out: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CP-lite fallback with NCCL collectives (TRTLLM_Q35_ATTN_CP_NCCL=1, or no peer memory for TPGroupExchange):
    group all-gather of the FP8 q rows, owner-draft partials, alltoall_helix of the per-destination head slices,
    q35_cp_combine. group: the global ranks of the KV group (cp_rank = index in it)."""
    from tensorrt_llm._torch.distributed.ops import _allgather, alltoall_helix

    cp = len(group)
    rows, hq, d = q.shape
    esz = q.element_size()
    q_bytes = q.view(torch.uint8).reshape(rows, hq * d * esz)
    q_all = _allgather(q_bytes, group, cp_rank, None, dim=0)
    q_all = q_all.view(torch.uint8).view(cp, rows, hq, d * esz).view(q.dtype)
    part_o, part_lse = q35_cp_local_partial(
        run_func,
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
        bmm1_scale_arg,
        bmm2_scale,
        workspace,
        counter,
        sm_count,
        None,
        enable_pdl,
        include_draft=False,
        out_dtype=out.dtype,
        draft_owner=True,
    )
    chunks_o = [part_o[:, p * hq : (p + 1) * hq].contiguous() for p in range(cp)]
    chunks_l = [part_lse[:, p * hq : (p + 1) * hq].contiguous() for p in range(cp)]
    recv_o, recv_l = alltoall_helix(chunks_o + chunks_l, group)
    q35_cp_combine(
        recv_o.contiguous(),
        recv_l.contiguous(),
        out.view(rows, hq, d),
        gate=gate,
        enable_pdl=enable_pdl,
    )
    return out
