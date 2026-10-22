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
"""Context-parallel ("CP-lite") generation attention for Qwen3.5 when TP replicates each KV head over a group of ranks.

Every rank still appends every token, but page p of a request is read only by CP rank (p mod CP) of the KV group:
  1. all-gather q within the KV group (distributed/ll_exchange.py),
  2. partial (O, LSE) of every group head over this rank's pages: fused kernel (q35_cp_fused.py, contract
     q35_cp_fused_partial_ref) or the merge-free trtllm-gen path (q35_cp_partial_merge_free),
  3. one-launch all-to-all + base-2 LSE merge (+ output gate) of each rank's own heads.
"""

import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from .q35_segmented import _gdc_wait_with_memory_clobber

# Rows of one fused call per KV head (q_len x group heads) the kernel supports (MMA N <= 128).
_CP_FUSED_MAX_ROWS = 128


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
):
    """(O, base-2 LSE) of one (row, group head) over the causal draft block (jmask holds at least j = 0)."""
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
    o_suf = tl.sum(p[:, None] * v_t, axis=0) * (bmm2 / l_suf)
    lse_suf = m_suf * 1.4426950408889634 + tl.log2(l_suf)
    return o_suf, lse_suf


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
    q_local: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """This rank's partials for TPGroupExchange.alltoall_merge with extra_o / extra_lse (no local merge kernel).

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
        BQL=max(2, triton.next_power_of_2(q_len)),
        HAS_BMM1_PTR=has_b1,
        HAS_BMM2_PTR=has_b2,
        DRAFT_PRE_WAIT=q_local is not None,
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
        False,  # enable_pdl: the FMHA may read seg_lens / seg_table before its dependency wait
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
        False,  # enable_block_sparse_attention
        None,  # sparse_mla_top_k_lens
        0,  # bf16q_fp8kv_transform_mode (FP8 query)
        None,  # use_fp16_softmax
    )
    # request s * b + r, folded head hh = t * GH + d * HQL + h  ->  [d, s, r * q_len + t, h]
    o = seg_out.view(S, num_tokens, CP, HQL, D).permute(2, 0, 1, 3, 4)
    lse = seg_lse.view(S, num_tokens, CP, HQL).permute(2, 0, 1, 3)
    return o, lse, extra_o, extra_lse, valid


def _cp_fused_shape_ok(
    q: torch.Tensor, kv_pool: torch.Tensor, cp: int, q_len: int, sm_count: int
) -> bool:
    """Whether the fused CP partial kernel (fmha/q35_cp_fused.py) takes this call."""
    rows, hq, d = q.shape
    if not (
        IS_CUTLASS_DSL_AVAILABLE
        and q.dtype == torch.float8_e4m3fn
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
    from .q35_cp_fused import q35_cp_fused_supported

    return q35_cp_fused_supported(q_len, cp, hq, rows // q_len, sm_count)


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
    """Contract of the fused CP partial kernel (fmha/q35_cp_fused.py); FP32 torch reference for tests.

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
) -> torch.Tensor:
    """CP-lite generation attention of this rank's heads with one-launch LL exchanges within the KV group.

    exchange: TPGroupExchange (distributed/ll_exchange.py) whose groups are the KV groups (member = CP rank);
    channel 0 carries the q all-gather, channel 1 the all-to-all + LSE merge (+ gate) of the per-destination
    (O, LSE) partials. q [b * q_len, Hq_local, D] (FP8), out [b * q_len, Hq_local, D]; gate optional
    (out * sigmoid(gate)). run_func: flashinfer trtllm_paged_attention_decode (merge-free path).
    """
    cp, cp_rank = exchange.group_size, exchange.grank
    rows, hq, d = q.shape
    # With PDL the all-gather releases its dependents right after its own wait on the QKV preprocessing: the fused
    # kernel reads only host data and the prefix KV before its wait, and the merge-free prep kernel's own-head draft
    # programs read the local q / draft K/V before theirs, overlapping the exchange. Relies on the predecessor's
    # completion flush being device-wide.
    trigger = "after_wait" if enable_pdl else False
    q_all = (
        exchange.allgather(q, channel=0, trigger=trigger).view(rows, cp, hq, d).permute(1, 0, 2, 3)
    )
    if _cp_fused_shape_ok(q, kv_pool, cp, q_len, sm_count):
        from .q35_cp_fused import q35_cp_fused_partial

        o = torch.empty(cp, rows, hq, d, dtype=out.dtype, device=q.device)
        lse = torch.empty(cp, rows, hq, dtype=torch.float32, device=q.device)
        q35_cp_fused_partial(
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
        exchange.alltoall_merge(o, lse, out.view(rows, hq, d), gate=gate, channel=1)
        return out
    # No local merge kernel: the FMHA partials (several per member) and the own-head draft partial go straight into
    # the one-launch all-to-all + merge.
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
    return out
