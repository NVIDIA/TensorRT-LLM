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
"""Segmented split-KV generation attention on top of the trtllm-gen decode kernel (small batch).

At small batch the trtllm-gen in-kernel split-KV reduction dominates. Instead:
  1. prep (Triton): fold the q_len draft tokens into the head dimension ([b*q_len, Hq, D] -> [b*S, Hkv*q_len*G, D])
     and split each request's context prefix KV[0, L - q_len) into S page-aligned segments;
  2. FMHA (trtllm-gen, one "request" per segment) returning the base-2 LSE per (segment, row);
  3. merge (Triton): LSE-weighted combine of the S segments plus the causal draft block KV[L - q_len, L) in fp32.
"""

import math
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

from ....modules.fla.op import gdc_wait_with_memory_clobber
from .flashinfer_trtllm_gen import _get_bmm1_scale_log2

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

LOG2E = 1.4426950408889634


@triton.jit
def seg_geometry(
    L, s, QL: tl.constexpr, S: tl.constexpr, PAGE: tl.constexpr, CP: tl.constexpr, RANK
):
    """Pages of the context prefix [0, L - QL) that member RANK of a CP-way page split owns (page p belongs to
    p mod CP), split into S segments; returns (first owned index, pages in segment s, tokens in segment s, prefix
    pages). Without a page split CP = 1, RANK = 0."""
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
def _seg_prep_kernel(
    q,
    q_stride_tok,
    q_stride_head,
    table,
    table_stride_req,
    table_stride_kv,
    seq_lens,
    seg_q,
    seg_table,
    seg_lens,
    QL: tl.constexpr,
    G: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    PAGE: tl.constexpr,
    P_SEG: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    pid = tl.program_id(0)  # r * S + s
    r = pid // S
    s = pid % S
    # Pre-wait: reads of host-uploaded lengths / tables only. Every write comes after the wait: the outputs are fresh
    # allocations, and the caching allocator may hand them a block that a kernel still running before the wait uses.
    L = tl.load(seq_lens + r)
    j0, npg, n, _ = seg_geometry(L, s, QL, S, PAGE, 1, 0)
    valid = n > 0
    kv = tl.arange(0, 2)
    p = tl.arange(0, BLOCK_P)
    src_p = tl.where(valid, j0 + p, 0)
    m1 = (p < P_SEG) & ((p < npg) | ((~valid) & (p == 0)))
    m = (kv[:, None] < 2) & m1[None, :]
    vals = tl.load(
        table + r.to(tl.int64) * table_stride_req + kv[:, None] * table_stride_kv + src_p[None, :],
        mask=m,
        other=0,
    )
    if LAUNCH_WITH_PDL:
        # q is written by the QKV preprocessing kernel right before this one.
        gdc_wait_with_memory_clobber()
    tl.store(seg_lens + pid, tl.where(valid, n, 1))
    tl.store(seg_table + (pid.to(tl.int64) * 2 + kv[:, None]) * P_SEG + p[None, :], vals, mask=m)
    HH: tl.constexpr = HKV * QL * G
    e = tl.arange(0, BLOCK_Q)
    em = e < HH * D
    hh = e // D
    dd = e % D
    kvh = hh // (QL * G)
    rem = hh % (QL * G)
    t = rem // G
    g = rem % G
    src = q + (r * QL + t).to(tl.int64) * q_stride_tok + (kvh * G + g) * q_stride_head + dd
    tl.store(seg_q + pid.to(tl.int64) * HH * D + e, tl.load(src, mask=em), mask=em)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def draft_block(
    q,
    q_stride_tok,
    q_stride_head,
    head,
    kv_pool,
    kvh,
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
    HKV: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    HAS_BMM1_PTR: tl.constexpr,
    HAS_BMM2_PTR: tl.constexpr,
):
    """(O, base-2 LSE) of query (row, head) over the causal q_len x q_len draft block of KV head kvh, fp32 from the
    FP8 q / K / V (jmask holds at least j = 0)."""
    if HAS_BMM1_PTR:
        bmm1 = tl.load(bmm1_ptr).to(tl.float32)
    else:
        bmm1 = bmm1_val
    if HAS_BMM2_PTR:
        bmm2 = tl.load(bmm2_ptr).to(tl.float32)
    else:
        bmm2 = bmm2_val
    qv = tl.load(q + row.to(tl.int64) * q_stride_tok + head * q_stride_head + d).to(tl.float32)
    k_t = tl.load(
        kv_pool + ((blk_k * HKV + kvh) * PAGE + off)[:, None] * D + d[None, :],
        mask=jmask[:, None],
        other=0.0,
    ).to(tl.float32)
    v_t = tl.load(
        kv_pool + ((blk_v * HKV + kvh) * PAGE + off)[:, None] * D + d[None, :],
        mask=jmask[:, None],
        other=0.0,
    ).to(tl.float32)
    s_j = tl.sum(k_t * qv[None, :], axis=1) * bmm1
    s_j = tl.where(jmask, s_j, float("-inf"))
    m_suf = tl.max(s_j, axis=0)
    p = tl.where(jmask, tl.exp(s_j - m_suf), 0.0)
    l_suf = tl.sum(p, axis=0)
    o_suf = tl.sum(p[:, None] * v_t, axis=0) * (bmm2 / l_suf)
    lse_suf = m_suf * 1.4426950408889634 + tl.log2(l_suf)
    return o_suf, lse_suf


@triton.jit
def _seg_merge_kernel(
    seg_out,
    seg_lse,
    q,
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
    out,
    out_stride_tok,
    out_stride_head,
    gate,
    gate_stride_tok,
    gate_stride_head,
    QL: tl.constexpr,
    BQL: tl.constexpr,
    G: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_BMM1_PTR: tl.constexpr,
    HAS_BMM2_PTR: tl.constexpr,
    HAS_GATE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr = False,
):
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)  # r * QL + t
    head = tl.program_id(1)  # kvh * G + g
    r = row // QL
    t = row % QL
    kvh = head // G
    g = head % G
    HH: tl.constexpr = HKV * QL * G
    hh = kvh * (QL * G) + t * G + g
    d = tl.arange(0, D)
    # Pre-wait: host-uploaded lengths / block tables only.
    L = tl.load(seq_lens + r)
    prefix = L - QL
    j = tl.arange(0, BQL)
    jmask = (j <= t) & (j < QL)
    pos = prefix + j
    tab = table + r.to(tl.int64) * table_stride_req
    blk_k = tl.load(tab + pos // PAGE, mask=jmask, other=0).to(tl.int64)
    blk_v = tl.load(tab + table_stride_kv + pos // PAGE, mask=jmask, other=0).to(tl.int64)
    off = (pos % PAGE).to(tl.int64)
    n_pages = (prefix + PAGE - 1) // PAGE
    seg_pages = (n_pages + S - 1) // S
    sv = tl.arange(0, BS)
    n_s = tl.minimum(tl.maximum(prefix - sv * seg_pages * PAGE, 0), seg_pages * PAGE)
    smask = (sv < S) & (n_s > 0)
    if LAUNCH_WITH_PDL:
        # Segment outputs / LSE and scales come from the preceding kernels, q and the draft K/V from the QKV
        # preprocessing.
        gdc_wait_with_memory_clobber()
    o_suf, lse_suf = draft_block(
        q,
        q_stride_tok,
        q_stride_head,
        head,
        kv_pool,
        kvh,
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
        HKV,
        PAGE,
        D,
        HAS_BMM1_PTR,
        HAS_BMM2_PTR,
    )
    seg_rows = (r * S + sv).to(tl.int64) * HH + hh
    lse_s = tl.load(seg_lse + seg_rows, mask=smask, other=float("-inf"))
    o_s = tl.load(seg_out + seg_rows[:, None] * D + d[None, :], mask=smask[:, None], other=0.0).to(
        tl.float32
    )

    m_all = tl.maximum(tl.max(lse_s, axis=0), lse_suf)
    w_s = tl.where(smask, tl.exp2(lse_s - m_all), 0.0)
    w_suf = tl.exp2(lse_suf - m_all)
    acc = tl.sum(w_s[:, None] * o_s, axis=0) + w_suf * o_suf
    res = (acc / (tl.sum(w_s, axis=0) + w_suf)).to(out.dtype.element_ty)
    if HAS_GATE:
        # Same arithmetic as the separate _fused_sigmoid_mul_kernel: rounded attention output x sigmoid(gate) in fp32.
        gv = tl.load(gate + row.to(tl.int64) * gate_stride_tok + head * gate_stride_head + d).to(
            tl.float32
        )
        res = (res.to(tl.float32) * tl.sigmoid(gv)).to(out.dtype.element_ty)
    tl.store(out + row.to(tl.int64) * out_stride_tok + head * out_stride_head + d, res)
    if LAUNCH_WITH_PDL and not EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


def segment_attention(
    seg_q: torch.Tensor,
    kv_pool: torch.Tensor,
    seg_table: torch.Tensor,
    seg_lens: torch.Tensor,
    max_seg_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    seg_out: torch.Tensor,
    seg_lse: torch.Tensor,
) -> None:
    """trtllm-gen decode of every segment as a one-token request of seg_q's folded heads, with its base-2 LSE.

    Calls FlashInfer's trtllm-gen decode op directly (the op behind ``trtllm_batch_decode_with_kv_cache``), skipping
    the public wrapper's per-call argument validation. Launched without PDL: the FMHA may read the segment tables /
    lengths, which the kernel right before it writes, before its dependency wait.
    """
    num_requests, num_heads, _ = seg_q.shape
    sm_count = flashinfer.decode.get_device_sm_count(seg_q.device)
    assert counter.numel() * counter.element_size() >= 4 * max(
        num_requests * num_heads, sm_count
    ), "multi-CTA counter buffer too small for the segment attention call"
    flashinfer.decode.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode(
        seg_out,
        None,  # out_scale_factor
        seg_q,
        kv_pool,
        kv_pool,
        workspace,
        counter,
        seg_table,
        seg_lens,
        1,  # max_q_len
        max_seg_len,
        _get_bmm1_scale_log2(bmm1_scale) if isinstance(bmm1_scale, torch.Tensor) else bmm1_scale,
        bmm2_scale,
        -1.0,  # o_sf_scale
        -1,  # o_sf_vec_size
        0,  # o_sf_start_index
        num_requests,
        -1,  # window_left
        0,  # sparse_mla_top_k
        sm_count,
        False,  # enable_pdl
        workspace.numel() * workspace.element_size(),
        None,  # sinks
        None,  # cum_seq_lens_q
        None,  # k_block_scales
        None,  # v_block_scales
        None,  # skip_softmax_threshold_scale_factor
        False,  # uses_shared_paged_kv_idx
        seg_lse,
        1.0,  # lse_scale
        seg_lse.stride(0),
        seg_lse.stride(1),
        False,  # enable_block_sparse_attention
        None,  # sparse_mla_top_k_lens
        0,  # bf16q_fp8kv_transform_mode (FP8 query)
        None,  # use_fp16_softmax
    )


def segmented_decode_supported(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_len: int,
    batch: int,
    num_segments: int,
    window_left: int,
    has_sinks: bool,
    tokens_per_block: int,
    q_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> bool:
    return (
        num_segments > 1
        and batch * num_segments <= 256
        and 1 <= q_len <= 16
        and num_heads % num_kv_heads == 0
        and q_len * (num_heads // num_kv_heads) <= 64
        and head_dim in (64, 128, 256)
        and window_left < 0
        and not has_sinks
        and tokens_per_block in (16, 32, 64, 128)
        and q_dtype in (torch.float8_e4m3fn, torch.bfloat16, torch.float16)
        and out_dtype in (torch.bfloat16, torch.float16)
    )


def segmented_decode(
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_kv_len: int,
    q_len: int,
    num_kv_heads: int,
    num_segments: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    workspace: torch.Tensor,
    counter: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: bool,
    gate: Optional[torch.Tensor] = None,
) -> None:
    """Segmented generation attention; writes `out` [b * q_len, Hq, D].

    Args:
        q: [b * q_len, Hq, D] (FP8 when the KV cache is FP8); kv_pool: [blocks, Hkv, page, D].
        block_tables: [b, 2, max_pages] K / V page indices; seq_lens: [b] KV lengths including the new tokens.
        bmm1_scale: float or the trtllm-gen [raw, log2] scale tensor.
        counter: zero-initialized trtllm-gen multi-CTA counters for b * num_segments requests of the folded heads.
        gate: [b * q_len, Hq, D] or [b * q_len, Hq * D]; the output is multiplied by sigmoid(gate).
    """
    S = num_segments
    num_tokens, Hq, D = q.shape
    b = num_tokens // q_len
    Hkv = num_kv_heads
    G = Hq // Hkv
    HH = Hkv * q_len * G
    page = kv_pool.shape[-2]
    assert kv_pool.is_contiguous() and kv_pool.shape[1] == Hkv and kv_pool.shape[-1] == D
    assert block_tables.dim() == 3 and block_tables.shape[1] == 2 and block_tables.stride(-1) == 1
    assert out.stride(-1) == 1 and q.stride(-1) == 1
    max_pages = math.ceil(max(max_kv_len, 1) / page)
    p_seg = math.ceil(max_pages / S) + 1
    dev = q.device
    seg_q = torch.empty(b * S, HH, D, dtype=q.dtype, device=dev)
    seg_table = torch.empty(b * S, 2, p_seg, dtype=torch.int32, device=dev)
    seg_lens = torch.empty(b * S, dtype=torch.int32, device=dev)
    seg_out = torch.empty(b * S, HH, D, dtype=out.dtype, device=dev)
    seg_lse = torch.empty(b * S, HH, dtype=torch.float32, device=dev)
    _seg_prep_kernel[(b * S,)](
        q,
        q.stride(0),
        q.stride(1),
        block_tables,
        block_tables.stride(0),
        block_tables.stride(1),
        seq_lens,
        seg_q,
        seg_table,
        seg_lens,
        QL=q_len,
        G=G,
        HKV=Hkv,
        D=D,
        S=S,
        PAGE=page,
        P_SEG=p_seg,
        BLOCK_P=triton.next_power_of_2(p_seg),
        BLOCK_Q=triton.next_power_of_2(HH * D),
        LAUNCH_WITH_PDL=enable_pdl,
        num_warps=4,
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
    has_b1 = isinstance(bmm1_scale, torch.Tensor)
    has_b2 = isinstance(bmm2_scale, torch.Tensor)
    if gate is not None:
        gate = gate.view(num_tokens, Hq, D)
        assert gate.stride(-1) == 1
    _seg_merge_kernel[(b * q_len, Hq)](
        seg_out,
        seg_lse,
        q,
        q.stride(0),
        q.stride(1),
        kv_pool,
        block_tables,
        block_tables.stride(0),
        block_tables.stride(1),
        seq_lens,
        bmm1_scale if has_b1 else seg_lens,
        0.0 if has_b1 else float(bmm1_scale),
        bmm2_scale if has_b2 else seg_lens,
        0.0 if has_b2 else float(bmm2_scale),
        out,
        out.stride(0),
        out.stride(1),
        gate if gate is not None else out,
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        QL=q_len,
        BQL=max(2, triton.next_power_of_2(q_len)),
        G=G,
        HKV=Hkv,
        D=D,
        S=S,
        BS=max(2, triton.next_power_of_2(S)),
        PAGE=page,
        HAS_BMM1_PTR=has_b1,
        HAS_BMM2_PTR=has_b2,
        HAS_GATE=gate is not None,
        LAUNCH_WITH_PDL=enable_pdl,
        EARLY_TRIGGER=True,
        # [BS, D] fp32 segment partials per program: more warps above 16 segments (fewer registers per thread)
        num_warps=4 if S <= 16 else (8 if S <= 64 else 16),
        launch_pdl=enable_pdl,
    )
