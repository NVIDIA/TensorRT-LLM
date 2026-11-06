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
"""Fused rejection sampling for one-model block drafters (DSpark and DFlash), as Triton kernels.

With ``TLLM_SPEC_FAST_RS=1``, `SpecWorkerBase` calls these kernels instead of its reference path
(`SpecWorkerBase._sample_and_accept_draft_tokens_rejection` and the rejection branch of
`SpecWorkerBase.advanced_sample_draft_block`). The reference path runs about ten flashinfer and
PyTorch launches per step, most of them one CTA per row over the full vocabulary, which leaves the
GPU nearly idle for hundreds of microseconds at a few rows per step. Here every row is split into
``_CHUNK``-wide pieces, so a few rows fill the GPU.

The kernels follow the reference temperature-only path (``AdvancedSamplingMode.NO_TOPK_NO_TOPP``):
flashinfer's ``softmax`` and ``sampling_from_probs`` for the drafts, and ``chain_speculative_sampling``
for the verify. They have static shapes and never sync with the host, so they capture into CUDA
graphs. All reductions run in a fixed order (no atomics), so every tensor-parallel rank holding the
same logits draws the same tokens. Logits of ``-inf`` (masked tokens) get probability 0, including
chunks with no finite logit.
"""

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

FAST_RS_ENV_VAR = "TLLM_SPEC_FAST_RS"

# Vocab elements per program (one load per program; 163840 = 40 chunks).
_CHUNK = 4096
_NUM_WARPS = 8
# cuRAND's _curand_uniform: x * 2^-32 + 2^-33 (2.3283064e-10f is exactly 2^-32). constexpr:
# @triton.jit code may only read module globals that are tl.constexpr.
_U32_TO_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
_U32_TO_UNIFORM_BIAS = tl.constexpr(1.1641532182693481e-10)
# Philox layout. The kernels draw cuRAND Philox4x32-10 uniforms (curand_init(seed, subsequence,
# offset) + curand_uniform), with each request's own seed[b] and offset[b]:
# - The verify of request b draws exactly as chain_speculative_sampling (the reference behind
#   rejection_sampling_one_model): curand_init(seed[b], PER_ROW_SUBSEQUENCE_BASE | offset[b], 0),
#   draw i for accept test i up to the first rejected position j, then the next draw (j + 1, or K
#   when every draft is accepted), as 1 - u, for the replacement or bonus token. The caller passes
#   the offset of the request's acceptance slot, as the reference does.
# - Draft position i of request b uses draw 0 of subsequence b * K + i at word offset + 1, keyed
#   with seed ^ DRAFT_KEY_XOR, so it never reuses a word of a sampler keyed with the plain seed.
DRAFT_KEY_XOR = 0x3C6EF372FE94F82B
_DRAFT_KEY_XOR = tl.constexpr(DRAFT_KEY_XOR)
# kPerRowSubsequenceBase (1 << 63) of the C++ samplers, as the int64 bit pattern.
PER_ROW_SUBSEQUENCE_BASE = -(1 << 63)
_PER_ROW_SUBSEQUENCE_BASE = tl.constexpr(PER_ROW_SUBSEQUENCE_BASE)


def fast_rs_enabled() -> bool:
    """Whether ``TLLM_SPEC_FAST_RS=1`` selects the fused rejection-sampling kernels."""
    return os.environ.get(FAST_RS_ENV_VAR, "0") == "1"


@triton.jit
def _curand_uniform(seed, offset, subsequence, draw):
    """Draw ``draw`` (0-based) of ``curand_init(seed, subsequence, offset)`` +
    ``curand_uniform``: Philox4x32-10 with key = seed, counter = (offset / 4 + (offset % 4 +
    draw) / 4, subsequence), output word (offset % 4 + draw) % 4, mapped to (0, 1]."""
    g = (offset & 3) + draw
    ctr = (offset >> 2) + (g >> 2)
    c0 = (ctr & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((ctr >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c2 = (subsequence & 0xFFFFFFFF).to(tl.uint32)
    c3 = ((subsequence >> 32) & 0xFFFFFFFF).to(tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, c2, c3, 10)
    e = g & 3
    word = tl.where(e == 0, r0, tl.where(e == 1, r1, tl.where(e == 2, r2, r3)))
    return tl.fma(word.to(tl.float32), _U32_TO_UNIFORM_SCALE, _U32_TO_UNIFORM_BIAS)


@triton.jit
def _verify_uniform(seed, offset, draw):
    """Draw ``draw`` (0-based) of a request's acceptance stream, as chain_speculative_sampling draws
    it: ``curand_init(seed, PER_ROW_SUBSEQUENCE_BASE | offset, 0)`` + ``curand_uniform``.
    ``offset - offset`` is an int64 zero, which keeps the counter arithmetic in 64 bits."""
    return _curand_uniform(seed, offset - offset, offset | _PER_ROW_SUBSEQUENCE_BASE, draw)


@triton.jit
def _inv_temperature(temp):
    # flashinfer: inv_temp = (T == 0) ? 0 : 1.f / T (IEEE division).
    return tl.where(temp == 0.0, 0.0, tl.math.div_rn(1.0, tl.where(temp == 0.0, 1.0, temp)))


@triton.jit
def _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2: tl.constexpr):
    """Combine a row's per-chunk (max, sum exp(x - max)) partials in a fixed order."""
    offs = tl.arange(0, NC_POW2)
    mask = offs < num_chunks
    m = tl.load(pmax_ptr + row * num_chunks + offs, mask=mask, other=float("-inf"))
    s = tl.load(psum_ptr + row * num_chunks + offs, mask=mask, other=0.0)
    big_m = tl.max(m, 0)
    # A chunk with no finite logit has m = -inf and s = 0; keep exp(m - big_m) defined.
    safe_m = tl.where(big_m == float("-inf"), 0.0, big_m)
    scale = tl.where(mask, tl.exp(m - safe_m), 0.0)
    big_s = tl.sum(s * scale, 0)
    return big_m, big_s


@triton.jit
def _load_scaled(
    logits_ptr,
    row,
    stride_row,
    bias_ptr,
    bias_row,
    stride_bias,
    offs,
    mask,
    inv_t,
    is_pad,
    HAS_BIAS: tl.constexpr,
):
    """``(logits + bias) / T`` for one chunk; padding requests read as zero logits (the reference
    zeroes them, which gives a uniform, legal distribution)."""
    x = tl.load(logits_ptr + row * stride_row + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        b = tl.load(bias_ptr + bias_row * stride_bias + offs, mask=mask, other=0.0)
        x = x + b.to(tl.float32)
    return tl.where(is_pad, 0.0, x * inv_t)


@triton.jit
def _rs_row_stats_kernel(
    logits_ptr,
    stride_row,
    bias_ptr,
    stride_bias,
    temps_ptr,
    slot_ptr,
    dummy_row,
    rows_per_req,
    temps_div,
    pmax_ptr,
    psum_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    inv_t = _inv_temperature(tl.load(temps_ptr + row // temps_div))
    is_pad = tl.load(slot_ptr + row // rows_per_req) == dummy_row
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = _load_scaled(
        logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask, inv_t, is_pad, HAS_BIAS
    )
    x = tl.where(mask, x, float("-inf"))
    m = tl.max(x, 0)
    # All -inf (masked) chunk: m stays -inf and the chunk adds nothing (exp(-inf - 0) = 0).
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    s = tl.sum(tl.where(mask, tl.exp(x - safe_m), 0.0), 0)
    tl.store(pmax_ptr + row * num_chunks + ch, m)
    tl.store(psum_ptr + row * num_chunks + ch, s)


@triton.jit
def _rs_draft_write_kernel(
    logits_ptr,
    stride_row,
    bias_ptr,
    stride_bias,
    temps_ptr,
    slot_ptr,
    dummy_row,
    rows_per_req,
    step0,
    pmax_ptr,
    psum_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    csum_ptr,
    clast_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    NC_POW2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    req = row // rows_per_req
    step = step0 + row % rows_per_req
    big_m, big_s = _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2)
    inv_s = tl.math.div_rn(1.0, big_s)
    inv_t = _inv_temperature(tl.load(temps_ptr + req))
    slot = tl.load(slot_ptr + req)
    is_pad = slot == dummy_row
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = _load_scaled(
        logits_ptr, row, stride_row, bias_ptr, row, stride_bias, offs, mask, inv_t, is_pad, HAS_BIAS
    )
    q = tl.exp(x - big_m) * inv_s
    q = tl.where(mask, q, 0.0)
    tl.store(probs_ptr + slot * stride_slot + step * stride_step + offs, q, mask=mask)
    tl.store(csum_ptr + row * num_chunks + ch, tl.sum(q, 0))
    last = tl.max(tl.where(q > 0.0, offs, -1), 0)
    tl.store(clast_ptr + row * num_chunks + ch, last)


@triton.jit
def _pick_chunk(csum_ptr, clast_ptr, base_row, num_chunks, u):
    """First chunk whose running mass exceeds ``u`` (sequential, flashinfer's rule), the mass
    before it, and the row's last nonzero index (the fallback when nothing exceeds ``u``)."""
    agg = 0.0
    kstar = -1
    base = 0.0
    last = -1
    for k in range(num_chunks):
        c = tl.load(csum_ptr + base_row * num_chunks + k)
        lk = tl.load(clast_ptr + base_row * num_chunks + k)
        hit = (kstar < 0) & (agg + c > u)
        kstar = tl.where(hit, k, kstar)
        base = tl.where(hit, agg, base)
        agg = agg + c
        last = tl.maximum(last, lk)
    return kstar, base, last


@triton.jit
def _rs_draft_pick_kernel(
    probs_ptr,
    stride_slot,
    stride_step,
    slot_ptr,
    rows_per_req,
    step0,
    steps_total,
    csum_ptr,
    clast_ptr,
    seed_ptr,
    offset_ptr,
    step_offset,
    tokens_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = row // rows_per_req
    step = step0 + row % rows_per_req
    seed = tl.load(seed_ptr + req) ^ _DRAFT_KEY_XOR
    offset = tl.load(offset_ptr + req) + step_offset
    # The block sampler's row (request-major, K per request) is the Philox subsequence; a
    # per-step launch (Markov feedback chain) keeps the same stream for each draft position.
    u = _curand_uniform(seed, offset, req * steps_total + step, 0)
    kstar, base, last = _pick_chunk(csum_ptr, clast_ptr, row, num_chunks, u)
    slot = tl.load(slot_ptr + req)
    kk = tl.maximum(kstar, 0)
    offs = kk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    q = tl.load(probs_ptr + slot * stride_slot + step * stride_step + offs, mask=mask, other=0.0)
    cdf = base + tl.cumsum(q, 0)
    hit = (q > 0.0) & (cdf > u) & mask
    first = tl.min(tl.where(hit, offs, 2147483647), 0)
    chunk_last = tl.max(tl.where((q > 0.0) & mask, offs, -1), 0)
    # Rounding can leave the in-chunk running sum just under u although the chunk total
    # crossed it: the draw then lands on the chunk's last nonzero entry.
    token = tl.where(first < 2147483647, first, chunk_last)
    token = tl.where(kstar >= 0, token, last)
    token = tl.where(token >= 0, token, 0)
    tl.store(tokens_ptr + req * steps_total + step, token.to(tl.int32))


@triton.jit
def _markov_feedback_emb_kernel(
    tokens_ptr,
    steps_total,
    step,
    anchor_ptr,
    w1_ptr,
    emb_ptr,
    vocab,
    RANK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """``markov_w1[prev]`` for draft position ``step``: prev is the anchor for position 0 and
    the token drawn at position ``step - 1`` otherwise; out-of-vocab ids give a zero row (as
    `markov_prev_embeddings`)."""
    req = tl.program_id(0).to(tl.int64)
    if step == 0:
        prev = tl.load(anchor_ptr + req).to(tl.int64)
    else:
        prev = tl.load(tokens_ptr + req * steps_total + step - 1).to(tl.int64)
    valid = (prev >= 0) & (prev < vocab)
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < RANK
    w = tl.load(w1_ptr + tl.where(valid, prev, 0) * RANK + offs_r, mask=mask_r & valid, other=0.0)
    tl.store(emb_ptr + req * RANK + offs_r, w, mask=mask_r)


@triton.jit
def _rs_accept_pos(
    tlogits_ptr,
    stride_row,
    temps_ptr,
    pmax_ptr,
    psum_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    req,
    slot,
    is_pad,
    seed,
    offset,
    num_draft,
    vocab,
    num_chunks,
    NC_POW2: tl.constexpr,
):
    """chain_speculative_sampling's accept loop: position i is accepted when
    ``u_i * q_draft(d_i) < p_target(d_i)``; returns the first rejected position (K if none)."""
    pos = num_draft
    for i in range(num_draft):
        row = req * (num_draft + 1) + i
        big_m, big_s = _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2)
        inv_s = tl.math.div_rn(1.0, big_s)
        inv_t = _inv_temperature(tl.load(temps_ptr + row))
        d = tl.load(draft_ptr + req * num_draft + i).to(tl.int64)
        d = tl.minimum(tl.maximum(d, 0), vocab - 1)
        x = tl.load(tlogits_ptr + row * stride_row + d)
        x = tl.where(is_pad, 0.0, x * inv_t)
        p_t = tl.exp(x - big_m) * inv_s
        q_d = tl.load(probs_ptr + slot * stride_slot + i * stride_step + d)
        u = _verify_uniform(seed, offset, i)
        reject = (pos == num_draft) & (~(u * q_d < p_t))
        pos = tl.where(reject, i, pos)
    return pos


@triton.jit
def _rs_verify_resid_kernel(
    tlogits_ptr,
    stride_row,
    temps_ptr,
    slot_ptr,
    dummy_row,
    num_draft,
    pmax_ptr,
    psum_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    seed_ptr,
    offset_ptr,
    pos_ptr,
    rsum_ptr,
    rlast_ptr,
    tsum_ptr,
    tlast_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    NC_POW2: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    seed = tl.load(seed_ptr + req)
    offset = tl.load(offset_ptr + req)
    slot = tl.load(slot_ptr + req)
    is_pad = slot == dummy_row
    pos = _rs_accept_pos(
        tlogits_ptr,
        stride_row,
        temps_ptr,
        pmax_ptr,
        psum_ptr,
        probs_ptr,
        stride_slot,
        stride_step,
        draft_ptr,
        req,
        slot,
        is_pad,
        seed,
        offset,
        num_draft,
        vocab,
        num_chunks,
        NC_POW2,
    )
    row = req * (num_draft + 1) + pos
    big_m, big_s = _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2)
    inv_s = tl.math.div_rn(1.0, big_s)
    inv_t = _inv_temperature(tl.load(temps_ptr + row))
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = tl.load(tlogits_ptr + row * stride_row + offs, mask=mask, other=0.0)
    x = tl.where(is_pad, 0.0, x * inv_t)
    p_t = tl.exp(x - big_m) * inv_s
    has_draft = pos < num_draft
    q_d = tl.load(
        probs_ptr + slot * stride_slot + tl.minimum(pos, num_draft - 1) * stride_step + offs,
        mask=mask & has_draft,
        other=0.0,
    )
    res = tl.where(mask, tl.maximum(p_t - q_d, 0.0), 0.0)
    tl.store(rsum_ptr + req * num_chunks + ch, tl.sum(res, 0))
    tl.store(rlast_ptr + req * num_chunks + ch, tl.max(tl.where(res > 0.0, offs, -1), 0))
    # The target alone, which the pick kernel samples when the residual has no mass.
    p_t = tl.where(mask, p_t, 0.0)
    tl.store(tsum_ptr + req * num_chunks + ch, tl.sum(p_t, 0))
    tl.store(tlast_ptr + req * num_chunks + ch, tl.max(tl.where(p_t > 0.0, offs, -1), 0))
    if ch == 0:
        tl.store(pos_ptr + req, pos)


@triton.jit
def _rs_verify_pick_kernel(
    tlogits_ptr,
    stride_row,
    temps_ptr,
    slot_ptr,
    dummy_row,
    num_draft,
    pmax_ptr,
    psum_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    seed_ptr,
    offset_ptr,
    pos_ptr,
    rsum_ptr,
    rlast_ptr,
    tsum_ptr,
    tlast_ptr,
    out_tokens_ptr,
    out_num_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    NC_POW2: tl.constexpr,
    KP1_POW2: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    seed = tl.load(seed_ptr + req)
    offset = tl.load(offset_ptr + req)
    slot = tl.load(slot_ptr + req)
    is_pad = slot == dummy_row
    pos = tl.load(pos_ptr + req)
    has_draft = pos < num_draft
    # The draw after the accept draws (see the Philox layout above DRAFT_KEY_XOR), as 1 - u in
    # [0, 1), like chain_speculative_sampling.
    u = 1.0 - _verify_uniform(seed, offset, tl.where(has_draft, pos + 1, num_draft))
    r_total = 0.0
    t_total = 0.0
    for k in range(num_chunks):
        r_total = r_total + tl.load(rsum_ptr + req * num_chunks + k)
        t_total = t_total + tl.load(tsum_ptr + req * num_chunks + k)
    # A residual without mass (target == draft) falls back to the target, as the reference does.
    target_only = (~has_draft) | (~(r_total > 0.0))
    u = u * tl.where(target_only, t_total, r_total)
    r_kstar, r_base, r_last = _pick_chunk(rsum_ptr, rlast_ptr, req, num_chunks, u)
    t_kstar, t_base, t_last = _pick_chunk(tsum_ptr, tlast_ptr, req, num_chunks, u)
    kstar = tl.where(target_only, t_kstar, r_kstar)
    base = tl.where(target_only, t_base, r_base)
    last = tl.where(target_only, t_last, r_last)
    row = req * (num_draft + 1) + pos
    big_m, big_s = _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2)
    inv_s = tl.math.div_rn(1.0, big_s)
    inv_t = _inv_temperature(tl.load(temps_ptr + row))
    kk = tl.maximum(kstar, 0)
    offs = kk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = tl.load(tlogits_ptr + row * stride_row + offs, mask=mask, other=0.0)
    x = tl.where(is_pad, 0.0, x * inv_t)
    p_t = tl.exp(x - big_m) * inv_s
    q_d = tl.load(
        probs_ptr + slot * stride_slot + tl.minimum(pos, num_draft - 1) * stride_step + offs,
        mask=mask & (~target_only),
        other=0.0,
    )
    res = tl.where(mask, tl.maximum(p_t - q_d, 0.0), 0.0)
    cdf = base + tl.cumsum(res, 0)
    hit = (res > 0.0) & (cdf > u)
    first = tl.min(tl.where(hit, offs, 2147483647), 0)
    chunk_last = tl.max(tl.where(res > 0.0, offs, -1), 0)
    token = tl.where(first < 2147483647, first, chunk_last)
    token = tl.where(kstar >= 0, token, last)
    token = tl.where(token >= 0, token, 0)
    cols = tl.arange(0, KP1_POW2)
    drafts = tl.load(draft_ptr + req * num_draft + cols, mask=cols < num_draft, other=0)
    out = tl.where(cols < pos, drafts, tl.where(cols == pos, token, -1))
    tl.store(
        out_tokens_ptr + req * (num_draft + 1) + cols, out.to(tl.int32), mask=cols < num_draft + 1
    )
    tl.store(out_num_ptr + req, (pos + 1).to(tl.int32))


class FastRejectionWorkspace:
    """Grow-only scratch buffers, one flat allocation per name. Smaller shapes are views of the
    same storage, so a buffer first sized by the largest CUDA-graph batch keeps one stable
    address for every captured batch size and for eager steps. A buffer outgrown later (an
    eager step or a capture larger than any before) is retired, never freed: graphs captured
    with it keep reading and writing its address."""

    def __init__(self):
        self._bufs = {}
        self._retired = []

    def get(
        self, name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        numel = 1
        for dim in shape:
            numel *= dim
        key = (name, dtype, torch.device(device))
        buf = self._bufs.get(key)
        if buf is None or buf.numel() < numel:
            if buf is not None:
                self._retired.append(buf)
            buf = torch.empty((max(numel, 1),), dtype=dtype, device=device)
            self._bufs[key] = buf
        return buf[:numel].view(shape)


_WORKSPACE = FastRejectionWorkspace()


def _num_chunks(vocab: int) -> Tuple[int, int]:
    nc = triton.cdiv(vocab, _CHUNK)
    return nc, triton.next_power_of_2(nc)


def _row_stats(
    logits: torch.Tensor,
    temps: torch.Tensor,
    temps_div: int,
    slot_ids: torch.Tensor,
    dummy_row: int,
    rows_per_req: int,
    tag: str,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, vocab = logits.shape
    nc, _ = _num_chunks(vocab)
    pmax = _WORKSPACE.get(tag + ".pmax", (rows, nc), torch.float32, logits.device)
    psum = _WORKSPACE.get(tag + ".psum", (rows, nc), torch.float32, logits.device)
    if rows > 0:
        _rs_row_stats_kernel[(rows, nc)](
            logits,
            logits.stride(0),
            bias if bias is not None else logits,
            bias.stride(0) if bias is not None else 0,
            temps,
            slot_ids,
            dummy_row,
            rows_per_req,
            temps_div,
            pmax,
            psum,
            vocab,
            nc,
            CHUNK=_CHUNK,
            HAS_BIAS=bias is not None,
            num_warps=_NUM_WARPS,
        )
    return pmax, psum


def _draft_rows(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    temperatures: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    step_offset: int,
    step0: int,
    steps_total: int,
    tokens: torch.Tensor,
) -> None:
    """Sample ``rows_per_req = rows / num_gens`` draft positions (from ``step0``) per request,
    writing the proposals to ``draft_probs[slot, step]`` and the tokens to
    ``tokens[req, step]`` (``tokens`` is ``[num_gens, steps_total]`` int32)."""
    rows, vocab = logits.shape
    num_gens = slot_ids.shape[0]
    rows_per_req = rows // num_gens
    _check_request_args(temperatures, slot_ids, seed, offset, num_gens, num_gens)
    nc, nc_pow2 = _num_chunks(vocab)
    # Per (row, chunk): online max and sum of exp(x / T - max).
    pmax, psum = _row_stats(
        logits, temperatures, rows_per_req, slot_ids, dummy_row, rows_per_req, "draft", bias
    )
    # Per-chunk mass and last nonzero index: [rows, num_chunks] fp32 and int32.
    csum = _WORKSPACE.get("draft.csum", (rows, nc), torch.float32, logits.device)
    clast = _WORKSPACE.get("draft.clast", (rows, nc), torch.int32, logits.device)
    # Per (row, chunk): q = exp(x / T - M) * (1 / S), written straight into the slot-indexed
    # draft_probs[slot, i, :] row (the reference scatters a temporary), plus csum and clast.
    _rs_draft_write_kernel[(rows, nc)](
        logits,
        logits.stride(0),
        bias if bias is not None else logits,
        bias.stride(0) if bias is not None else 0,
        temperatures,
        slot_ids,
        dummy_row,
        rows_per_req,
        step0,
        pmax,
        psum,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        csum,
        clast,
        vocab,
        nc,
        CHUNK=_CHUNK,
        NC_POW2=nc_pow2,
        HAS_BIAS=bias is not None,
        num_warps=_NUM_WARPS,
    )
    # Per row: inverse-CDF draw, the first x with q(x) > 0 and cumsum(q)(x) > u (flashinfer's
    # sampling_from_probs rule).
    _rs_draft_pick_kernel[(rows,)](
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        slot_ids,
        rows_per_req,
        step0,
        steps_total,
        csum,
        clast,
        seed,
        offset,
        step_offset,
        tokens,
        vocab,
        nc,
        CHUNK=_CHUNK,
        num_warps=_NUM_WARPS,
    )


def _check_draft_args(logits: torch.Tensor, draft_probs: torch.Tensor) -> None:
    assert logits.dtype == torch.float32 and logits.stride(-1) == 1
    assert draft_probs.dtype == torch.float32 and draft_probs.stride(2) == 1
    assert draft_probs.shape[2] == logits.shape[-1]


def _check_request_args(
    temperatures: torch.Tensor,
    slot_ids: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    num_temps: int,
    num_gens: int,
) -> None:
    """The kernels index ``temperatures[r]``, ``slot_ids[b]``, ``seed[b]`` and ``offset[b]``
    directly (int64 slot ids keep ``slot * stride`` in 64 bits)."""
    assert temperatures.dtype == torch.float32 and temperatures.stride(0) == 1
    assert temperatures.numel() >= num_temps
    for t in (slot_ids, seed, offset):
        assert t.dtype == torch.int64 and t.dim() == 1 and t.stride(0) == 1
        assert t.numel() >= num_gens


def fast_rs_draft_block(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    step_offset: int,
) -> torch.Tensor:
    """Sample one block of draft tokens and store their proposal rows.

    Args:
        logits: ``[num_gens * K, vocab]`` fp32 draft logits, request-major (row ``b * K + i``).
        temperatures: ``[num_gens]`` fp32 per-request temperature.
        slot_ids: ``[num_gens]`` int64 ``draft_probs`` slot of each request.
        dummy_row: slot of CUDA-graph padding requests (their logits are treated as zeros);
            ``-1`` when there is none.
        draft_probs: ``[slots, max_draft_len, vocab]`` fp32; rows ``[slot, 0:K, :]`` are written.
        seed, offset: ``[num_gens]`` int64 per-request Philox seed and offset (the draft stream
            is keyed ``seed[b] ^ DRAFT_KEY_XOR``).
        step_offset: added to ``offset[b]`` (1 for the block draft sampler).
    Returns:
        ``[num_gens * K]`` int32 tokens, distributed as ``softmax(logits / T)``.
    """
    rows, _ = logits.shape
    num_gens = slot_ids.shape[0]
    tokens = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return tokens
    _check_draft_args(logits, draft_probs)
    k = rows // num_gens
    assert k * num_gens == rows and draft_probs.shape[1] >= k
    _draft_rows(
        logits,
        None,
        temperatures,
        slot_ids,
        dummy_row,
        draft_probs,
        seed,
        offset,
        step_offset,
        0,
        k,
        tokens,
    )
    return tokens


def fast_rs_draft_markov_feedback(
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    step_offset: int,
) -> torch.Tensor:
    """DSpark vanilla-Markov block chain that feeds back the sampled token.

    Position ``i`` proposes ``q_i = softmax((base_i + markov_w2 @ markov_w1[x_{i-1}]) / T)``
    with ``x_{-1}`` the anchor and ``x_{i-1}`` the token drawn at position ``i - 1`` (SGLang's
    DSpark draft chain). ``q_i`` is stored for verification, so rejection sampling stays
    lossless; the bias GEMV is the reference ``F.linear`` (bit-identical bias).

    Args:
        base_logits: ``[num_gens, K, vocab]`` fp32 shared-lm_head block logits.
        anchor_tokens: ``[num_gens]`` last accepted token per request.
        markov_w1, markov_w2: ``[vocab, rank]`` Markov head weights.
        Other arguments as in `fast_rs_draft_block`.
    Returns:
        ``[num_gens, K]`` int32 tokens.
    """
    num_gens, k, vocab = base_logits.shape
    tokens = torch.empty((num_gens, k), dtype=torch.int32, device=base_logits.device)
    if num_gens == 0 or k == 0:
        return tokens
    _check_draft_args(base_logits, draft_probs)
    assert draft_probs.shape[1] >= k
    rank = markov_w1.shape[1]
    anchor = anchor_tokens.contiguous()
    emb = _WORKSPACE.get("markov.emb", (num_gens, rank), markov_w1.dtype, base_logits.device)
    for step in range(k):
        _markov_feedback_emb_kernel[(num_gens,)](
            tokens,
            k,
            step,
            anchor,
            markov_w1,
            emb,
            vocab,
            RANK=rank,
            BLOCK_R=triton.next_power_of_2(rank),
            num_warps=4,
        )
        bias = torch.nn.functional.linear(emb, markov_w2)
        _draft_rows(
            base_logits[:, step],
            bias,
            temperatures,
            slot_ids,
            dummy_row,
            draft_probs,
            seed,
            offset,
            step_offset,
            step,
            k,
            tokens,
        )
    return tokens


def fast_rs_verify(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    draft_tokens: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample the generation requests' drafts against the target.

    With the same seed and offset, this draws the same uniforms as ``chain_speculative_sampling``
    (`rejection_sampling_one_model`) and applies the same accept test and inverse-CDF rule, so the
    accept decisions and the drawn tokens are the reference's. The exception is a float sum that
    lands on the other side of a threshold because it is reduced in another order (per chunk here,
    per CTA in the reference): the mass sums and the softmax denominator.

    Args:
        logits: ``[num_gens * (K + 1), vocab]`` fp32 target logits, request-major.
        temperatures: ``[num_gens * (K + 1)]`` fp32 per-row temperature.
        draft_tokens: ``[num_gens, K]`` int32 draft tokens (unit inner stride).
        slot_ids: ``[num_gens]`` int64 ``draft_probs`` slot of each request.
        dummy_row: slot of CUDA-graph padding requests (their logits are treated as zeros).
        draft_probs: ``[slots, >= K, vocab]`` fp32 proposals stored by the previous draft step.
        seed, offset: ``[num_gens]`` int64 per-request Philox seed and offset of the acceptance
            draws (the request draws from subsequence ``PER_ROW_SUBSEQUENCE_BASE | offset``).
    Returns:
        ``accepted_tokens [num_gens, K + 1]`` int32 (accepted drafts, the drawn token, then
        ``-1``) and ``num_accepted [num_gens]`` int32 (accepted drafts + 1).
    """
    num_gens, k = draft_tokens.shape
    rows, vocab = logits.shape
    device = logits.device
    accepted = torch.empty((num_gens, k + 1), dtype=torch.int32, device=device)
    num_acc = torch.empty((num_gens,), dtype=torch.int32, device=device)
    if num_gens == 0:
        return accepted, num_acc
    assert rows == num_gens * (k + 1), (rows, num_gens, k)
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert draft_probs.dtype == torch.float32 and draft_probs.stride(2) == 1
    assert draft_probs.shape[2] == vocab and draft_probs.shape[1] >= k
    assert draft_tokens.stride(1) == 1
    _check_request_args(temperatures, slot_ids, seed, offset, rows, num_gens)
    nc, nc_pow2 = _num_chunks(vocab)
    # Row statistics of the K + 1 target rows.
    pmax, psum = _row_stats(logits, temperatures, 1, slot_ids, dummy_row, k + 1, "target")
    pos = _WORKSPACE.get("verify.pos", (num_gens,), torch.int64, device)
    rsum = _WORKSPACE.get("verify.rsum", (num_gens, nc), torch.float32, device)
    rlast = _WORKSPACE.get("verify.rlast", (num_gens, nc), torch.int32, device)
    tsum = _WORKSPACE.get("verify.tsum", (num_gens, nc), torch.float32, device)
    tlast = _WORKSPACE.get("verify.tlast", (num_gens, nc), torch.int32, device)
    # Per (request, chunk): the accept loop (u_i * q(d_i) < p(d_i), a few scalar loads), then the
    # chunk mass of max(p - q, 0) at the first rejected position (p alone for the bonus position)
    # and the chunk mass of p.
    _rs_verify_resid_kernel[(num_gens, nc)](
        logits,
        logits.stride(0),
        temperatures,
        slot_ids,
        dummy_row,
        k,
        pmax,
        psum,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_tokens,
        seed,
        offset,
        pos,
        rsum,
        rlast,
        tsum,
        tlast,
        vocab,
        nc,
        CHUNK=_CHUNK,
        NC_POW2=nc_pow2,
        num_warps=_NUM_WARPS,
    )
    # Per request: inverse-CDF draw from the residual, then the [K + 1] output row (accepted drafts,
    # the drawn token, -1 padding).
    _rs_verify_pick_kernel[(num_gens,)](
        logits,
        logits.stride(0),
        temperatures,
        slot_ids,
        dummy_row,
        k,
        pmax,
        psum,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_tokens,
        seed,
        offset,
        pos,
        rsum,
        rlast,
        tsum,
        tlast,
        accepted,
        num_acc,
        vocab,
        nc,
        CHUNK=_CHUNK,
        NC_POW2=nc_pow2,
        KP1_POW2=triton.next_power_of_2(k + 1),
        num_warps=_NUM_WARPS,
    )
    return accepted, num_acc


def fast_rs_supported(
    logits: torch.Tensor, draft_probs: Optional[torch.Tensor], vocab_stored: int
) -> bool:
    """Shapes the fused path covers: full-vocab fp32 target logits with unit inner stride and a
    full-vocab fp32 proposal buffer (shared draft / target vocabulary)."""
    return (
        draft_probs is not None
        and logits.is_cuda
        and logits.dim() == 2
        and logits.dtype == torch.float32
        and logits.stride(1) == 1
        and draft_probs.dtype == torch.float32
        and draft_probs.dim() == 3
        and draft_probs.stride(2) == 1
        and draft_probs.shape[2] == logits.shape[1]
        and vocab_stored == logits.shape[1]
    )
