# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of this target's post-target step: the DSpark draft block's attention, its
latent and query rope tails, the recurrent Markov head's per-step glue, and the vocabulary
gather of the soft-capped logits.

Every kernel is launched with `enable_fp_fusion=False`: the elementwise chains they replace
round each torch op to its dtype, and a contracted multiply-add would skip one of those
roundings.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _round_bf16(x):
    """fp32 rounded to bf16 (to nearest, ties to even) and widened back. The rounding is an
    inline `cvt.rn.bf16.f32`: the compiler folds a bare `.to(bf16).to(fp32)` pair into
    nothing, which skips a rounding the chain specifies."""
    r = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.bfloat16, is_pure=True, pack=1
    )
    return r.to(tl.float32)


@triton.jit
def _attend(q_lat, q_pe, k_ptrs, valid, m_i, l_i, acc, scale, LAT: tl.constexpr, PE: tl.constexpr):
    """One key tile of the online softmax. The scores round to bf16 before the scale, as the
    bf16 score GEMM they replace does; masked keys load as zero so an uninitialized pool row
    cannot reach the weighted sum."""
    d_lat = tl.arange(0, LAT)
    d_pe = tl.arange(0, PE)
    k_lat = tl.load(k_ptrs[:, None] + d_lat[None, :], mask=valid[:, None], other=0.0)
    k_pe = tl.load(k_ptrs[:, None] + LAT + d_pe[None, :], mask=valid[:, None], other=0.0)
    s = tl.dot(q_pe, tl.trans(k_pe), tl.dot(q_lat, tl.trans(k_lat)))
    s = _round_bf16(s) * scale
    s = tl.where(valid[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_lat)
    return m_new, l_i, acc


@triton.jit
def draft_attention_partial_kernel(
    q_ptr,
    pool_ptr,
    pool_s_page,
    pool_s_slot,
    table_ptr,
    table_s_row,
    row_ptr,
    len_ptr,
    blk_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    scale,
    max_pos,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    QROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ROWS: tl.constexpr,
    LAT: tl.constexpr,
    PE: tl.constexpr,
):
    """Split `split < NSPLIT` of sequence `b` attends its `WINDOW / NSPLIT` keys of the
    request's last `WINDOW` pool rows; split `NSPLIT` attends the `BLOCK` keys of the query
    block itself. Each writes its (max, sum, unnormalized output) for `ROWS` query rows."""
    b = tl.program_id(0)
    split = tl.program_id(1)
    group = tl.program_id(2)
    rows = group * ROWS + tl.arange(0, ROWS)
    live = rows < QROWS
    d_lat = tl.arange(0, LAT)
    d_pe = tl.arange(0, PE)
    q_rows = q_ptr + (b * QROWS + rows).to(tl.int64)[:, None] * (LAT + PE)
    q_lat = tl.load(q_rows + d_lat[None, :], mask=live[:, None], other=0.0)
    q_pe = tl.load(q_rows + LAT + d_pe[None, :], mask=live[:, None], other=0.0)
    m_i = tl.full([ROWS], -1.0e30, tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, LAT], tl.float32)
    t = tl.arange(0, BLOCK_T)
    if split < NSPLIT:
        length = tl.load(len_ptr + b)
        table_row = tl.load(row_ptr + b)
        start = tl.maximum(length - WINDOW, 0)
        keys: tl.constexpr = WINDOW // NSPLIT
        for t0 in range(split * keys, split * keys + keys, BLOCK_T):
            pos = start + t0 + t
            valid = pos < length
            safe = tl.minimum(pos, max_pos)
            page = tl.load(table_ptr + table_row * table_s_row + safe // PAGE)
            page = tl.maximum(page, 0).to(tl.int64)
            k_ptrs = pool_ptr + page * pool_s_page + (safe % PAGE) * pool_s_slot
            m_i, l_i, acc = _attend(q_lat, q_pe, k_ptrs, valid, m_i, l_i, acc, scale, LAT, PE)
    else:
        valid = t < BLOCK
        k_ptrs = blk_ptr + (b * BLOCK + t).to(tl.int64) * (LAT + PE)
        m_i, l_i, acc = _attend(q_lat, q_pe, k_ptrs, valid, m_i, l_i, acc, scale, LAT, PE)
    part = (b * (NSPLIT + 1) + split) * QROWS + rows
    tl.store(m_ptr + part, m_i, mask=live)
    tl.store(l_ptr + part, l_i, mask=live)
    tl.store(acc_ptr + part.to(tl.int64)[:, None] * LAT + d_lat[None, :], acc, mask=live[:, None])


@triton.jit
def draft_attention_combine_kernel(
    m_ptr, l_ptr, acc_ptr, out_ptr, QROWS: tl.constexpr, SPLITS: tl.constexpr, LAT: tl.constexpr
):
    """Merge one query row's split partials and round the normalized output to bf16."""
    row = tl.program_id(0)
    b = row // QROWS
    base = b * SPLITS * QROWS + row % QROWS
    d = tl.arange(0, LAT)
    m_max = tl.load(m_ptr + base)
    for j in tl.static_range(1, SPLITS):
        m_max = tl.maximum(m_max, tl.load(m_ptr + base + j * QROWS))
    total = tl.zeros([1], tl.float32)
    out = tl.zeros([LAT], tl.float32)
    for j in tl.static_range(SPLITS):
        idx = base + j * QROWS
        w = tl.exp(tl.load(m_ptr + idx) - m_max)
        total += w * tl.load(l_ptr + idx)
        out += w * tl.load(acc_ptr + idx.to(tl.int64) * LAT + d)
    tl.store(out_ptr + row.to(tl.int64) * LAT + d, (out / total).to(tl.bfloat16))


@triton.jit
def latent_tail_kernel(
    kv_ptr,
    kv_s_row,
    kv_s_layer,
    w_ptr,
    out_ptr,
    out_s_row,
    out_s_layer,
    q_ptr,
    q_s_row,
    query_ptr,
    query_s_row,
    pos_ptr,
    rope_ptr,
    eps,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    QK: tl.constexpr,
    LAT: tl.constexpr,
    PE: tl.constexpr,
    HAS_Q: tl.constexpr,
):
    """Row `r`, layer `l`: `out = [rmsnorm(kv[:LAT]) * w | rope(kv[LAT:])]`; with `HAS_Q`,
    also each head's `query[..., LAT:] = rope(q[..., NOPE:])`. The rope is the interleaved
    pair rotation in fp32, each product and sum rounded as the torch chain rounds them."""
    r = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1)
    half = tl.arange(0, PE // 2)
    pos = tl.load(pos_ptr + r).to(tl.int64)
    cos = tl.load(rope_ptr + pos * PE + half)
    sin = tl.load(rope_ptr + pos * PE + PE // 2 + half)
    d = tl.arange(0, LAT)
    kv = kv_ptr + r * kv_s_row + layer * kv_s_layer
    x = tl.load(kv + d).to(tl.float32)
    rstd = libdevice.rsqrt(tl.sum(x * x, axis=0) / LAT + eps)
    w = tl.load(w_ptr + layer * LAT + d).to(tl.float32)
    out = out_ptr + r * out_s_row + layer * out_s_layer
    tl.store(out + d, (x * rstd * w).to(tl.bfloat16))
    even = tl.load(kv + LAT + 2 * half).to(tl.float32)
    odd = tl.load(kv + LAT + 2 * half + 1).to(tl.float32)
    tl.store(out + LAT + 2 * half, (even * cos - odd * sin).to(tl.bfloat16))
    tl.store(out + LAT + 2 * half + 1, (even * sin + odd * cos).to(tl.bfloat16))
    if HAS_Q:
        h = tl.arange(0, HEADS)
        q = q_ptr + r * q_s_row + h[:, None] * QK + NOPE
        even = tl.load(q + 2 * half[None, :]).to(tl.float32)
        odd = tl.load(q + 2 * half[None, :] + 1).to(tl.float32)
        c = cos[None, :]
        s = sin[None, :]
        dst = query_ptr + r * query_s_row + h[:, None] * (LAT + PE) + LAT
        tl.store(dst + 2 * half[None, :], (even * c - odd * s).to(tl.bfloat16))
        tl.store(dst + 2 * half[None, :] + 1, (even * s + odd * c).to(tl.bfloat16))


@triton.jit
def markov_input_kernel(
    prev_ptr,
    emb_ptr,
    vocab,
    state_ptr,
    hid_ptr,
    hid_s_row,
    out_ptr,
    HAS_STATE: tl.constexpr,
    RANK: tl.constexpr,
    HIDDEN: tl.constexpr,
    HID_BLOCK: tl.constexpr,
):
    """`out[b] = [state | embedding(prev) masked to 0 off the vocabulary | hidden]`, the
    recurrent head's joint input; a zero state on the chain's first step."""
    b = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, RANK)
    prev = tl.load(prev_ptr + b)
    valid = (prev >= 0) & (prev < vocab)
    idx = tl.minimum(tl.maximum(prev, 0), vocab - 1).to(tl.int64)
    emb = tl.load(emb_ptr + idx * RANK + d)
    emb = tl.where(valid, emb, tl.zeros_like(emb))
    out = out_ptr + b * (2 * RANK + HIDDEN)
    if HAS_STATE:
        state = tl.load(state_ptr + b * RANK + d)
    else:
        state = tl.zeros_like(emb)
    tl.store(out + d, state)
    tl.store(out + RANK + d, emb)
    for c in tl.static_range(0, HIDDEN, HID_BLOCK):
        hd = c + tl.arange(0, HID_BLOCK)
        tl.store(out + 2 * RANK + hd, tl.load(hid_ptr + b * hid_s_row + hd))


@triton.jit
def markov_gate_kernel(
    joint_ptr, state_ptr, state_out_ptr, emit_ptr, HAS_STATE: tl.constexpr, RANK: tl.constexpr
):
    """`gate, cand, emit = joint.chunk(3)`; `state = sigmoid(gate) * state + (1 -
    sigmoid(gate)) * tanh(cand)`; `emit = tanh(emit)`. Every op rounds to bf16 as the bf16
    torch chain does; sigmoid is `1 / (1 + exp(-x))` in fp32, the torch kernel's form."""
    b = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, RANK)
    joint = joint_ptr + b * 3 * RANK
    g = tl.load(joint + d).to(tl.float32)
    c = tl.load(joint + RANK + d).to(tl.float32)
    e = tl.load(joint + 2 * RANK + d).to(tl.float32)
    gate = _round_bf16(libdevice.div_rn(1.0, 1.0 + libdevice.exp(-g)))
    if HAS_STATE:
        state = tl.load(state_ptr + b * RANK + d).to(tl.float32)
    else:
        state = tl.zeros([RANK], tl.float32)
    kept = _round_bf16(gate * state)
    update = _round_bf16(_round_bf16(1.0 - gate) * _round_bf16(libdevice.tanh(c)))
    tl.store(state_out_ptr + b * RANK + d, (kept + update).to(tl.bfloat16))
    tl.store(emit_ptr + b * RANK + d, libdevice.tanh(e).to(tl.bfloat16))


@triton.jit
def markov_logits_kernel(
    base_ptr,
    base_s_row,
    bias_ptr,
    out_ptr,
    out_s_row,
    part_max_ptr,
    part_idx_ptr,
    vocab,
    CHUNKS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """`out[b] = base[b] + bias[b]` (fp32 plus a widened bf16) over one vocabulary chunk, and
    the chunk's maximum with its first index."""
    b = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    v = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v < vocab
    x = tl.load(base_ptr + b * base_s_row + v, mask=mask, other=0.0)
    x = x + tl.load(bias_ptr + b * vocab + v, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + b * out_s_row + v, x, mask=mask)
    x = tl.where(mask, x, float("-inf"))
    m = tl.max(x, axis=0)
    first = tl.min(tl.where(x == m, v, 2147483647), axis=0)
    tl.store(part_max_ptr + b * CHUNKS + chunk, m)
    tl.store(part_idx_ptr + b * CHUNKS + chunk, first)


@triton.jit
def argmax_finish_kernel(
    part_max_ptr, part_idx_ptr, out_ptr, CHUNKS: tl.constexpr, CHUNKS_P2: tl.constexpr
):
    """The row's first maximum over its chunks: the smallest index among the chunks holding
    the largest value, torch.argmax's tie-break."""
    b = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, CHUNKS_P2)
    mask = c < CHUNKS
    m = tl.load(part_max_ptr + b * CHUNKS + c, mask=mask, other=float("-inf"))
    idx = tl.load(part_idx_ptr + b * CHUNKS + c, mask=mask, other=2147483647)
    best = tl.max(m, axis=0)
    first = tl.min(tl.where((m == best) & mask, idx, 2147483647), axis=0)
    tl.store(out_ptr + b, first.to(tl.int64))


@triton.jit
def _mul_rn(a, b):
    """fp32 product rounded to nearest with subnormals kept, as the torch kernels compute it."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def scatter_shard_kernel(
    src_ptr, src_s_row, peers_ptr, rank, rows_cap, shard, WORLD: tl.constexpr, BLOCK_V: tl.constexpr
):
    """Write this rank's bf16 vocabulary shard of row `r` into block `rank` of every rank's
    `[world, rows_cap, shard]` symmetric slot, its own included: NVLink stores, no reads."""
    r = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    v = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v < shard
    x = tl.load(src_ptr + r * src_s_row + v, mask=mask)
    offset = (rank * rows_cap + r) * shard + v
    for peer in tl.static_range(WORLD):
        dst = tl.load(peers_ptr + peer).to(tl.pointer_type(tl.bfloat16))
        tl.store(dst + offset, x, mask=mask)


@triton.jit
def softcap_gather_kernel(
    buf_ptr, rows_cap, shard, out_ptr, out_s_row, inv_cap, cap, BLOCK_V: tl.constexpr
):
    """`out[r, p * shard + v] = cap * tanh(float(shard_p[r, v]) * inv_cap)` from this rank's own
    slot, which every rank's scatter has filled. `x * inv_cap` is the torch division by a Python
    float; tanh is libdevice's `tanhf`, which Triton links flushing subnormals, so a subnormal
    argument (where tanhf is the identity) bypasses it."""
    r = tl.program_id(0).to(tl.int64)
    peer = tl.program_id(1)
    chunk = tl.program_id(2)
    v = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v < shard
    x = tl.load(buf_ptr + (peer * rows_cap + r) * shard + v, mask=mask, other=0.0)
    y = _mul_rn(x.to(tl.float32), inv_cap)
    t = tl.where(tl.abs(y) < 1.1754943508222875e-38, y, libdevice.tanh(y))
    tl.store(out_ptr + r * out_s_row + peer * shard + v, _mul_rn(t, cap), mask=mask)
