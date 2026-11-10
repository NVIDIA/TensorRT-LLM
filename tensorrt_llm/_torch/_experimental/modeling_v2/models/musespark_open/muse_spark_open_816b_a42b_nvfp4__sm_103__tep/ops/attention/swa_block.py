# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""A sliding-window layer's decode attention block, from the QKV projection to o_proj's input.

Between the QKV projection and the output projection a sliding-window decode
step of this target launches four dependent kernels: ``preattn_prep`` (the
weightless Q/K norms, the query pre-scale, RoPE and the gate copy),
``paged_decode_attention``'s split and combine, and the output norm-gate. The
window's FP8 K/V is half a megabyte, so the block is launch edges and the
dependent loads behind each one, not bytes.

This entry is that block with fewer edges, in the same arithmetic:

* The split kernel reads what does not depend on its predecessor *before*
  ``griddepcontrol.wait``: the sequence length, the block-table entries of its
  key block, the cached K/V of the keys before this step (``j < past``), and,
  with ``fused_prep``, the step's positions and their RoPE table rows. Only the
  projection operand is read after the wait.
* ``fused_prep``: the split programs normalize, scale and rotate their own
  query rows and the step's keys from the raw projection, with
  ``preattn_prep``'s statements (its thread-order sum, ``rsqrt.approx.ftz``,
  BF16 roundings, ``fma.rn.ftz`` / ``mul.rn.ftz`` rotation), so the prep
  launch is gone. The gate stays where the projection put it.
* ``gate``: the combine kernel applies the per-head RMSNorm and the BF16
  sigmoid gate to the head it has just merged, with the built-in norm-gate
  kernel's summation tree written out, and lets the output projection launch
  right after its own wait, as the PDL norm-gate copy does.

Every value is the one the four-kernel chain computes, bit for bit: the KV
cache bytes, the attention output and the gated o_proj input.

Each program of the split grid owns exactly one 32-key block (``num_splits``
must cover the window's largest block count, which ``window_splits`` does);
that is what lets its cached K/V be read before the wait.

``blocks_per_program`` > 1 runs the split as fewer programs: each owns that
many consecutive key blocks, reads the cached K/V of all of them before the
wait, prepares its query rows once, then runs each block with the
one-block program's statements from a fresh online-softmax state and stores
that block's partial in the block's own slot. The partials, the combine and
every output bit are the ones of ``blocks_per_program = 1``; what changes is
how many programs the grid has (``ceil(num_splits / blocks_per_program)`` per
head and row group), which decides how many rounds of resident programs a
step of several sequences needs.

``split_max_registers`` (with ``blocks_per_program`` > 1) compiles that split
kernel under a register cap per thread (Triton's ``maxnreg``): register
allocation only, so the same instructions on the same values and the same
bits, with more programs resident per SM (``SPLIT_REGISTER_CAP``: three).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # the pre-3.2 layout
    from triton.language.extra.cuda import libdevice

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .paged_decode_attention import (
    BLOCK_N,
    MAX_ROWS,
    NUM_WARPS,
    Q_PER_KV,
    ROWS_PER_PROGRAM,
    _to_cache_fp8,
)
from .paged_decode_attention import HEAD_DIM as _HEAD_DIM
from .paged_decode_attention import paged_decode_attention as _paged_decode_attention
from .preattn_prep import _normalize, _rotate, _rsqrt_approx_ftz
from .preattn_prep import preattn_prep as _preattn_prep

HEAD_DIM = _HEAD_DIM
#: Stages of the split kernel's key loop: it runs at most once per program.
NUM_STAGES = 1
#: The key blocks one split program may own (``blocks_per_program``).
MAX_BLOCKS_PER_PROGRAM = 4
#: Registers per thread at which three 128-thread split programs fit an SM's 64K registers
#: (65536 / 384 = 170, rounded down to the allocation granule of 8).
SPLIT_REGISTER_CAP = 168
#: The register caps ``split_max_registers`` takes.
MIN_SPLIT_REGISTERS = 128
MAX_SPLIT_REGISTERS = 255


@triton.jit
def _prep_rows(
    ptr,
    row_off,
    col_off,
    mask,
    cos,
    sin,
    scale,
    eps,
    N: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    """``preattn_prep`` of ``N`` 64-wide heads of the raw projection, as BF16 ``[N, 64]``.

    Head ``n`` starts at ``ptr + row_off[n] + col_off[n]``. Thread ``l`` of a head
    owns elements ``8 l .. 8 l + 7``; its squares are summed in order with
    ``add.rn``, then threads pair over bit 0, 1, 2 -- ``preattn_prep``'s tree.
    The norm, the scale and the rotation are its elementwise statements on the
    same values, so each element rounds exactly as there. Masked heads read 0.
    """
    d = tl.arange(0, 64)
    lane = tl.arange(0, 8)
    head = ptr + row_off[:, None] + col_off[:, None]
    m = mask[:, None]
    x0 = tl.load(head + lane[None, :] * 8 + 0, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(head + lane[None, :] * 8 + 1, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(head + lane[None, :] * 8 + 2, mask=m, other=0.0).to(tl.float32)
    x3 = tl.load(head + lane[None, :] * 8 + 3, mask=m, other=0.0).to(tl.float32)
    x4 = tl.load(head + lane[None, :] * 8 + 4, mask=m, other=0.0).to(tl.float32)
    x5 = tl.load(head + lane[None, :] * 8 + 5, mask=m, other=0.0).to(tl.float32)
    x6 = tl.load(head + lane[None, :] * 8 + 6, mask=m, other=0.0).to(tl.float32)
    x7 = tl.load(head + lane[None, :] * 8 + 7, mask=m, other=0.0).to(tl.float32)
    s = libdevice.add_rn(x0 * x0, x1 * x1)
    s = libdevice.add_rn(s, x2 * x2)
    s = libdevice.add_rn(s, x3 * x3)
    s = libdevice.add_rn(s, x4 * x4)
    s = libdevice.add_rn(s, x5 * x5)
    s = libdevice.add_rn(s, x6 * x6)
    s = libdevice.add_rn(s, x7 * x7)
    # lane = 4a + 2b + c: pair over c (bit 0), then b, then a.
    s = tl.reshape(s, [N, 2, 2, 2])
    total = tl.sum(tl.sum(tl.sum(s, axis=3), axis=2), axis=1)
    inverse_rms = _rsqrt_approx_ftz(total / 64.0 + eps)
    x = tl.load(head + d[None, :], mask=m, other=0.0).to(tl.float32)
    y = _normalize(x, inverse_rms[:, None], scale)
    if APPLY_ROPE:
        even, odd = tl.split(tl.reshape(y, [N, 32, 2]))
        even, odd = _rotate(even, odd, cos, sin)
        y = tl.reshape(tl.join(even, odd), [N, 64])
    return y.to(tl.bfloat16)


@triton.jit
def _prep_rows_load(ptr, row_off, col_off, mask, N: tl.constexpr):
    """``_prep_rows``' loads alone (BF16, as stored), so a caller can issue them early."""
    d = tl.arange(0, 64)
    lane = tl.arange(0, 8)
    head = ptr + row_off[:, None] + col_off[:, None]
    m = mask[:, None]
    x0 = tl.load(head + lane[None, :] * 8 + 0, mask=m, other=0.0)
    x1 = tl.load(head + lane[None, :] * 8 + 1, mask=m, other=0.0)
    x2 = tl.load(head + lane[None, :] * 8 + 2, mask=m, other=0.0)
    x3 = tl.load(head + lane[None, :] * 8 + 3, mask=m, other=0.0)
    x4 = tl.load(head + lane[None, :] * 8 + 4, mask=m, other=0.0)
    x5 = tl.load(head + lane[None, :] * 8 + 5, mask=m, other=0.0)
    x6 = tl.load(head + lane[None, :] * 8 + 6, mask=m, other=0.0)
    x7 = tl.load(head + lane[None, :] * 8 + 7, mask=m, other=0.0)
    x = tl.load(head + d[None, :], mask=m, other=0.0)
    return x0, x1, x2, x3, x4, x5, x6, x7, x


@triton.jit
def _prep_rows_math(x0, x1, x2, x3, x4, x5, x6, x7, x, cos, sin, scale, eps, N: tl.constexpr,
                    APPLY_ROPE: tl.constexpr):  # fmt: skip
    """``_prep_rows``' arithmetic, statement for statement, on ``_prep_rows_load``'s values."""
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)
    x3 = x3.to(tl.float32)
    x4 = x4.to(tl.float32)
    x5 = x5.to(tl.float32)
    x6 = x6.to(tl.float32)
    x7 = x7.to(tl.float32)
    s = libdevice.add_rn(x0 * x0, x1 * x1)
    s = libdevice.add_rn(s, x2 * x2)
    s = libdevice.add_rn(s, x3 * x3)
    s = libdevice.add_rn(s, x4 * x4)
    s = libdevice.add_rn(s, x5 * x5)
    s = libdevice.add_rn(s, x6 * x6)
    s = libdevice.add_rn(s, x7 * x7)
    s = tl.reshape(s, [N, 2, 2, 2])
    total = tl.sum(tl.sum(tl.sum(s, axis=3), axis=2), axis=1)
    inverse_rms = _rsqrt_approx_ftz(total / 64.0 + eps)
    y = _normalize(x.to(tl.float32), inverse_rms[:, None], scale)
    if APPLY_ROPE:
        even, odd = tl.split(tl.reshape(y, [N, 32, 2]))
        even, odd = _rotate(even, odd, cos, sin)
        y = tl.reshape(tl.join(even, odd), [N, 64])
    return y.to(tl.bfloat16)


@triton.jit
def _rope_rows(cos_sin_ptr, positions_ptr, index, mask, APPLY_ROPE: tl.constexpr):
    """``(cos, sin)`` ``[N, 32]`` of the positions ``positions[index]`` (zeros where masked)."""
    half = tl.arange(0, 32)
    if APPLY_ROPE:
        position = tl.load(positions_ptr + index, mask=mask, other=0).to(tl.int64)
        row = cos_sin_ptr + position[:, None] * 64 + half[None, :]
        cos = tl.load(row, mask=mask[:, None], other=0.0)
        sin = tl.load(row + 32, mask=mask[:, None], other=0.0)
    else:
        cos = tl.zeros([index.shape[0], 32], tl.float32)
        sin = tl.zeros([index.shape[0], 32], tl.float32)
    return cos, sin


@triton.jit
def _swa_split_kernel(
    qkv_ptr,
    qkv_stride,
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    seq_len_ptr,
    scale_ptr,
    cos_sin_ptr,
    positions_ptr,
    prep_eps,
    q_scale,
    part_o_ptr,
    part_ml_ptr,
    sm_scale_log2,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    Q_FP8: tl.constexpr,
    FUSED_PREP: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    REQS: tl.constexpr = 1,
    EARLY_LOADS: tl.constexpr = False,
):
    G: tl.constexpr = HQ // HKV
    PAGE: tl.constexpr = HKV * P * D
    split = tl.program_id(0)
    h = tl.program_id(1)
    row_group = tl.program_id(2)
    if REQS > 1:
        # Several sequences of M rows each, request-major in the operand: program
        # (split, h, req * RG + row_group) is the one-sequence program of request
        # ``req`` on its own rows, length, block-table row, positions and partials.
        RG: tl.constexpr = (M * G + R - 1) // R
        req = row_group // RG
        row_group = row_group % RG
        qkv_ptr += req * M * qkv_stride
        bt_ptr += req * 2 * bt_stride_kv  # [requests, 2, pages], K then V rows
        seq_len_ptr += req
        if APPLY_ROPE:
            positions_ptr += req * M
        part_o_ptr += req * num_splits * HKV * M * G * D
        part_ml_ptr += req * num_splits * HKV * M * G * 2
    d = tl.arange(0, D)

    # Column offsets of this program's heads in the operand: the prep output
    # [q | k | v], or the raw projection (checkpoint order [q_0 | gate_0 | ... |
    # k | v], or the low-latency layout [q | k | v | gate]).
    q_stride = D
    k_col = HQ * D + h * D
    v_col = (HQ + HKV) * D + h * D
    if FUSED_PREP:
        if not QKV_PERMUTED:
            q_stride = 2 * D
            k_col = 2 * HQ * D + h * D
            v_col = 2 * HQ * D + HKV * D + h * D

    # ---- Before the wait: graph inputs and cache bytes no grid in flight writes.
    # seq_lens, the block table and the positions are written before the graph;
    # the cached keys (j < past) were written by earlier steps; this step's own
    # keys (j >= past) are written by this grid's (0, h, 0) program only.
    L = tl.load(seq_len_ptr)
    past = L - M
    if WINDOW > 0:
        lo = tl.maximum(past - WINDOW + 1, 0)
    else:
        lo = 0
    base = (lo // BLOCK) * BLOCK
    n_blocks = tl.cdiv(L - base, BLOCK)
    per_split = tl.cdiv(n_blocks, num_splits)
    b0 = split * per_split
    b1 = tl.minimum(b0 + per_split, n_blocks)
    scale = tl.load(scale_ptr)
    scale_bf16 = scale.to(tl.bfloat16)
    dequant = 1.0 / scale

    j = base + b0 * BLOCK + tl.arange(0, BLOCK)
    live = b0 < b1
    # Never touch a page outside the attended range: a sliding-window layer's
    # block table need not hold a valid page there.
    cached = (j >= lo) & (j < past) & live
    new = (j >= past) & (j < L) & live
    page_k = tl.load(bt_ptr + j // P, mask=cached, other=0).to(tl.int64)
    page_v = tl.load(bt_ptr + bt_stride_kv + j // P, mask=cached, other=0).to(tl.int64)
    in_page = (h * P + j % P) * D
    k_old = tl.load(
        pool_ptr + (page_k * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )
    v_old = tl.load(
        pool_ptr + (page_v * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )

    writer = (split == 0) & (row_group == 0)
    tok = tl.arange(0, 8)
    tok_ok = tok < M
    write_ok = tok_ok & writer
    pos = past + tok
    w_page_k = tl.load(bt_ptr + pos // P, mask=write_ok, other=0).to(tl.int64)
    w_page_v = tl.load(bt_ptr + bt_stride_kv + pos // P, mask=write_ok, other=0).to(tl.int64)
    slot = (h * P + pos % P) * D

    r = row_group * R + tl.arange(0, R)  # row = token * G + query head within the KV group
    t = r // G
    row_ok = t < M
    new_row = j - past
    if FUSED_PREP:
        cos_q, sin_q = _rope_rows(cos_sin_ptr, positions_ptr, t, row_ok, APPLY_ROPE)
        cos_k, sin_k = _rope_rows(cos_sin_ptr, positions_ptr, new_row, new, APPLY_ROPE)
        cos_w, sin_w = _rope_rows(cos_sin_ptr, positions_ptr, tok, write_ok, APPLY_ROPE)

    if LAUNCH_WITH_PDL:
        # From here on the operand: the previous kernel's output. The combine
        # kernel may launch now and waits for this grid.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if EARLY_LOADS:
        # Every operand load of this program at once, right after the wait (with the fused
        # prep: the query rows, the key block's step K/V rows, the writer's step K/V rows;
        # masked rows load nothing): one dependent round trip instead of one per prep. The
        # arithmetic below is the same statements on the same values.
        e_q = _prep_rows_load(qkv_ptr, t * qkv_stride, (h * G + r % G) * q_stride, row_ok, R)
        e_k = _prep_rows_load(qkv_ptr, new_row * qkv_stride, tl.zeros([BLOCK], tl.int32) + k_col,
                              new, BLOCK)  # fmt: skip
        e_v = tl.load(
            qkv_ptr + new_row[:, None] * qkv_stride + v_col + d[None, :],
            mask=new[:, None],
            other=0.0,
        )
        e_wk = _prep_rows_load(
            qkv_ptr, tok * qkv_stride, tl.zeros([8], tl.int32) + k_col, write_ok, 8
        )
        e_wv = tl.load(
            qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :],
            mask=write_ok[:, None],
            other=0.0,
        )

    if writer:
        # The step's K/V into the paged cache: what applyBiasRopeUpdateKVCacheV2 stores.
        if EARLY_LOADS:
            wk0, wk1, wk2, wk3, wk4, wk5, wk6, wk7, wkx = e_wk
            k_new = _prep_rows_math(wk0, wk1, wk2, wk3, wk4, wk5, wk6, wk7, wkx, cos_w, sin_w, 1.0,
                                    prep_eps, 8, APPLY_ROPE)  # fmt: skip
        elif FUSED_PREP:
            k_new = _prep_rows(
                qkv_ptr,
                tok * qkv_stride,
                tl.zeros([8], tl.int32) + k_col,
                tok_ok,
                cos_w,
                sin_w,
                1.0,
                prep_eps,
                8,
                APPLY_ROPE,
            )
        else:
            k_new = tl.load(
                qkv_ptr + tok[:, None] * qkv_stride + k_col + d[None, :],
                mask=tok_ok[:, None],
                other=0.0,
            )
        if EARLY_LOADS:
            v_new = e_wv
        else:
            v_new = tl.load(
                qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :],
                mask=tok_ok[:, None],
                other=0.0,
            )
        tl.store(
            pool_ptr + (w_page_k * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(k_new, scale_bf16),
            mask=tok_ok[:, None],
        )
        tl.store(
            pool_ptr + (w_page_v * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(v_new, scale_bf16),
            mask=tok_ok[:, None],
        )

    q_head = h * G + r % G
    if EARLY_LOADS:
        qx0, qx1, qx2, qx3, qx4, qx5, qx6, qx7, qxx = e_q
        q = _prep_rows_math(qx0, qx1, qx2, qx3, qx4, qx5, qx6, qx7, qxx, cos_q, sin_q, q_scale, prep_eps, R,
                            APPLY_ROPE)  # fmt: skip
        q = tl.where(row_ok[:, None], q, 0.0)
    elif FUSED_PREP:
        q = _prep_rows(
            qkv_ptr,
            t * qkv_stride,
            q_head * q_stride,
            row_ok,
            cos_q,
            sin_q,
            q_scale,
            prep_eps,
            R,
            APPLY_ROPE,
        )
        q = tl.where(row_ok[:, None], q, 0.0)
    else:
        q = tl.load(
            qkv_ptr + t[:, None] * qkv_stride + q_head[:, None] * D + d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
    qk_scale = sm_scale_log2 * dequant
    if Q_FP8:
        q = _to_cache_fp8(q, scale_bf16).to(tl.bfloat16)
        qk_scale = qk_scale * dequant
    q_pos = past + t
    m_i = tl.full([R], -1.0e30, tl.float32)
    l_i = tl.zeros([R], tl.float32)
    acc = tl.zeros([R, D], tl.float32)
    # Whether this program's key block holds any of the step's own keys.
    has_new = base + b0 * BLOCK + BLOCK > past
    for b in range(b0, b1):  # at most once: one key block per program
        if EARLY_LOADS:
            k_new = tl.zeros([BLOCK, D], tl.bfloat16)
            if has_new:
                kx0, kx1, kx2, kx3, kx4, kx5, kx6, kx7, kxx = e_k
                k_new = _prep_rows_math(kx0, kx1, kx2, kx3, kx4, kx5, kx6, kx7, kxx, cos_k, sin_k, 1.0,
                                        prep_eps, BLOCK, APPLY_ROPE)  # fmt: skip
        elif FUSED_PREP:
            k_new = tl.zeros([BLOCK, D], tl.bfloat16)
            if has_new:
                k_new = _prep_rows(
                    qkv_ptr,
                    new_row * qkv_stride,
                    tl.zeros([BLOCK], tl.int32) + k_col,
                    new,
                    cos_k,
                    sin_k,
                    1.0,
                    prep_eps,
                    BLOCK,
                    APPLY_ROPE,
                )
        else:
            k_new = tl.load(
                qkv_ptr + new_row[:, None] * qkv_stride + k_col + d[None, :],
                mask=new[:, None],
                other=0.0,
            )
        if EARLY_LOADS:
            v_new = e_v
        else:
            v_new = tl.load(
                qkv_ptr + new_row[:, None] * qkv_stride + v_col + d[None, :],
                mask=new[:, None],
                other=0.0,
            )
        k = tl.where(new[:, None], _to_cache_fp8(k_new, scale_bf16), k_old)
        v = tl.where(new[:, None], _to_cache_fp8(v_new, scale_bf16), v_old)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
        keep = (j[None, :] <= q_pos[:, None]) & (j[None, :] >= lo)
        if WINDOW > 0:
            keep = keep & (j[None, :] > q_pos[:, None] - WINDOW)
        s = tl.where(keep, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    # An empty split (b0 >= b1) stores l = 0, which the combine weighs as zero.
    part = (split * HKV + h) * (M * G) + r
    tl.store(part_o_ptr + part[:, None] * D + d[None, :], acc * dequant, mask=row_ok[:, None])
    tl.store(part_ml_ptr + part * 2, m_i, mask=row_ok)
    tl.store(part_ml_ptr + part * 2 + 1, l_i, mask=row_ok)


@triton.jit
def _block_kv(
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    h,
    base,
    blk,
    n_blocks,
    lo,
    past,
    L,
    D: tl.constexpr,
    P: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Key block ``blk``'s keys ``j``, its step-key mask and its cached K/V (zeros elsewhere).

    ``_swa_split_kernel``'s pre-wait statements for a program whose block is ``blk``: the
    block-table entries and the FP8 K/V of the cached keys ``lo <= j < past``; a block at or
    past ``n_blocks`` is empty and loads nothing.
    """
    d = tl.arange(0, D)
    j = base + blk * BLOCK + tl.arange(0, BLOCK)
    live = blk < n_blocks
    cached = (j >= lo) & (j < past) & live
    new = (j >= past) & (j < L) & live
    page_k = tl.load(bt_ptr + j // P, mask=cached, other=0).to(tl.int64)
    page_v = tl.load(bt_ptr + bt_stride_kv + j // P, mask=cached, other=0).to(tl.int64)
    in_page = (h * P + j % P) * D
    k_old = tl.load(
        pool_ptr + (page_k * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )
    v_old = tl.load(
        pool_ptr + (page_v * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )
    return j, new, k_old, v_old


@triton.jit
def _block_attend(
    m_i,
    l_i,
    acc,
    q,
    qk_scale,
    q_pos,
    j,
    new,
    k_old,
    v_old,
    blk,
    n_blocks,
    base,
    lo,
    past,
    qkv_ptr,
    qkv_stride,
    k_col,
    v_col,
    cos_sin_ptr,
    positions_ptr,
    prep_eps,
    scale_bf16,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    FUSED_PREP: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    """One key block's online-softmax update of ``(m_i, l_i, acc)``: ``_swa_split_kernel``'s
    key loop for a program whose block is ``blk``, statement for statement (the step's keys
    of the block are prepared from the projection here; their RoPE rows are read here too,
    which is after the wait and the same values)."""
    d = tl.arange(0, D)
    new_row = j - past
    # Whether this key block holds any of the step's own keys.
    has_new = base + blk * BLOCK + BLOCK > past
    for b in range(blk, tl.minimum(blk + 1, n_blocks)):  # at most once
        if FUSED_PREP:
            k_new = tl.zeros([BLOCK, D], tl.bfloat16)
            if has_new:
                cos_k, sin_k = _rope_rows(cos_sin_ptr, positions_ptr, new_row, new, APPLY_ROPE)
                k_new = _prep_rows(
                    qkv_ptr,
                    new_row * qkv_stride,
                    tl.zeros([BLOCK], tl.int32) + k_col,
                    new,
                    cos_k,
                    sin_k,
                    1.0,
                    prep_eps,
                    BLOCK,
                    APPLY_ROPE,
                )
        else:
            k_new = tl.load(
                qkv_ptr + new_row[:, None] * qkv_stride + k_col + d[None, :],
                mask=new[:, None],
                other=0.0,
            )
        v_new = tl.load(
            qkv_ptr + new_row[:, None] * qkv_stride + v_col + d[None, :],
            mask=new[:, None],
            other=0.0,
        )
        k = tl.where(new[:, None], _to_cache_fp8(k_new, scale_bf16), k_old)
        v = tl.where(new[:, None], _to_cache_fp8(v_new, scale_bf16), v_old)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
        keep = (j[None, :] <= q_pos[:, None]) & (j[None, :] >= lo)
        if WINDOW > 0:
            keep = keep & (j[None, :] > q_pos[:, None] - WINDOW)
        s = tl.where(keep, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _store_partial(
    part_o_ptr,
    part_ml_ptr,
    slot,
    slots,
    h,
    r,
    row_ok,
    acc,
    m_i,
    l_i,
    dequant,
    HKV: tl.constexpr,
    MG: tl.constexpr,
    D: tl.constexpr,
):
    """Store one partial at ``slot`` (nothing at or past ``slots``), as the split kernel does."""
    d = tl.arange(0, D)
    ok = row_ok & (slot < slots)
    part = (slot * HKV + h) * MG + r
    tl.store(part_o_ptr + part[:, None] * D + d[None, :], acc * dequant, mask=ok[:, None])
    tl.store(part_ml_ptr + part * 2, m_i, mask=ok)
    tl.store(part_ml_ptr + part * 2 + 1, l_i, mask=ok)


@triton.jit
def _swa_split_bpp_kernel(
    qkv_ptr,
    qkv_stride,
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    seq_len_ptr,
    scale_ptr,
    cos_sin_ptr,
    positions_ptr,
    prep_eps,
    q_scale,
    part_o_ptr,
    part_ml_ptr,
    sm_scale_log2,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    Q_FP8: tl.constexpr,
    FUSED_PREP: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    BPP: tl.constexpr,
    REQS: tl.constexpr = 1,
    MERGE: tl.constexpr = False,
):
    """``_swa_split_kernel`` with ``BPP`` (2..4) consecutive key blocks per program.

    Program ``(s, h, row group)`` owns blocks ``s * BPP .. s * BPP + BPP - 1``. Before the
    wait it reads what the one-block programs of those blocks read there (their cached K/V,
    the graph inputs) except the RoPE rows of the step's keys, which it reads with the
    projection; after the wait it writes the step's K/V when it owns block 0 and prepares its
    query rows once. Each block then runs that block's one-block statements from the initial
    online-softmax state and stores its partial in slot ``block`` (``num_splits`` slots, the
    combine unchanged), so every partial and every cache byte is the one-block kernel's.

    ``MERGE`` (not used by the ops; the profile bench measures it) instead carries the
    softmax state across the program's blocks and stores one partial per program in slot
    ``s`` of ``ceil(num_splits / BPP)``: a different association of the same FP32 merge,
    which does not reproduce the one-block bits.
    """
    G: tl.constexpr = HQ // HKV
    PAGE: tl.constexpr = HKV * P * D
    MG: tl.constexpr = M * G
    pid = tl.program_id(0)
    h = tl.program_id(1)
    row_group = tl.program_id(2)
    if MERGE:
        slots = tl.cdiv(num_splits, BPP)
    else:
        slots = num_splits
    if REQS > 1:
        # As _swa_split_kernel: request ``req``'s rows, length, table row, positions, partials.
        RG: tl.constexpr = (M * G + R - 1) // R
        req = row_group // RG
        row_group = row_group % RG
        qkv_ptr += req * M * qkv_stride
        bt_ptr += req * 2 * bt_stride_kv  # [requests, 2, pages], K then V rows
        seq_len_ptr += req
        if APPLY_ROPE:
            positions_ptr += req * M
        part_o_ptr += req * slots * HKV * MG * D
        part_ml_ptr += req * slots * HKV * MG * 2
    d = tl.arange(0, D)

    q_stride = D
    k_col = HQ * D + h * D
    v_col = (HQ + HKV) * D + h * D
    if FUSED_PREP:
        if not QKV_PERMUTED:
            q_stride = 2 * D
            k_col = 2 * HQ * D + h * D
            v_col = 2 * HQ * D + HKV * D + h * D

    # ---- Before the wait: as _swa_split_kernel, for every block this program owns.
    L = tl.load(seq_len_ptr)
    past = L - M
    if WINDOW > 0:
        lo = tl.maximum(past - WINDOW + 1, 0)
    else:
        lo = 0
    base = (lo // BLOCK) * BLOCK
    n_blocks = tl.cdiv(L - base, BLOCK)
    scale = tl.load(scale_ptr)
    scale_bf16 = scale.to(tl.bfloat16)
    dequant = 1.0 / scale

    blk = pid * BPP
    j0, new0, k_old0, v_old0 = _block_kv(
        pool_ptr, bt_ptr, bt_stride_kv, h, base, blk, n_blocks, lo, past, L, D, P, PAGE, BLOCK
    )
    if BPP > 1:
        j1, new1, k_old1, v_old1 = _block_kv(
            pool_ptr, bt_ptr, bt_stride_kv, h, base, blk + 1, n_blocks, lo, past, L, D, P, PAGE,
            BLOCK,
        )  # fmt: skip
    if BPP > 2:
        j2, new2, k_old2, v_old2 = _block_kv(
            pool_ptr, bt_ptr, bt_stride_kv, h, base, blk + 2, n_blocks, lo, past, L, D, P, PAGE,
            BLOCK,
        )  # fmt: skip
    if BPP > 3:
        j3, new3, k_old3, v_old3 = _block_kv(
            pool_ptr, bt_ptr, bt_stride_kv, h, base, blk + 3, n_blocks, lo, past, L, D, P, PAGE,
            BLOCK,
        )  # fmt: skip

    writer = (pid == 0) & (row_group == 0)
    tok = tl.arange(0, 8)
    tok_ok = tok < M
    write_ok = tok_ok & writer
    pos = past + tok
    w_page_k = tl.load(bt_ptr + pos // P, mask=write_ok, other=0).to(tl.int64)
    w_page_v = tl.load(bt_ptr + bt_stride_kv + pos // P, mask=write_ok, other=0).to(tl.int64)
    slot = (h * P + pos % P) * D

    r = row_group * R + tl.arange(0, R)  # row = token * G + query head within the KV group
    t = r // G
    row_ok = t < M
    if FUSED_PREP:
        cos_q, sin_q = _rope_rows(cos_sin_ptr, positions_ptr, t, row_ok, APPLY_ROPE)
        cos_w, sin_w = _rope_rows(cos_sin_ptr, positions_ptr, tok, write_ok, APPLY_ROPE)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if writer:
        # The step's K/V into the paged cache, as _swa_split_kernel's (0, h, 0) program.
        if FUSED_PREP:
            k_new = _prep_rows(
                qkv_ptr,
                tok * qkv_stride,
                tl.zeros([8], tl.int32) + k_col,
                tok_ok,
                cos_w,
                sin_w,
                1.0,
                prep_eps,
                8,
                APPLY_ROPE,
            )
        else:
            k_new = tl.load(
                qkv_ptr + tok[:, None] * qkv_stride + k_col + d[None, :],
                mask=tok_ok[:, None],
                other=0.0,
            )
        v_new = tl.load(
            qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        tl.store(
            pool_ptr + (w_page_k * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(k_new, scale_bf16),
            mask=tok_ok[:, None],
        )
        tl.store(
            pool_ptr + (w_page_v * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(v_new, scale_bf16),
            mask=tok_ok[:, None],
        )

    # The query rows, once for all the program's blocks.
    q_head = h * G + r % G
    if FUSED_PREP:
        q = _prep_rows(
            qkv_ptr,
            t * qkv_stride,
            q_head * q_stride,
            row_ok,
            cos_q,
            sin_q,
            q_scale,
            prep_eps,
            R,
            APPLY_ROPE,
        )
        q = tl.where(row_ok[:, None], q, 0.0)
    else:
        q = tl.load(
            qkv_ptr + t[:, None] * qkv_stride + q_head[:, None] * D + d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
    qk_scale = sm_scale_log2 * dequant
    if Q_FP8:
        q = _to_cache_fp8(q, scale_bf16).to(tl.bfloat16)
        qk_scale = qk_scale * dequant
    q_pos = past + t

    m_i = tl.full([R], -1.0e30, tl.float32)
    l_i = tl.zeros([R], tl.float32)
    acc = tl.zeros([R, D], tl.float32)
    m_i, l_i, acc = _block_attend(
        m_i, l_i, acc, q, qk_scale, q_pos, j0, new0, k_old0, v_old0, blk, n_blocks, base, lo,
        past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr, prep_eps,
        scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
    )  # fmt: skip
    if not MERGE:
        _store_partial(part_o_ptr, part_ml_ptr, blk, slots, h, r, row_ok, acc, m_i, l_i,
                       dequant, HKV, MG, D)  # fmt: skip
    if BPP > 1:
        if not MERGE:
            m_i = tl.full([R], -1.0e30, tl.float32)
            l_i = tl.zeros([R], tl.float32)
            acc = tl.zeros([R, D], tl.float32)
        m_i, l_i, acc = _block_attend(
            m_i, l_i, acc, q, qk_scale, q_pos, j1, new1, k_old1, v_old1, blk + 1, n_blocks,
            base, lo, past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr,
            prep_eps, scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
        )  # fmt: skip
        if not MERGE:
            _store_partial(part_o_ptr, part_ml_ptr, blk + 1, slots, h, r, row_ok, acc, m_i,
                           l_i, dequant, HKV, MG, D)  # fmt: skip
    if BPP > 2:
        if not MERGE:
            m_i = tl.full([R], -1.0e30, tl.float32)
            l_i = tl.zeros([R], tl.float32)
            acc = tl.zeros([R, D], tl.float32)
        m_i, l_i, acc = _block_attend(
            m_i, l_i, acc, q, qk_scale, q_pos, j2, new2, k_old2, v_old2, blk + 2, n_blocks,
            base, lo, past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr,
            prep_eps, scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
        )  # fmt: skip
        if not MERGE:
            _store_partial(part_o_ptr, part_ml_ptr, blk + 2, slots, h, r, row_ok, acc, m_i,
                           l_i, dequant, HKV, MG, D)  # fmt: skip
    if BPP > 3:
        if not MERGE:
            m_i = tl.full([R], -1.0e30, tl.float32)
            l_i = tl.zeros([R], tl.float32)
            acc = tl.zeros([R, D], tl.float32)
        m_i, l_i, acc = _block_attend(
            m_i, l_i, acc, q, qk_scale, q_pos, j3, new3, k_old3, v_old3, blk + 3, n_blocks,
            base, lo, past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr,
            prep_eps, scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
        )  # fmt: skip
        if not MERGE:
            _store_partial(part_o_ptr, part_ml_ptr, blk + 3, slots, h, r, row_ok, acc, m_i,
                           l_i, dequant, HKV, MG, D)  # fmt: skip
    if MERGE:
        _store_partial(part_o_ptr, part_ml_ptr, pid, slots, h, r, row_ok, acc, m_i, l_i,
                       dequant, HKV, MG, D)  # fmt: skip


def launch_split_bpp(
    qkv: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    table_stride_kv: int,
    seq_lens: torch.Tensor,
    kv_scale_orig_quant: torch.Tensor,
    cos_sin: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    prep_eps: float,
    q_scale: float,
    part_o: torch.Tensor,
    part_ml: torch.Tensor,
    num_splits: int,
    *,
    rows: int,
    reqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    window: int,
    q_fp8: bool,
    fused_prep: bool,
    qkv_permuted: bool,
    enable_pdl: bool,
    blocks_per_program: int,
    merge: bool = False,
    max_registers: Optional[int] = None,
):
    """Launch ``_swa_split_bpp_kernel`` of a ``reqs``-sequence step into ``part_o`` / ``part_ml``.

    ``blocks_per_program`` 2..4 key blocks per program, ``ceil(num_splits /
    blocks_per_program)`` programs per head and row group, with the arguments the ops pass
    ``_swa_split_kernel`` (``blocks_per_program = 1`` is that kernel, which the ops launch
    themselves). ``cos_sin`` / ``positions`` are ``None`` without RoPE. ``merge`` is for
    measurements only (see the kernel); the ops never pass it. ``max_registers``: the compile's
    register cap per thread (``maxnreg``; register allocation only, the same instructions'
    results). Returns the compiled kernel.
    """
    rope = cos_sin is not None
    row_groups = reqs * triton.cdiv(rows * Q_PER_KV, ROWS_PER_PROGRAM)
    args = (
        qkv,
        qkv.stride(0),
        kv_pool,
        block_tables,
        table_stride_kv,
        seq_lens,
        kv_scale_orig_quant,
        cos_sin if rope else kv_scale_orig_quant,
        positions if rope else seq_lens,
        prep_eps,
        q_scale,
        part_o,
        part_ml,
        math.log2(math.e) / math.sqrt(HEAD_DIM),
        num_splits,
    )
    meta = dict(
        M=rows,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        D=HEAD_DIM,
        P=kv_pool.shape[2],
        WINDOW=window,
        BLOCK=BLOCK_N,
        R=ROWS_PER_PROGRAM,
        Q_FP8=q_fp8,
        FUSED_PREP=fused_prep,
        QKV_PERMUTED=qkv_permuted,
        APPLY_ROPE=rope,
        LAUNCH_WITH_PDL=enable_pdl,
    )
    if reqs > 1:
        meta["REQS"] = reqs
    if max_registers is not None:
        meta["maxnreg"] = max_registers
    programs = triton.cdiv(num_splits, blocks_per_program)
    return _swa_split_bpp_kernel[(programs, num_kv_heads, row_groups)](
        *args,
        **meta,
        BPP=blocks_per_program,
        MERGE=merge,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        launch_pdl=enable_pdl,
    )


def check_blocks_per_program(
    blocks_per_program: int, split_max_registers: Optional[int] = None
) -> None:
    """Raise ValueError unless ``blocks_per_program`` is 1..``MAX_BLOCKS_PER_PROGRAM`` and
    ``split_max_registers`` is ``None`` or, with ``blocks_per_program`` > 1, a cap in
    ``[MIN_SPLIT_REGISTERS, MAX_SPLIT_REGISTERS]`` (the one-block kernel has no cap)."""
    if (
        isinstance(blocks_per_program, bool)
        or not isinstance(blocks_per_program, int)
        or not 1 <= blocks_per_program <= MAX_BLOCKS_PER_PROGRAM
    ):
        raise ValueError(
            f"swa_block takes 1..{MAX_BLOCKS_PER_PROGRAM} key blocks per split program, got "
            f"{blocks_per_program!r}"
        )
    if split_max_registers is None:
        return
    if (
        blocks_per_program == 1
        or isinstance(split_max_registers, bool)
        or not isinstance(split_max_registers, int)
        or not MIN_SPLIT_REGISTERS <= split_max_registers <= MAX_SPLIT_REGISTERS
    ):
        raise ValueError(
            f"swa_block's split register cap must be None or {MIN_SPLIT_REGISTERS}.."
            f"{MAX_SPLIT_REGISTERS} with 2..{MAX_BLOCKS_PER_PROGRAM} key blocks per split "
            f"program, got {split_max_registers!r} at {blocks_per_program!r}"
        )


@triton.jit
def _norm_gate_sum_of_squares(x):
    """``sum(x * x)`` of one 64-wide head in the built-in norm-gate kernel's add order.

    That kernel (4 warps, one element per thread) reduces with a butterfly
    across the 32 threads of a warp, thread-index bit 4 first, then joins the two warps
    holding the head: element ``32 w + 16 b4 + 8 b3 + 4 b2 + 2 b1 + b0`` is
    paired over ``b4``, ``b3``, ``b2``, ``b1``, ``b0``, then ``w``. Each size-2
    sum is one IEEE add, which commutes, so the result is that kernel's bits.
    """
    sq = tl.reshape(x * x, [2, 2, 2, 2, 2, 2])
    sq = tl.sum(sq, axis=1)  # b4
    sq = tl.sum(sq, axis=1)  # b3
    sq = tl.sum(sq, axis=1)  # b2
    sq = tl.sum(sq, axis=1)  # b1
    sq = tl.sum(sq, axis=1)  # b0
    return tl.sum(sq, axis=0)  # w


@triton.jit
def _swa_combine_kernel(
    part_o_ptr,
    part_ml_ptr,
    out_ptr,
    out_stride,
    gate_ptr,
    gate_token_stride,
    gate_head_stride,
    gate_eps,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    FUSED_GATE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    REQS: tl.constexpr = 1,
    EARLY_LOADS: tl.constexpr = False,
):
    G: tl.constexpr = HQ // HKV
    row = tl.program_id(0)  # token * G + query head within the KV group
    h = tl.program_id(1)
    if REQS > 1:
        # Request ``program_id(2)``'s partials, output rows and gate rows.
        req = tl.program_id(2)
        part_o_ptr += req * num_splits * HKV * M * G * D
        part_ml_ptr += req * num_splits * HKV * M * G * 2
        out_ptr += req * M * out_stride
        if FUSED_GATE:
            gate_ptr += req * M * gate_token_stride
    if EARLY_LOADS:
        if FUSED_GATE:
            # Before the wait: the gate is the QKV projection's (or the prep's), complete
            # before this grid launched -- the split launches it only after its own wait,
            # which returns once that kernel has completed. Read at L2.
            g_early = tl.load(
                gate_ptr + (row // G) * gate_token_stride + (h * G + row % G) * gate_head_stride
                + tl.arange(0, D),
                cache_modifier=".cg",
            )  # fmt: skip
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
        if FUSED_GATE:
            # As the PDL norm-gate copy: the output projection may launch once
            # this grid has passed its wait.
            tl.extra.cuda.gdc_launch_dependents()
    sp = tl.arange(0, BLOCK_S)
    s_ok = sp < num_splits
    d = tl.arange(0, D)
    part = (sp * HKV + h) * (M * G) + row
    m_split = tl.load(part_ml_ptr + part * 2, mask=s_ok, other=-1.0e30)
    l_split = tl.load(part_ml_ptr + part * 2 + 1, mask=s_ok, other=0.0)
    if EARLY_LOADS:
        # The partial O with the (max, sum) words: one dependent round trip.
        o = tl.load(part_o_ptr + part[:, None] * D + d[None, :], mask=s_ok[:, None], other=0.0)
    m_max = tl.max(m_split, 0)
    w = tl.where(s_ok & (l_split > 0), tl.math.exp2(m_split - m_max), 0.0)
    den = tl.sum(w * l_split, 0)
    if not EARLY_LOADS:
        o = tl.load(part_o_ptr + part[:, None] * D + d[None, :], mask=s_ok[:, None], other=0.0)
    num = tl.sum(o * w[:, None], 0)
    t = row // G
    head = h * G + row % G
    attention = (num / den).to(tl.bfloat16)
    if FUSED_GATE:
        # The built-in _attention_norm_gate_kernel's statements on this head.
        x = attention.to(tl.float32)
        variance = _norm_gate_sum_of_squares(x) / D
        normed = (x * tl.rsqrt(variance + gate_eps)).to(tl.bfloat16)
        if EARLY_LOADS:
            g = g_early.to(tl.float32)
        else:
            g = tl.load(gate_ptr + t * gate_token_stride + head * gate_head_stride + d).to(
                tl.float32
            )
        sigmoid = tl.sigmoid(g).to(tl.bfloat16)
        result = normed.to(tl.float32) * sigmoid.to(tl.float32)
        tl.store(out_ptr + t * out_stride + head * D + d, result)
    else:
        tl.store(out_ptr + t * out_stride + head * D + d, attention)


def max_blocks(window: int, rows: int) -> int:
    """The most 32-key blocks a ``window``-key step of ``rows`` rows can attend."""
    return triton.cdiv(window + rows - 2, BLOCK_N) + 1


def gate_head_stride(gate: torch.Tensor) -> int:
    """The stride between heads of ``gate`` as the built-in norm-gate launch computes it."""
    return gate.stride(1) if gate.ndim == 3 else HEAD_DIM


#: ``swa_block_single_early_loads``, pushed here by ``lever_settings.apply`` (default off):
#: ``swa_block_attention`` (decode steps of one sequence) builds its split kernel with
#: ``EARLY_LOADS`` -- with the fused prep, every projection load the program makes after its
#: wait is issued at once (the query rows, the key block's step K/V rows and, in program
#: (0, h, 0), the step's K/V rows it writes to the cache) before any prep arithmetic: one
#: dependent round trip, not one per prep -- and its combine with ``EARLY_LOADS``: the partial O
#: is loaded with the (max, sum) words and, with the gate, the gate head before its wait (the
#: projection that wrote it completed before the split, which launches the combine, passed its
#: own wait). The same statements on the same values: the same bits.
_SINGLE_EARLY_LOADS = False


def configure(*, single_early_loads: bool = False) -> None:
    """Set this module's lever-backed settings (``lever_settings.apply``), once per process."""
    global _SINGLE_EARLY_LOADS
    _SINGLE_EARLY_LOADS = bool(single_early_loads)


class SwaBlockAttention(OpWrapper):
    """A sliding-window layer's decode attention block, with fewer launches, the same bits.

    With ``M`` rows (positions ``L - M .. L - 1``, ``L = seq_lens[0]``), ``H``
    query and ``G = H / 8`` KV heads of width 64:

    * ``fused_prep=False``: ``qkv`` is ``preattn_prep``'s fused operand
      ``[M, (H + 2G) * 64]`` and the result is ``paged_decode_attention``'s.
    * ``fused_prep=True``: ``qkv`` is the raw QKV projection
      ``[M, 2 (H + G) * 64]`` (checkpoint order, or ``qkv_permuted``), and the
      result is ``paged_decode_attention(preattn_prep(qkv, ...)[0], ...)``
      with ``cos_sin``, ``position_ids``, ``prep_eps``, ``q_scale`` as
      ``preattn_prep`` takes them (``cos_sin=None``: NoPE).
    * ``gate`` (``[M, H * 64]`` or a ``[M, H, 64]`` view, unit last stride):
      the result is additionally the built-in output norm-gate,
      ``bf16(rmsnorm(head, gate_eps)) * bf16(sigmoid(gate head))`` per head.

    The step's K/V are written into the paged FP8 cache in place, as
    ``paged_decode_attention`` writes them. Returns a fresh ``[M, H * 64]``
    BF16 tensor. Batch 1, one sequence.

    Fusion boundary: the QKV projection before, the output projection after.

    ``enable_pdl``: both kernels launch with programmatic dependent launch. The
    split kernel reads its graph inputs and the cached K/V before
    ``griddepcontrol.wait`` and the operand after it, then lets the combine
    launch; the combine waits and, with ``gate``, lets its dependent launch.

    ``blocks_per_program`` (1..4): key blocks per split program (see the module
    docstring); 1 is the one-block kernel, every value computes the same bits.
    ``split_max_registers`` (``None`` or, with 2..4 blocks, a register cap per thread):
    the split kernel's compile register cap, the same bits.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: seq_lens[0], the sequence's K and V page ids, the
    positions (with fused_prep and RoPE), all on the device and read before
    griddepcontrol.wait; the layer's pool view; the FP8 KV scale. Call site:
    the target's sliding-window attention layers on decode steps of one
    sequence and at most 8 rows, under the swa_block fast path. Grids: split
    (ceil(num_splits / blocks_per_program), G, M*8/32), combine (M*8, G).

    blocks_per_program (1..4, default 1): key blocks per split program. 1 is
    the one-block kernel; 2..4, uncapped and under split_max_registers =
    SPLIT_REGISTER_CAP, are gated bit for bit against it (cache bytes,
    output, gated output, 1..4 sequences, window and page boundaries, a late
    PDL operand, CUDA-graph replays).

    Gated by its own tests, bit for bit against the chain it replaces
    (preattn_prep -> paged_decode_attention -> the built-in norm-gate kernel):
    KV cache bytes, attention output and gated output, at 1..8 rows across the
    window and page boundaries, eagerly and through CUDA-graph replays at
    changing contexts, behind a PDL predecessor that writes the operand late
    (the negative control); the PTX audit lists every global load before the
    wait and traces each to the seq_lens, block-table, pool, positions,
    cos/sin or scale parameter.

    Loud preconditions (ValueError): those of paged_decode_attention and, with
    fused_prep, of preattn_prep; a sliding window (window > 0); num_splits >=
    the window's largest block count; with gate, a BF16 [M, H*64] or
    [M, H, 64] gate of unit last stride.

    Not certified: full (window 0) attention, more than one sequence, head
    width other than 64, a query/KV head ratio other than 8.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> torch.Tensor:
        check_blocks_per_program(blocks_per_program, split_max_registers)
        rows = qkv.shape[0]
        page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
        width = (2 if fused_prep else 1) * (num_q_heads + num_kv_heads) * HEAD_DIM
        if not fused_prep:
            width += num_kv_heads * HEAD_DIM
        if (
            qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.stride(-1) != 1
            or qkv.shape[1] != width
            or num_q_heads != Q_PER_KV * num_kv_heads
            or not 1 <= rows <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[1:]) != (num_kv_heads, page, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or page & (page - 1) != 0
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 2
            or block_tables.shape[0] != 2
            or block_tables.stride(-1) != 1
            or seq_lens.dtype != torch.int32
            or kv_scale_orig_quant.dtype != torch.float32
            or window <= 0
            or num_splits < max_blocks(window, rows)
        ):
            raise ValueError(
                "swa_block takes BF16 qkv [1..8, "
                + ("2 (H + G) * 64" if fused_prep else "(H + 2G) * 64")
                + "] with H = 8 G, an FP8 E4M3 pool [pages, G, P, 64] (P a power of two), an int32 "
                "[2, pages] block table, int32 seq_lens, an fp32 scale, a sliding window and at least "
                f"one split per key block; got qkv {qkv.dtype} {tuple(qkv.shape)}, H={num_q_heads} "
                f"G={num_kv_heads}, pool {kv_pool.dtype} {tuple(kv_pool.shape)}, table "
                f"{block_tables.dtype} {tuple(block_tables.shape)}, seq_lens {seq_lens.dtype}, scale "
                f"{kv_scale_orig_quant.dtype}, window {window}, splits {num_splits} (need "
                f"{max_blocks(window, rows) if window > 0 else 'a window'})"
            )
        rope = fused_prep and cos_sin is not None
        positions = None
        if rope:
            positions = position_ids.reshape(-1)
            if (
                positions.numel() != rows
                or cos_sin.dtype != torch.float32
                or cos_sin.dim() != 3
                or tuple(cos_sin.shape[1:]) != (2, HEAD_DIM // 2)
                or not cos_sin.is_contiguous()
            ):
                raise ValueError(
                    f"swa_block RoPE needs one position per row and a contiguous fp32 "
                    f"[max_positions, 2, {HEAD_DIM // 2}] table, got {positions.numel()} positions "
                    f"for {rows} rows and a {cos_sin.dtype} {tuple(cos_sin.shape)} table"
                )
        if gate is not None and (
            gate.dtype != torch.bfloat16
            or gate.ndim not in (2, 3)
            or gate.stride(-1) != 1
            or gate.shape[0] != rows
            or gate.numel() != rows * num_q_heads * HEAD_DIM
            or (gate.ndim == 3 and gate.shape[-1] != HEAD_DIM)
        ):
            raise ValueError(
                f"swa_block's gate must be BF16 [rows, H * {HEAD_DIM}] or [rows, H, {HEAD_DIM}] "
                f"with unit last stride, got {gate.dtype} {tuple(gate.shape)}"
            )
        out = torch.empty((rows, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
        group_rows = rows * Q_PER_KV
        part_o = torch.empty(
            (num_splits * num_kv_heads * group_rows, HEAD_DIM),
            dtype=torch.float32,
            device=qkv.device,
        )
        part_ml = torch.empty(
            (num_splits * num_kv_heads * group_rows, 2), dtype=torch.float32, device=qkv.device
        )
        # ``swa_block_single_early_loads``: the one-request block's two launches issue their
        # operand loads at once. The blocks-per-program split keeps the served launch.
        early = _SINGLE_EARLY_LOADS and blocks_per_program <= 1
        if blocks_per_program > 1:
            launch_split_bpp(
                qkv, kv_pool, block_tables, block_tables.stride(0), seq_lens, kv_scale_orig_quant,
                cos_sin if rope else None, positions, prep_eps, q_scale, part_o, part_ml,
                num_splits, rows=rows, reqs=1, num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads, window=window, q_fp8=q_fp8, fused_prep=fused_prep,
                qkv_permuted=qkv_permuted, enable_pdl=enable_pdl,
                blocks_per_program=blocks_per_program, max_registers=split_max_registers,
            )  # fmt: skip
        else:
            if early:
                from tensorrt_llm.logger import logger

                logger.info_once(
                    "[musespark_open] swa_block_single_early_loads: early loads on the "
                    "one-request block (split: one round trip for its projection loads"
                    f"{'' if fused_prep else ' -- off without the fused prep'}; combine: O with "
                    f"the (max, sum) words"
                    f"{', the gate before its wait' if gate is not None else ''})",
                    key="musespark_open.swa_block.single_early_loads",
                )
            _swa_split_kernel[
                (num_splits, num_kv_heads, triton.cdiv(group_rows, ROWS_PER_PROGRAM))
            ](
                qkv,
                qkv.stride(0),
                kv_pool,
                block_tables,
                block_tables.stride(0),
                seq_lens,
                kv_scale_orig_quant,
                cos_sin if rope else kv_scale_orig_quant,
                positions if rope else seq_lens,
                prep_eps,
                q_scale,
                part_o,
                part_ml,
                math.log2(math.e) / math.sqrt(HEAD_DIM),
                num_splits,
                M=rows,
                HQ=num_q_heads,
                HKV=num_kv_heads,
                D=HEAD_DIM,
                P=page,
                WINDOW=window,
                BLOCK=BLOCK_N,
                R=ROWS_PER_PROGRAM,
                Q_FP8=q_fp8,
                FUSED_PREP=fused_prep,
                QKV_PERMUTED=qkv_permuted,
                APPLY_ROPE=rope,
                LAUNCH_WITH_PDL=enable_pdl,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
                launch_pdl=enable_pdl,
                # Passed only when on: the default launch is the kernel as before. The split's
                # early loads are the fused prep's operands, so they need it.
                **({"EARLY_LOADS": True} if early and fused_prep else {}),
            )
        _swa_combine_kernel[(group_rows, num_kv_heads)](
            part_o,
            part_ml,
            out,
            out.stride(0),
            out if gate is None else gate,
            0 if gate is None else gate.stride(0),
            0 if gate is None else gate_head_stride(gate),
            gate_eps,
            num_splits,
            M=rows,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            D=HEAD_DIM,
            BLOCK_S=triton.next_power_of_2(num_splits),
            FUSED_GATE=gate is not None,
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=4,
            launch_pdl=enable_pdl,
            **({"EARLY_LOADS": True} if early else {}),
        )
        return out

    def reference(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> torch.Tensor:
        """The three references in sequence: ``preattn_prep``'s (with ``fused_prep``), then
        ``paged_decode_attention``'s (which writes the step's rows into ``kv_pool``), then the
        norm-gate in float64 between its BF16 rounding points (with ``gate``).

        ``num_splits``, ``enable_pdl``, ``blocks_per_program`` and ``split_max_registers`` change
        how the op is cut and launched, not what it computes, and are ignored here.
        """
        if fused_prep:
            rope = cos_sin is not None
            qkv, _ = _preattn_prep.reference(
                qkv,
                cos_sin,
                position_ids if rope else None,
                num_q_heads,
                num_kv_heads,
                prep_eps,
                q_scale,
                qkv_permuted,
            )
        out = _paged_decode_attention.reference(
            qkv,
            kv_pool,
            block_tables,
            seq_lens,
            kv_scale_orig_quant,
            num_q_heads,
            num_kv_heads,
            window,
            num_splits,
            q_fp8,
        )
        if gate is None:
            return out
        rows = out.shape[0]
        x = out.reshape(rows, num_q_heads, HEAD_DIM).double()
        normed = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + gate_eps)).to(torch.bfloat16)
        g = gate.reshape(rows, num_q_heads, HEAD_DIM).double()
        sigmoid = torch.sigmoid(g).to(torch.bfloat16)
        gated = (normed.double() * sigmoid.double()).to(torch.bfloat16)
        return gated.reshape(rows, num_q_heads * HEAD_DIM)

    def is_valid(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> None:
        """What the kernels take and answer wrongly with no error: those of
        ``paged_decode_attention`` (a context shorter than the step, a block table too short, page
        ids outside the pool) and, with RoPE, a position outside the table."""
        rows = qkv.shape[0]
        length = int(seq_lens[0])
        page = kv_pool.shape[2]
        if length < rows:
            raise ValueError(f"seq_lens[0] = {length} is shorter than the step's {rows} rows")
        pages = (length + page - 1) // page
        if block_tables.shape[1] < pages:
            raise ValueError(
                f"a [2, {block_tables.shape[1]}] block table cannot address {length} keys "
                f"of {page}-token pages"
            )
        lo = max(length - rows - window + 1, 0) if window > 0 else 0
        used = block_tables[:, lo // page : pages]
        if used.numel() and (int(used.min()) < 0 or int(used.max()) >= kv_pool.shape[0]):
            raise ValueError(
                f"page ids [{int(used.min())}, {int(used.max())}] outside the {kv_pool.shape[0]}-page pool"
            )
        if fused_prep and cos_sin is not None and position_ids is not None and position_ids.numel():
            low, high = int(position_ids.min()), int(position_ids.max())
            if low < 0 or high >= cos_sin.shape[0]:
                raise ValueError(
                    f"positions [{low}, {high}] outside the RoPE table of {cos_sin.shape[0]}"
                )


swa_block_attention = SwaBlockAttention()
