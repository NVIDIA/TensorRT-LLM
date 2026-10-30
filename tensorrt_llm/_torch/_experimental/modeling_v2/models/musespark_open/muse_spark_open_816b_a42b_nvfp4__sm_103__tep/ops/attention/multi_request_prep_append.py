# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A multi-request decode step's pre-attention prep and generation preprocess, in one launch.

On a decode step of ``B >= 2`` sequences of ``Q`` query rows each (the target's
K = 7 verify blocks: ``Q = 8``), each target layer's built-in attention runs
three launches between the QKV projection and the trtllm-gen generation FMHA:
the pre-attention prep (``preattn_prep``), then the attention op's generation
preprocess ``thop.trtllm_gen_generation_preprocess``, whose two kernels on
this path are

* the decoder-info kernel ``computeSeqAndPaddingOffsets`` (launched because
  ``B > 1`` and ``Q > 1``), which writes the cumulative query and key lengths
  and a per-token (sequence, index) table into the attention workspace, and
* ``applyBiasRopeUpdateKVCacheV2`` without rotation (the prep applied RoPE),
  which writes every row's K and V into the paged FP8 cache, Q in E4M3 into the
  workspace's query buffer and the FMHA's two scale words into the workspace.

This entry is ``preattn_prep`` (on RoPE and NoPE layers alike) with the writes
the FMHA reads folded in, so the FMHA can be launched straight after it:

* row ``r`` of sequence ``b = r // Q`` (``t = r % Q``) sits at cache position
  ``p = seq_lens[b] - Q + t`` -- the append kernel's ``past + t`` with
  ``past = cache_seq_len - max_input_seq_len`` when the call carries no
  per-sequence generation lengths -- and its K and V go to
  ``pool[block_tables[b][kv][p >> log2 P], kv_head, p & (P - 1), :]`` as
  ``e4m3_rn_satfinite(bf16(x * bf16(s)))`` of the BF16 value the prep writes
  (``KVBlockArray`` addressing; the conversion is ``paged_decode_attention``'s,
  byte-identical to the built-in append's);
* Q in E4M3, the same conversion, ``[rows, H, 64]`` into the query buffer;
* the scale words ``bmm1 = [s_q * s_k * bmm1_host, that * log2(e)]`` and
  ``bmm2 = o_scale * s_v`` (``s`` the cache's dequant scale), in the built-in
  kernel's operation order;
* with ``Q > 1``, the cumulative query lengths ``[0, Q, 2 Q, ..., B Q]`` -- the
  decoder-info kernel's ``cu_seqlens`` for fixed query lengths, which the real
  preprocess returns as a view of the workspace (with ``Q = 1`` it launches no
  decoder-info kernel and returns none).

Not written: the decoder-info kernel's cumulative key lengths and its
per-token table. Without multi-token generation (on Blackwell the trtllm-gen
path enables spec-dec masks only for dynamic trees, so a linear draft chain
passes no generation lengths) the append kernel addresses rows itself and the
FMHA takes ``q_len_per_req`` instead of ``cu_seqlens``: nothing reads either.

Fusion boundary: the QKV projection producing ``qkv`` is outside; the
trtllm-gen generation FMHA reading the query buffer, the cache and the scales,
and the output gate's norm and sigmoid, are outside. The fused QKV operand and
the gate are still written, bit for bit ``preattn_prep``'s, so a call that ends
up running the built-in preprocess after all reads the same operand.

``prefetch`` (optional): a tensor whose bytes the kernel asks the L2 to fetch --
the layer's attention output projection weight, read after the FMHA. Every
program issues one ``cp.async.bulk.prefetch.L2`` of an equal slice right after
its wait, and a prefetching call runs one program per (row, head) item -- the
built-in grid's program count, so the requests and their slice sizes are the
built-in grid's -- and they leave every SM's TMA unit within the prep, before the
FMHA's own TMA loads of K/V start. It reads nothing back and writes nothing: the values computed are the same
with and without it.

Launch geometry, per SM version (``LAUNCH_KIND_BY_SM``). On sm103 one program
takes several consecutive (row, head) items of the row-major ``rows x (H + G)``
grid (``PACKED_GEOMETRY``), each item running the per-head code of a
one-program-per-(row, head) grid element for element (the same loads, the same
additions in the same order, the same rounding points and stores), so the bytes
written do not depend on the geometry. On sm107 the grid is one program per
(row, head), one warp each (``_multi_request_prep_append_grid_kernel``).
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from ..gemm.dense_gemm import _bulk_prefetch_l2
from .paged_decode_attention import _to_cache_fp8
from .preattn_prep import (
    HEAD_DIM,
    _inverse_rms,
    _load8,
    _normalize,
    _rotate,
    _rsqrt_approx_ftz,
    is_rope_table,
    libdevice,
    preattn_prep,
)

#: ``log2(e)`` as the built-in kernel's ``kLog2e`` float constant.
LOG2E = 1.4426950408889634
#: Query rows per sequence the entry takes: one accepted token and up to 7 drafts.
MAX_ROWS_PER_SEQ = 8
#: Sequences one call takes: 128 verify blocks of 8 rows (``preattn_prep``'s 1024-row cap), the
#: largest decode step of a generation worker of ``max_batch_size`` 128. The grid covers the
#: (row, head) items, so a call's rows do not depend on how many sequences it carries.
MAX_SEQS = 128
#: Rows one call takes.
MAX_ROWS = MAX_SEQS * MAX_ROWS_PER_SEQ
#: The launch geometry by row count: ``(last row count, (row, head) items per program, warps per
#: program, vector lanes)``, the first entry whose row count reaches the call's. An item is one
#: head of one row; a program's items are consecutive in row-major (row, head) order. With vector
#: lanes a lane loads and stores its 8 consecutive values as one vector (the same bytes). The
#: entries above 32 rows are the fastest geometries of a GB300 sweep at 16..1024 rows; up to 32
#: rows one item per program (the built-in grid) is faster in a served decode step on GB300.
PACKED_GEOMETRY = (
    (32, 1, 1, False),
    (128, 2, 1, False),
    (256, 4, 1, True),
    (768, 8, 2, True),
    (MAX_ROWS, 4, 1, True),
)


def packed_geometry(rows: int, heads: int, programs_at_least: int = 0) -> Tuple[int, int, bool]:
    """``(items per program, warps, vector lanes)`` of a call of ``rows`` rows of ``heads`` heads.

    ``programs_at_least`` lowers the items per program (never below one) until the grid has at
    least that many programs -- the prefetching call's one bulk request per program then leaves
    from at least that many programs.
    """
    items, warps, vec = next((i, w, v) for last, i, w, v in PACKED_GEOMETRY if rows <= last)
    while items > 1 and -(-rows * heads // items) < programs_at_least:
        items //= 2
    warps = max(1, min(warps, items * 8 // 32))
    return items, warps, vec


@triton.jit
def _fp8_store8(ptr, y0, y1, y2, y3, y4, y5, y6, y7, scale_bf16, mask):
    """Eight BF16 values one norm thread owns, as E4M3 at ``ptr + 0..7`` where ``mask``."""
    tl.store(ptr + 0, _to_cache_fp8(y0, scale_bf16), mask=mask)
    tl.store(ptr + 1, _to_cache_fp8(y1, scale_bf16), mask=mask)
    tl.store(ptr + 2, _to_cache_fp8(y2, scale_bf16), mask=mask)
    tl.store(ptr + 3, _to_cache_fp8(y3, scale_bf16), mask=mask)
    tl.store(ptr + 4, _to_cache_fp8(y4, scale_bf16), mask=mask)
    tl.store(ptr + 5, _to_cache_fp8(y5, scale_bf16), mask=mask)
    tl.store(ptr + 6, _to_cache_fp8(y6, scale_bf16), mask=mask)
    tl.store(ptr + 7, _to_cache_fp8(y7, scale_bf16), mask=mask)


@triton.jit
def _split8(v):
    """The 8 elements of ``v``'s last axis (shape ``[ITEMS, 2, 2, 2, 8]``), in order, each
    ``[ITEMS, 2, 2, 2]``."""
    v = tl.reshape(v, (v.shape[0], 2, 2, 2, 2, 2, 2))
    # Element k = 4 a + 2 b + c of the last three axes (a, b, c): split c, then b, then a.
    c0, c1 = tl.split(v)
    c0b0, c0b1 = tl.split(c0)
    c1b0, c1b1 = tl.split(c1)
    x0, x4 = tl.split(c0b0)
    x2, x6 = tl.split(c0b1)
    x1, x5 = tl.split(c1b0)
    x3, x7 = tl.split(c1b1)
    return x0, x1, x2, x3, x4, x5, x6, x7


@triton.jit
def _join8(x0, x1, x2, x3, x4, x5, x6, x7):
    """``_split8``'s inverse: ``[ITEMS, 2, 2, 2, 8]`` with element k = ``xk``."""
    c0 = tl.join(tl.join(x0, x4), tl.join(x2, x6))
    c1 = tl.join(tl.join(x1, x5), tl.join(x3, x7))
    v = tl.join(c0, c1)
    return tl.reshape(v, (v.shape[0], 2, 2, 2, 8))


@triton.jit
def _inverse_rms_per_item(x0, x1, x2, x3, x4, x5, x6, x7, eps):
    """``_inverse_rms`` of each item of ``[ITEMS, 2, 2, 2]`` lane-shaped values, as ``[ITEMS]``.

    The same additions in the same order: in order within a lane (``add.rn``), then lane
    bit 0, 1, 2, each a single commutative add of a size-2 axis.
    """
    s = libdevice.add_rn(x0 * x0, x1 * x1)
    s = libdevice.add_rn(s, x2 * x2)
    s = libdevice.add_rn(s, x3 * x3)
    s = libdevice.add_rn(s, x4 * x4)
    s = libdevice.add_rn(s, x5 * x5)
    s = libdevice.add_rn(s, x6 * x6)
    s = libdevice.add_rn(s, x7 * x7)
    total = tl.sum(tl.sum(tl.sum(s, axis=3), axis=2), axis=1)
    return _rsqrt_approx_ftz(total / 64.0 + eps)


@triton.jit
def _multi_request_prep_append_kernel(
    qkv_ptr,
    qkv_out_ptr,
    gate_out_ptr,
    cos_sin_ptr,
    positions_ptr,
    q8_ptr,
    pool_ptr,
    bt_ptr,
    bt_stride_seq,
    bt_stride_kv,
    seq_len_ptr,
    orig_quant_ptr,
    quant_orig_ptr,
    bmm1_ptr,
    bmm2_ptr,
    cu_seqlens_ptr,
    bmm1_host,
    o_scale,
    qkv_stride,
    qkv_out_stride,
    gate_out_stride,
    eps,
    q_scale,
    num_seqs,
    num_items,
    pf_base,
    pf_bytes,
    pf_chunk,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    ROWS_PER_SEQ: tl.constexpr,
    P: tl.constexpr,
    LOG2_P: tl.constexpr,
    SEQS_BLOCK: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    ITEMS: tl.constexpr = 1,
    LANES: tl.constexpr = 32,
    VEC: tl.constexpr = False,
    PREFETCH: tl.constexpr = False,
):
    """Program ``p`` takes the ``ITEMS`` items ``p * ITEMS ..`` of the ``rows * (H + G)``
    (row, head) items in row-major order; every item runs the one-head code of the
    one-program-per-(row, head) grid it replaces, element for element."""
    D: tl.constexpr = 64
    HALF: tl.constexpr = 32
    HEADS: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS
    PAGE: tl.constexpr = NUM_KV_HEADS * P * D
    if LAUNCH_WITH_PDL:
        # Before any global read: the projection is the previous kernel's output.
        # No early trigger: a dependent (the FMHA) launches when this grid ends.
        tl.extra.cuda.gdc_wait()
    program = tl.program_id(0)
    if PREFETCH:
        # One bulk L2 prefetch of this program's slice of ``pf_bytes`` (element 0 issues it;
        # ``LANES`` is the program's thread count, so no element is held twice). The asm
        # yields 0, so the store never happens; it keeps the request live.
        lane = tl.arange(0, LANES)
        start = program.to(tl.int64) * pf_chunk
        size = tl.minimum(pf_bytes - start, pf_chunk)
        size = tl.where((lane == 0) & (size > 0), size, 0).to(tl.int32)
        issued = _bulk_prefetch_l2(tl.full([LANES], 0, tl.int64) + pf_base + start, size)
        tl.store(cu_seqlens_ptr + lane, issued, mask=issued != 0)

    # The program's items; a tail item past the last one computes item 0's addresses
    # (every load stays in bounds) and stores nothing.
    item = program * ITEMS + tl.arange(0, ITEMS)
    valid = item < num_items
    item = tl.where(valid, item, 0)
    row = item // HEADS
    head = item % HEADS
    is_q = head < NUM_Q_HEADS
    is_kv = head >= NUM_Q_HEADS
    kv_head = head - NUM_Q_HEADS
    row64 = row.to(tl.int64)
    scale_bf16 = tl.load(orig_quant_ptr).to(tl.bfloat16)

    # The row's cache position: the sequence's last ROWS_PER_SEQ positions
    # (seq_lens counts the step's rows), addressed as KVBlockArray does.
    seq = row // ROWS_PER_SEQ
    position = tl.load(seq_len_ptr + seq) - ROWS_PER_SEQ + row % ROWS_PER_SEQ
    table = bt_ptr + seq.to(tl.int64) * bt_stride_seq + (position >> LOG2_P)
    slot = (kv_head * P + (position & (P - 1))) * D

    # The pass-through half first, so its load is in flight during the norm:
    # a query item copies its gate head, a key item its value head (and stores it
    # into the cache's V page).
    d = tl.arange(0, D)[None, :]
    if QKV_PERMUTED:
        gate_in = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * D + head * D
        v_in = NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
    else:
        gate_in = head * 2 * D + D
        v_in = 2 * NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
    pass_in = tl.where(is_q, gate_in, v_in)
    passed = tl.load(qkv_ptr + (row64 * qkv_stride + pass_in)[:, None] + d)
    q_store = (valid & is_q)[:, None]
    kv_store = (valid & is_kv)[:, None]
    tl.store(gate_out_ptr + (row64 * gate_out_stride + head * D)[:, None] + d, passed, mask=q_store)
    tl.store(
        qkv_out_ptr
        + (row64 * qkv_out_stride + (NUM_Q_HEADS + NUM_KV_HEADS) * D + kv_head * D)[:, None]
        + d,
        passed,
        mask=kv_store,
    )
    page_v = tl.load(table + bt_stride_kv, mask=valid & is_kv, other=0).to(tl.int64)
    tl.store(
        pool_ptr + (page_v * PAGE + slot)[:, None] + d,
        _to_cache_fp8(passed, scale_bf16),
        mask=kv_store,
    )

    if QKV_PERMUTED:
        in_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    else:
        in_off = tl.where(is_q, head * 2 * D, 2 * NUM_Q_HEADS * D + kv_head * D)
    out_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    scale = tl.where(is_q, q_scale, 1.0)[:, None, None, None]

    lane8 = 8 * (
        4 * tl.arange(0, 2)[None, :, None, None]
        + 2 * tl.arange(0, 2)[None, None, :, None]
        + tl.arange(0, 2)[None, None, None, :]
    )
    in_lane = qkv_ptr + (row64 * qkv_stride + in_off)[:, None, None, None] + lane8
    if VEC:
        # One 16-byte load per lane: the same 8 values.
        x0, x1, x2, x3, x4, x5, x6, x7 = _split8(
            tl.load(in_lane[:, :, :, :, None] + tl.arange(0, 8)[None, None, None, None, :]).to(
                tl.float32
            )
        )
    else:
        x0, x1, x2, x3, x4, x5, x6, x7 = _load8(in_lane)
    inverse_rms = _inverse_rms_per_item(x0, x1, x2, x3, x4, x5, x6, x7, eps)[:, None, None, None]
    y0 = _normalize(x0, inverse_rms, scale)
    y1 = _normalize(x1, inverse_rms, scale)
    y2 = _normalize(x2, inverse_rms, scale)
    y3 = _normalize(x3, inverse_rms, scale)
    y4 = _normalize(x4, inverse_rms, scale)
    y5 = _normalize(x5, inverse_rms, scale)
    y6 = _normalize(x6, inverse_rms, scale)
    y7 = _normalize(x7, inverse_rms, scale)

    if APPLY_ROPE:
        # Pair p = 4 * lane + i rotates elements 2p, 2p + 1 by table row
        # [cos(32) | sin(32)] of this token's RoPE position.
        rope_position = tl.load(positions_ptr + row).to(tl.int64)
        rope = cos_sin_ptr + (rope_position * D)[:, None, None, None] + lane8 // 2
        y0, y1 = _rotate(y0, y1, tl.load(rope + 0), tl.load(rope + HALF + 0))
        y2, y3 = _rotate(y2, y3, tl.load(rope + 1), tl.load(rope + HALF + 1))
        y4, y5 = _rotate(y4, y5, tl.load(rope + 2), tl.load(rope + HALF + 2))
        y6, y7 = _rotate(y6, y7, tl.load(rope + 3), tl.load(rope + HALF + 3))

    # The BF16 operand values: what the prep stores and what the append converts.
    b0 = y0.to(tl.bfloat16)
    b1 = y1.to(tl.bfloat16)
    b2 = y2.to(tl.bfloat16)
    b3 = y3.to(tl.bfloat16)
    b4 = y4.to(tl.bfloat16)
    b5 = y5.to(tl.bfloat16)
    b6 = y6.to(tl.bfloat16)
    b7 = y7.to(tl.bfloat16)
    store = valid[:, None, None, None]
    out = qkv_out_ptr + (row64 * qkv_out_stride + out_off)[:, None, None, None] + lane8
    q8 = q8_ptr + ((row64 * NUM_Q_HEADS + head) * D)[:, None, None, None] + lane8
    page_k = tl.load(table, mask=valid & is_kv, other=0).to(tl.int64)
    k8 = pool_ptr + (page_k * PAGE + slot)[:, None, None, None] + lane8
    if VEC:
        # One vector store per lane each: the same bytes.
        e = tl.arange(0, 8)[None, None, None, None, :]
        b = _join8(b0, b1, b2, b3, b4, b5, b6, b7)
        tl.store(out[:, :, :, :, None] + e, b, mask=store[:, :, :, :, None])
        b8 = _to_cache_fp8(b, scale_bf16)
        tl.store(q8[:, :, :, :, None] + e, b8, mask=(valid & is_q)[:, None, None, None, None])
        tl.store(k8[:, :, :, :, None] + e, b8, mask=(valid & is_kv)[:, None, None, None, None])
    else:
        tl.store(out + 0, b0, mask=store)
        tl.store(out + 1, b1, mask=store)
        tl.store(out + 2, b2, mask=store)
        tl.store(out + 3, b3, mask=store)
        tl.store(out + 4, b4, mask=store)
        tl.store(out + 5, b5, mask=store)
        tl.store(out + 6, b6, mask=store)
        tl.store(out + 7, b7, mask=store)
        # Their E4M3 copies: Q into the FMHA's query buffer, K into the cache's K page.
        _fp8_store8(
            q8, b0, b1, b2, b3, b4, b5, b6, b7, scale_bf16, (valid & is_q)[:, None, None, None]
        )
        _fp8_store8(
            k8, b0, b1, b2, b3, b4, b5, b6, b7, scale_bf16, (valid & is_kv)[:, None, None, None]
        )

    if program == 0:
        # The FMHA's scale words, in the built-in kernel's operation order:
        # (s_q * s_k) * bmm1_host, then that * log2(e); o_scale * s_v.
        s = tl.load(quant_orig_ptr)
        bmm1 = s * s * bmm1_host
        tl.store(bmm1_ptr, bmm1)
        tl.store(bmm1_ptr + 1, bmm1 * 1.4426950408889634)
        tl.store(bmm2_ptr, o_scale * s)
        if ROWS_PER_SEQ > 1:
            # The decoder-info kernel's cumulative query lengths for fixed query lengths
            # (it runs only for more than one query row per sequence).
            i = tl.arange(0, SEQS_BLOCK)
            tl.store(cu_seqlens_ptr + i, i * ROWS_PER_SEQ, mask=i <= num_seqs)


#: The kernel ``multi_request_prep_append`` launches per SM version (``launch_kind``): on sm103
#: the packed kernel above on ``PACKED_GEOMETRY``; on sm107 the grid of one program per
#: (row, head), one warp each (``_multi_request_prep_append_grid_kernel``). Both compute every
#: item with the same per-head code, so the bytes written are the same.
LAUNCH_KIND_BY_SM = {103: "packed", 107: "grid"}


@functools.lru_cache(maxsize=None)
def _sm_of(index: int) -> int:
    major, minor = torch.cuda.get_device_capability(index)
    return 10 * major + minor


def launch_kind(device: torch.device) -> str:
    """``LAUNCH_KIND_BY_SM`` for ``device``.

    Raises:
        ValueError: on a device ``LAUNCH_KIND_BY_SM`` does not list.
    """
    sm = _sm_of(device.index if device.index is not None else torch.cuda.current_device())
    kind = LAUNCH_KIND_BY_SM.get(sm)
    if kind is None:
        raise ValueError(
            f"multi_request_prep_append: no launch for sm{sm} "
            f"(supported: {sorted(LAUNCH_KIND_BY_SM)})"
        )
    return kind


@triton.jit
def _fp8_store8_all(ptr, y0, y1, y2, y3, y4, y5, y6, y7, scale_bf16):
    """Eight BF16 values one norm thread owns, as E4M3 at ``ptr + 0..7`` (no mask)."""
    tl.store(ptr + 0, _to_cache_fp8(y0, scale_bf16))
    tl.store(ptr + 1, _to_cache_fp8(y1, scale_bf16))
    tl.store(ptr + 2, _to_cache_fp8(y2, scale_bf16))
    tl.store(ptr + 3, _to_cache_fp8(y3, scale_bf16))
    tl.store(ptr + 4, _to_cache_fp8(y4, scale_bf16))
    tl.store(ptr + 5, _to_cache_fp8(y5, scale_bf16))
    tl.store(ptr + 6, _to_cache_fp8(y6, scale_bf16))
    tl.store(ptr + 7, _to_cache_fp8(y7, scale_bf16))


@triton.jit
def _multi_request_prep_append_grid_kernel(
    qkv_ptr,
    qkv_out_ptr,
    gate_out_ptr,
    cos_sin_ptr,
    positions_ptr,
    q8_ptr,
    pool_ptr,
    bt_ptr,
    bt_stride_seq,
    bt_stride_kv,
    seq_len_ptr,
    orig_quant_ptr,
    quant_orig_ptr,
    bmm1_ptr,
    bmm2_ptr,
    cu_seqlens_ptr,
    bmm1_host,
    o_scale,
    qkv_stride,
    qkv_out_stride,
    gate_out_stride,
    eps,
    q_scale,
    num_seqs,
    pf_base,
    pf_bytes,
    pf_chunk,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    ROWS_PER_SEQ: tl.constexpr,
    P: tl.constexpr,
    LOG2_P: tl.constexpr,
    SEQS_BLOCK: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    PREFETCH: tl.constexpr = False,
):
    D: tl.constexpr = 64
    HALF: tl.constexpr = 32
    PAGE: tl.constexpr = NUM_KV_HEADS * P * D
    if LAUNCH_WITH_PDL:
        # Before any global read: the projection is the previous kernel's output.
        # No early trigger: a dependent (the FMHA) launches when this grid ends.
        tl.extra.cuda.gdc_wait()
    if PREFETCH:
        # One bulk L2 prefetch of this program's slice of ``pf_bytes`` (lane 0 issues it). The
        # asm yields 0, so the store never happens; it keeps the request live.
        lane = tl.arange(0, 32)
        program = tl.program_id(0).to(tl.int64) * tl.num_programs(1) + tl.program_id(1)
        start = program * pf_chunk
        size = tl.minimum(pf_bytes - start, pf_chunk)
        size = tl.where((lane == 0) & (size > 0), size, 0).to(tl.int32)
        issued = _bulk_prefetch_l2(tl.full([32], 0, tl.int64) + pf_base + start, size)
        tl.store(cu_seqlens_ptr + lane, issued, mask=issued != 0)
    row = tl.program_id(0)
    head = tl.program_id(1)
    is_q = head < NUM_Q_HEADS
    kv_head = head - NUM_Q_HEADS
    row64 = row.to(tl.int64)
    in_row = qkv_ptr + row64 * qkv_stride
    out_row = qkv_out_ptr + row64 * qkv_out_stride
    scale_bf16 = tl.load(orig_quant_ptr).to(tl.bfloat16)

    # The row's cache position: the sequence's last ROWS_PER_SEQ positions
    # (seq_lens counts the step's rows), addressed as KVBlockArray does.
    seq = row // ROWS_PER_SEQ
    position = tl.load(seq_len_ptr + seq) - ROWS_PER_SEQ + row % ROWS_PER_SEQ
    table = bt_ptr + seq.to(tl.int64) * bt_stride_seq + (position >> LOG2_P)
    slot = (kv_head * P + (position & (P - 1))) * D

    # The pass-through half first, so its load is in flight during the norm:
    # a query program copies its gate head, a key program its value head (and
    # stores it into the cache's V page).
    d = tl.arange(0, D)
    if is_q:
        if QKV_PERMUTED:
            gate_in = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * D + head * D
        else:
            gate_in = head * 2 * D + D
        gate = tl.load(in_row + gate_in + d)
        tl.store(gate_out_ptr + row64 * gate_out_stride + head * D + d, gate)
    else:
        if QKV_PERMUTED:
            v_in = NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
        else:
            v_in = 2 * NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
        value = tl.load(in_row + v_in + d)
        tl.store(out_row + (NUM_Q_HEADS + NUM_KV_HEADS) * D + kv_head * D + d, value)
        page_v = tl.load(table + bt_stride_kv).to(tl.int64)
        tl.store(pool_ptr + page_v * PAGE + slot + d, _to_cache_fp8(value, scale_bf16))

    if QKV_PERMUTED:
        in_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    else:
        in_off = tl.where(is_q, head * 2 * D, 2 * NUM_Q_HEADS * D + kv_head * D)
    out_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    scale = tl.where(is_q, q_scale, 1.0)

    lane = (
        4 * tl.arange(0, 2)[:, None, None]
        + 2 * tl.arange(0, 2)[None, :, None]
        + tl.arange(0, 2)[None, None, :]
    )
    x0, x1, x2, x3, x4, x5, x6, x7 = _load8(in_row + in_off + lane * 8)
    inverse_rms = _inverse_rms(x0, x1, x2, x3, x4, x5, x6, x7, eps)
    y0 = _normalize(x0, inverse_rms, scale)
    y1 = _normalize(x1, inverse_rms, scale)
    y2 = _normalize(x2, inverse_rms, scale)
    y3 = _normalize(x3, inverse_rms, scale)
    y4 = _normalize(x4, inverse_rms, scale)
    y5 = _normalize(x5, inverse_rms, scale)
    y6 = _normalize(x6, inverse_rms, scale)
    y7 = _normalize(x7, inverse_rms, scale)

    if APPLY_ROPE:
        # Pair p = 4 * lane + i rotates elements 2p, 2p + 1 by table row
        # [cos(32) | sin(32)] of this token's RoPE position.
        rope_position = tl.load(positions_ptr + row).to(tl.int64)
        rope = cos_sin_ptr + rope_position * D + lane * 4
        y0, y1 = _rotate(y0, y1, tl.load(rope + 0), tl.load(rope + HALF + 0))
        y2, y3 = _rotate(y2, y3, tl.load(rope + 1), tl.load(rope + HALF + 1))
        y4, y5 = _rotate(y4, y5, tl.load(rope + 2), tl.load(rope + HALF + 2))
        y6, y7 = _rotate(y6, y7, tl.load(rope + 3), tl.load(rope + HALF + 3))

    # The BF16 operand values: what the prep stores and what the append converts.
    b0 = y0.to(tl.bfloat16)
    b1 = y1.to(tl.bfloat16)
    b2 = y2.to(tl.bfloat16)
    b3 = y3.to(tl.bfloat16)
    b4 = y4.to(tl.bfloat16)
    b5 = y5.to(tl.bfloat16)
    b6 = y6.to(tl.bfloat16)
    b7 = y7.to(tl.bfloat16)
    out = out_row + out_off + lane * 8
    tl.store(out + 0, b0)
    tl.store(out + 1, b1)
    tl.store(out + 2, b2)
    tl.store(out + 3, b3)
    tl.store(out + 4, b4)
    tl.store(out + 5, b5)
    tl.store(out + 6, b6)
    tl.store(out + 7, b7)

    # Their E4M3 copies: Q into the FMHA's query buffer, K into the cache's K page.
    if is_q:
        q8 = q8_ptr + (row64 * NUM_Q_HEADS + head) * D + lane * 8
        _fp8_store8_all(q8, b0, b1, b2, b3, b4, b5, b6, b7, scale_bf16)
    else:
        page_k = tl.load(table).to(tl.int64)
        k8 = pool_ptr + page_k * PAGE + slot + lane * 8
        _fp8_store8_all(k8, b0, b1, b2, b3, b4, b5, b6, b7, scale_bf16)

    if (row == 0) & (head == 0):
        # The FMHA's scale words, in the built-in kernel's operation order:
        # (s_q * s_k) * bmm1_host, then that * log2(e); o_scale * s_v.
        s = tl.load(quant_orig_ptr)
        bmm1 = s * s * bmm1_host
        tl.store(bmm1_ptr, bmm1)
        tl.store(bmm1_ptr + 1, bmm1 * 1.4426950408889634)
        tl.store(bmm2_ptr, o_scale * s)
        if ROWS_PER_SEQ > 1:
            # The decoder-info kernel's cumulative query lengths for fixed query lengths
            # (it runs only for more than one query row per sequence).
            i = tl.arange(0, SEQS_BLOCK)
            tl.store(cu_seqlens_ptr + i, i * ROWS_PER_SEQ, mask=i <= num_seqs)


def _cache_positions(seq_lens: torch.Tensor, num_seqs: int, rows_per_seq: int) -> torch.Tensor:
    """Every row's cache position ``seq_lens[b] - Q + t``, as int64 ``[B * Q]``."""
    lengths = seq_lens[:num_seqs].long()
    token = torch.arange(rows_per_seq, device=seq_lens.device)
    return (lengths[:, None] - rows_per_seq + token[None, :]).reshape(-1)


class MultiRequestPrepAppend(OpWrapper):
    """``preattn_prep`` for ``B`` sequences, plus their FP8 K/V append, E4M3 Q, FMHA scales and
    cumulative query lengths.

    With ``H = num_q_heads``, ``G = num_kv_heads``, head width 64, ``B`` =
    ``num_seqs`` sequences of ``Q`` rows each (row ``r`` of sequence ``b = r //
    Q`` at cache position ``p = seq_lens[b] - Q + r % Q``), ``s =
    kv_scale_orig_quant[0]`` and ``u = kv_scale_quant_orig[0]``::

        qkv_out, gate_out = preattn_prep(qkv, cos_sin, position_ids, ...)   # bit for bit
        e4m3(x)  = e4m3_rn_satfinite(bf16(x * bf16(s)))
        q_fp8[r, h, :]                                              = e4m3(q'[r, h])
        kv_pool[block_tables[b][0][p >> log2 P], g, p & (P - 1), :] = e4m3(k'[r, g])
        kv_pool[block_tables[b][1][p >> log2 P], g, p & (P - 1), :] = e4m3(v[r, g])
        bmm1_scale[0:2] = [u * u * bmm1_host_scale, that * log2(e)]
        bmm2_scale[0]   = o_scale * u
        cu_seqlens[0:B + 1] = [0, Q, ..., B Q]                      # only when Q > 1

    ``q'``, ``k'`` are ``qkv_out``'s query and key heads (normalized, scaled and,
    with ``cos_sin``, rotated). ``qkv_out`` and ``gate_out`` are fresh and
    contiguous; ``q_fp8`` (at least ``B Q H 64`` bytes, any dtype), ``kv_pool``,
    ``bmm1_scale`` (fp32, at least 2), ``bmm2_scale`` (fp32, at least 1) and
    ``cu_seqlens`` (int32, at least ``B + 1``) are written in place.
    ``block_tables`` is the TRT-LLM table ``[>= B, 2, pages]`` (K page ids, then
    V page ids) into ``kv_pool [pages, G, P, 64]``.

    ``enable_pdl`` launches with programmatic dependent launch: the grid may
    start while the projection's kernel finishes and waits before its first
    global read; it triggers its dependents only by completing.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    note = """
    Metadata consumed: seq_lens[0:B] (device, read by the kernel), the
    sequences' K and V page ids, the layer's pool view, the two FP8 KV
    scales, the FMHA's host bmm1 scale and the output scale (1.0 for a BF16
    output); with RoPE the per-row positions and the cos/sin table. Call site:
    every target layer (46 RoPE + 16 NoPE) on decode steps of 2..128 sequences
    of the same number of rows (fast path multi_request_prep_append), whose
    trtllm-gen generation preprocess launch (decoder-info + append kernels)
    the target then skips. Grid on sm103: the rows * (H + G) (row, head) items in
    row-major order, PACKED_GEOMETRY's items, warps and lanes per program by row count
    (with a prefetch, one program per item); on sm107: (rows, H + G), one warp per
    program.

    Bit-identical, gated by its own tests: qkv_out and gate_out against
    preattn_prep; the cache bytes, the E4M3 query bytes, both scale words and
    the cumulative query lengths against thop.trtllm_gen_generation_preprocess
    run on the same projection, pool and tables (every pool byte, including
    the untouched ones), including saturating, NaN, zero and special heads;
    and the FMHA output on both paths, eager and from a captured graph.

    Loud preconditions (ValueError before any launch): BF16 projection with
    unit last stride and width 2 * (H + G) * 64, B * Q rows with 2 <= B <= 128
    and 1 <= Q <= 8, FP8 E4M3 pool [pages, G, P, 64] with P a power of two,
    int32 [>= B, 2, pages] block table with unit last stride, int32 seq_lens
    of at least B, int32 cu_seqlens of at least B + 1, fp32 scales, a query
    buffer of at least B Q H 64 bytes, and with RoPE a contiguous fp32
    [positions, 2, 32] table and one position per row.

    Not certified: sequences of different row counts (the target routes only
    uniform steps), a sequence shorter than its rows, head width other than
    64, NVFP4 or BF16 caches, multi-token generation (per-sequence generation
    lengths and the per-token table).
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        num_seqs: int,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        q_fp8: torch.Tensor,
        bmm1_scale: torch.Tensor,
        bmm2_scale: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bmm1_host_scale: float,
        o_scale: float = 1.0,
        enable_pdl: bool = False,
        prefetch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, width = qkv.shape
        page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
        rows_per_seq = rows // num_seqs if num_seqs > 0 else 0
        if (
            qkv.dtype != torch.bfloat16
            or qkv.stride(-1) != 1
            or width != 2 * (num_q_heads + num_kv_heads) * HEAD_DIM
            or not 2 <= num_seqs <= MAX_SEQS
            or rows != num_seqs * rows_per_seq
            or not 1 <= rows_per_seq <= MAX_ROWS_PER_SEQ
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[1:]) != (num_kv_heads, page, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or page <= 0
            or page & (page - 1) != 0
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 3
            or block_tables.shape[0] < num_seqs
            or block_tables.shape[1] != 2
            or block_tables.stride(-1) != 1
            or seq_lens.dtype != torch.int32
            or seq_lens.numel() < num_seqs
            or kv_scale_orig_quant.dtype != torch.float32
            or kv_scale_quant_orig.dtype != torch.float32
            or bmm1_scale.dtype != torch.float32
            or bmm1_scale.numel() < 2
            or bmm2_scale.dtype != torch.float32
            or bmm2_scale.numel() < 1
            or cu_seqlens.dtype != torch.int32
            or cu_seqlens.numel() < num_seqs + 1
            or not q_fp8.is_contiguous()
            or q_fp8.numel() * q_fp8.element_size() < rows * num_q_heads * HEAD_DIM
        ):
            raise ValueError(
                "multi_request_prep_append takes a BF16 [B Q, 2 (H + G) 64] projection of "
                f"2..{MAX_SEQS} sequences of 1..{MAX_ROWS_PER_SEQ} rows, an FP8 E4M3 pool "
                "[pages, G, P, 64] (P a power of two), an int32 [>= B, 2, pages] block table, "
                "int32 seq_lens and cu_seqlens, fp32 scales and a contiguous query buffer of "
                f"B Q H 64 bytes; got qkv {qkv.dtype} {tuple(qkv.shape)} strides {qkv.stride()}, "
                f"B={num_seqs}, H={num_q_heads} G={num_kv_heads}, pool {kv_pool.dtype} "
                f"{tuple(kv_pool.shape)}, table {block_tables.dtype} {tuple(block_tables.shape)}, "
                f"seq_lens {seq_lens.dtype}[{seq_lens.numel()}], cu_seqlens {cu_seqlens.dtype}"
                f"[{cu_seqlens.numel()}], scales {kv_scale_orig_quant.dtype}/"
                f"{kv_scale_quant_orig.dtype}, bmm {bmm1_scale.dtype}[{bmm1_scale.numel()}]/"
                f"{bmm2_scale.dtype}[{bmm2_scale.numel()}], query buffer "
                f"{q_fp8.numel() * q_fp8.element_size()} bytes"
            )
        positions = None
        if cos_sin is not None:
            positions = position_ids.reshape(-1)
            if positions.numel() != rows or not is_rope_table(cos_sin):
                raise ValueError(
                    "multi_request_prep_append RoPE needs one position per row and a contiguous "
                    f"fp32 [max_positions, 2, {HEAD_DIM // 2}] table, got {positions.numel()} "
                    f"positions for {rows} rows and a {cos_sin.dtype} {tuple(cos_sin.shape)} table"
                )
        elif position_ids is not None:
            raise ValueError("multi_request_prep_append: position_ids without a cos/sin table")
        heads = num_q_heads + num_kv_heads
        num_items = rows * heads
        kind = launch_kind(qkv.device)
        pf_base, pf_bytes, pf_chunk = 0, 0, 16
        programs_at_least = 0
        if prefetch is not None:
            pf_bytes = prefetch.numel() * prefetch.element_size()
            if not (
                prefetch.is_cuda
                and prefetch.is_contiguous()
                and prefetch.data_ptr() % 16 == 0
                and pf_bytes % 16 == 0
            ):
                raise ValueError(
                    "multi_request_prep_append prefetches a contiguous, 16-byte aligned CUDA tensor "
                    f"of a multiple of 16 bytes, got {tuple(prefetch.shape)} {prefetch.dtype} at "
                    f"{prefetch.data_ptr():#x}"
                )
            # One bulk request per program, one program per (row, head) item: the o_proj
            # weight is requested in the one-program-per-(row, head) grid's slices, from
            # every SM, whatever the table's packing at these rows.
            programs_at_least = num_items
        if kind == "grid":
            items, warps, vec = 1, 1, False
        else:
            items, warps, vec = packed_geometry(rows, heads, programs_at_least)
        programs = -(-num_items // items)
        if prefetch is not None:
            pf_base = prefetch.data_ptr()
            pf_chunk = ((pf_bytes + programs - 1) // programs + 15) // 16 * 16
        qkv_out = torch.empty(
            (rows, (num_q_heads + 2 * num_kv_heads) * HEAD_DIM), dtype=qkv.dtype, device=qkv.device
        )
        gate_out = torch.empty((rows, num_q_heads * HEAD_DIM), dtype=qkv.dtype, device=qkv.device)
        if kind == "grid":
            _multi_request_prep_append_grid_kernel[(rows, heads)](
                qkv,
                qkv_out,
                gate_out,
                cos_sin,
                positions,
                q_fp8.view(torch.uint8).view(torch.float8_e4m3fn),
                kv_pool,
                block_tables,
                block_tables.stride(0),
                block_tables.stride(1),
                seq_lens,
                kv_scale_orig_quant,
                kv_scale_quant_orig,
                bmm1_scale,
                bmm2_scale,
                cu_seqlens,
                bmm1_host_scale,
                o_scale,
                qkv.stride(0),
                qkv_out.stride(0),
                gate_out.stride(0),
                eps,
                q_scale,
                num_seqs,
                pf_base,
                pf_bytes,
                pf_chunk,
                NUM_Q_HEADS=num_q_heads,
                NUM_KV_HEADS=num_kv_heads,
                QKV_PERMUTED=qkv_permuted,
                APPLY_ROPE=cos_sin is not None,
                ROWS_PER_SEQ=rows_per_seq,
                P=page,
                LOG2_P=page.bit_length() - 1,
                SEQS_BLOCK=triton.next_power_of_2(num_seqs + 1),
                LAUNCH_WITH_PDL=enable_pdl,
                PREFETCH=prefetch is not None,
                num_warps=1,
                launch_pdl=enable_pdl,
            )
            return qkv_out, gate_out
        _multi_request_prep_append_kernel[(programs,)](
            qkv,
            qkv_out,
            gate_out,
            cos_sin,
            positions,
            q_fp8.view(torch.uint8).view(torch.float8_e4m3fn),
            kv_pool,
            block_tables,
            block_tables.stride(0),
            block_tables.stride(1),
            seq_lens,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            bmm1_scale,
            bmm2_scale,
            cu_seqlens,
            bmm1_host_scale,
            o_scale,
            qkv.stride(0),
            qkv_out.stride(0),
            gate_out.stride(0),
            eps,
            q_scale,
            num_seqs,
            num_items,
            pf_base,
            pf_bytes,
            pf_chunk,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            QKV_PERMUTED=qkv_permuted,
            APPLY_ROPE=cos_sin is not None,
            ROWS_PER_SEQ=rows_per_seq,
            P=page,
            LOG2_P=page.bit_length() - 1,
            SEQS_BLOCK=triton.next_power_of_2(num_seqs + 1),
            LAUNCH_WITH_PDL=enable_pdl,
            ITEMS=items,
            LANES=32 * warps,
            VEC=vec,
            PREFETCH=prefetch is not None,
            num_warps=warps,
            launch_pdl=enable_pdl,
        )
        return qkv_out, gate_out

    def reference(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        num_seqs: int,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        q_fp8: torch.Tensor,
        bmm1_scale: torch.Tensor,
        bmm2_scale: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bmm1_host_scale: float,
        o_scale: float = 1.0,
        enable_pdl: bool = False,
        prefetch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``preattn_prep.reference`` (float64 between its rounding points), then the same
        in-place writes from its BF16 values.

        The E4M3 conversions round the BF16 product with the BF16 scale, then to
        E4M3 with saturation; the scale words are fp32 products in the kernel's
        order. ``enable_pdl`` changes how the op is launched and ``prefetch`` only asks
        the L2 for bytes; both are ignored here.
        """
        rows = qkv.shape[0]
        page = kv_pool.shape[2]
        rows_per_seq = rows // num_seqs
        qkv_out, gate_out = preattn_prep.reference(
            qkv,
            cos_sin,
            position_ids,
            num_q_heads,
            num_kv_heads,
            eps,
            q_scale,
            qkv_permuted,
        )
        q_width, kv_width = num_q_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
        q = qkv_out[:, :q_width].reshape(rows, num_q_heads, HEAD_DIM)
        k = qkv_out[:, q_width : q_width + kv_width].reshape(rows, num_kv_heads, HEAD_DIM)
        v = qkv_out[:, q_width + kv_width :].reshape(rows, num_kv_heads, HEAD_DIM)
        s_bf16 = kv_scale_orig_quant.float()[0].to(torch.bfloat16).float()

        def e4m3(x: torch.Tensor) -> torch.Tensor:
            # E4M3 round-to-nearest with saturation: a finite overflow lands on
            # +-448 rather than torch's NaN; NaN stays NaN.
            y = (x.float() * s_bf16).to(torch.bfloat16).float()
            return y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)

        q8 = q_fp8.view(torch.uint8).reshape(-1)
        q8[: rows * num_q_heads * HEAD_DIM] = e4m3(q).reshape(-1)
        pool_bytes = kv_pool.view(torch.uint8)
        positions = _cache_positions(seq_lens, num_seqs, rows_per_seq)
        seqs = torch.arange(num_seqs, device=qkv.device).repeat_interleave(rows_per_seq)
        index = torch.div(positions, page, rounding_mode="floor")
        k_pages = block_tables[seqs, 0, index].long()
        v_pages = block_tables[seqs, 1, index].long()
        slots = positions - index * page
        for g in range(num_kv_heads):
            pool_bytes[k_pages, g, slots] = e4m3(k[:, g])
            pool_bytes[v_pages, g, slots] = e4m3(v[:, g])
        u = kv_scale_quant_orig.float()[0]
        bmm1 = u * u * torch.tensor(bmm1_host_scale, dtype=torch.float32, device=u.device)
        bmm1_scale[0] = bmm1
        bmm1_scale[1] = bmm1 * torch.tensor(LOG2E, dtype=torch.float32, device=u.device)
        bmm2_scale[0] = torch.tensor(o_scale, dtype=torch.float32, device=u.device) * u
        if rows_per_seq > 1:
            cu_seqlens[: num_seqs + 1] = torch.arange(
                0, (num_seqs + 1) * rows_per_seq, rows_per_seq, dtype=torch.int32, device=u.device
            )
        return qkv_out, gate_out

    def is_valid(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        num_seqs: int,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        kv_scale_quant_orig: torch.Tensor,
        q_fp8: torch.Tensor,
        bmm1_scale: torch.Tensor,
        bmm2_scale: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bmm1_host_scale: float,
        o_scale: float = 1.0,
        enable_pdl: bool = False,
        prefetch: Optional[torch.Tensor] = None,
    ) -> None:
        """Inputs the kernel takes and answers wrongly, with no error.

        A sequence shorter than its rows puts keys at negative positions; a block
        table shorter than a sequence reads page ids past its row; a page id outside
        the pool writes outside it; a RoPE position outside the table is read from
        outside it.
        """
        rows = qkv.shape[0]
        rows_per_seq = rows // num_seqs
        page = kv_pool.shape[2]
        lengths = seq_lens[:num_seqs].long()
        if int(lengths.min()) < rows_per_seq:
            raise ValueError(
                f"a sequence of {int(lengths.min())} keys is shorter than its {rows_per_seq} rows"
            )
        last = (lengths - 1) // page
        if block_tables.shape[2] <= int(last.max()):
            raise ValueError(
                f"a [.., 2, {block_tables.shape[2]}] block table cannot address "
                f"{int(lengths.max())} keys of {page}-token pages"
            )
        positions = _cache_positions(seq_lens, num_seqs, rows_per_seq)
        seqs = torch.arange(num_seqs, device=qkv.device).repeat_interleave(rows_per_seq)
        used = block_tables[seqs, :, positions // page]
        if int(used.min()) < 0 or int(used.max()) >= kv_pool.shape[0]:
            raise ValueError(
                f"page ids [{int(used.min())}, {int(used.max())}] outside the "
                f"{kv_pool.shape[0]}-page pool"
            )
        if cos_sin is not None and position_ids is not None and position_ids.numel():
            low, high = int(position_ids.min()), int(position_ids.max())
            if low < 0 or high >= cos_sin.shape[0]:
                raise ValueError(
                    f"positions [{low}, {high}] outside the RoPE table of {cos_sin.shape[0]}"
                )


multi_request_prep_append = MultiRequestPrepAppend()
