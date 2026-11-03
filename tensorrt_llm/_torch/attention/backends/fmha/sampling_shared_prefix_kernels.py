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
"""Triton kernels around the FlashInfer calls of the n-sample shared-prefix path.

One kernel before the attention calls writes every row's current K/V into its
cache page and gathers the queries into group order; one kernel after them
merges the prefix and tail states and scatters the result back to batch order
in the output dtype. Each replaces a handful of small PyTorch ops per layer.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _write_kv_gather_q_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_row_stride,
    kv_row_stride,
    pool_ptr,
    pool_page_stride,
    pool_kv_stride,
    pool_head_stride,
    pool_slot_stride,
    write_page_ptr,
    write_slot_ptr,
    perm_ptr,
    q_out_ptr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_KV_HEADS: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, HEAD_DIM)

    kv_heads = tl.arange(0, BLOCK_KV_HEADS)
    kv_mask = kv_heads[:, None] < NUM_KV_HEADS
    kv_offsets = kv_heads[:, None] * HEAD_DIM + dims[None, :]
    page = tl.load(write_page_ptr + row).to(tl.int64)
    slot = tl.load(write_slot_ptr + row).to(tl.int64)
    cache = (
        pool_ptr
        + page * pool_page_stride
        + slot * pool_slot_stride
        + kv_heads[:, None].to(tl.int64) * pool_head_stride
        + dims[None, :]
    )
    k = tl.load(k_ptr + row * kv_row_stride + kv_offsets, mask=kv_mask)
    v = tl.load(v_ptr + row * kv_row_stride + kv_offsets, mask=kv_mask)
    tl.store(cache, k, mask=kv_mask)
    tl.store(cache + pool_kv_stride, v, mask=kv_mask)

    if HAS_PERM:
        source = tl.load(perm_ptr + row).to(tl.int64)
    else:
        source = row.to(tl.int64)
    heads = tl.arange(0, BLOCK_HEADS)
    q_mask = heads[:, None] < NUM_HEADS
    q_offsets = heads[:, None] * HEAD_DIM + dims[None, :]
    q = tl.load(q_ptr + source * q_row_stride + q_offsets, mask=q_mask)
    tl.store(q_out_ptr + row.to(tl.int64) * (NUM_HEADS * HEAD_DIM) + q_offsets, q, mask=q_mask)


@triton.jit
def _merge_scatter_kernel(
    prefix_out_ptr,
    prefix_lse_ptr,
    tail_out_ptr,
    tail_lse_ptr,
    empty_ptr,
    perm_ptr,
    out_ptr,
    out_row_stride,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_EMPTY: tl.constexpr,
    Q_LEN: tl.constexpr,
):
    # One program per grouped query token; token ``row`` belongs to grouped row
    # ``row // Q_LEN`` (``Q_LEN`` tokens per row, 1 outside speculative decoding).
    row = tl.program_id(0)
    heads = tl.arange(0, BLOCK_HEADS)
    dims = tl.arange(0, HEAD_DIM)
    head_mask = heads < NUM_HEADS
    mask = head_mask[:, None]
    offsets = row.to(tl.int64) * (NUM_HEADS * HEAD_DIM) + heads[:, None] * HEAD_DIM + dims[None, :]

    tail_out = tl.load(tail_out_ptr + offsets, mask=mask).to(tl.float32)
    tail_lse = tl.load(tail_lse_ptr + row * NUM_HEADS + heads, mask=head_mask, other=0.0)
    prefix_out = tl.load(prefix_out_ptr + offsets, mask=mask).to(tl.float32)
    prefix_lse = tl.load(prefix_lse_ptr + row * NUM_HEADS + heads, mask=head_mask, other=0.0)
    # Base-2 log-sum-exp states (FlashInfer's convention).
    peak = tl.maximum(prefix_lse, tail_lse)
    w_prefix = tl.exp2(prefix_lse - peak)
    w_tail = tl.exp2(tail_lse - peak)
    merged = (prefix_out * w_prefix[:, None] + tail_out * w_tail[:, None]) / (w_prefix + w_tail)[
        :, None
    ]
    if HAS_EMPTY:
        # A placeholder prefix state must not contribute, even if it is not finite.
        empty = tl.load(empty_ptr + row // Q_LEN) != 0
        merged = tl.where(empty, tail_out, merged)

    if HAS_PERM:
        target = tl.load(perm_ptr + row // Q_LEN).to(tl.int64) * Q_LEN + row % Q_LEN
    else:
        target = row.to(tl.int64)
    tl.store(
        out_ptr + target * out_row_stride + heads[:, None] * HEAD_DIM + dims[None, :],
        merged.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def write_kv_gather_q(
    pool: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    write_page: torch.Tensor,
    write_slot: torch.Tensor,
    perm: Optional[torch.Tensor],
) -> torch.Tensor:
    """Write each row's K/V at (``write_page``, ``write_slot``) and return the
    queries in group order (``q[perm]``) as a contiguous tensor.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND),
            innermost dimension contiguous.
        q: [rows, heads, head_dim]; k, v: [rows, kv_heads, head_dim]; each row's
            heads contiguous (views of the packed QKV input are fine).
    """
    rows, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(2) != 1 or tensor.stride(1) != head_dim:
            raise ValueError(f"{name} must have contiguous heads, got strides {tensor.stride()}.")
    if k.stride(0) != v.stride(0):
        raise ValueError("k and v must share a row stride.")
    if pool.stride(4) != 1:
        raise ValueError(f"pool must be contiguous in head_dim, got strides {pool.stride()}.")
    q_out = torch.empty((rows, num_heads, head_dim), dtype=q.dtype, device=q.device)
    if rows == 0:
        return q_out
    _write_kv_gather_q_kernel[(rows,)](
        q,
        k,
        v,
        q.stride(0),
        k.stride(0),
        pool,
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        write_page,
        write_slot,
        perm if perm is not None else write_page,
        q_out,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEADS=triton.next_power_of_2(num_heads),
        BLOCK_KV_HEADS=triton.next_power_of_2(num_kv_heads),
        HAS_PERM=perm is not None,
    )
    return q_out


def merge_scatter(
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    tail_out: torch.Tensor,
    tail_lse: torch.Tensor,
    out: torch.Tensor,
    perm: Optional[torch.Tensor],
    prefix_empty: Optional[torch.Tensor],
    q_len: int = 1,
) -> None:
    """Merge two attention states (grouped order) and write ``out[perm]``.

    With ``q_len`` > 1 every grouped row owns ``q_len`` consecutive query tokens;
    ``perm`` and ``prefix_empty`` index rows, the tensors tokens.

    Args:
        prefix_out, tail_out: [rows, heads, head_dim] normalized partial outputs,
            contiguous.
        prefix_lse, tail_lse: [rows, heads] fp32 base-2 log-sum-exp, contiguous.
        out: [rows, heads, head_dim] output in batch order; each row contiguous.
        perm: [rows] batch row of each grouped position, or None for identity.
        prefix_empty: [rows] bool, grouped order, or None: rows that take the
            tail state alone.
    """
    rows, num_heads, head_dim = prefix_out.shape
    for name, tensor in (("prefix_out", prefix_out), ("tail_out", tail_out)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if out.stride(2) != 1 or out.stride(1) != head_dim:
        raise ValueError(f"out must have contiguous heads, got strides {out.stride()}.")
    if rows == 0:
        return
    _merge_scatter_kernel[(rows,)](
        prefix_out,
        prefix_lse,
        tail_out,
        tail_lse,
        prefix_empty if prefix_empty is not None else prefix_lse,
        perm if perm is not None else prefix_lse,
        out,
        out.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEADS=triton.next_power_of_2(num_heads),
        HAS_PERM=perm is not None,
        HAS_EMPTY=prefix_empty is not None,
        Q_LEN=q_len,
    )


@triton.jit
def _step_slots_kernel(
    kv_lens_ptr,
    perm_ptr,
    write_slot_ptr,
    tail_last_page_len_ptr,
    rows,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    # Clamp offsets before indexed loads so masked lanes never speculatively
    # access an unmapped address on SM90 (which lacks hardware-predicated TMA).
    safe = tl.minimum(offsets, rows - 1)
    kv_len = tl.maximum(tl.load(kv_lens_ptr + safe, mask=mask, other=1), 1)
    tl.store(write_slot_ptr + safe, ((kv_len - 1) % TOKENS_PER_BLOCK).to(tl.int64), mask=mask)
    source = tl.load(perm_ptr + safe).to(tl.int64)  # perm[safe] is always a valid row index
    kv_len = tl.maximum(tl.load(kv_lens_ptr + source), 1)  # source < rows, always in-bounds
    last = (kv_len - 1) % TOKENS_PER_BLOCK + 1
    tl.store(tail_last_page_len_ptr + safe, last.to(tl.int32), mask=mask)


def step_slots(
    kv_lens: torch.Tensor,
    perm: torch.Tensor,
    write_slot: torch.Tensor,
    tail_last_page_len: torch.Tensor,
    tokens_per_block: int,
) -> None:
    """Derive the per-step slot indices from the device KV lengths.

    Writes, for ``rows = kv_lens.numel()`` rows, ``write_slot[r]`` (int64, batch
    order): the slot of row ``r``'s current token in its page, and
    ``tail_last_page_len[i]`` (int32, grouped order): the filled length of the
    last page of row ``perm[i]``. Both move every step while the pages stay.
    """
    rows = kv_lens.numel()
    if rows == 0:
        return
    block = 256
    _step_slots_kernel[(triton.cdiv(rows, block),)](
        kv_lens,
        perm,
        write_slot,
        tail_last_page_len,
        rows,
        TOKENS_PER_BLOCK=tokens_per_block,
        BLOCK=block,
    )


@triton.jit
def _write_kv_gather_q_spec_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_row_stride,
    kv_row_stride,
    pool_ptr,
    pool_page_stride,
    pool_kv_stride,
    pool_head_stride,
    pool_slot_stride,
    kv_lens_ptr,
    perm_ptr,
    tail_indptr_ptr,
    shared_ptr,
    tail_pages_ptr,
    tail_capacity,
    q_out_ptr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_KV_HEADS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    Q_LEN: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    # Program ``g`` handles grouped token g: row i = g // Q_LEN of the grouped
    # order, token j = g % Q_LEN of that row, batch row perm[i].
    g = tl.program_id(0)
    i = g // Q_LEN
    j = g % Q_LEN
    dims = tl.arange(0, HEAD_DIM)
    if HAS_PERM:
        row = tl.load(perm_ptr + i).to(tl.int64)
    else:
        row = i.to(tl.int64)
    token = row * Q_LEN + j

    # The row's tokens of this step sit at kv_len - Q_LEN + j on the device KV
    # length, which can be below the host length the page tables were built for.
    kv_len = tl.load(kv_lens_ptr + row).to(tl.int64)
    pos = kv_len - Q_LEN + j
    page_in_tail = pos // TOKENS_PER_BLOCK - tl.load(shared_ptr + i).to(tl.int64)
    entry = tl.load(tail_indptr_ptr + i).to(tl.int64) + page_in_tail
    valid = (pos >= 0) & (page_in_tail >= 0) & (entry < tail_capacity)
    page = tl.load(tail_pages_ptr + entry, mask=valid, other=0).to(tl.int64)
    slot = pos % TOKENS_PER_BLOCK

    kv_heads = tl.arange(0, BLOCK_KV_HEADS)
    kv_mask = (kv_heads[:, None] < NUM_KV_HEADS) & valid
    kv_offsets = kv_heads[:, None] * HEAD_DIM + dims[None, :]
    cache = (
        pool_ptr
        + page * pool_page_stride
        + slot * pool_slot_stride
        + kv_heads[:, None].to(tl.int64) * pool_head_stride
        + dims[None, :]
    )
    k = tl.load(k_ptr + token * kv_row_stride + kv_offsets, mask=kv_mask)
    v = tl.load(v_ptr + token * kv_row_stride + kv_offsets, mask=kv_mask)
    tl.store(cache, k, mask=kv_mask)
    tl.store(cache + pool_kv_stride, v, mask=kv_mask)

    heads = tl.arange(0, BLOCK_HEADS)
    q_mask = heads[:, None] < NUM_HEADS
    q_offsets = heads[:, None] * HEAD_DIM + dims[None, :]
    q = tl.load(q_ptr + token * q_row_stride + q_offsets, mask=q_mask)
    tl.store(q_out_ptr + g.to(tl.int64) * (NUM_HEADS * HEAD_DIM) + q_offsets, q, mask=q_mask)


def write_kv_gather_q_spec(
    pool: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_lens: torch.Tensor,
    perm: Optional[torch.Tensor],
    tail_indptr: torch.Tensor,
    shared: torch.Tensor,
    tail_pages: torch.Tensor,
    q_len: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Speculative-decoding variant of ``write_kv_gather_q``: ``q_len`` tokens per row.

    Token ``j`` of batch row ``r`` is stored at position ``kv_lens[r] - q_len + j``
    (device lengths, which include this step's tokens), in the page that the
    row's tail table names for it, and the queries are returned in group order.

    Args:
        q: [rows * q_len, heads, head_dim]; k, v: [rows * q_len, kv_heads, head_dim].
        kv_lens: [rows] device KV lengths in batch order.
        perm: [rows] batch row of each grouped row, or None.
        tail_indptr: [rows + 1] int32: offset of each grouped row's first tail
            page in ``tail_pages``.
        shared: [rows] int32, grouped order: shared pages before the tail.
        tail_pages: int32 pool pages of every row's tail.
    """
    tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(2) != 1 or tensor.stride(1) != head_dim:
            raise ValueError(f"{name} must have contiguous heads, got strides {tensor.stride()}.")
    if k.stride(0) != v.stride(0):
        raise ValueError("k and v must share a row stride.")
    if pool.stride(4) != 1:
        raise ValueError(f"pool must be contiguous in head_dim, got strides {pool.stride()}.")
    q_out = torch.empty((tokens, num_heads, head_dim), dtype=q.dtype, device=q.device)
    if tokens == 0:
        return q_out
    _write_kv_gather_q_spec_kernel[(tokens,)](
        q,
        k,
        v,
        q.stride(0),
        k.stride(0),
        pool,
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        kv_lens,
        perm if perm is not None else kv_lens,
        tail_indptr,
        shared,
        tail_pages,
        tail_pages.numel(),
        q_out,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEADS=triton.next_power_of_2(num_heads),
        BLOCK_KV_HEADS=triton.next_power_of_2(num_kv_heads),
        HAS_PERM=perm is not None,
        Q_LEN=q_len,
        TOKENS_PER_BLOCK=tokens_per_block,
    )
    return q_out


@triton.jit
def _step_last_len_kernel(
    kv_lens_ptr,
    perm_ptr,
    base_len_ptr,
    tail_last_page_len_ptr,
    rows,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    # Clamp offsets before indexed loads so masked lanes never speculatively
    # access an unmapped address on SM90 (which lacks hardware-predicated TMA).
    safe = tl.minimum(offsets, rows - 1)
    source = tl.load(perm_ptr + safe).to(tl.int64)  # perm[safe] is always a valid row index
    kv_len = tl.load(kv_lens_ptr + source)  # source < rows, always in-bounds
    base = tl.load(base_len_ptr + safe, mask=mask, other=0)
    tl.store(tail_last_page_len_ptr + safe, (kv_len - base).to(tl.int32), mask=mask)


def step_last_len(
    kv_lens: torch.Tensor,
    perm: torch.Tensor,
    base_len: torch.Tensor,
    tail_last_page_len: torch.Tensor,
) -> None:
    """Derive the tail last-page lengths of a speculative step from device KV lengths.

    ``tail_last_page_len[i] = kv_lens[perm[i]] - base_len[i]`` (int32, grouped
    order), where ``base_len`` is the first position of the row's last host
    page. The value may be zero or negative when the host page count exceeds
    what the device length needs; FlashInfer adds it to ``(pages - 1) *
    page_size`` in unsigned arithmetic, which yields the true length.
    """
    rows = kv_lens.numel()
    if rows == 0:
        return
    block = 256
    _step_last_len_kernel[(triton.cdiv(rows, block),)](
        kv_lens, perm, base_len, tail_last_page_len, rows, BLOCK=block
    )


@triton.jit
def _rope_qkv_kernel(
    qkv_ptr,
    qkv_row_stride,
    cos_sin_ptr,
    kv_lens_ptr,
    NUM_ROPE_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
):
    # Program ``t`` rotates token t = row * Q_LEN + j in place. The query and key
    # heads are adjacent in the packed QKV row, so they form one run of
    # NUM_ROPE_HEADS heads; V is not touched.
    t = tl.program_id(0)
    row = t // Q_LEN
    j = t % Q_LEN
    # Device KV length includes this step's tokens: token j sits at kv_len - Q_LEN + j.
    pos = tl.load(kv_lens_ptr + row).to(tl.int64) - Q_LEN + j
    pos = tl.maximum(pos, 0)
    heads = tl.arange(0, BLOCK_HEADS)
    pairs = tl.arange(0, HALF)
    mask = (heads[:, None] < NUM_ROPE_HEADS) & (pairs[None, :] < HALF)
    cos = tl.load(cos_sin_ptr + pos * (2 * HALF) + 2 * pairs)
    sin = tl.load(cos_sin_ptr + pos * (2 * HALF) + 2 * pairs + 1)
    base = qkv_ptr + t.to(tl.int64) * qkv_row_stride + heads[:, None] * HEAD_DIM + pairs[None, :]
    x1 = tl.load(base, mask=mask, other=0.0)
    x2 = tl.load(base + HALF, mask=mask, other=0.0)
    f1 = x1.to(tl.float32)
    f2 = x2.to(tl.float32)
    tl.store(base, (f1 * cos[None, :] - f2 * sin[None, :]).to(x1.dtype), mask=mask)
    tl.store(base + HALF, (f2 * cos[None, :] + f1 * sin[None, :]).to(x1.dtype), mask=mask)


def rope_qkv_inplace(
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_len: int,
) -> None:
    """Apply NeoX-style rotary embedding to the Q and K heads of packed QKV, in place.

    This is what the fused attention op does before it writes K to the cache, so
    the library can take over from that op for models whose RoPE it would
    otherwise apply. Token ``j`` of row ``r`` (token index ``r * q_len + j``)
    is at position ``kv_lens[r] - q_len + j`` (device lengths, which include the
    step's tokens), so no host value is read and the launch is capturable.

    Args:
        qkv: [rows * q_len, (heads + 2 * kv_heads) * head_dim], row stride
            arbitrary, columns contiguous.
        cos_sin: float32 table of the attention op (``attn.rotary_cos_sin``),
            laid out [position][pair][cos, sin] with ``rotary_dim / 2`` pairs.
        kv_lens: [rows] device KV lengths.
        rotary_dim: rotated leading dimensions of each head (a power of two,
            at most ``head_dim``); the rest of the head is left as is.
    """
    tokens = qkv.shape[0]
    if tokens == 0:
        return
    half = rotary_dim // 2
    if rotary_dim % 2 or half & (half - 1) or rotary_dim > head_dim:
        raise ValueError(f"unsupported rotary dim {rotary_dim} for head dim {head_dim}.")
    if qkv.stride(1) != 1:
        raise ValueError(f"qkv columns must be contiguous, got strides {qkv.stride()}.")
    rope_heads = num_heads + num_kv_heads
    _rope_qkv_kernel[(tokens,)](
        qkv,
        qkv.stride(0),
        cos_sin,
        kv_lens,
        NUM_ROPE_HEADS=rope_heads,
        HEAD_DIM=head_dim,
        HALF=half,
        BLOCK_HEADS=triton.next_power_of_2(rope_heads),
        Q_LEN=q_len,
    )
