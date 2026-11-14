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
"""indexer_topk exact decode top-k selection over the indexer logits arena (Triton).

Experimental and opt-in (``decode_topk(selector="radix")``); the adapter
path uses the native ``TopK`` module. For the 6-row decode shape it is
exact, and the native self-sampling engine stays the default, so it is kept
as a tested reference only.

Same contract as the native decode ``TopK`` (self-sampling GVR) on the
uniform-width GEN path: row ``r`` of ``B * next_n`` rows has
``n_r = kv_len[r // next_n] - next_n + r % next_n + 1`` valid logits; the
output row holds the indices of the ``k`` largest logits in any order
(tie-interchangeable: the selected value multiset equals ``torch.topk``),
rows with ``0 < n_r <= k`` hold ``0 .. n_r - 1`` then ``-1``, and rows with
``n_r <= 0`` hold ``-1``. No host synchronization; graph-capturable.

Algorithm (MSD radix select on order-preserving 32-bit keys):

1. ``_hist_kernel``, grid ``(chunks, rows)``: per-row histogram of the top
   12 key bits, accumulated per CTA and flushed with atomics; the last CTA of
   a row (acq-rel ticket) finds the threshold bin and the count above it,
   publishes both, and clears the histogram for the next call.
2. ``_select_kernel``, same grid: every CTA writes the indices of its keys
   above the threshold bin (slots from an atomic counter) and appends the
   threshold-bin keys to a candidate buffer; the last CTA of a row resolves
   the remaining 20 key bits over the candidates (or, if they overflowed the
   buffer, over the whole row) and writes the rest, then resets the counters.
"""

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

TOP_BITS = 12
NUM_TOP_BINS = 1 << TOP_BITS
DEFAULT_CANDIDATE_CAPACITY = 16384
_NUM_WARPS = 8


@triton.jit
def _keys(x):
    """Order-preserving uint32 keys of fp32 values (larger value, larger key)."""
    i = x.to(tl.int32, bitcast=True)
    # Arithmetic shift: all ones for negative values, else only the sign bit.
    flip = (i >> 31) | -2147483648
    return (i ^ flip).to(tl.uint32, bitcast=True)


@triton.jit
def _row_len(kv_lens_ptr, row, next_n: tl.constexpr):
    kv = tl.load(kv_lens_ptr + row // next_n)
    return tl.maximum(kv - next_n + row % next_n + 1, 0)


@triton.jit
def _fence_all(NUM_THREADS: tl.constexpr):
    # Every thread publishes its prior global writes at GPU scope (one
    # element per thread, so each thread runs the fence once).
    tl.inline_asm_elementwise(
        "fence.acq_rel.gpu;\n\tmov.u32 $0, $1;",
        "=r,r",
        [tl.arange(0, NUM_THREADS)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hist_kernel(
    logits_ptr,
    stride_row,
    kv_lens_ptr,
    hist_ptr,
    thr_ptr,
    ctr_ptr,
    next_n: tl.constexpr,
    K: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    NUM_THREADS: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    n = _row_len(kv_lens_ptr, row, next_n)
    if n > K:
        span = tl.cdiv(tl.cdiv(n, CHUNKS), BLOCK) * BLOCK
        start = chunk * span
        end = tl.minimum(start + span, n)
        offs = tl.arange(0, BLOCK)
        local = tl.zeros([NB], dtype=tl.int32)
        invalid = tl.full([], 0, tl.int32)
        for base in range(start, end, BLOCK):
            idx = base + offs
            valid = idx < end
            x = tl.load(logits_ptr + row * stride_row + idx, mask=valid, other=0.0)
            b = tl.where(valid, (_keys(x) >> (32 - 12)).to(tl.int32), 0)
            local += tl.histogram(b, NB)
            invalid += tl.sum((~valid).to(tl.int32))
        bins = tl.arange(0, NB)
        local = local - tl.where(bins == 0, invalid, 0)
        tl.atomic_add(hist_ptr + row * NB + bins, local, mask=local != 0, sem="relaxed")
        _fence_all(NUM_THREADS)
        tl.debug_barrier()
        ticket = tl.atomic_add(ctr_ptr + row * 4, 1, sem="acq_rel", scope="gpu")
        if ticket == CHUNKS - 1:
            tl.debug_barrier()
            h = tl.load(hist_ptr + row * NB + bins, cache_modifier=".cg")
            total = tl.sum(h)
            prefix = tl.cumsum(h, 0)
            at_or_above = total - prefix + h
            b1 = tl.max(tl.where(at_or_above >= K, bins, -1))
            above = tl.sum(tl.where(bins > b1, h, 0))
            tl.store(thr_ptr + row * 2, b1)
            tl.store(thr_ptr + row * 2 + 1, above)
            tl.store(hist_ptr + row * NB + bins, tl.zeros([NB], dtype=tl.int32))
            tl.store(ctr_ptr + row * 4, 0)


@triton.jit
def _load_source(
    logits_ptr,
    stride_row,
    cand_key_ptr,
    cand_idx_ptr,
    row,
    i,
    inside,
    from_buffer,
    CAP: tl.constexpr,
):
    """Keys and row positions of the candidate buffer or of the full row."""
    if from_buffer:
        key = tl.load(cand_key_ptr + row * CAP + i, mask=inside, other=0, cache_modifier=".cg")
        idx = tl.load(cand_idx_ptr + row * CAP + i, mask=inside, other=0, cache_modifier=".cg")
    else:
        x = tl.load(logits_ptr + row * stride_row + i, mask=inside, other=0.0)
        key = _keys(x)
        idx = i
    return key, idx


@triton.jit
def _digit_pass(
    logits_ptr,
    stride_row,
    cand_key_ptr,
    cand_idx_ptr,
    row,
    src_len,
    from_buffer,
    prefix_key,
    prefix_mask,
    want,
    SHIFT: tl.constexpr,
    NBITS: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One MSD digit: histogram of keys matching the prefix, then the digit
    ``t`` holding the ``want``-th largest key. Returns the extended prefix,
    mask and the number still wanted among keys equal to the new prefix."""
    NUM: tl.constexpr = 1 << NBITS
    offs = tl.arange(0, BLOCK)
    bins = tl.arange(0, NUM)
    h = tl.zeros([NUM], dtype=tl.int32)
    for base in range(0, src_len, BLOCK):
        i = base + offs
        inside = i < src_len
        key, _ = _load_source(
            logits_ptr, stride_row, cand_key_ptr, cand_idx_ptr, row, i, inside, from_buffer, CAP
        )
        active = inside & ((key & prefix_mask) == prefix_key)
        d = tl.where(active, ((key >> SHIFT) & (NUM - 1)).to(tl.int32), 0)
        h += tl.histogram(d, NUM)
        h -= tl.where(bins == 0, tl.sum((~active).to(tl.int32)), 0)
    total = tl.sum(h)
    prefix = tl.cumsum(h, 0)
    at_or_above = total - prefix + h
    t = tl.max(tl.where(at_or_above >= want, bins, -1))
    above = tl.sum(tl.where(bins > t, h, 0))
    new_key = prefix_key | (t.to(tl.uint32) << SHIFT)
    new_mask = prefix_mask | ((NUM - 1) << SHIFT)
    return new_key, new_mask, want - above


@triton.jit
def _final_select(
    logits_ptr,
    stride_row,
    cand_key_ptr,
    cand_idx_ptr,
    out_ptr,
    row,
    n,
    b1,
    count_above,
    num_cand,
    K: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pick ``K - count_above`` of the threshold-bin keys (last CTA of a row)."""
    from_buffer = num_cand <= CAP
    src_len = tl.where(from_buffer, num_cand, n)
    offs = tl.arange(0, BLOCK)
    want = K - count_above
    prefix_key = b1.to(tl.uint32) << 20
    prefix_mask = tl.full([], 0xFFF00000, tl.uint32)
    # Remaining 20 bits in digits of 8, 8 and 4 bits.
    prefix_key, prefix_mask, want = _digit_pass(
        logits_ptr,
        stride_row,
        cand_key_ptr,
        cand_idx_ptr,
        row,
        src_len,
        from_buffer,
        prefix_key,
        prefix_mask,
        want,
        12,
        8,
        CAP,
        BLOCK,
    )
    prefix_key, prefix_mask, want = _digit_pass(
        logits_ptr,
        stride_row,
        cand_key_ptr,
        cand_idx_ptr,
        row,
        src_len,
        from_buffer,
        prefix_key,
        prefix_mask,
        want,
        4,
        8,
        CAP,
        BLOCK,
    )
    prefix_key, prefix_mask, want = _digit_pass(
        logits_ptr,
        stride_row,
        cand_key_ptr,
        cand_idx_ptr,
        row,
        src_len,
        from_buffer,
        prefix_key,
        prefix_mask,
        want,
        0,
        4,
        CAP,
        BLOCK,
    )
    # Emit keys above the exact threshold key, then `want` of the ties.
    threshold = prefix_key
    slot = count_above
    ties_left = want
    for base in range(0, src_len, BLOCK):
        i = base + offs
        inside = i < src_len
        key, idx = _load_source(
            logits_ptr, stride_row, cand_key_ptr, cand_idx_ptr, row, i, inside, from_buffer, CAP
        )
        in_bin = inside & ((key >> 20) == (threshold >> 20))
        gt = in_bin & (key > threshold)
        eq = in_bin & (key == threshold)
        eq_rank = tl.cumsum(eq.to(tl.int32), 0)
        take_eq = eq & (eq_rank <= ties_left)
        sel = gt | take_eq
        pos = slot + tl.cumsum(sel.to(tl.int32), 0) - 1
        tl.store(out_ptr + row * K + pos, idx, mask=sel)
        slot += tl.sum(sel.to(tl.int32))
        ties_left -= tl.sum(take_eq.to(tl.int32))


@triton.jit
def _select_kernel(
    logits_ptr,
    stride_row,
    kv_lens_ptr,
    out_ptr,
    thr_ptr,
    ctr_ptr,
    cand_key_ptr,
    cand_idx_ptr,
    next_n: tl.constexpr,
    K: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    NUM_THREADS: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    n = _row_len(kv_lens_ptr, row, next_n)
    offs = tl.arange(0, BLOCK)
    if n <= K:
        if chunk == 0:
            for base in range(0, K, BLOCK):
                j = base + offs
                tl.store(out_ptr + row * K + j, tl.where(j < n, j, -1), mask=j < K)
    else:
        b1 = tl.load(thr_ptr + row * 2, cache_modifier=".cg")
        count_above = tl.load(thr_ptr + row * 2 + 1, cache_modifier=".cg")
        span = tl.cdiv(tl.cdiv(n, CHUNKS), BLOCK) * BLOCK
        start = chunk * span
        end = tl.minimum(start + span, n)
        for base in range(start, end, BLOCK):
            idx = base + offs
            valid = idx < end
            x = tl.load(logits_ptr + row * stride_row + idx, mask=valid, other=0.0)
            key = _keys(x)
            b = (key >> 20).to(tl.int32)
            gt = valid & (b > b1)
            eq = valid & (b == b1)
            n_gt = tl.sum(gt.to(tl.int32))
            n_eq = tl.sum(eq.to(tl.int32))
            if n_gt > 0:
                slot = tl.atomic_add(ctr_ptr + row * 4 + 1, n_gt, sem="relaxed")
                pos = slot + tl.cumsum(gt.to(tl.int32), 0) - 1
                tl.store(out_ptr + row * K + pos, idx, mask=gt)
            if n_eq > 0:
                cslot = tl.atomic_add(ctr_ptr + row * 4 + 2, n_eq, sem="relaxed")
                cpos = cslot + tl.cumsum(eq.to(tl.int32), 0) - 1
                keep = eq & (cpos < CAP)
                tl.store(cand_key_ptr + row * CAP + cpos, key, mask=keep)
                tl.store(cand_idx_ptr + row * CAP + cpos, idx, mask=keep)
        _fence_all(NUM_THREADS)
        tl.debug_barrier()
        ticket = tl.atomic_add(ctr_ptr + row * 4 + 3, 1, sem="acq_rel", scope="gpu")
        if ticket == CHUNKS - 1:
            tl.debug_barrier()
            num_cand = tl.load(ctr_ptr + row * 4 + 2, cache_modifier=".cg")
            _final_select(
                logits_ptr,
                stride_row,
                cand_key_ptr,
                cand_idx_ptr,
                out_ptr,
                row,
                n,
                b1,
                count_above,
                num_cand,
                K,
                CAP,
                BLOCK,
            )
            tl.store(ctr_ptr + row * 4 + 1, 0)
            tl.store(ctr_ptr + row * 4 + 2, 0)
            tl.store(ctr_ptr + row * 4 + 3, 0)


class RadixTopKWorkspace:
    """Graph-stable scratch of the radix select for up to ``max_rows`` rows.

    Counters and histograms are zero-initialized once and left zeroed by
    every call, so the scratch can be reused across calls and CUDA-graph
    replays on one stream. Concurrent calls on different streams need
    separate workspaces.
    """

    def __init__(self, max_rows: int, device, capacity: int = DEFAULT_CANDIDATE_CAPACITY):
        self.max_rows = max_rows
        self.capacity = capacity
        self.hist = torch.zeros(max_rows, NUM_TOP_BINS, dtype=torch.int32, device=device)
        self.thr = torch.zeros(max_rows, 2, dtype=torch.int32, device=device)
        self.ctr = torch.zeros(max_rows, 4, dtype=torch.int32, device=device)
        self.cand_key = torch.empty(max_rows, capacity, dtype=torch.uint32, device=device)
        self.cand_idx = torch.empty(max_rows, capacity, dtype=torch.int32, device=device)


_WORKSPACES: Dict[Tuple[int, int], RadixTopKWorkspace] = {}


def get_workspace(rows: int, device: torch.device) -> RadixTopKWorkspace:
    """Per-device workspace sized for at least ``rows`` rows (grown on demand)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WORKSPACES.get(index)
    if ws is None or ws.max_rows < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("radix top-k workspace must be created before graph capture")
        ws = RadixTopKWorkspace(max(rows, 8), device)
        _WORKSPACES[index] = ws
    return ws


def default_chunks(rows: int, num_sms: int) -> int:
    """CTAs per row: about two waves of the grid over the SMs."""
    return max(1, min(64, (2 * num_sms) // max(rows, 1)))


def radix_topk_decode(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    next_n: int,
    chunks: int,
    block: int = 4096,
    workspace: RadixTopKWorkspace = None,
) -> torch.Tensor:
    """Exact decode top-k over a row-major fp32 logits view.

    Args:
        logits: ``[rows, >= max n_r]`` float32 view with unit inner stride.
        kv_lens: ``[rows // next_n]`` int32 logical KV lengths.
        out: ``[rows, k]`` int32 contiguous output.
        next_n: Verify rows per request.
        chunks: CTAs per row (fixed per captured graph).
        block: Elements per CTA iteration.
        workspace: Scratch; defaults to the per-device workspace.

    Returns:
        ``out``.
    """
    rows, k = out.shape
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert out.dtype == torch.int32 and out.is_contiguous() and kv_lens.dtype == torch.int32
    assert rows % next_n == 0 and kv_lens.numel() == rows // next_n
    ws = workspace or get_workspace(rows, logits.device)
    grid = (chunks, rows)
    _hist_kernel[grid](
        logits,
        logits.stride(0),
        kv_lens,
        ws.hist,
        ws.thr,
        ws.ctr,
        next_n=next_n,
        K=k,
        CHUNKS=chunks,
        BLOCK=block,
        NB=NUM_TOP_BINS,
        NUM_THREADS=32 * _NUM_WARPS,
        num_warps=_NUM_WARPS,
    )
    _select_kernel[grid](
        logits,
        logits.stride(0),
        kv_lens,
        out,
        ws.thr,
        ws.ctr,
        ws.cand_key,
        ws.cand_idx,
        next_n=next_n,
        K=k,
        CHUNKS=chunks,
        BLOCK=block,
        CAP=ws.capacity,
        NUM_THREADS=32 * _NUM_WARPS,
        num_warps=_NUM_WARPS,
    )
    return out
