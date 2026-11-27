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
"""Triton kernels of the vocab-parallel token selection and their launchers.

The selection reduces each rank's vocab shard of the logits to ``(value, global index)`` pairs
(:func:`local_select`, :func:`markov_add_local_argmax`), exchanges them, and reduces the exchanged
pairs to tokens (:func:`reduce_pairs`, :func:`markov_reduce_embed`); :func:`markov_chain` runs a
whole DSpark Markov chain in one launch. ``dflash.K3VocabParallelHead`` owns the exchange
buffers and calls these through the ``ops/spec`` entries.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

_INT32_MAX = tl.constexpr(2**31 - 1)
# Index reported by a split without valid columns; exact in float32 and above
# every supported vocab id.
_EMPTY_INDEX = tl.constexpr(2**24)
_LOCAL_BLOCK_V = 2048
# Shard columns per program of the one-launch Markov chain.
_MARKOV_BLOCK_V = 128
# Capacity of the one-launch Markov chain's exchange sets.
_CHAIN_MAX_STEPS = 16
_CHAIN_MAX_REQS = 16
_REDUCE_BLOCK_ROWS = 32
# Programs the local reductions aim to launch (rows x column splits).
_TARGET_PROGRAMS = 256


@triton.jit
def _argmax_combine(val_a, idx_a, val_b, idx_b):
    # torch.argmax order: NaN beats any number, equal values go to the lower index.
    nan_a = val_a != val_a
    nan_b = val_b != val_b
    take_a = tl.where(
        nan_a,
        (~nan_b) | (idx_a < idx_b),
        (~nan_b) & ((val_a > val_b) | ((val_a == val_b) & (idx_a < idx_b))),
    )
    return tl.where(take_a, val_a, val_b), tl.where(take_a, idx_a, idx_b)


@triton.jit
def _gumbel(r):
    # Standard Gumbel sample -log(-log(u)) from 32 random bits. 23 bits map to
    # the open interval (0, 1); with 23 bits the midpoint (k + 0.5) / 2^23 is
    # exact in float32, so u never rounds to 0 or 1.
    u = ((r >> 9).to(tl.float32) + 0.5) * (1.0 / 8388608.0)
    return -tl.log(-tl.log(u))


@triton.jit
def _store_pair(
    out_ptr,
    row,
    stride_out,
    slot,
    best_val,
    gidx,
    zero_lo,
    zero_hi,
    fill,
    ZERO_WIDTH: tl.constexpr,
):
    # Publishes (value, global index) at columns [2 * slot, 2 * slot + 1] of the
    # row. With ZERO_WIDTH > 0 the program flagged `fill` also zeroes the row
    # outside [zero_lo, zero_hi) -- the other ranks' columns -- so that an
    # all-reduce sum acts as an all-gather.
    base = out_ptr + row.to(tl.int64) * stride_out
    tl.store(base + 2 * slot, best_val)
    tl.store(base + 2 * slot + 1, gidx.to(tl.float32))
    if ZERO_WIDTH > 0:
        if fill:
            offs = tl.arange(0, ZERO_WIDTH)
            tl.store(
                base + offs,
                tl.zeros([ZERO_WIDTH], tl.float32),
                mask=(offs < zero_lo) | (offs >= zero_hi),
            )


@triton.jit
def _global_index(best_idx, col_base):
    # A split without valid columns reports an index above any vocab id, so it
    # never wins a tie.
    return tl.where(best_idx == _INT32_MAX, _EMPTY_INDEX, col_base + best_idx)


@triton.jit
def _local_argmax_kernel(
    x_ptr,
    stride_x,
    n_cols,
    col_base,
    chunk,
    out_ptr,
    stride_out,
    slot0,
    zero_lo,
    zero_hi,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Program (row, split) reduces columns [split * chunk, (split + 1) * chunk).
    row = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, n_cols)
    offs = tl.arange(0, BLOCK_V)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(lo, hi, BLOCK_V):
        cols = start + offs
        mask = cols < hi
        vals = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = _argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, _argmax_combine)
    _store_pair(
        out_ptr,
        row,
        stride_out,
        slot0 + split,
        val,
        _global_index(idx, col_base),
        zero_lo,
        zero_hi,
        split == 0,
        ZERO_WIDTH,
    )


@triton.jit
def _local_sample_kernel(
    x_ptr,
    stride_x,
    n_cols,
    col_base,
    chunk,
    temp_ptr,
    seed_ptr,
    offset_ptr,
    greedy_threshold,
    out_ptr,
    stride_out,
    slot0,
    zero_lo,
    zero_hi,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Gumbel-max over one split of this rank's shard. The noise of global vocab
    # index v is output (v % 4) of Philox(seed, counter=(v // 4, row, offset
    # lo, offset hi)): disjoint streams per row and per step, identical on
    # every rank. col_base and chunk are multiples of 4, so 4-column groups
    # never straddle splits or shards.
    GROUPS: tl.constexpr = BLOCK_V // 4
    row = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, n_cols)
    offs_g = tl.arange(0, GROUPS)
    offs_l = tl.arange(0, 4)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    temperature = tl.load(temp_ptr + row).to(tl.float32)
    is_greedy = temperature <= greedy_threshold
    inv_t = 1.0 / tl.where(is_greedy, 1.0, temperature)
    seed = tl.load(seed_ptr + row).to(tl.int64)
    offset = tl.load(offset_ptr + row).to(tl.int64)
    c1 = tl.full([GROUPS], 0, tl.uint32) + row.to(tl.uint32)
    c2 = tl.full([GROUPS], 0, tl.uint32) + (offset & 0xFFFFFFFF).to(tl.uint32)
    c3 = tl.full([GROUPS], 0, tl.uint32) + ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    best_val = tl.full([GROUPS, 4], float("-inf"), tl.float32)
    best_idx = tl.full([GROUPS, 4], _INT32_MAX, tl.int32)
    for start in range(lo, hi, BLOCK_V):
        cols = start + offs_g[:, None] * 4 + offs_l[None, :]
        mask = cols < hi
        vals = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if not is_greedy:
            counter = ((col_base + start) // 4 + offs_g).to(tl.uint32)
            r0, r1, r2, r3 = tl.philox(seed, counter, c1, c2, c3)
            # [GROUPS, 2, 2] -> [GROUPS, 4] in lane order r0, r1, r2, r3.
            noise = tl.join(
                tl.join(_gumbel(r0), _gumbel(r2)), tl.join(_gumbel(r1), _gumbel(r3))
            ).reshape([GROUPS, 4])
            vals = vals * inv_t + noise
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = _argmax_combine(best_val, best_idx, vals, idx)
    best_val, best_idx = tl.reduce((best_val, best_idx), 1, _argmax_combine)
    val, idx = tl.reduce((best_val, best_idx), 0, _argmax_combine)
    _store_pair(
        out_ptr,
        row,
        stride_out,
        slot0 + split,
        val,
        _global_index(idx, col_base),
        zero_lo,
        zero_hi,
        split == 0,
        ZERO_WIDTH,
    )


@triton.jit
def _reduce_pairs_kernel(
    pairs_ptr,
    stride_row,
    stride_slot,
    n_rows,
    n_slots,
    out_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    slots = tl.arange(0, BLOCK_SLOTS)
    mask = (rows[:, None] < n_rows) & (slots[None, :] < n_slots)
    ptrs = pairs_ptr + rows[:, None].to(tl.int64) * stride_row + slots[None, :] * stride_slot
    vals = tl.load(ptrs, mask=mask, other=float("-inf"))
    gidx = tl.load(ptrs + 1, mask=mask, other=0.0).to(tl.int32)
    gidx = tl.where(mask, gidx, _INT32_MAX)
    _, best = tl.reduce((vals, gidx), 1, _argmax_combine)
    tl.store(out_ptr + rows, best.to(out_ptr.dtype.element_ty), mask=rows < n_rows)


@triton.jit
def _markov_reduce_embed_kernel(
    pairs_ptr,
    stride_row,
    stride_slot,
    n_slots,
    anchor_ptr,
    w1_ptr,
    emb_ptr,
    tokens_ptr,
    stride_tokens,
    vocab,
    step,
    RANK: tl.constexpr,
    FROM_ANCHOR: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Resolves prev_step for one request (the anchor at step 0, else the global
    # argmax of the previous step, which is also draft token step - 1) and
    # gathers its markov_w1 row; an out-of-vocab prev gives a zero row.
    req = tl.program_id(0)
    if FROM_ANCHOR:
        prev = tl.load(anchor_ptr + req).to(tl.int64)
    else:
        slots = tl.arange(0, BLOCK_SLOTS)
        smask = slots < n_slots
        ptrs = pairs_ptr + req.to(tl.int64) * stride_row + slots * stride_slot
        vals = tl.load(ptrs, mask=smask, other=float("-inf"))
        gidx = tl.load(ptrs + 1, mask=smask, other=0.0).to(tl.int32)
        gidx = tl.where(smask, gidx, _INT32_MAX)
        _, best = tl.reduce((vals, gidx), 0, _argmax_combine)
        prev = best.to(tl.int64)
        tl.store(tokens_ptr + req.to(tl.int64) * stride_tokens + step - 1, prev.to(tl.int32))
    valid = (prev >= 0) & (prev < vocab)
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < RANK
    w1_row = tl.load(
        w1_ptr + tl.where(valid, prev, 0) * RANK + offs_r, mask=mask_r & valid, other=0.0
    )
    tl.store(emb_ptr + req * RANK + offs_r, w1_row, mask=mask_r)


@triton.jit
def _markov_add_local_argmax_kernel(
    base_ptr,
    stride_base,
    bias_ptr,
    stride_bias,
    n_cols,
    col_base,
    chunk,
    out_ptr,
    stride_out,
    slot0,
    zero_lo,
    zero_hi,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # base (shard logits of block position `step`) + bias, upcast to fp32 like
    # the gathered reference, then the argmax of one split of the shard.
    req = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, n_cols)
    offs = tl.arange(0, BLOCK_V)
    base_row = base_ptr + req.to(tl.int64) * stride_base
    bias_row = bias_ptr + req.to(tl.int64) * stride_bias
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(lo, hi, BLOCK_V):
        cols = start + offs
        mask = cols < hi
        base = tl.load(base_row + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
        vals = tl.where(mask, base + bias, float("-inf"))
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = _argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, _argmax_combine)
    _store_pair(
        out_ptr,
        req,
        stride_out,
        slot0 + split,
        val,
        _global_index(idx, col_base),
        zero_lo,
        zero_hi,
        split == 0,
        ZERO_WIDTH,
    )


@triton.jit
def _gdc_wait():
    # griddepcontrol.wait with a memory clobber, so that no load is moved above
    # it (tl.extra.cuda.gdc_wait has none).
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _argmax_pairs(vals, idx, axis: tl.constexpr):
    # The (value, index) that a _argmax_combine reduction along ``axis`` returns,
    # from native reductions: NaN beats every number and the lowest index wins
    # among NaNs; otherwise the largest value, lowest index among equal values
    # (+0.0 == -0.0). The selected index is unique, so it does not depend on the
    # reduction order; the value is the selected entry's (NaN, or the maximum).
    is_nan = vals != vals
    any_nan = tl.max(is_nan.to(tl.int32), axis=axis) > 0
    vmax = tl.max(tl.where(is_nan, float("-inf"), vals), axis=axis)
    cand = tl.where(
        tl.expand_dims(any_nan, axis), is_nan, (vals == tl.expand_dims(vmax, axis)) & ~is_nan
    )
    best = tl.min(tl.where(cand, idx, _INT32_MAX), axis=axis)
    return tl.where(any_nan, float("nan"), vmax), best


_NEG_ZERO_BITS = tl.constexpr(-(2**31))


@triton.jit
def _wait_pairs_written(xptr, xm):
    # Spin until no pair of the tile holds the -0.0 sentinel. Only a scalar is
    # loop-carried: Triton 3.8 fails to compile (TritonGPURemoveLayoutConversions)
    # a while loop that carries the [16, TP] tiles once TP >= 8.
    pending = tl.full((), 1, tl.int32)
    while pending > 0:
        xv = tl.load(xptr, mask=xm, other=0.0, volatile=True)
        xi = tl.load(xptr + 1, mask=xm, other=0.0, volatile=True)
        dirty = xm & (
            (xv.to(tl.int32, bitcast=True) == _NEG_ZERO_BITS)
            | (xi.to(tl.int32, bitcast=True) == _NEG_ZERO_BITS)
        )
        pending = tl.max(tl.max(dirty.to(tl.int32), axis=1), axis=0)


@triton.jit
def _markov_chain_kernel(
    anchor_ptr,
    w1_ptr,
    w2_ptr,
    stride_w2,
    base_ptr,
    stride_base_req,
    stride_base_step,
    tokens_ptr,
    stride_tokens,
    partial_ptr,
    ticket_ptr,
    launch_ptr,
    mc_ptr,
    uc_ptr,
    num_reqs,
    vocab,
    n_cols,
    col_base,
    tp_rank,
    STEPS: tl.constexpr,
    RANK: tl.constexpr,
    TP: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_P: tl.constexpr,
    SET_FLOATS: tl.constexpr,
):
    # The whole DSpark Markov chain of one draft block in one launch. Every
    # step: resolve prev, gather its markov_w1 row, compute this program's
    # markov_w2 bias columns, add the step's shard logits and reduce to a
    # (value, global index) candidate; the last program of the rank to finish
    # reduces the rank's candidates and multicasts the rank's pair per request
    # into every rank's exchange buffer; every program then polls its local copy
    # until all ranks' pairs of the step are present (Lamport: the buffer holds
    # -0.0 where nothing was written yet; published values never are -0.0).
    #
    # Exchange sets: launch L uses set L % 3 (STEPS regions of [request, rank,
    # 2]) and clears set (L - 1) % 3, which every program of this rank finished
    # reading in launch L - 1; another rank writes that set again only in launch
    # L + 2, which needs this rank's pairs of launch L + 1, published after this
    # launch completed -- so the clear needs no fence. All programs read L
    # before their first ticket; program 0 advances it after the last step.
    # Every program must be resident at once (the caller launches at most one
    # per SM), since each one waits for all programs' pairs every step.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    reqs = tl.arange(0, BLOCK_B)
    rmask = reqs < num_reqs
    cols = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    cmask = cols < n_cols
    r = tl.arange(0, RANK)
    ranks = tl.arange(0, TP)
    launch = tl.load(launch_ptr, volatile=True)
    cur_set = launch % 3
    if pid == 0:
        # The -0.0 sentinel is written as its bit pattern: a -0.0 float constant
        # is not guaranteed to keep its sign through compilation.
        z = tl.arange(0, SET_FLOATS)
        tl.store(
            uc_ptr.to(tl.pointer_type(tl.int32)) + ((launch + 2) % 3) * SET_FLOATS + z,
            tl.full([SET_FLOATS], _NEG_ZERO_BITS, tl.int32),
        )
    # The step logits come from the previous grid; nothing is read before this.
    _gdc_wait()
    prev = tl.load(anchor_ptr + reqs, mask=rmask, other=-1).to(tl.int64)
    xm = rmask[:, None] & (ranks[None, :] < TP)
    for step in tl.static_range(STEPS + 1):
        if step < STEPS:
            # Issued before the wait for the previous step's pairs: neither
            # depends on it.
            w2 = tl.load(
                w2_ptr + cols[:, None].to(tl.int64) * stride_w2 + r[None, :], mask=cmask[:, None]
            )
            base = tl.load(
                base_ptr
                + reqs[:, None].to(tl.int64) * stride_base_req
                + step * stride_base_step
                + cols[None, :],
                mask=rmask[:, None] & cmask[None, :],
                other=0.0,
            )
        if step > 0:
            # Wait for every rank's pair of the previous step, then take the winner.
            region = cur_set * SET_FLOATS + (step - 1) * (BLOCK_B * TP * 2)
            xptr = uc_ptr + region + (reqs[:, None] * TP + ranks[None, :]) * 2
            _wait_pairs_written(xptr, xm)
            # Read again once all are written: a slot is written once per launch
            # and cleared only in a later one, so this read sees the polled values.
            xv = tl.load(xptr, mask=xm, other=0.0, volatile=True)
            xi = tl.load(xptr + 1, mask=xm, other=0.0, volatile=True)
            _, best = _argmax_pairs(
                tl.where(xm, xv, float("-inf")), tl.where(xm, xi.to(tl.int32), _INT32_MAX), 1
            )
            prev = best.to(tl.int64)
            if pid == 0:
                tl.store(
                    tokens_ptr + reqs.to(tl.int64) * stride_tokens + step - 1,
                    prev.to(tl.int32),
                    mask=rmask,
                )
        if step < STEPS:
            valid = rmask & (prev >= 0) & (prev < vocab)
            w1 = tl.load(
                w1_ptr + tl.where(valid, prev, 0)[:, None] * RANK + r[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            # F.linear(markov_w1[prev], w2) rounds the FP32 dot to the weights' dtype.
            bias = tl.dot(w1, tl.trans(w2)).to(w2.dtype).to(tl.float32)
            vals = tl.where(
                rmask[:, None] & cmask[None, :], base.to(tl.float32) + bias, float("-inf")
            )
            idx = tl.where(cmask[None, :], cols[None, :], _INT32_MAX) + tl.zeros(
                [BLOCK_B, BLOCK_V], tl.int32
            )
            cand_val, cand_idx = _argmax_pairs(vals, idx, 1)
            cand_idx = _global_index(cand_idx, col_base)
            pbase = partial_ptr + ((step * nprog + pid) * BLOCK_B + reqs) * 2
            tl.store(pbase, cand_val)
            tl.store(pbase + 1, cand_idx.to(tl.float32))
            tl.debug_barrier()
            ticket = tl.atomic_add(ticket_ptr + step, 1, sem="acq_rel", scope="gpu")
            if ticket == nprog - 1:
                p = tl.arange(0, BLOCK_P)
                pm = (p[:, None] < nprog) & rmask[None, :]
                qptrs = partial_ptr + ((step * nprog + p[:, None]) * BLOCK_B + reqs[None, :]) * 2
                qv = tl.load(qptrs, mask=pm, other=float("-inf"), volatile=True)
                qi = tl.where(
                    pm,
                    tl.load(qptrs + 1, mask=pm, other=0.0, volatile=True).to(tl.int32),
                    _INT32_MAX,
                )
                win_val, win_idx = _argmax_pairs(qv, qi, 0)
                # -0.0 marks an empty slot; publish +0.0 instead (argmax-equal).
                win_val = tl.where(
                    win_val.to(tl.int32, bitcast=True) == _NEG_ZERO_BITS, 0.0, win_val
                )
                tl.atomic_xchg(ticket_ptr + step, 0, sem="relaxed", scope="gpu")
                out = (
                    mc_ptr
                    + cur_set * SET_FLOATS
                    + step * (BLOCK_B * TP * 2)
                    + (reqs * TP + tp_rank) * 2
                )
                tl.store(out, win_val, mask=rmask)
                tl.store(out + 1, win_idx.to(tl.float32), mask=rmask)
    if pid == 0:
        tl.store(launch_ptr, launch + 1)


class _PairLayout(NamedTuple):
    """Where the local kernels publish pairs, for one exchange."""

    buffer: torch.Tensor
    stride_out: int
    slot0: int  # first slot owned by this rank
    zero_lo: int  # columns [zero_lo, zero_hi) are this rank's; the rest is zero-filled
    zero_hi: int
    zero_width: int  # 0: no zero fill (NCCL all-gather)
    splits: int  # column splits per rank (slots per rank)
    chunk: int  # columns per split


def local_select(
    local: torch.Tensor,
    col_base: int,
    layout: _PairLayout,
    sampling: Optional[tuple] = None,
) -> None:
    """Publish each split's ``(value, global index)`` pair of the shard logits ``local``
    ``[rows, shard]`` into ``layout``: the argmax, or with ``sampling = (temperatures, seeds,
    offsets, greedy_threshold)`` the Gumbel-max draw of temperature-only sampling."""
    rows, n_cols = local.shape
    if rows == 0:
        return
    kernel = _local_argmax_kernel if sampling is None else _local_sample_kernel
    extra = () if sampling is None else tuple(sampling)
    kernel[(rows, layout.splits)](
        local,
        local.stride(0),
        n_cols,
        col_base,
        layout.chunk,
        *extra,
        layout.buffer,
        layout.stride_out,
        layout.slot0,
        layout.zero_lo,
        layout.zero_hi,
        ZERO_WIDTH=layout.zero_width,
        BLOCK_V=_LOCAL_BLOCK_V,
        num_warps=8,
    )


def reduce_pairs(
    pairs: torch.Tensor, stride_row: int, stride_slot: int, n_slots: int, out: torch.Tensor
) -> torch.Tensor:
    """``out[r]`` = the global index of the winning exchanged pair of row ``r``."""
    rows = out.shape[0]
    if rows > 0:
        _reduce_pairs_kernel[(triton.cdiv(rows, _REDUCE_BLOCK_ROWS),)](
            pairs,
            stride_row,
            stride_slot,
            rows,
            n_slots,
            out,
            BLOCK_ROWS=_REDUCE_BLOCK_ROWS,
            BLOCK_SLOTS=triton.next_power_of_2(n_slots),
            num_warps=4,
        )
    return out


def markov_reduce_embed(
    pairs: torch.Tensor,
    stride_row: int,
    stride_slot: int,
    n_slots: int,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    emb: torch.Tensor,
    tokens: torch.Tensor,
    vocab: int,
    step: int,
) -> None:
    """Resolve ``prev_step`` per request (the anchor at step 0, else the winner of the exchanged
    pairs, also stored as draft token ``step - 1``) and gather its ``markov_w1`` row into
    ``emb``."""
    num_reqs = emb.shape[0]
    rank = markov_w1.shape[1]
    _markov_reduce_embed_kernel[(num_reqs,)](
        pairs,
        stride_row,
        stride_slot,
        n_slots,
        anchors,
        markov_w1,
        emb,
        tokens,
        tokens.stride(0),
        vocab,
        step,
        RANK=rank,
        FROM_ANCHOR=step == 0,
        BLOCK_SLOTS=triton.next_power_of_2(n_slots),
        BLOCK_R=triton.next_power_of_2(rank),
        num_warps=4,
    )


def markov_add_local_argmax(
    step_logits: torch.Tensor, bias: torch.Tensor, col_base: int, layout: _PairLayout
) -> None:
    """Publish each split's argmax pair of ``step_logits + bias`` (``[requests, shard]``, added
    in FP32) into ``layout``."""
    num_reqs, shard = step_logits.shape
    _markov_add_local_argmax_kernel[(num_reqs, layout.splits)](
        step_logits,
        step_logits.stride(0),
        bias,
        bias.stride(0),
        shard,
        col_base,
        layout.chunk,
        layout.buffer,
        layout.stride_out,
        layout.slot0,
        layout.zero_lo,
        layout.zero_hi,
        ZERO_WIDTH=layout.zero_width,
        BLOCK_V=_LOCAL_BLOCK_V,
        num_warps=8,
    )


def markov_chain(
    local_block: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    w2_shard: torch.Tensor,
    tokens: torch.Tensor,
    partial: torch.Tensor,
    tickets: torch.Tensor,
    launch: torch.Tensor,
    mcast: torch.Tensor,
    local: torch.Tensor,
    set_floats: int,
    vocab: int,
    col_base: int,
    tp: int,
    tp_rank: int,
) -> torch.Tensor:
    """The DSpark greedy Markov chain of one draft block in one launch (``_markov_chain_kernel``):
    ``tokens`` ``[B, K]`` int32 from the shard block logits ``[B, K, shard]``."""
    num_reqs, block_size, shard = local_block.shape
    programs = triton.cdiv(shard, _MARKOV_BLOCK_V)
    _markov_chain_kernel[(programs,)](
        first_prev_tokens,
        markov_w1,
        w2_shard,
        w2_shard.stride(0),
        local_block,
        local_block.stride(0),
        local_block.stride(1),
        tokens,
        tokens.stride(0),
        partial,
        tickets,
        launch,
        mcast,
        local,
        num_reqs,
        vocab,
        shard,
        col_base,
        tp_rank,
        STEPS=block_size,
        RANK=markov_w1.shape[1],
        TP=tp,
        BLOCK_B=_CHAIN_MAX_REQS,
        BLOCK_V=_MARKOV_BLOCK_V,
        BLOCK_P=triton.next_power_of_2(programs),
        SET_FLOATS=set_floats,
        num_warps=4,
        launch_pdl=True,
    )
    return tokens
