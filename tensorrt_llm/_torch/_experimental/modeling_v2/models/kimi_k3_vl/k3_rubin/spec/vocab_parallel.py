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
"""Vocab-parallel token selection for a vocab-sharded LM head.

A column-parallel LM head leaves every TP rank with a contiguous vocab shard of
the logits. Gathering the full logits to pick one token per row moves
``rows * vocab`` values; this module instead reduces each rank's shard to one
``(value, global index)`` pair per row, exchanges only those pairs, and takes
the winner. The result equals ``torch.argmax`` over the gathered logits bit for
bit: ``NaN`` beats every number, and equal maxima resolve to the lowest global
index (rank shards are contiguous and ordered, so the lowest rank's lowest
column wins).

Three consumers:

* :func:`vocab_parallel_argmax` -- greedy target sampling.
* :func:`vocab_parallel_sample` -- temperature-only sampling through the
  Gumbel-max trick: ``argmax(logits / T + G)`` with ``G`` standard Gumbel noise
  is an exact draw from ``softmax(logits / T)``. The noise of every vocab entry
  comes from a counter-based Philox stream keyed by the row's seed, so all
  ranks agree on the winner. The distribution matches the gathered sampler; the
  individual draws do not (different random stream).
* :func:`vocab_parallel_markov_tokens` -- the DSpark vanilla-Markov greedy
  block chain on shard logits: step ``i`` adds ``markov_w2 @ markov_w1[prev_i]``
  to block position ``i`` and takes the global argmax, which becomes
  ``prev_{i+1}`` and draft token ``i``.

The pair exchange is an all-reduce (sum) of a zero-padded ``[rows, 2 * tp]``
float32 buffer in which each rank fills only its own two columns (``x + 0`` is
exact), over the MNNVL all-reduce when available and NCCL otherwise. Every
launch has a shape fixed by ``(rows, shard width)`` and nothing synchronizes
with the host, so all paths are CUDA-graph capturable.

Switches (read once per process):

* ``TLLM_SPEC_VOCAB_PARALLEL=1`` enables the greedy target argmax and the
  draft Markov chain.
* ``TLLM_SPEC_VOCAB_PARALLEL_SAMPLING=1`` additionally samples non-greedy
  target rows vocab-parallel, only when the deployment's
  ``advanced_sampling_mode`` is ``no_topk_no_topp``.
* ``TLLM_SPEC_VOCAB_PARALLEL_COMM=auto|mnnvl|nccl`` picks the pair exchange.
* ``TLLM_DSPARK_MARKOV_V3=1`` runs the draft Markov chain of up to 16 requests
  and 16 block positions as the one-launch CuTe DSL kernel
  (:meth:`VocabParallelHead.markov_tokens_v3`) on SM100-family devices.
"""

import os
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..features import active as _features

# The pair exchange: auto (MNNVL when available, else NCCL), mnnvl or nccl.
_VOCAB_PARALLEL_COMM = "auto"

_INT32_MAX = tl.constexpr(2**31 - 1)
# Index reported by a split without valid columns; exact in float32 and above
# every supported vocab id.
_EMPTY_INDEX = tl.constexpr(2**24)
_LOCAL_BLOCK_V = 2048
# Shard columns per program of the fused Markov step.
_MARKOV_BLOCK_V = 128
# Floats per request row of the fused Markov step's exchange (multi-rank). The
# one-shot all-reduce runs width / 4 threads per row, and those threads also
# clear the Lamport buffer that the workspace's previous all-reduce wrote --
# before step 0, the drafter's last all-reduce (~450 KB at 14 tokens), which
# 8 threads per row (a 32-float row) take ~40 us to clear on GB300. 1024
# floats run 4 CTAs of 64 threads per row.
_MARKOV_EXCHANGE_WIDTH = 1024
# Capacity of the one-launch Markov chain's exchange sets.
_CHAIN_MAX_STEPS = 16
_CHAIN_MAX_REQS = 16
_REDUCE_BLOCK_ROWS = 32
# Programs the local reductions aim to launch (rows x column splits).
_TARGET_PROGRAMS = 256


def vocab_parallel_enabled() -> bool:
    """Whether ``TLLM_SPEC_VOCAB_PARALLEL=1`` selects the vocab-parallel paths."""
    return _features().spec_vocab_parallel


def markov_v2_enabled() -> bool:
    """Whether ``TLLM_DSPARK_MARKOV_V2=1`` runs each DSpark Markov step as one
    fused kernel (see :meth:`VocabParallelHead.markov_tokens_fused`), or ``=2``
    the whole chain as one kernel (:meth:`VocabParallelHead.markov_tokens_chain`)."""
    return vocab_parallel_enabled() and _features().dspark_markov_v2 in ("1", "2")


def markov_chain_kernel_enabled() -> bool:
    """Whether ``TLLM_DSPARK_MARKOV_V2=2`` selects the one-launch Markov chain."""
    return vocab_parallel_enabled() and _features().dspark_markov_v2 == "2"


def markov_v3_enabled() -> bool:
    """Whether ``TLLM_DSPARK_MARKOV_V3=1`` selects the one-launch CuTe DSL Markov
    chain (:meth:`VocabParallelHead.markov_tokens_v3`) where it applies."""
    return vocab_parallel_enabled() and _features().dspark_markov_v3


def vocab_parallel_sampling_enabled() -> bool:
    """Whether non-greedy target rows may be sampled vocab-parallel."""
    return vocab_parallel_enabled() and _features().spec_vocab_parallel_sampling


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
def _gdc_launch_dependents():
    tl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // dummy $0",
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


@triton.jit
def _markov_round_kernel(
    pairs_ptr,
    stride_pairs,
    n_slots,
    anchor_ptr,
    w1_ptr,
    w2_ptr,
    stride_w2,
    base_ptr,
    stride_base,
    tokens_ptr,
    stride_tokens,
    step,
    partial_ptr,
    ticket_ptr,
    out_ptr,
    stride_out,
    slot,
    num_reqs,
    vocab,
    n_cols,
    col_base,
    RANK: tl.constexpr,
    FROM_ANCHOR: tl.constexpr,
    ZERO_WIDTH: tl.constexpr,
    ZERO_CHUNK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # One Markov step for every request over this program's BLOCK_V shard
    # columns: resolve prev (the anchor at step 0, else the global argmax of the
    # previous step's exchanged pairs), gather its markov_w1 row, compute the
    # columns' markov_w2 bias, add the step's shard logits and reduce to a
    # (value, global index) candidate. The last program to finish reduces every
    # program's candidates to this rank's pair per request and publishes it for
    # the all-reduce exchange; every program zeroes ZERO_CHUNK columns of the
    # rest of the row.
    #
    # PDL: the program triggers its dependents before its own wait. The
    # dependent (the exchange all-reduce, or at a single rank the next step's
    # kernel) reads this grid's output only after its own griddepcontrol.wait,
    # which returns once this grid has completed, so the early trigger only
    # lets it launch and read its own weights sooner.
    pid = tl.program_id(0)
    reqs = tl.arange(0, BLOCK_B)
    rmask = reqs < num_reqs
    cols = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    cmask = cols < n_cols
    r = tl.arange(0, RANK)
    # The weight block is older than any predecessor in the chain: read it
    # before waiting on the previous grid.
    w2 = tl.load(w2_ptr + cols[:, None].to(tl.int64) * stride_w2 + r[None, :], mask=cmask[:, None])
    if not FROM_ANCHOR:
        # Step > 0: this step's shard logits were written by the block's LM head,
        # which completed before step 0's kernel (it waits on it) ran.
        base = tl.load(
            base_ptr + reqs[:, None].to(tl.int64) * stride_base + cols[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        )
    _gdc_launch_dependents()
    _gdc_wait()
    if FROM_ANCHOR:
        base = tl.load(
            base_ptr + reqs[:, None].to(tl.int64) * stride_base + cols[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        )
        prev = tl.load(anchor_ptr + reqs, mask=rmask, other=-1).to(tl.int64)
    else:
        s = tl.arange(0, BLOCK_SLOTS)
        smask = rmask[:, None] & (s[None, :] < n_slots)
        pptrs = pairs_ptr + reqs[:, None].to(tl.int64) * stride_pairs + 2 * s[None, :]
        vals = tl.load(pptrs, mask=smask, other=float("-inf"))
        gidx = tl.where(smask, tl.load(pptrs + 1, mask=smask, other=0.0).to(tl.int32), _INT32_MAX)
        _, best = _argmax_pairs(vals, gidx, 1)
        prev = best.to(tl.int64)
        if pid == 0:
            tl.store(
                tokens_ptr + reqs.to(tl.int64) * stride_tokens + step - 1,
                prev.to(tl.int32),
                mask=rmask,
            )
    valid = rmask & (prev >= 0) & (prev < vocab)
    w1 = tl.load(
        w1_ptr + tl.where(valid, prev, 0)[:, None] * RANK + r[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    # F.linear(markov_w1[prev], w2) rounds the FP32 dot to the weights' dtype.
    bias = tl.dot(w1, tl.trans(w2)).to(w2.dtype).to(tl.float32)
    vals = tl.where(rmask[:, None] & cmask[None, :], base.to(tl.float32) + bias, float("-inf"))
    idx = tl.where(cmask[None, :], cols[None, :], _INT32_MAX) + tl.zeros(
        [BLOCK_B, BLOCK_V], tl.int32
    )
    cand_val, cand_idx = _argmax_pairs(vals, idx, 1)
    cand_idx = _global_index(cand_idx, col_base)
    pbase = partial_ptr + (pid * BLOCK_B + reqs) * 2
    tl.store(pbase, cand_val)
    tl.store(pbase + 1, cand_idx.to(tl.float32))
    # After the wait: the output can reuse memory that the previous grid read.
    row = out_ptr + reqs.to(tl.int64) * stride_out
    z = pid * ZERO_CHUNK + tl.arange(0, ZERO_CHUNK)
    zmask = (
        rmask[:, None]
        & (z[None, :] < ZERO_WIDTH)
        & ((z[None, :] < 2 * slot) | (z[None, :] >= 2 * slot + 2))
    )
    tl.store(row[:, None] + z[None, :], tl.zeros([BLOCK_B, ZERO_CHUNK], tl.float32), mask=zmask)
    # Last-arriving program: every program's candidates are visible to it once
    # its acquire-release ticket observes their releases.
    tl.debug_barrier()
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == tl.num_programs(0) - 1:
        p = tl.arange(0, BLOCK_P)
        pm = (p[:, None] < tl.num_programs(0)) & rmask[None, :]
        qptrs = partial_ptr + (p[:, None] * BLOCK_B + reqs[None, :]) * 2
        qv = tl.load(qptrs, mask=pm, other=float("-inf"), volatile=True)
        qi = tl.where(
            pm, tl.load(qptrs + 1, mask=pm, other=0.0, volatile=True).to(tl.int32), _INT32_MAX
        )
        win_val, win_idx = _argmax_pairs(qv, qi, 0)
        tl.store(row + 2 * slot, win_val, mask=rmask)
        tl.store(row + 2 * slot + 1, win_idx.to(tl.float32), mask=rmask)
        tl.atomic_xchg(ticket_ptr, 0, sem="relaxed", scope="gpu")


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


class _PairExchange:
    """Exchanges ``(value, global index)`` float32 pairs per row across TP.

    Over MNNVL each rank owns ``splits`` consecutive pair slots of a zero-padded
    ``[rows, width]`` buffer that an all-reduce sums; over NCCL each rank
    publishes one pair per row and the pairs are all-gathered.
    """

    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self._mnnvl = None
        backend = _VOCAB_PARALLEL_COMM
        if self.tp_size > 1 and backend in ("auto", "mnnvl"):
            from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

            if MNNVLAllReduce.is_mnnvl(mapping, torch.float32):
                try:
                    self._mnnvl = MNNVLAllReduce(mapping, torch.float32)
                except (RuntimeError, ValueError) as e:
                    if backend == "mnnvl":
                        raise
                    logger.info(
                        f"Vocab-parallel pair exchange: MNNVL unavailable ({e}); using NCCL."
                    )
            elif backend == "mnnvl":
                raise RuntimeError(
                    "vocab-parallel pair exchange over MNNVL, but MNNVL is unavailable"
                )
        self.backend = "mnnvl" if self._mnnvl is not None else "nccl"

    def chain_buffers(self, device):
        """``(multicast view, local view, set floats)`` of the one-launch Markov
        chain's exchange: three sets of ``[steps, requests, ranks, 2]`` float32
        pairs, -0.0 where nothing was written. Allocated collectively on first
        use (every rank reaches it at the same point of the warmup), outside
        graph capture. A single rank uses one local buffer for both views."""
        if getattr(self, "_chain", None) is not None:
            return self._chain
        assert not torch.cuda.is_current_stream_capturing()
        set_floats = _CHAIN_MAX_STEPS * _CHAIN_MAX_REQS * max(self.tp_size, 1) * 2
        numel = 3 * set_floats
        if self.tp_size == 1:
            buf = torch.full((numel,), -0.0, dtype=torch.float32, device=device)
            self._chain = (buf, buf, set_floats)
            return self._chain
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert self.backend == "mnnvl"
        comm = MNNVLAllReduce.allreduce_mnnvl_workspaces[self.mapping]["comm"]
        force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
        handle = McastGPUBuffer(
            numel * 4,
            self.tp_size,
            self.tp_rank,
            self.mapping.local_rank,
            force_mn or self.mapping.is_multi_node(),
            comm.py2f(),
        )
        local = handle.get_uc_buffer(self.tp_rank, [numel], torch.float32, 0)
        mcast = handle.get_mc_buffer([numel], torch.float32, 0)
        local.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()
        self._chain_handle = handle
        self._chain = (mcast, local, set_floats)
        return self._chain

    def markov_v3_buffers(self, device, entries: int):
        """``(local address, multicast address, state)`` of the CuTe DSL Markov
        chain's key exchange for ``entries`` (ranks x CTAs per rank): its
        rotating sets, zero-filled, and this rank's int32 launch state. Allocated
        collectively on first use, outside graph capture. A single rank uses one
        local buffer for both addresses."""
        cached = getattr(self, "_markov_v3", None)
        if cached is not None and cached[0] == entries:
            return cached[1:4]
        assert cached is None, "the Markov exchange is sized once per process"
        assert not torch.cuda.is_current_stream_capturing()
        from ..kernels.dspark_markov_greedy import NUM_SETS, STATE_WORDS, set_bytes

        words = NUM_SETS * set_bytes(entries) // 4
        state = torch.zeros((STATE_WORDS,), dtype=torch.int32, device=device)
        if self.tp_size == 1:
            buf = torch.zeros((words,), dtype=torch.int32, device=device)
            self._markov_v3 = (entries, buf.data_ptr(), buf.data_ptr(), state, buf)
            return self._markov_v3[1:4]
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert self.backend == "mnnvl"
        comm = MNNVLAllReduce.allreduce_mnnvl_workspaces[self.mapping]["comm"]
        force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
        handle = McastGPUBuffer(
            words * 4,
            self.tp_size,
            self.tp_rank,
            self.mapping.local_rank,
            force_mn or self.mapping.is_multi_node(),
            comm.py2f(),
        )
        local = handle.get_uc_buffer(self.tp_rank, [words], torch.int32, 0)
        mcast = handle.get_mc_buffer([words], torch.int32, 0)
        local.zero_()
        torch.cuda.synchronize()
        comm.Barrier()
        self._markov_v3 = (entries, local.data_ptr(), mcast.data_ptr(), state, handle)
        return self._markov_v3[1:4]

    def layout(self, rows: int, n_cols: int, device) -> _PairLayout:
        # Split each row over enough programs to fill the GPU when rows are few;
        # the per-split pairs travel as extra slots of the same exchange.
        max_splits = triton.cdiv(n_cols, _LOCAL_BLOCK_V)
        splits = 1
        if self.backend == "mnnvl" and self.tp_size > 1:
            splits = max(1, min(max_splits, triton.cdiv(_TARGET_PROGRAMS, max(rows, 1))))
        chunk = triton.cdiv(triton.cdiv(n_cols, splits), _LOCAL_BLOCK_V) * _LOCAL_BLOCK_V
        splits = max(1, triton.cdiv(n_cols, chunk))
        if self.tp_size == 1 or self.backend == "nccl":
            buf = torch.empty((rows, 2 * splits), dtype=torch.float32, device=device)
            return _PairLayout(buf, 2 * splits, 0, 0, 0, 0, splits, chunk)
        width = triton.next_power_of_2(2 * self.tp_size * splits)
        buf = torch.empty((rows, width), dtype=torch.float32, device=device)
        lo = 2 * self.tp_rank * splits
        return _PairLayout(
            buf, width, self.tp_rank * splits, lo, lo + 2 * splits, width, splits, chunk
        )

    def exchange(self, layout: _PairLayout) -> tuple[torch.Tensor, int, int, int]:
        """Every rank's pairs: (tensor, row stride, slot stride, number of slots)."""
        buf = layout.buffer
        rows = buf.shape[0]
        if self.tp_size == 1:
            return buf, layout.stride_out, 2, layout.splits
        if self.backend == "mnnvl":
            out = self._mnnvl(buf, all_reduce_params=AllReduceParams())
            return out, layout.stride_out, 2, self.tp_size * layout.splits
        from tensorrt_llm._torch.distributed.ops import allgather

        assert layout.splits == 1
        out = allgather(buf, self.mapping, dim=0)
        return out, 2, 2 * rows, self.tp_size


class VocabParallelHead:
    """Vocab-parallel token selection over one column-parallel LM head.

    Args:
        lm_head: an ``LMHead`` (or ``Linear``) in COLUMN tensor-parallel mode
            whose weight holds this rank's contiguous vocab shard.
        vocab_size: the full vocabulary size.
    """

    def __init__(self, lm_head, vocab_size: int):
        self.lm_head = lm_head
        self.mapping = lm_head.mapping
        self.vocab_size = int(vocab_size)
        self.shard_width = int(lm_head.out_features)
        self.col_base = self.mapping.tp_rank * self.shard_width
        self.exchange = _PairExchange(self.mapping)
        self._markov_w2_shard = None
        self._markov_w2_src_ptr = None
        logger.info(
            f"Vocab-parallel LM-head token selection: tp={self.mapping.tp_size}, "
            f"shard={self.shard_width}, vocab={self.vocab_size}, exchange={self.exchange.backend}"
        )

    @staticmethod
    def supported(lm_head, vocab_size: int) -> bool:
        """Whether ``lm_head`` is a plain vocab-sharded head this class handles."""
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        mapping = getattr(lm_head, "mapping", None)
        if mapping is None or mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return False
        if getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN:
            return False
        if not getattr(lm_head, "gather_output", False):
            return False
        if getattr(lm_head, "padding_size", 0) != 0 or getattr(lm_head, "has_any_quant", False):
            return False
        # The sampler draws Philox noise for 4-column groups of the global vocab.
        shard = int(lm_head.out_features)
        return shard % 4 == 0 and shard * mapping.tp_size == int(vocab_size) and vocab_size < 2**24

    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``[rows, shard]`` logits in the head's output dtype."""
        return self.lm_head.apply_linear(hidden_states, self.lm_head.bias)

    def shard_logits(self, hidden_states: torch.Tensor) -> "VocabShardedLogits":
        """This rank's logits wrapped for a spec worker (see ``VocabShardedLogits``)."""
        return VocabShardedLogits(self, self.local_logits(hidden_states))

    def gather_full_logits(self, local: torch.Tensor) -> torch.Tensor:
        """Full float32 logits, identical to the gathered head + ``.float()``."""
        from tensorrt_llm._torch.distributed import allgather

        return allgather(local, self.mapping, sizes=self.lm_head.gather_output_sizes).float()

    def _launch_local(self, kernel, local: torch.Tensor, extra_args: tuple) -> _PairLayout:
        rows, n_cols = local.shape
        layout = self.exchange.layout(rows, n_cols, local.device)
        if rows > 0:
            kernel[(rows, layout.splits)](
                local,
                local.stride(0),
                n_cols,
                self.col_base,
                layout.chunk,
                *extra_args,
                layout.buffer,
                layout.stride_out,
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=_LOCAL_BLOCK_V,
                num_warps=8,
            )
        return layout

    def _exchange_and_reduce(self, layout: _PairLayout, rows: int, out: torch.Tensor):
        pairs, stride_row, stride_slot, n_slots = self.exchange.exchange(layout)
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

    def selection_pairs(self, local: torch.Tensor, spec_metadata) -> tuple:
        """Every rank's ``(value, global index)`` pairs of the target selection,
        not yet reduced: ``(pairs, row stride, slot stride, slots)``.

        The same local kernel and exchange as :meth:`VocabShardedLogits.select_tokens`
        (argmax for an all-greedy batch, else the Gumbel-max draw), for a caller
        that reduces the pairs inside a fused consumer kernel.
        """
        assert local.dim() == 2 and local.stride(-1) == 1
        if spec_metadata.is_all_greedy_sample:
            layout = self._launch_local(_local_argmax_kernel, local, ())
        else:
            from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
                GREEDY_TEMPERATURE_THRESHOLD,
            )

            rows = local.shape[0]
            layout = self._launch_local(
                _local_sample_kernel,
                local,
                (
                    spec_metadata.temperatures[:rows],
                    spec_metadata.seeds[:rows],
                    spec_metadata.offsets[:rows],
                    float(GREEDY_TEMPERATURE_THRESHOLD),
                ),
            )
        return self.exchange.exchange(layout)

    def argmax(self, local: torch.Tensor, out_dtype=torch.int64) -> torch.Tensor:
        """Global argmax over the vocab of shard logits ``[rows, shard]``."""
        assert local.dim() == 2 and local.stride(-1) == 1
        layout = self._launch_local(_local_argmax_kernel, local, ())
        out = torch.empty((local.shape[0],), dtype=out_dtype, device=local.device)
        return self._exchange_and_reduce(layout, local.shape[0], out)

    def sample(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
        out_dtype=torch.int64,
    ) -> torch.Tensor:
        """Temperature-only sampling (Gumbel-max); rows at or below the greedy
        temperature threshold take the plain argmax."""
        assert local.dim() == 2 and local.stride(-1) == 1
        layout = self._launch_local(
            _local_sample_kernel, local, (temperatures, seeds, offsets, float(greedy_threshold))
        )
        out = torch.empty((local.shape[0],), dtype=out_dtype, device=local.device)
        return self._exchange_and_reduce(layout, local.shape[0], out)

    def _markov_round_buffers(self, num_reqs: int, device):
        """Persistent scratch of the fused Markov steps (candidates, ticket), and
        the per-rank-pair exchange layout: one slot per rank."""
        programs = triton.cdiv(self.shard_width, _MARKOV_BLOCK_V)
        block_b = max(16, triton.next_power_of_2(max(num_reqs, 1)))
        key = (programs, block_b, str(device))
        cache = getattr(self, "_markov_round_cache", None)
        if cache is None or cache[0] != key:
            partial = torch.empty((programs * block_b * 2,), dtype=torch.float32, device=device)
            ticket = torch.zeros((1,), dtype=torch.int32, device=device)
            self._markov_round_cache = (key, partial, ticket)
        return programs, block_b, self._markov_round_cache[1], self._markov_round_cache[2]

    def markov_tokens_fused(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`markov_tokens` with one launch per step (plus the exchange).

        Each step's kernel resolves prev, gathers its ``markov_w1`` row, computes
        this rank's ``markov_w2`` bias columns, adds the step's shard logits and
        publishes the rank's (value, global index) pair; the exchange is the same
        all-reduce. The bias is an FP32 dot of the BF16 row and weights rounded
        to BF16, like the ``F.linear`` of the unfused chain, but its summation
        order differs, so a draft token can differ where two vocab entries tie to
        within that rounding. Requires the MNNVL (or single-rank) exchange.
        """
        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        if num_reqs == 0 or block_size == 0:
            return tokens
        rank = markov_w1.shape[1]
        w2_shard = self.markov_w2_shard(markov_w2)
        assert (
            w2_shard.stride(-1) == 1 and markov_w1.is_contiguous() and local_block.stride(-1) == 1
        )
        programs, block_b, partial, ticket = self._markov_round_buffers(num_reqs, device)
        tp = self.exchange.tp_size
        width = max(_MARKOV_EXCHANGE_WIDTH, triton.next_power_of_2(2 * tp)) if tp > 1 else 2
        pairs, stride_pairs, n_slots = first_prev_tokens, 0, 1
        for step in range(block_size + 1):
            if step == block_size:
                # The last step's pairs only resolve the final draft token.
                emb = torch.empty((num_reqs, rank), dtype=markov_w1.dtype, device=device)
                _markov_reduce_embed_kernel[(num_reqs,)](
                    pairs,
                    stride_pairs,
                    2,
                    n_slots,
                    first_prev_tokens,
                    markov_w1,
                    emb,
                    tokens,
                    tokens.stride(0),
                    self.vocab_size,
                    step,
                    RANK=rank,
                    FROM_ANCHOR=False,
                    BLOCK_SLOTS=triton.next_power_of_2(n_slots),
                    BLOCK_R=triton.next_power_of_2(rank),
                    num_warps=4,
                )
                break
            buf = torch.empty((num_reqs, width), dtype=torch.float32, device=device)
            step_logits = local_block[:, step]
            _markov_round_kernel[(programs,)](
                pairs,
                stride_pairs,
                n_slots,
                first_prev_tokens,
                markov_w1,
                w2_shard,
                w2_shard.stride(0),
                step_logits,
                step_logits.stride(0),
                tokens,
                tokens.stride(0),
                step,
                partial,
                ticket,
                buf,
                width,
                self.mapping.tp_rank if tp > 1 else 0,
                num_reqs,
                self.vocab_size,
                shard,
                self.col_base,
                RANK=rank,
                FROM_ANCHOR=step == 0,
                ZERO_WIDTH=width,
                ZERO_CHUNK=triton.next_power_of_2(triton.cdiv(width, programs)),
                BLOCK_B=block_b,
                BLOCK_SLOTS=triton.next_power_of_2(max(tp, 1)),
                BLOCK_V=_MARKOV_BLOCK_V,
                BLOCK_P=triton.next_power_of_2(programs),
                num_warps=4,
                launch_pdl=True,
            )
            if tp > 1:
                buf = self.exchange._mnnvl(buf, all_reduce_params=AllReduceParams())
            pairs, stride_pairs, n_slots = buf, width, tp
        return tokens

    def markov_tokens_chain(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`markov_tokens_fused` with the whole chain in one launch.

        Each step's per-rank pairs travel through a multicast buffer that every
        program polls (see ``_markov_chain_kernel``) instead of an all-reduce
        launch, and the steps share one launch. Same values as
        :meth:`markov_tokens_fused`. Needs ``num_reqs <= 16`` and ``block <= 16``
        (the caller falls back otherwise) and the MNNVL (or single-rank) exchange.
        """
        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        if num_reqs == 0 or block_size == 0:
            return tokens
        assert num_reqs <= _CHAIN_MAX_REQS and block_size <= _CHAIN_MAX_STEPS
        rank = markov_w1.shape[1]
        w2_shard = self.markov_w2_shard(markov_w2)
        assert (
            w2_shard.stride(-1) == 1 and markov_w1.is_contiguous() and local_block.stride(-1) == 1
        )
        block_v = _MARKOV_BLOCK_V
        programs = triton.cdiv(shard, block_v)
        # Every program polls for the others' pairs: all of them must be resident.
        assert programs <= torch.cuda.get_device_properties(device).multi_processor_count
        mcast, local, set_floats = self.exchange.chain_buffers(device)
        state = getattr(self, "_chain_state", None)
        if state is None or state[0] != (programs, str(device)):
            partial = torch.empty(
                (_CHAIN_MAX_STEPS * programs * _CHAIN_MAX_REQS * 2,),
                dtype=torch.float32,
                device=device,
            )
            tickets = torch.zeros((_CHAIN_MAX_STEPS,), dtype=torch.int32, device=device)
            launch = torch.zeros((1,), dtype=torch.int32, device=device)
            self._chain_state = ((programs, str(device)), partial, tickets, launch)
        _, partial, tickets, launch = self._chain_state
        tp = max(self.exchange.tp_size, 1)
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
            self.vocab_size,
            shard,
            self.col_base,
            self.mapping.tp_rank if tp > 1 else 0,
            STEPS=block_size,
            RANK=rank,
            TP=tp,
            BLOCK_B=_CHAIN_MAX_REQS,
            BLOCK_V=block_v,
            BLOCK_P=triton.next_power_of_2(programs),
            SET_FLOATS=set_floats,
            num_warps=4,
            launch_pdl=True,
        )
        return tokens

    def markov_v3_supported(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: Optional[torch.Tensor],
        markov_w2: Optional[torch.Tensor],
    ) -> bool:
        """Whether :meth:`markov_tokens_v3` serves these operands: BF16 logits
        and Markov weights (rank a multiple of 16, at most 256), at most 16
        requests and 16 block positions, an SM100-family device with a CTA per
        256 shard rows resident at once, and the MNNVL (or single-rank)
        exchange."""
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if not IS_CUTLASS_DSL_AVAILABLE or markov_w1 is None or markov_w2 is None:
            return False
        if self.exchange.tp_size > 1 and self.exchange.backend != "mnnvl":
            return False
        num_reqs, block_size, shard = local_block.shape
        rank_dim = markov_w1.shape[1]
        device = local_block.device
        if not (1 <= num_reqs <= 16 and 1 <= block_size <= 16) or shard != self.shard_width:
            return False
        if (
            local_block.dtype != torch.bfloat16
            or markov_w1.dtype != torch.bfloat16
            or markov_w2.dtype != torch.bfloat16
            or local_block.stride(-1) != 1
            or not markov_w1.is_contiguous()
            or not markov_w2.is_contiguous()
            or markov_w2.shape[1] != rank_dim
            or rank_dim % 16
            or rank_dim > 256
            or first_prev_tokens.dtype not in (torch.int32, torch.int64)
            or first_prev_tokens.stride(0) != 1
            or markov_w2.data_ptr() % 16
        ):
            return False
        from ..kernels.dspark_markov_greedy import ctas_for, smem_bytes

        props = torch.cuda.get_device_properties(device)
        return (
            props.major == 10
            and ctas_for(shard) <= props.multi_processor_count
            and smem_bytes(rank_dim) <= props.shared_memory_per_block_optin
        )

    def markov_tokens_v3(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`markov_tokens_chain` as one CuTe DSL launch without the
        intra-GPU reduction.

        Each CTA keeps its 256 ``markov_w2`` shard rows in shared memory, adds
        the BF16-rounded bias of every request to the step's shard logits in
        FP32 (the arithmetic of the other chains) and stores its own
        (value, global index) key into every rank's exchange; every CTA polls
        all keys of the step (``dspark_markov_greedy``). Same tokens as
        :meth:`markov_tokens_chain` wherever both compute the same BF16 bias.
        The caller checks :meth:`markov_v3_supported`.
        """
        from ..kernels.dspark_markov_greedy import ctas_for, run_dspark_markov_greedy

        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        tp = max(self.exchange.tp_size, 1)
        box_uc, box_mc, state = self.exchange.markov_v3_buffers(device, tp * ctas_for(shard))
        run_dspark_markov_greedy(
            local_block,
            first_prev_tokens,
            markov_w1,
            self.markov_w2_shard(markov_w2),
            tokens,
            vocab=self.vocab_size,
            tp=tp,
            rank=self.mapping.tp_rank if tp > 1 else 0,
            state=state,
            box_uc=box_uc,
            box_mc=box_mc,
            multicast=tp > 1,
        )
        return tokens

    def markov_w2_shard(self, markov_w2: torch.Tensor) -> torch.Tensor:
        """This rank's rows of ``markov_w2`` (cached; a contiguous row slice)."""
        if self._markov_w2_src_ptr != markov_w2.data_ptr():
            self._markov_w2_shard = markov_w2[self.col_base : self.col_base + self.shard_width]
            self._markov_w2_src_ptr = markov_w2.data_ptr()
        return self._markov_w2_shard

    def markov_tokens(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: Optional[torch.Tensor],
        markov_w2: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Greedy DSpark block draft tokens from shard block logits.

        Args:
            local_block: ``[B, K, shard]`` shard logits of the block positions.
            first_prev_tokens: ``[B]`` anchor token ids (step 0's prev).
            markov_w1, markov_w2: ``[vocab, rank]`` Markov head weights, or
                ``None`` for a drafter without a Markov head (plain argmax).
        Returns:
            ``[B, K]`` int32 draft tokens, bit-identical to the per-position
            argmax of ``dspark_markov_chain_logits`` on the gathered logits.
        """
        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        if markov_w1 is None:
            flat = local_block.reshape(num_reqs * block_size, shard)
            return self.argmax(flat, out_dtype=torch.int32).view(num_reqs, block_size)
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        if num_reqs == 0 or block_size == 0:
            return tokens
        rank = markov_w1.shape[1]
        w2_shard = self.markov_w2_shard(markov_w2)
        emb = torch.empty((num_reqs, rank), dtype=markov_w1.dtype, device=device)
        pairs, stride_row, stride_slot, n_slots = first_prev_tokens, 0, 0, 1
        for step in range(block_size + 1):
            _markov_reduce_embed_kernel[(num_reqs,)](
                pairs,
                stride_row,
                stride_slot,
                n_slots,
                first_prev_tokens,
                markov_w1,
                emb,
                tokens,
                tokens.stride(0),
                self.vocab_size,
                step,
                RANK=rank,
                FROM_ANCHOR=step == 0,
                BLOCK_SLOTS=triton.next_power_of_2(n_slots),
                BLOCK_R=triton.next_power_of_2(rank),
                num_warps=4,
            )
            if step == block_size:
                break
            # Same F.linear as the reference chain, on this rank's w2 rows.
            bias = F.linear(emb, w2_shard)
            step_logits = local_block[:, step]
            layout = self.exchange.layout(num_reqs, shard, device)
            _markov_add_local_argmax_kernel[(num_reqs, layout.splits)](
                step_logits,
                step_logits.stride(0),
                bias,
                bias.stride(0),
                shard,
                self.col_base,
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
            pairs, stride_row, stride_slot, n_slots = self.exchange.exchange(layout)
        return tokens


class VocabShardedLogits:
    """This rank's vocab shard of the target logits, handed to a spec worker in
    place of the gathered float32 logits.

    The worker either selects tokens from it vocab-parallel
    (:meth:`select_tokens`) or, for a batch the vocab-parallel samplers do not
    cover, materializes the gathered logits with :meth:`gather`.
    """

    __slots__ = ("head", "local")

    def __init__(self, head: VocabParallelHead, local: torch.Tensor):
        self.head = head
        self.local = local

    @property
    def device(self) -> torch.device:
        return self.local.device

    def gather(self) -> torch.Tensor:
        """The gathered ``[rows, vocab]`` float32 logits."""
        return self.head.gather_full_logits(self.local)

    def select_pairs(self, spec_metadata) -> tuple:
        """The unreduced exchanged pairs of :meth:`select_tokens` (see
        :meth:`VocabParallelHead.selection_pairs`)."""
        return self.head.selection_pairs(self.local, spec_metadata)

    def select_tokens(self, spec_metadata) -> torch.Tensor:
        """One target token per row: the argmax for an all-greedy batch, else a
        temperature-only draw (the caller guarantees the sampling mode)."""
        if spec_metadata.is_all_greedy_sample:
            return self.head.argmax(self.local)
        from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

        rows = self.local.shape[0]
        return self.head.sample(
            self.local,
            spec_metadata.temperatures[:rows],
            spec_metadata.seeds[:rows],
            spec_metadata.offsets[:rows],
            GREEDY_TEMPERATURE_THRESHOLD,
        )
