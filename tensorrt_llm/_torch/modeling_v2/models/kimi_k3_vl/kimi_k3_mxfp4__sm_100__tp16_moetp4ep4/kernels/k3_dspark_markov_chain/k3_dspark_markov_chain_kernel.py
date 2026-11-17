# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark vanilla-Markov greedy block chain over (a vocab shard of) the block logits, in Triton: per block position
one launch resolves the previous token and gathers its ``markov_w1`` row, a GEMV forms the bias, and one launch adds
it to the position's logits and reduces each column split to a ``(value, global index)`` pair. The pairs of a TP
group's shards are exchanged by one float32 all-reduce of a zero-padded buffer in which each rank writes only its own
columns (``x + 0`` is exact and indices below ``2**24`` are exact in float32, so the sum acts as an all-gather); the
winner is taken in ``torch.argmax`` order (NaN beats every number, equal maxima go to the lowest global index).
Launch shapes depend only on ``(rows, shard width)`` and nothing synchronizes with the host, so the chain is
CUDA-graph capturable.

Called through ``ops/spec/k3_dspark_markov_chain`` (`K3DSparkMarkovChain`)."""

from typing import Callable, NamedTuple, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_INT32_MAX = tl.constexpr(2**31 - 1)
# Index published by a split without valid columns: exact in float32 and above
# every supported vocab id, so it never wins a tie.
_EMPTY_INDEX = tl.constexpr(2**24)
_LOCAL_BLOCK_V = 2048
# Programs the local reductions aim for (rows x column splits): rows are few in
# decode, so each row is split over several programs.
_TARGET_PROGRAMS = 256


@triton.jit
def argmax_combine(val_a, idx_a, val_b, idx_b):
    """``tl.reduce`` combine in ``torch.argmax`` order: NaN beats any number,
    equal values go to the lower index."""
    nan_a = val_a != val_a
    nan_b = val_b != val_b
    take_a = tl.where(
        nan_a,
        (~nan_b) | (idx_a < idx_b),
        (~nan_b) & ((val_a > val_b) | ((val_a == val_b) & (idx_a < idx_b))),
    )
    return tl.where(take_a, val_a, val_b), tl.where(take_a, idx_a, idx_b)


@triton.jit
def _global_index(best_idx, col_base):
    return tl.where(best_idx == _INT32_MAX, _EMPTY_INDEX, col_base + best_idx)


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
    # Publishes (value, global index) at columns [2 * slot, 2 * slot + 1]. With
    # ZERO_WIDTH > 0 the program flagged `fill` also zeroes the row outside
    # [zero_lo, zero_hi), i.e. the other ranks' columns.
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


# The pair layout follows the row count, so the pair-buffer row stride and the slot count change with the batch.
# They are not specialized: a specialization flips when a value becomes 1 or a multiple of 16 and would compile a
# kernel variant mid-serving.
_PAIR_LAYOUT_ARGS = ["stride_out", "slot0", "zero_lo", "zero_hi"]


@triton.jit(do_not_specialize=["stride_row", "n_slots"])
def _markov_reduce_embed_kernel(
    pairs_ptr,
    stride_row,
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
    # Resolves prev_step for one request -- the anchor at step 0, else the
    # global argmax of the previous step, which is also draft token step - 1 --
    # and gathers its markov_w1 row; an out-of-vocab prev gives a zero row.
    req = tl.program_id(0)
    if FROM_ANCHOR:
        prev = tl.load(anchor_ptr + req).to(tl.int64)
    else:
        slots = tl.arange(0, BLOCK_SLOTS)
        smask = slots < n_slots
        ptrs = pairs_ptr + req.to(tl.int64) * stride_row + slots * 2
        vals = tl.load(ptrs, mask=smask, other=float("-inf"))
        gidx = tl.load(ptrs + 1, mask=smask, other=0.0).to(tl.int32)
        gidx = tl.where(smask, gidx, _INT32_MAX)
        _, best = tl.reduce((vals, gidx), 0, argmax_combine)
        prev = best.to(tl.int64)
        tl.store(tokens_ptr + req.to(tl.int64) * stride_tokens + step - 1, prev.to(tl.int32))
    valid = (prev >= 0) & (prev < vocab)
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < RANK
    w1_row = tl.load(
        w1_ptr + tl.where(valid, prev, 0) * RANK + offs_r, mask=mask_r & valid, other=0.0
    )
    tl.store(emb_ptr + req * RANK + offs_r, w1_row, mask=mask_r)


@triton.jit(do_not_specialize=_PAIR_LAYOUT_ARGS)
def _markov_add_local_argmax_kernel(
    base_ptr,
    stride_base,
    n_cols,
    col_base,
    chunk,
    bias_ptr,
    stride_bias,
    biased_ptr,
    stride_biased,
    out_ptr,
    stride_out,
    slot0,
    zero_lo,
    zero_hi,
    STORE_BIASED: tl.constexpr,
    ZERO_WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # base (logits of block position `step`) + bias, both upcast to float32
    # like the gathered reference, optionally stored, then the argmax of one
    # split.
    req = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * chunk
    hi = tl.minimum(lo + chunk, n_cols)
    offs = tl.arange(0, BLOCK_V)
    base_row = base_ptr + req.to(tl.int64) * stride_base
    bias_row = bias_ptr + req.to(tl.int64) * stride_bias
    biased_row = biased_ptr + req.to(tl.int64) * stride_biased
    best_val = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_V], _INT32_MAX, tl.int32)
    for start in range(lo, hi, BLOCK_V):
        cols = start + offs
        mask = cols < hi
        base = tl.load(base_row + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
        if STORE_BIASED:
            tl.store(biased_row + cols, base + bias, mask=mask)
        vals = tl.where(mask, base + bias, float("-inf"))
        idx = tl.where(mask, cols, _INT32_MAX)
        best_val, best_idx = argmax_combine(best_val, best_idx, vals, idx)
    val, idx = tl.reduce((best_val, best_idx), 0, argmax_combine)
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


class _PairLayout(NamedTuple):
    """Where the local kernels of one exchange publish their pairs."""

    buffer: torch.Tensor  # [rows, width] float32
    slot0: int  # first pair slot owned by this rank
    zero_lo: int  # columns [zero_lo, zero_hi) are this rank's; the rest is zeroed
    zero_hi: int
    zero_width: int  # 0 when there is nothing to zero (a single shard)
    splits: int  # column splits per rank, i.e. pair slots per rank
    chunk: int  # columns per split


class _PairExchange:
    """Lays out and exchanges ``(value, global index)`` pairs across a TP group.

    Each rank owns ``splits`` consecutive pair slots of a zero-padded
    ``[rows, width]`` float32 buffer and one all-reduce sums the buffers. A
    single shard (``all_reduce`` None) publishes its pairs and exchanges nothing.
    """

    def __init__(
        self,
        all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]],
        tp_size: int,
        tp_rank: int,
    ):
        self.all_reduce = all_reduce
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def layout(self, rows: int, n_cols: int, device: torch.device) -> _PairLayout:
        max_splits = triton.cdiv(n_cols, _LOCAL_BLOCK_V)
        splits = max(1, min(max_splits, triton.cdiv(_TARGET_PROGRAMS, max(rows, 1))))
        chunk = triton.cdiv(triton.cdiv(n_cols, splits), _LOCAL_BLOCK_V) * _LOCAL_BLOCK_V
        splits = max(1, triton.cdiv(n_cols, chunk))
        if self.tp_size == 1:
            buf = torch.empty((rows, 2 * splits), dtype=torch.float32, device=device)
            return _PairLayout(buf, 0, 0, 0, 0, splits, chunk)
        width = triton.next_power_of_2(2 * self.tp_size * splits)
        buf = torch.empty((rows, width), dtype=torch.float32, device=device)
        lo = 2 * self.tp_rank * splits
        return _PairLayout(buf, self.tp_rank * splits, lo, lo + 2 * splits, width, splits, chunk)

    def exchange(self, layout: _PairLayout) -> tuple[torch.Tensor, int]:
        """Every rank's pairs, ``[rows, width]``, and the number of pair slots."""
        if self.all_reduce is None:
            return layout.buffer, layout.splits
        return self.all_reduce(layout.buffer), self.tp_size * layout.splits


def _markov_chain_tokens(
    block_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2_rows: torch.Tensor,
    col_base: int,
    exchange: _PairExchange,
    biased_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Greedy vanilla-Markov block chain over (a shard of) the block logits.

    Args:
        block_logits: ``[B, K, n_cols]`` logits, unit vocab stride.
        first_prev_tokens: ``[B]`` anchor token ids.
        markov_w1: ``[vocab, rank]``, contiguous.
        markov_w2_rows: ``[n_cols, rank]`` rows of ``markov_w2`` for the columns
            of ``block_logits``.
        col_base: global vocab id of column 0.
        exchange: the TP pair exchange.
        biased_out: optional ``[B, K, n_cols]`` float32 destination of the
            biased logits (unit vocab stride).
    Returns:
        ``[B, K]`` int32 draft tokens.
    """
    num_reqs, block_size, n_cols = block_logits.shape
    device = block_logits.device
    tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
    if num_reqs == 0 or block_size == 0:
        return tokens
    assert markov_w1.is_contiguous() and block_logits.stride(-1) == 1
    vocab, rank = markov_w1.shape
    emb = torch.empty((num_reqs, rank), dtype=markov_w1.dtype, device=device)
    layout = exchange.layout(num_reqs, n_cols, device)
    pairs, n_slots = first_prev_tokens, 1
    for step in range(block_size + 1):
        _markov_reduce_embed_kernel[(num_reqs,)](
            pairs,
            pairs.stride(0) if step > 0 else 0,
            n_slots,
            first_prev_tokens,
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
        if step == block_size:
            break
        # The same F.linear as the reference chain, on this shard's w2 rows.
        bias = F.linear(emb, markov_w2_rows)
        step_logits = block_logits[:, step]
        step_biased = step_logits if biased_out is None else biased_out[:, step]
        _markov_add_local_argmax_kernel[(num_reqs, layout.splits)](
            step_logits,
            step_logits.stride(0),
            n_cols,
            col_base,
            layout.chunk,
            bias,
            bias.stride(0),
            step_biased,
            step_biased.stride(0),
            layout.buffer,
            layout.buffer.stride(0),
            layout.slot0,
            layout.zero_lo,
            layout.zero_hi,
            STORE_BIASED=biased_out is not None,
            ZERO_WIDTH=layout.zero_width,
            BLOCK_V=_LOCAL_BLOCK_V,
            num_warps=8,
        )
        pairs, n_slots = exchange.exchange(layout)
    return tokens


def markov_chain_logits(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    tp_exchange=None,
) -> torch.Tensor:
    """The DSpark vanilla-Markov greedy block chain, returning the biased logits.

    Each block position takes three launches: resolve the previous token and
    gather its ``markov_w1`` row, the bias GEMV, add + argmax.

    Args:
        base_logits: ``[B, K, n]`` float32 block logits, unit vocab stride:
            the full vocab, or with ``tp_exchange`` this rank's contiguous
            vocab shard.
        first_prev_tokens: ``[B]`` anchor token ids.
        markov_w1, markov_w2: ``[vocab, rank]`` Markov head weights of one
            dtype, ``markov_w1`` contiguous.
        tp_exchange: when ``base_logits`` is a vocab shard, the TP group's
            pair exchange (``tp_size``, ``tp_rank``, ``layout`` and
            ``exchange`` as `_PairExchange` has them; the speculative worker
            builds it).
    Returns:
        ``[B, K, n]`` float32 biased logits: (this rank's shard of) the eager
        chain over the raw Markov tensors, advanced by a plain argmax of the
        full vocab. The bias is one ``F.linear`` GEMV over this shard's
        ``markov_w2`` rows.
    """
    assert base_logits.dtype == torch.float32
    n_cols = base_logits.shape[-1]
    if tp_exchange is None:
        col_base, exchange = 0, _PairExchange(None, 1, 0)
    else:
        assert n_cols * tp_exchange.tp_size == markov_w2.shape[0]
        col_base, exchange = tp_exchange.tp_rank * n_cols, tp_exchange
    biased = torch.empty_like(base_logits)
    _markov_chain_tokens(
        base_logits,
        first_prev_tokens,
        markov_w1,
        markov_w2[col_base : col_base + n_cols],
        col_base,
        exchange,
        biased_out=biased,
    )
    return biased
