# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch building blocks shared by the references of the speculative-decoding entries."""

from __future__ import annotations

from typing import Optional

import torch

#: Index a vocab-parallel split without valid columns reports (above every vocab id).
EMPTY_INDEX = 2**24
INT32_MAX = 2**31 - 1


def argmax_pairs(vals: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(value, index)`` the ``torch.argmax`` order selects along the last dim of ``(vals,
    idx)``: a NaN beats every number and the lowest index wins among NaNs; otherwise the largest
    value wins and the lowest index among equal values (``-0.0 == +0.0``)."""
    if vals.dtype != torch.float64:
        vals = vals.float()
    idx = idx.to(torch.int64)
    is_nan = torch.isnan(vals)
    any_nan = is_nan.any(dim=-1, keepdim=True)
    vmax = torch.where(is_nan, float("-inf"), vals).amax(dim=-1, keepdim=True)
    cand = torch.where(any_nan, is_nan, (vals == vmax) & ~is_nan)
    best = torch.where(cand, idx, INT32_MAX).amin(dim=-1)
    value = torch.where(any_nan.squeeze(-1), float("nan"), vmax.squeeze(-1))
    return value, best


def interleaved_pairs(
    buf: torch.Tensor, stride_row: int, stride_slot: int, n_slots: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(values, indices)`` ``[rows, n_slots]`` of exchanged ``(value, global index)`` float32
    pairs at ``buf.view(-1)[r * stride_row + s * stride_slot + {0, 1}]``."""
    flat = buf.reshape(-1)
    r = torch.arange(rows, device=buf.device)[:, None] * stride_row
    s = torch.arange(n_slots, device=buf.device)[None, :] * stride_slot
    at = (r + s).reshape(-1)
    vals = flat[at].view(rows, n_slots)
    # The kernels convert the float32 index with a truncating cast; the indices are integers.
    idx = flat[at + 1].to(torch.int32).view(rows, n_slots)
    return vals, idx


def strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    force: Optional[tuple] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``SpecWorkerBase._sample_and_accept_draft_tokens_base`` on already-selected target
    tokens ``[num_contexts + num_gens * (K + 1)]``: ``accepted [batch, K + 1]`` int32 (context
    rows define column 0 only; the rest is 0 here) and ``num_accepted [batch]`` int32.

    ``force``: None, ``(base_total,)``, or ``(base_total, frac, max_total, pool, counter,
    counter_stride, slot_stride)``, in which case the draw reads ``counter + 1`` (the kernels
    advance ``counter`` by one) and ``counter`` is not written here.
    """
    num_gens = batch_size - num_contexts
    k = draft_tokens.shape[-1]
    device = target_tokens.device
    accepted = torch.zeros((batch_size, k + 1), dtype=torch.int32, device=device)
    num_accepted = torch.ones((batch_size,), dtype=torch.int32, device=device)
    accepted[:num_contexts, 0] = target_tokens[:num_contexts].to(torch.int32)
    if num_gens == 0:
        return accepted, num_accepted
    t_gen = target_tokens[num_contexts : num_contexts + num_gens * (k + 1)].to(torch.int32)
    t_gen = t_gen.view(num_gens, k + 1)
    accepted[num_contexts:] = t_gen
    match = (draft_tokens[:num_gens].to(torch.int32) == t_gen[:, :k]).to(torch.int32)
    n = 1 + match.cumprod(dim=1).sum(dim=1)
    if force is not None and len(force) == 1:
        n = torch.full_like(n, int(force[0]))
    elif force is not None:
        base_total, frac, max_total, pool, counter, counter_stride, slot_stride = force
        nxt = counter.reshape(-1)[0].to(torch.int64) + 1
        rows = torch.arange(num_gens, device=device, dtype=torch.int64)
        at = (nxt * counter_stride + rows * slot_stride) & (pool.numel() - 1)
        extra = (pool.reshape(-1)[at] < frac).to(torch.int64)
        n = (base_total + extra).clamp(max=max_total)
    num_accepted[num_contexts:] = n.to(torch.int32)
    return accepted, num_accepted


def u32_mul(a: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(high, low)`` 32-bit words of ``a * b`` for int64 tensors holding uint32 values."""
    b_hi, b_lo = b >> 16, b & 0xFFFF
    p_lo = a * b_lo
    p_hi = a * b_hi
    high = (p_hi + (p_lo >> 16)) >> 16
    low = (p_lo + ((p_hi & 0xFFFF) << 16)) & 0xFFFFFFFF
    return high, low


def philox4x32(seed: torch.Tensor, c0, c1, c2, c3, rounds: int = 10):
    """Philox4x32 (``tl.philox``) on int64 tensors holding uint32 counters; ``seed`` int64."""
    mask = 0xFFFFFFFF
    k0 = seed & mask
    k1 = (seed >> 32) & mask
    for _ in range(rounds):
        hi_b, lo_b = u32_mul(c2, 0xCD9E8D57)
        hi_a, lo_a = u32_mul(c0, 0xD2511F53)
        c0, c1, c2, c3 = hi_b ^ c1 ^ k0, lo_b, hi_a ^ c3 ^ k1, lo_a
        k0 = (k0 + 0x9E3779B9) & mask
        k1 = (k1 + 0xBB67AE85) & mask
    return c0, c1, c2, c3


def gumbel(bits: torch.Tensor) -> torch.Tensor:
    """``-log(-log(u))`` with ``u = ((bits >> 9) + 0.5) / 2**23``, in float64."""
    u = ((bits >> 9).double() + 0.5) / 8388608.0
    return -torch.log(-torch.log(u))


def publish_pairs(layout, vals: torch.Tensor, col_base: int) -> torch.Tensor:
    """The pair buffer of ``layout`` (a ``vocab_parallel_kernels._PairLayout``) after each
    split of ``vals`` ``[rows, shard]`` published its argmax ``(value, col_base + column)`` at
    slot ``slot0 + split``, and the columns outside ``[zero_lo, zero_hi)`` were zeroed when
    ``zero_width > 0``. A clone; ``layout.buffer`` is not written."""
    buf = layout.buffer.clone()
    rows, n_cols = vals.shape
    if rows == 0:
        return buf
    if layout.zero_width > 0:
        cols = torch.arange(layout.zero_width, device=buf.device)
        outside = (cols < layout.zero_lo) | (cols >= layout.zero_hi)
        buf[:, : layout.zero_width][:, outside] = 0.0
    for split in range(layout.splits):
        lo = split * layout.chunk
        hi = min(lo + layout.chunk, n_cols)
        slot = layout.slot0 + split
        if lo >= hi:
            value = torch.full((rows,), float("-inf"), device=buf.device)
            index = torch.full((rows,), EMPTY_INDEX, device=buf.device)
        else:
            cols = torch.arange(lo, hi, device=buf.device).expand(rows, -1)
            value, index = argmax_pairs(vals[:, lo:hi], cols)
            index = index + col_base
        buf[:, 2 * slot] = value.to(buf.dtype)
        buf[:, 2 * slot + 1] = index.to(buf.dtype)
    return buf


def markov_chain_tokens(
    block_logits: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    vocab: int,
) -> torch.Tensor:
    """Greedy DSpark vanilla-Markov chain: ``[B, K]`` int32 tokens of the block logits
    ``[B, K, V]`` (``V`` the columns of ``markov_w2`` rows given). Step ``i`` adds the bias
    ``markov_w2 @ markov_w1[prev_i]`` (FP32 product rounded to the weights' dtype, as
    ``F.linear`` returns it) to position ``i`` in FP32 and takes the ``torch.argmax``; ``prev_0``
    is the anchor, and a ``prev`` outside ``[0, vocab)`` gives a zero bias."""
    num_reqs, block_size, width = block_logits.shape
    device = block_logits.device
    tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
    prev = anchors.to(torch.int64)
    cols = torch.arange(width, device=device).expand(num_reqs, -1)
    w2 = markov_w2.double()
    for step in range(block_size):
        valid = (prev >= 0) & (prev < vocab)
        row = markov_w1[prev.clamp(0, markov_w1.shape[0] - 1)].double() * valid[:, None]
        bias = (row @ w2.t()).to(markov_w2.dtype).float()
        _, prev = argmax_pairs(block_logits[:, step].float() + bias, cols)
        tokens[:, step] = prev.to(torch.int32)
    return tokens
