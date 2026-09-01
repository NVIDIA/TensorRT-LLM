# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused argmax + scatter for greedy sampling.

``torch.argmax`` over a vocabulary row, the cast to the token buffer's dtype and
the scatter into that buffer are four device operations (a zero-fill, the
reduction, the cast and the scatter). At decode batch sizes only one of them
does real work, and the reduction itself runs on a single CTA because there is
one row to reduce -- so the vocabulary is walked by one block while the rest of
the GPU idles, and the host pays three more launches for the surrounding
bookkeeping.

This splits the reduction across a fixed number of blocks per row and folds the
cast and the scatter into the merge, leaving two launches whose combined grid
covers the device.

Ordering matches ``torch.argmax``: the lowest index among equal maxima wins.
That is obtained without a second comparison pass by reducing a single packed
key per candidate -- the value in the high half, the complemented index in the
low half -- so one integer maximum resolves both.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["greedy_argmax_scatter", "supports_greedy_argmax_scatter", "ARGMAX_SPLITS"]

#: Blocks the vocabulary reduction is split across, per row. Sized so a
#: single-row decode step fills a meaningful fraction of the device while each
#: block still has enough elements to amortize its own launch. Must be a power
#: of two: the merge reduces exactly this many candidates in one tile.
ARGMAX_SPLITS = 64

# Triton only lets a kernel read module globals that are constexpr.
_INT64_MIN = tl.constexpr(-9223372036854775808)
_UINT32_MAX = tl.constexpr(4294967295)
_SIGN_BIT = tl.constexpr(2147483648)


@triton.jit
def _pack_key(values, indices, vocab_size):
    """Pack (value, index) into one int64 whose maximum is the argmax.

    The float is mapped to an order-preserving 32-bit unsigned pattern (set the
    sign bit for non-negatives, invert everything for negatives), so integer
    comparison agrees with float comparison -- including -inf and NaN, which
    keep the ordering ``torch.max`` gives them. It is then re-biased into signed
    range so the packed key stays a well-ordered int64.

    The index occupies the low half, counted down from ``vocab_size``, so that
    among equal values the *smallest* index yields the largest key --
    ``torch.argmax``'s first-occurrence rule.
    """
    bits = values.to(tl.int32, bitcast=True)
    unsigned = bits.to(tl.int64) & _UINT32_MAX
    ordered = tl.where(bits >= 0, unsigned + _SIGN_BIT, _UINT32_MAX - unsigned)
    return ((ordered - _SIGN_BIT) << 32) | (vocab_size - 1 - indices).to(tl.int64)


@triton.jit
def _argmax_partial_kernel(
    logits_ptr,
    partials_ptr,
    stride_logits_row,
    stride_partials_row,
    vocab_size,
    chunk_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    start = split * chunk_size
    stop = tl.minimum(start + chunk_size, vocab_size)

    # Accumulate lane-wise and reduce once at the end, rather than reducing
    # every tile.
    best = tl.full([BLOCK], _INT64_MIN, tl.int64)
    row_ptr = logits_ptr + row * stride_logits_row
    for base in range(start, stop, BLOCK):
        offsets = base + tl.arange(0, BLOCK)
        mask = offsets < stop
        values = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        # Masked lanes carry a valid index so the packed key stays well formed;
        # they are discarded by the where rather than by their value.
        keys = _pack_key(values, tl.where(mask, offsets, 0), vocab_size)
        best = tl.maximum(best, tl.where(mask, keys, _INT64_MIN))

    tl.store(partials_ptr + row * stride_partials_row + split, tl.max(best, axis=0))


@triton.jit
def _argmax_merge_scatter_kernel(
    partials_ptr,
    dest_ptr,
    new_tokens_ptr,
    next_tokens_ptr,
    stride_partials_row,
    stride_new_tokens_row,
    stride_new_tokens_beam,
    vocab_size,
    splits: tl.constexpr,
    beam_width: tl.constexpr,
    BEAM_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    keys = tl.load(partials_ptr + row * stride_partials_row + tl.arange(0, splits))
    best = tl.max(keys, axis=0)
    token = (vocab_size - 1 - (best & _UINT32_MAX)).to(next_tokens_ptr.dtype.element_ty)

    tl.store(next_tokens_ptr + row, token)

    dest = tl.load(dest_ptr + row).to(tl.int64)
    beams = tl.arange(0, BEAM_BLOCK)
    # Broadcast the scalar token across the beam lanes explicitly.
    broadcast = token.to(new_tokens_ptr.dtype.element_ty) + tl.zeros(
        [BEAM_BLOCK], dtype=new_tokens_ptr.dtype.element_ty
    )
    tl.store(
        new_tokens_ptr + dest * stride_new_tokens_row + beams * stride_new_tokens_beam,
        broadcast,
        mask=beams < beam_width,
    )


def supports_greedy_argmax_scatter(logits: torch.Tensor, new_tokens: torch.Tensor) -> bool:
    """Check the layouts the fused path indexes directly.

    The reduction walks each logits row with unit stride and the scatter treats
    ``new_tokens`` as ``[row, beam]``; anything else keeps the tensor-op path.
    """
    return (
        logits.is_cuda
        and logits.dim() == 2
        and logits.stride(1) == 1
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and new_tokens.dim() == 3
    )


def greedy_argmax_scatter(
    logits: torch.Tensor,
    new_tokens: torch.Tensor,
    dest_indices: torch.Tensor,
    beam_width: int,
    out: torch.Tensor | None = None,
    partials: torch.Tensor | None = None,
) -> torch.Tensor:
    """Argmax each row of ``logits`` and scatter the result into ``new_tokens``.

    Equivalent to::

        next_tokens = torch.argmax(logits, dim=-1).to(new_tokens.dtype)
        new_tokens.view(-1, *new_tokens.shape[2:]).scatter_(
            0,
            dest_indices.unsqueeze(1).expand(-1, beam_width),
            next_tokens.unsqueeze(1).expand(-1, beam_width),
        )

    Args:
        logits: Shape ``[num_rows, vocab_size]``.
        new_tokens: Token buffer, ``[max_tokens, num_slots, beam_width]``.
        dest_indices: Flat row of ``new_tokens.view(-1, beam_width)`` each
          logits row writes to. Shape ``[num_rows]``.
        beam_width: Beams each row is broadcast across.
        out: Destination for the sampled tokens. Allocated when omitted.
        partials: Scratch for the split reduction, ``[num_rows,
          ARGMAX_SPLITS]`` int64. Allocated when omitted. Both are exposed so
          a caller replaying this sequence from a CUDA graph can supply
          buffers that outlive the capture.

    Returns:
        The sampled tokens, shape ``[num_rows]``, dtype of ``new_tokens``.
    """
    num_rows, vocab_size = logits.shape
    flat_tokens = new_tokens.view(-1, *new_tokens.shape[2:])
    next_tokens = (
        torch.empty(num_rows, dtype=new_tokens.dtype, device=new_tokens.device)
        if out is None
        else out
    )
    # Scratch for the split reduction. Every entry is written before it is read,
    # so it needs no initialization; at 8 bytes per split this is a caching
    # allocator hit rather than a device operation.
    if partials is None:
        partials = torch.empty((num_rows, ARGMAX_SPLITS), dtype=torch.int64, device=logits.device)

    _argmax_partial_kernel[(num_rows, ARGMAX_SPLITS)](
        logits,
        partials,
        logits.stride(0),
        partials.stride(0),
        vocab_size,
        triton.cdiv(vocab_size, ARGMAX_SPLITS),
        BLOCK=1024,
        num_warps=8,
    )
    _argmax_merge_scatter_kernel[(num_rows,)](
        partials,
        dest_indices,
        flat_tokens,
        next_tokens,
        partials.stride(0),
        flat_tokens.stride(0),
        flat_tokens.stride(1),
        vocab_size,
        splits=ARGMAX_SPLITS,
        beam_width=beam_width,
        BEAM_BLOCK=max(16, triton.next_power_of_2(beam_width)),
        num_warps=2,
    )
    return next_tokens
