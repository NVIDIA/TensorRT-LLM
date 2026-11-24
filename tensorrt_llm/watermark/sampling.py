# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keyed Gumbel-max token selection.

Given a row's probabilities ``p`` (already filtered by temperature, top-k, top-p and
min-p) and a per-row seed, the token is ``argmin_v (-ln u_v) / p_v`` where
``u_v = philox_uniform(seed, v)``. With ``u_v`` uniform and independent this is an exact
sample from ``p``; with ``u_v`` derived from a secret key it is deterministic given the
context, which is what the detector exploits.

The random numbers are integer-derived and bit-identical to ``prf.philox_uniform``. Only
the argmin involves floating point, and the detector never recomputes it.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .prf import philox_uniform_torch

__all__ = ["gumbel_sample_from_probs", "gumbel_sample_reference", "keyed_recovery"]

_BLOCK = 1024
_INT32_MAX = 2147483647


@triton.jit
def _philox_uniform(token, key0, key1):
    """First Philox4x32-10 output word for counter ``(token, 0, 0, 0)`` mapped to (0, 1)."""
    c0 = token
    c1 = tl.zeros_like(token)
    c2 = tl.zeros_like(token)
    c3 = tl.zeros_like(token)
    k0 = key0
    k1 = key1
    for _ in tl.static_range(10):
        hi0 = tl.umulhi(c0, 0xD2511F53)
        lo0 = c0 * 0xD2511F53
        hi1 = tl.umulhi(c2, 0xCD9E8D57)
        lo1 = c2 * 0xCD9E8D57
        n0 = hi1 ^ c1 ^ k0
        n2 = hi0 ^ c3 ^ k1
        c0 = n0
        c1 = lo1
        c2 = n2
        c3 = lo0
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85
    return ((c0 >> 9) * 2 + 1).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _gumbel_scan_kernel(
    probs_ptr,
    seeds_ptr,
    active_ptr,
    id_map_ptr,
    out_ptr,
    partial_score_ptr,
    partial_index_ptr,
    vocab_size,
    stride_row,
    chunk,
    num_splits,
    HAS_ACTIVE: tl.constexpr,
    HAS_ID_MAP: tl.constexpr,
    WRITE_PARTIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    # An inactive row scans nothing and reports -1.
    limit = vocab_size
    if HAS_ACTIVE:
        limit = tl.where(tl.load(active_ptr + row) != 0, vocab_size, 0)
    low = split * chunk
    high = tl.minimum(low + chunk, limit)
    seed = tl.load(seeds_ptr + row)
    key0 = (seed & 0xFFFFFFFF).to(tl.uint32)
    key1 = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    inf = float("inf")
    best_score = tl.full([BLOCK], inf, dtype=tl.float32)
    best_index = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(low, high, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        in_range = offsets < high
        prob = tl.load(
            probs_ptr + row.to(tl.int64) * stride_row + offsets, mask=in_range, other=0.0
        )
        counters = offsets
        if HAS_ID_MAP:
            counters = tl.load(id_map_ptr + offsets, mask=in_range, other=0)
        uniform = _philox_uniform(counters.to(tl.uint32), key0, key1)
        score = tl.where(in_range & (prob > 0.0), -tl.log(uniform) / prob, inf)
        better = score < best_score
        best_score = tl.where(better, score, best_score)
        best_index = tl.where(better, offsets, best_index)
    lowest = tl.min(best_score, axis=0)
    # Lowest index among exact ties, so the result does not depend on how lanes were laid out.
    winner = tl.min(tl.where(best_score == lowest, best_index, 2147483647), axis=0)
    if WRITE_PARTIAL:
        tl.store(partial_score_ptr + row * num_splits + split, lowest)
        tl.store(partial_index_ptr + row * num_splits + split, winner)
    else:
        tl.store(out_ptr + row, tl.where(limit > 0, winner, -1))


@triton.jit
def _gumbel_reduce_kernel(
    partial_score_ptr,
    partial_index_ptr,
    active_ptr,
    out_ptr,
    num_splits,
    HAS_ACTIVE: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SPLITS)
    in_range = lane < num_splits
    score = tl.load(partial_score_ptr + row * num_splits + lane, mask=in_range, other=float("inf"))
    index = tl.load(partial_index_ptr + row * num_splits + lane, mask=in_range, other=2147483647)
    lowest = tl.min(score, axis=0)
    winner = tl.min(tl.where(score == lowest, index, 2147483647), axis=0)
    if HAS_ACTIVE:
        winner = tl.where(tl.load(active_ptr + row) != 0, winner, -1)
    tl.store(out_ptr + row, winner)


_SM_COUNT: dict[int, int] = {}


def _num_splits(device: torch.device, num_rows: int, vocab_size: int) -> int:
    """How many programs share one row, so the scan fills the device even for small batches."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    target_programs = 2 * _SM_COUNT[index]
    splits = max(1, min(32, -(-target_programs // max(num_rows, 1))))
    return min(splits, max(1, vocab_size // (4 * _BLOCK)))


def gumbel_sample_from_probs(
    probs: torch.Tensor,
    seeds: torch.Tensor,
    active: Optional[torch.Tensor] = None,
    id_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample one token per row with the keyed Gumbel-max rule.

    Args:
        probs: ``[R, V]`` float32 probabilities (rows sum to 1, zeros allowed).
        seeds: ``[R]`` int64 per-row seeds from ``prf.context_seeds_torch``.
        active: optional ``[R]`` bool; rows that are False are skipped and yield -1.
        id_map: optional ``[V]`` integer tensor giving the id each column takes in the random
            stream (the target-vocabulary id of a draft-vocabulary column). The returned
            tokens are still column indices.

    Returns:
        ``[R]`` int32 token ids.
    """
    assert probs.dim() == 2 and probs.dtype == torch.float32
    assert seeds.shape == (probs.shape[0],) and seeds.dtype == torch.int64
    assert probs.stride(1) == 1
    if id_map is not None:
        assert id_map.shape == (probs.shape[1],) and not id_map.dtype.is_floating_point
    if active is not None:
        assert active.shape == seeds.shape and active.dtype == torch.bool
    num_rows, vocab_size = probs.shape
    out = torch.empty(num_rows, dtype=torch.int32, device=probs.device)
    if num_rows == 0:
        return out
    splits = _num_splits(probs.device, num_rows, vocab_size)
    chunk = -(-vocab_size // splits)
    chunk = -(-chunk // _BLOCK) * _BLOCK
    splits = -(-vocab_size // chunk)
    if splits > 1:
        partial_score = torch.empty((num_rows, splits), dtype=torch.float32, device=probs.device)
        partial_index = torch.empty((num_rows, splits), dtype=torch.int32, device=probs.device)
    else:
        partial_score = partial_index = out
    _gumbel_scan_kernel[(num_rows, splits)](
        probs,
        seeds,
        active if active is not None else seeds,
        id_map if id_map is not None else seeds,
        out,
        partial_score,
        partial_index,
        vocab_size,
        probs.stride(0),
        chunk,
        splits,
        HAS_ACTIVE=active is not None,
        HAS_ID_MAP=id_map is not None,
        WRITE_PARTIAL=splits > 1,
        BLOCK=_BLOCK,
    )
    if splits > 1:
        _gumbel_reduce_kernel[(num_rows,)](
            partial_score,
            partial_index,
            active if active is not None else seeds,
            out,
            splits,
            HAS_ACTIVE=active is not None,
            BLOCK_SPLITS=triton.next_power_of_2(splits),
        )
    return out.clamp_(max=vocab_size - 1)


def gumbel_sample_reference(
    probs: torch.Tensor, seeds: torch.Tensor, id_map: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pure-torch float64 reference of ``gumbel_sample_from_probs`` (test oracle)."""
    vocab = torch.arange(probs.shape[1], device=probs.device) if id_map is None else id_map
    uniform = philox_uniform_torch(seeds, vocab).to(torch.float64)
    scores = -torch.log(uniform) / probs.to(torch.float64)
    scores = torch.where(probs > 0, scores, torch.full_like(scores, float("inf")))
    return scores.argmin(dim=1).to(torch.int32)


def keyed_recovery(
    accepted_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    seeds: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Redraw the token that follows the accepted drafts, as a keyed Gumbel-max draw.

    Rejection sampling accepts a prefix of the draft and then emits one more token: a sample
    from ``norm(max(0, p_target - p_draft))`` at the first rejected position, or from the
    target itself when every draft token was accepted. That token is the last of the
    ``num_accepted`` committed ones. Drawing it with a keyed Gumbel-max rule instead of the
    kernel's inverse-CDF draw watermarks it; the accept decisions are untouched.

    The output distribution stays the target's when ``seeds`` are independent of whatever
    keyed the draft tokens (the accept coins are independent of both), which is why dual-key
    watermarking uses a second key here.

    Args:
        accepted_tokens: ``[B, K + 1]`` int32 tokens from the rejection kernel.
        num_accepted: ``[B]`` committed tokens per request (at least 1).
        target_probs: ``[B, K + 1, V]`` float32 target distributions.
        draft_probs: ``[B, K, V]`` float32 draft distributions (target vocabulary).
        seeds: ``[B, K + 1]`` int64 per-position seeds of the key used for this token.
        active: ``[B, K + 1]`` bool; positions that do not carry a watermark keep the kernel's token.

    Returns:
        ``[B, K + 1]`` tokens with the last committed token of each request redrawn.
    """
    batch, window, vocab = target_probs.shape
    num_draft = window - 1
    position = (num_accepted.to(torch.int64) - 1).clamp(0, num_draft)
    target = target_probs.gather(1, position.view(batch, 1, 1).expand(batch, 1, vocab)).squeeze(1)
    if num_draft > 0:
        index = position.clamp(max=num_draft - 1)
        draft = draft_probs.gather(1, index.view(batch, 1, 1).expand(batch, 1, vocab)).squeeze(1)
        draft = torch.where((position < num_draft).view(batch, 1), draft, torch.zeros_like(draft))
    else:
        draft = torch.zeros_like(target)
    residual = (target - draft).clamp(min=0)
    # The kernel falls back to the target when the residual has no mass.
    residual = torch.where(residual.sum(dim=-1, keepdim=True) > 0, residual, target)
    row_seed = seeds.gather(1, position.unsqueeze(1)).squeeze(1)
    row_active = active.gather(1, position.unsqueeze(1)).squeeze(1)
    keyed = gumbel_sample_from_probs(residual.contiguous(), row_seed, row_active)
    current = accepted_tokens.gather(1, position.unsqueeze(1)).squeeze(1)
    redrawn = torch.where(row_active, keyed, current)
    out = accepted_tokens.clone()
    out.scatter_(1, position.unsqueeze(1), redrawn.unsqueeze(1).to(out.dtype))
    return out
