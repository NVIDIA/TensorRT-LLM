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
"""Fused kernels for the per-slot watermark state.

The two kernels replace what would otherwise be dozens of tiny torch operations per sampling
step (the context hash alone is a chain of integer operations per context token):

* :func:`plan_windows` derives, for ``B`` requests and ``K1`` consecutive positions each, the
  seed of every position's context (and of the second key), and whether the position takes the
  keyed draw once repeated contexts are sent back to ordinary sampling.
* :func:`commit_windows` advances the per-slot state (context tail, repeat ring, position
  count) by the tokens that were committed.

The hash is the one of :mod:`.prf`; the unit tests check both kernels against the plain torch
implementation in ``tensorrt_llm._torch.pyexecutor.sampler.watermark``.
"""

import torch
import triton
import triton.language as tl

__all__ = ["plan_windows", "commit_windows"]


@triton.jit
def _mix64(x):
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    return x ^ (x >> 31)


@triton.jit
def _context_seed(
    tail_row_ptr, drafts_row_ptr, position, key, WIDTH: tl.constexpr, K: tl.constexpr
):
    """Seed of the window position ``position``: hash of ``(tail ++ drafts)[position : position + WIDTH]``."""
    seed = key
    for c in tl.static_range(WIDTH):
        index = position + c
        from_tail = index < WIDTH
        token = tl.load(tail_row_ptr + tl.minimum(index, WIDTH - 1), mask=from_tail, other=0)
        if K > 0:
            draft = tl.load(
                drafts_row_ptr + tl.maximum(index - WIDTH, 0), mask=index >= WIDTH, other=0
            )
            token = tl.where(from_tail, token, draft)
        token = token.to(tl.uint64) & 0xFFFFFFFF
        seed = _mix64((seed ^ token) + 0x9E3779B97F4A7C15)
    return seed


@triton.jit
def _plan_kernel(
    slots_ptr,
    enabled_ptr,
    tail_ptr,
    drafts_ptr,
    count_ptr,
    ring_ptr,
    seeds_ptr,
    seeds2_ptr,
    active_ptr,
    keys_ptr,
    stride_tail,
    stride_drafts,
    stride_ring,
    ring_size,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    DEDUPE: tl.constexpr,
    DUAL: tl.constexpr,
    BLOCK_RING: tl.constexpr,
):
    request = tl.program_id(0)
    position = tl.program_id(1)
    num_positions = K + 1
    slot = tl.load(slots_ptr + request).to(tl.int64)
    tail_row = tail_ptr + slot * stride_tail
    drafts_row = drafts_ptr + request.to(tl.int64) * stride_drafts
    key = tl.load(keys_ptr).to(tl.uint64)
    seed = _context_seed(tail_row, drafts_row, position, key, WIDTH, K)
    seed_signed = seed.to(tl.int64, bitcast=True)
    out = request * num_positions + position
    tl.store(seeds_ptr + out, seed_signed)
    if DUAL:
        key2 = tl.load(keys_ptr + 1).to(tl.uint64)
        seed2 = _context_seed(tail_row, drafts_row, position, key2, WIDTH, K)
        tl.store(seeds2_ptr + out, seed2.to(tl.int64, bitcast=True))
    active = tl.load(enabled_ptr + request) != 0
    if DEDUPE:
        filled = tl.minimum(tl.load(count_ptr + slot), ring_size)
        ring_row = ring_ptr + slot * stride_ring
        seen = tl.zeros([BLOCK_RING], dtype=tl.int32)
        for start in range(0, ring_size, BLOCK_RING):
            offsets = start + tl.arange(0, BLOCK_RING)
            values = tl.load(ring_row + offsets, mask=offsets < filled, other=0)
            seen |= ((values == seed_signed) & (offsets < filled)).to(tl.int32)
        repeated = tl.max(seen, axis=0) != 0
        # A context equal to the context of an earlier position of the same window.
        for earlier in range(0, position):
            other = _context_seed(tail_row, drafts_row, earlier, key, WIDTH, K)
            repeated = repeated | (other == seed)
        active = active & (~repeated)
    tl.store(active_ptr + out, active.to(tl.int8))


def plan_windows(
    slots: torch.Tensor,
    enabled: torch.Tensor,
    tail: torch.Tensor,
    count: torch.Tensor,
    ring: torch.Tensor | None,
    ring_size: int,
    keys: torch.Tensor,
    dual: bool,
    drafts: torch.Tensor | None,
):
    """Seeds and activity of ``B`` windows of ``K + 1`` positions.

    Args:
        slots: ``[B]`` int64 sequence slots.
        enabled: ``[B]`` bool.
        tail: ``[rows, width]`` int64 per-slot context tail.
        count: ``[rows]`` int64 positions recorded per slot.
        ring: ``[rows, ring_size + 1]`` int64 repeat ring (``None`` disables the repeat check).
        keys: ``[2]`` int64 tensor holding the primary and the second key (bit patterns of
            values in ``[0, 2**64)``); the second is used only when ``dual``.
        dual: whether to also return the seeds of the second key.
        drafts: optional ``[B, K]`` draft tokens the window verifies.

    Returns:
        ``(seeds [B, K+1] int64, seeds2 [B, K+1] int64 or None, active [B, K+1] bool)``.
    """
    batch = slots.shape[0]
    num_drafts = 0 if drafts is None else drafts.shape[1]
    width = tail.shape[1]
    seeds = torch.empty((batch, num_drafts + 1), dtype=torch.int64, device=slots.device)
    seeds2 = torch.empty_like(seeds) if dual else seeds
    active = torch.empty((batch, num_drafts + 1), dtype=torch.int8, device=slots.device)
    if drafts is not None:
        drafts = drafts.to(torch.int64).contiguous()
    _plan_kernel[(batch, num_drafts + 1)](
        slots,
        enabled,
        tail,
        drafts if drafts is not None else tail,
        count,
        ring if ring is not None else tail,
        seeds,
        seeds2,
        active,
        keys,
        tail.stride(0),
        drafts.stride(0) if drafts is not None else 0,
        ring.stride(0) if ring is not None else 0,
        ring_size,
        WIDTH=width,
        K=num_drafts,
        DEDUPE=ring is not None,
        DUAL=dual,
        BLOCK_RING=1024,
    )
    return seeds, (seeds2 if dual else None), active.to(torch.bool)


@triton.jit
def _commit_kernel(
    slots_ptr,
    enabled_ptr,
    accepted_ptr,
    num_accepted_ptr,
    seeds_ptr,
    tail_ptr,
    count_ptr,
    ring_ptr,
    stride_accepted,
    stride_tail,
    stride_ring,
    ring_size,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    K1: tl.constexpr,
    DEDUPE: tl.constexpr,
):
    request = tl.program_id(0)
    slot = tl.load(slots_ptr + request).to(tl.int64)
    enabled = tl.load(enabled_ptr + request) != 0
    committed = tl.where(enabled, tl.load(num_accepted_ptr + request).to(tl.int64), 0)
    # New tail: the last WIDTH tokens of (old tail ++ the committed tokens).
    lane = tl.arange(0, BLOCK_WIDTH)
    lane_mask = lane < WIDTH
    index = committed + lane
    tail_row = tail_ptr + slot * stride_tail
    old = tl.load(
        tail_row + tl.minimum(index, WIDTH - 1), mask=lane_mask & (index < WIDTH), other=0
    )
    new = tl.load(
        accepted_ptr + request.to(tl.int64) * stride_accepted + tl.maximum(index - WIDTH, 0),
        mask=lane_mask & (index >= WIDTH),
        other=0,
    ).to(tl.int64)
    advanced = tl.where(index < WIDTH, old, new)
    count = tl.load(count_ptr + slot)
    tl.store(tail_row + lane, advanced, mask=lane_mask)
    if DEDUPE:
        ring_row = ring_ptr + slot * stride_ring
        for position in tl.static_range(K1):
            seed = tl.load(seeds_ptr + request * K1 + position)
            target = (count + position) % ring_size
            tl.store(ring_row + target, seed, mask=position < committed)
    tl.store(count_ptr + slot, count + committed)


def commit_windows(
    slots: torch.Tensor,
    enabled: torch.Tensor,
    accepted_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    seeds: torch.Tensor,
    tail: torch.Tensor,
    count: torch.Tensor,
    ring: torch.Tensor | None,
    ring_size: int,
) -> None:
    """Advance the per-slot state by the committed tokens of ``B`` windows.

    ``accepted_tokens`` is ``[B, K1]`` (the tokens of each window), ``num_accepted`` ``[B]`` how
    many of them were committed and ``seeds`` the ``[B, K1]`` seeds :func:`plan_windows`
    returned. Requests that are not ``enabled`` leave their slot untouched. Slots must be
    distinct among the enabled requests.
    """
    batch, num_positions = seeds.shape
    accepted_tokens = accepted_tokens.contiguous()
    width = tail.shape[1]
    _commit_kernel[(batch,)](
        slots,
        enabled,
        accepted_tokens,
        num_accepted,
        seeds,
        tail,
        count,
        ring if ring is not None else tail,
        accepted_tokens.stride(0),
        tail.stride(0),
        ring.stride(0) if ring is not None else 0,
        ring_size,
        WIDTH=width,
        BLOCK_WIDTH=triton.next_power_of_2(width),
        K1=num_positions,
        DEDUPE=ring is not None,
    )
