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
"""One-launch decode step: plan the seed, draw the keyed token, commit the slot state.

A plain decode step samples one token per request. For the rows that carry a watermark this
kernel does, in a single launch and one program per row:

1. derive the row's seed from its slot's context tail (and, with ``single_turn``
   deduplication, drop the keyed draw when the context was seen before),
2. take the Gumbel-max draw over the row's values, and
3. advance the slot's context tail, position count and repeat ring by the token it produced.

The values are either the raw logits of a row that applies no top-k, top-p or min-p filter
(temperature only), or the filtered probabilities of any other row. In both cases the keyed
noise is the one of :func:`tensorrt_llm.watermark.sampling.gumbel_sample_from_probs`: the
token is ``argmax_v ln(p_v) + G_v`` with ``G_v = -ln(-ln u_v)``, and a row that takes no
keyed draw keeps the token of the ordinary sampler it is given as ``fallback``.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .sampling import _philox_uniform
from .state_kernels import _context_seed

__all__ = ["StepScratch", "watermark_step"]

_BLOCK = 2048
_CHUNK = 16384  # vocabulary entries one program scans
_MAX_SPLITS = 32


_COMPILED: dict = {}


class StepScratch:
    """Per-row partial results of the split scan; ``counter`` is zero between launches."""

    def __init__(self, num_rows: int, device: torch.device):
        self.num_rows = num_rows
        self.value = torch.empty((num_rows, _MAX_SPLITS), dtype=torch.float32, device=device)
        self.index = torch.empty((num_rows, _MAX_SPLITS), dtype=torch.int32, device=device)
        self.counter = torch.zeros(num_rows, dtype=torch.int32, device=device)


@triton.jit
def _step_kernel(
    values_ptr,
    temperature_ptr,
    slots_ptr,
    enabled_ptr,
    fallback_ptr,
    out_ptr,
    tail_ptr,
    count_ptr,
    ring_ptr,
    keys_ptr,
    partial_value_ptr,
    partial_index_ptr,
    counter_ptr,
    stride_values,
    stride_tail,
    stride_ring,
    ring_size,
    vocab_size,
    NUM_SPLITS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    FROM_PROBS: tl.constexpr,
    DEDUPE: tl.constexpr,
    HAS_FALLBACK: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_RING: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    enabled = tl.load(enabled_ptr + row) != 0
    tail_row = tail_ptr + slot * stride_tail
    key = tl.load(keys_ptr).to(tl.uint64)
    seed = _context_seed(tail_row, tail_row, 0, key, WIDTH, 0)
    seed_signed = seed.to(tl.int64, bitcast=True)

    # Scan this program's slice of the row (a row without a watermark scans nothing).
    low = split * CHUNK
    high = tl.minimum(low + CHUNK, vocab_size)
    if not enabled:
        high = low
    key0 = (seed_signed & 0xFFFFFFFF).to(tl.uint32)
    key1 = ((seed_signed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    neg_inf = float("-inf")
    best_value = tl.full([BLOCK], neg_inf, dtype=tl.float32)
    best_index = tl.zeros([BLOCK], dtype=tl.int32)
    row_values = values_ptr + row.to(tl.int64) * stride_values
    if not FROM_PROBS:
        inv_temperature = 1.0 / tl.load(temperature_ptr + row)
    for start in range(low, high, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        in_range = offsets < high
        x = tl.load(row_values + offsets, mask=in_range, other=0.0).to(tl.float32)
        uniform = _philox_uniform(offsets.to(tl.uint32), key0, key1)
        if FROM_PROBS:
            # Same ordering as argmin(-ln(u) / p).
            value = tl.where(in_range & (x > 0.0), tl.log(uniform) / x, neg_inf)
        else:
            value = tl.where(in_range, x * inv_temperature - tl.log(-tl.log(uniform)), neg_inf)
        better = value > best_value
        best_value = tl.where(better, value, best_value)
        best_index = tl.where(better, offsets, best_index)
    highest = tl.max(best_value, axis=0)
    # Lowest index among exact ties, so the result does not depend on the lane layout.
    local = tl.min(tl.where(best_value == highest, best_index, 2147483647), axis=0)
    partial = row * MAX_SPLITS + split
    tl.store(partial_value_ptr + partial, highest)
    tl.store(partial_index_ptr + partial, local)
    tl.debug_barrier()
    # The last program of the row reduces the slices and commits.
    arrived = tl.atomic_add(counter_ptr + row, 1)
    if arrived == NUM_SPLITS - 1:
        tl.store(counter_ptr + row, 0)
        lane = tl.arange(0, BLOCK_SPLITS)
        lane_mask = lane < NUM_SPLITS
        values = tl.load(
            partial_value_ptr + row * MAX_SPLITS + lane,
            mask=lane_mask,
            other=neg_inf,
            volatile=True,
        )
        indices = tl.load(
            partial_index_ptr + row * MAX_SPLITS + lane,
            mask=lane_mask,
            other=2147483647,
            volatile=True,
        )
        best = tl.max(values, axis=0)
        keyed = tl.min(tl.where(values == best, indices, 2147483647), axis=0)

        count = tl.load(count_ptr + slot)
        active = enabled
        if DEDUPE:
            filled = tl.minimum(count, ring_size)
            ring_row = ring_ptr + slot * stride_ring
            seen = tl.zeros([BLOCK_RING], dtype=tl.int32)
            for start in range(0, ring_size, BLOCK_RING):
                offsets = start + tl.arange(0, BLOCK_RING)
                ring_values = tl.load(ring_row + offsets, mask=offsets < filled, other=0)
                seen |= ((ring_values == seed_signed) & (offsets < filled)).to(tl.int32)
            active = active & (tl.max(seen, axis=0) == 0)
        token = keyed.to(tl.int32)
        if HAS_FALLBACK:
            token = tl.where(active, token, tl.load(fallback_ptr + row))
        tl.store(out_ptr + row, token)

        # Commit: advance the slot by the token this row produced.
        if enabled:
            tl.debug_barrier()
            tail_lane = tl.arange(0, BLOCK_WIDTH)
            shifted = tl.load(tail_row + tail_lane + 1, mask=tail_lane + 1 < WIDTH, other=0)
            advanced = tl.where(tail_lane == WIDTH - 1, token.to(tl.int64), shifted)
            tl.debug_barrier()
            tl.store(tail_row + tail_lane, advanced, mask=tail_lane < WIDTH)
            if DEDUPE:
                tl.store(ring_ptr + slot * stride_ring + count % ring_size, seed_signed)
            tl.store(count_ptr + slot, count + 1)


def watermark_step(
    values: torch.Tensor,
    temperature: Optional[torch.Tensor],
    slots: torch.Tensor,
    enabled: torch.Tensor,
    fallback: Optional[torch.Tensor],
    tail: torch.Tensor,
    count: torch.Tensor,
    ring: Optional[torch.Tensor],
    ring_size: int,
    keys: torch.Tensor,
    scratch: StepScratch,
    *,
    from_probs: bool,
) -> torch.Tensor:
    """One decode step for ``R`` rows. Device-only; safe to capture in a CUDA graph.

    Args:
        values: ``[R, V]`` raw logits (``from_probs=False``; any float dtype) or filtered
            probabilities (``from_probs=True``, float32).
        temperature: ``[R]`` float32 temperatures; read only for logits.
        slots: ``[R]`` int64 sequence slots. Rows of one call must be on distinct slots.
        enabled: ``[R]`` bool, rows that carry a watermark. The others leave their slot
            untouched and return their ``fallback`` token.
        fallback: ``[R]`` int32 tokens of the ordinary sampler, for rows that do not carry
            a watermark and for repeated contexts; ``None`` when every row is enabled and
            no deduplication is configured.
        tail, count, ring, ring_size, keys: the slot state of
            :class:`~tensorrt_llm._torch.pyexecutor.sampler.watermark.WatermarkHandler`.
        scratch: partial-result buffers with at least ``R`` rows.

    Returns:
        ``[R]`` int32 tokens.
    """
    assert values.dim() == 2
    if values.stride(1) != 1:
        values = values.contiguous()
    num_rows, vocab_size = values.shape
    num_splits = triton.cdiv(vocab_size, _CHUNK)
    assert num_splits <= _MAX_SPLITS and num_rows <= scratch.num_rows
    out = torch.empty(num_rows, dtype=torch.int32, device=values.device)
    width = tail.shape[1]
    runtime_args = (
        values,
        temperature if temperature is not None else values,
        slots,
        enabled,
        fallback if fallback is not None else out,
        out,
        tail,
        count,
        ring if ring is not None else tail,
        keys,
        scratch.value,
        scratch.index,
        scratch.counter,
        values.stride(0),
        tail.stride(0),
        ring.stride(0) if ring is not None else 0,
        ring_size,
        vocab_size,
    )
    constexprs = dict(
        NUM_SPLITS=num_splits,
        WIDTH=width,
        BLOCK_WIDTH=triton.next_power_of_2(width),
        FROM_PROBS=from_probs,
        DEDUPE=ring is not None,
        HAS_FALLBACK=fallback is not None,
        BLOCK=_BLOCK,
        CHUNK=_CHUNK,
        BLOCK_RING=1024,
        MAX_SPLITS=_MAX_SPLITS,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    # Triton specializes a launch on its integer arguments and on pointer alignment. A decode
    # step repeats the same specialization, so compile once per key and launch the compiled
    # kernel directly, which skips the per-launch argument inspection (tens of microseconds).
    key = (
        values.dtype,
        values.data_ptr() % 16 == 0,
        tuple(runtime_args[13:]),
        tuple(constexprs.values()),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        # The first launch goes through the normal path and hands back the compiled kernel.
        _COMPILED[key] = _step_kernel[(num_rows, num_splits)](*runtime_args, **constexprs)
    else:
        compiled[(num_rows, num_splits, 1)](*runtime_args)
    return out
