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
"""Bit-exact host model of the M3.1 ungrouped noaux_tc selection and weights.

Models ``deepseek_v3_topk_block<float, float, float, int32_t, 128, false, 32>``
(``noAuxTcKernels.cu``) and ``reduce_topk::reduceTopK``
(``moeTopKFuncs.cuh``) from the per-expert sigmoid onward. The sigmoid itself
(``0.5f * tanhf(0.5f * x) + 0.5f``) uses the device ``tanhf`` and is an input.

* Selection key: ``(cub TwiddleIn(float bits) << 32) | (65535 - expert)``,
  largest first. Equal values therefore go to the lower expert index, ``+0``
  ranks above ``-0``, and a positive NaN ranks above ``+inf``.
* Slot ``k`` of the output is the k-th largest key; TopkReduce sums in this
  slot order.
* ``redNorm`` is a 32-lane ``cg::reduce`` with lanes ``>= top_k`` holding 0.
* The weight is evaluated in fp64 (``routedScalingFactor`` is a double) and
  then rounded to binary32.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from _torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.exact_arith import (
    add_f32,
    f32,
    f32_bits,
)

WARP_SIZE = 32
MAX_TOP_K = 32
_MAX_IDX = 65535


def _is_f32(x: float) -> bool:
    return math.isnan(x) or f32(x) == x


def twiddle_in(x: float) -> int:
    """cub::Traits<float>::TwiddleIn: maps float order onto unsigned order."""
    bits = f32_bits(x)
    return (~bits) & 0xFFFFFFFF if bits & 0x80000000 else bits | 0x80000000


def selection_key(value: float, expert: int) -> int:
    return (twiddle_in(value) << 32) | (0xFFFF & (_MAX_IDX - expert))


def warp_sum_shfl_down(lanes: Sequence[float]) -> float:
    """Lane-0 result of a binary32 shuffle-down tree with offsets 16, 8, 4, 2, 1."""
    if len(lanes) != WARP_SIZE:
        raise ValueError(f"expected {WARP_SIZE} lanes")
    values = list(lanes)
    offset = WARP_SIZE // 2
    while offset:
        values = [
            add_f32(values[i], values[i + offset]) if i + offset < WARP_SIZE else values[i]
            for i in range(WARP_SIZE)
        ]
        offset //= 2
    return values[0]


def warp_sum_shfl_xor_ascending(lanes: Sequence[float]) -> float:
    """Lane-0 result of a binary32 butterfly with offsets 1, 2, 4, 8, 16."""
    if len(lanes) != WARP_SIZE:
        raise ValueError(f"expected {WARP_SIZE} lanes")
    values = list(lanes)
    offset = 1
    while offset < WARP_SIZE:
        values = [add_f32(values[i], values[i ^ offset]) for i in range(WARP_SIZE)]
        offset *= 2
    return values[0]


def noaux_topk(
    sigmoid: Sequence[float],
    bias: Sequence[float],
    top_k: int,
    routed_scaling_factor: float,
    warp_sum: Callable[[Sequence[float]], float] = warp_sum_shfl_down,
) -> tuple[list[int], list[float]]:
    """Return ``(experts, weights)`` in slot order for one token.

    ``sigmoid`` and ``bias`` are binary32 values per expert. ``warp_sum`` is
    the ``cg::reduce`` lane order; the shuffle-down default is to be confirmed
    by the GPU oracle against the device kernel.
    """
    num_experts = len(sigmoid)
    if len(bias) != num_experts:
        raise ValueError("sigmoid and bias must have one value per expert")
    if not 0 < top_k <= min(MAX_TOP_K, num_experts):
        raise ValueError(f"top_k={top_k} outside [1, min({MAX_TOP_K}, {num_experts})]")
    if num_experts > _MAX_IDX:
        raise ValueError("expert index does not fit the 16-bit selection key")
    if any(not _is_f32(v) for v in (*sigmoid, *bias)):
        raise ValueError("sigmoid and bias must be binary32 values")
    biased = [add_f32(sigmoid[e], bias[e]) for e in range(num_experts)]
    order = sorted(range(num_experts), key=lambda e: selection_key(biased[e], e), reverse=True)
    experts = order[:top_k]

    lanes = [sigmoid[e] for e in experts] + [0.0] * (WARP_SIZE - top_k)
    red_norm = warp_sum(lanes)
    denominator = red_norm + 1e-20
    weights = [f32((sigmoid[e] * routed_scaling_factor) / denominator) for e in experts]
    return experts, weights


def expert_histogram(topk_experts: Sequence[Sequence[int]], num_experts: int) -> list[int]:
    """Rows routed to each expert; experts nobody selected stay at zero."""
    counts = [0] * num_experts
    for row in topk_experts:
        for expert in row:
            if not 0 <= expert < num_experts:
                raise ValueError(f"expert {expert} outside [0, {num_experts})")
            counts[expert] += 1
    return counts
