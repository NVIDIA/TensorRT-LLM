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

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Initialize exhaustive candidates when the physical key domain is small."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from ..pdl import pdl_prologue
from .selection import candidate_score, score_key, key_score


class ShortDomainSetup:
    def __init__(self, heads, safe, fp4, prepare_sf, pdl):
        self.heads = heads
        self.safe = safe
        self.fp4 = fp4
        self.prepare_sf = prepare_sf
        self.pdl = pdl

    @cute.jit
    def launch(self, scales, weights, thresholds, counts, samples, queue, m, n, stream):
        flat_weights = cute.make_tensor(weights.iterator, cute.make_layout(m * self.heads))
        self.kernel(scales, flat_weights, thresholds, counts, samples, queue, m, n).launch(
            grid=(cute.ceil_div(max(m * self.heads, n), 256), 1, 1),
            block=(256, 1, 1), stream=stream, use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(self, scales, weights, thresholds, counts, samples, queue, m, n):
        pdl_prologue(self.pdl)
        index = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        offset = cute.ceil_div(m, 32) * 32
        if index < m:
            thresholds[index] = cutlass.Float32(float('-inf'))
            counts[index] = 0
            # K3's exact fallback remains valid without a sampled upper bound.
            samples[index, samples.shape[1] - 30] = samples.element_type(0)
        if cutlass.const_expr(self.safe and (not self.fp4 or self.heads in (4, 32))):
            if index < m * self.heads:
                thresholds[offset + index] = weights[index] * cutlass.Float32(0.5)
        if cutlass.const_expr(self.prepare_sf):
            if index < n:
                prepared = cute.make_tensor(
                    cute.recast_ptr(thresholds.iterator, dtype=cutlass.Int32),
                    cute.make_layout(thresholds.shape[0]),
                )
                source = index // 128 * 128 + index % 4 * 32 + index % 128 // 4
                prepared[offset + index] = scales[source]
        if index == 0:
            queue[0] = 0
            queue[queue.shape[0] - 1] = 0


class ShortDomainFP8(ShortDomainSetup):
    @cute.jit
    def __call__(self, k, scales, q, weights, logits, starts, ends,
                 thresholds, counts, samples, out, queue,
                 m: cutlass.Int32, n: cutlass.Int32, stream: cuda.CUstream):
        self.launch(scales, weights, thresholds, counts, samples, queue, m, n, stream)


class ShortDomainFP4(ShortDomainSetup):
    @cute.jit
    def __call__(self, k, scales, q, q_scales, weights, logits, starts, ends,
                 thresholds, counts, samples, out, queue,
                 m: cutlass.Int32, n: cutlass.Int32, stream: cuda.CUstream):
        self.launch(scales, weights, thresholds, counts, samples, queue, m, n, stream)


@cute.jit
def candidate_bounds(candidates, row, bank, local, bank_count, span, hist,
                     dtype: cutlass.Constexpr, packed: cutlass.Constexpr,
                     threads: cutlass.Constexpr, bank_threads: cutlass.Constexpr,
                     barrier_id: cutlass.Constexpr):
    tid = cute.arch.thread_idx()[0]
    lane = cute.arch.lane_idx()
    warp = tid // 32
    minimum = cutlass.Uint32(0xFFFFFFFF)
    maximum = cutlass.Uint32(0)
    for offset in cutlass.range(local, bank_count, bank_threads):
        score = candidate_score(candidates[row, 8 + bank * span + offset], dtype, packed)
        if score == score:
            key = score_key(score)
            minimum = min(minimum, key)
            maximum = max(maximum, key)
    minimum = cute.arch.warp_redux_sync(minimum, 'umin')
    maximum = cute.arch.warp_redux_sync(maximum, 'umax')
    if lane == 0:
        hist[warp] = cutlass.Int32(minimum)
        hist[warp + 32] = cutlass.Int32(maximum)
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)
    minimum = cutlass.Uint32(0xFFFFFFFF)
    maximum = cutlass.Uint32(0)
    if lane < threads // 32:
        minimum = cutlass.Uint32(hist[lane])
        maximum = cutlass.Uint32(hist[lane + 32])
    minimum = cute.arch.warp_redux_sync(minimum, 'umin')
    maximum = cute.arch.warp_redux_sync(maximum, 'umax')
    lower = cutlass.Float32(float('inf'))
    upper = cutlass.Float32(float('-inf'))
    if minimum != cutlass.Uint32(0xFFFFFFFF):
        lower = key_score(maximum)
        upper = key_score(minimum)
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)
    return lower, upper
