# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional merge-compatible MLA statistics without changing output reduction."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64

from .ops import fmax_f32


@cute.jit
def partial_max_tensor(acc_lse, softmax_stats):
    """Bind optional raw QK maxima; the matching sums follow this tensor."""
    result = None
    if cutlass.const_expr(softmax_stats is not None and acc_lse is not None):
        result = cute.make_tensor(
            acc_lse.iterator + Int64(cute.cosize(acc_lse.layout)), acc_lse.layout
        )
    return result


@cute.jit
def store_softmax_stats(softmax_stats, row, raw_maximum, denominator, softmax_scale):
    """Publish actual natural-log max and unscaled exp-sum from one owner."""
    if cutlass.const_expr(softmax_stats is not None):
        maximum = raw_maximum * softmax_scale
        if denominator == Float32(0.0):
            maximum = Float32(-Float32.inf)
        ptr = softmax_stats.iterator.raw_ptr() + Int64(row) * Int64(2)
        ptr.store(maximum)
        (ptr + Int64(1)).store(denominator)


@cute.jit
def store_reduced_softmax_stats(
    softmax_stats, row, acc_lse, row_lse, active_splits, softmax_scale
):
    """Merge actual max/sum states on the final statistics writer lane.

    Existing output reducers use LSE to rescale partial O. Their LSE maximum
    is not the logit maximum. Absolute FP32 LSE also loses small log-sums at
    large finite maxima, so retain both maximum and denominator separately.
    Store raw QK maxima and scale only their differences: scaling absolute
    maxima first would round away small gaps between large finite scores.
    Inactive split slots are deliberately never read.
    """
    if cutlass.const_expr(softmax_stats is not None):
        scale_log2 = softmax_scale * Float32(1.4426950408889634)
        partial_rows = Int64(cute.cosize(acc_lse.layout))
        max_ptr = row_lse.iterator.raw_ptr() + partial_rows
        sum_ptr = max_ptr + partial_rows
        maximum = Float32(-Float32.inf)
        for split_idx in cutlass.range(active_splits):
            if row_lse[split_idx] != Float32(-Float32.inf):
                value = (max_ptr + Int64(split_idx)).load()
                maximum = fmax_f32(maximum, value)
        denominator = Float32(0.0)
        for split_idx in cutlass.range(active_splits):
            if row_lse[split_idx] != Float32(-Float32.inf):
                partial_max = (max_ptr + Int64(split_idx)).load()
                partial_sum = (sum_ptr + Int64(split_idx)).load()
                denominator += partial_sum * cute.math.exp2(
                    (partial_max - maximum) * scale_log2, fastmath=True
                )
        store_softmax_stats(softmax_stats, row, maximum, denominator, softmax_scale)
