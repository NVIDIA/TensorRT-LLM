# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Common arithmetic for standalone split-KV GMEM reducers.

Every producer using this interface publishes the same mathematical state:

* one FP32 log2-LSE value per split and output row; and
* one normalized 16-bit partial-O vector for that split and row.

Reducer policies still own row addressing, active-split discovery, cluster
topology, PDL ordering, and final-output stores. Those details differ between
FMHA, MLA 1CTA, and MLA 2CTA and are deliberately kept out of this module.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32


@cute.jit
def finalize_log2_sum_exp(frame: Float32, exp_sum: Float32) -> Float32:
    """Return ``frame + log2(exp_sum)`` or the neutral ``-inf`` LSE."""

    has_mass = exp_sum == exp_sum and exp_sum != Float32(0.0)
    return (
        frame + cute.math.log2(exp_sum, fastmath=True)
        if has_mass
        else Float32(-Float32.inf)
    )


@cute.jit
def normalized_lse_weight(partial_lse: Float32, global_lse: Float32) -> Float32:
    """Return the weight of one normalized partial in the global output."""

    has_mass = global_lse != Float32(-Float32.inf)
    return (
        Float32(
            cute.math.exp2(
                partial_lse - global_lse,
                fastmath=True,
            )
        )
        if has_mass
        else Float32(0.0)
    )


@cute.jit
def merge_log2_lse(
    lhs_lse: Float32,
    rhs_lse: Float32,
) -> tuple[Float32, Float32, Float32]:
    """Merge two normalized states and return LSE plus output weights."""

    neg_inf = Float32(-Float32.inf)
    frame = cute.math.max(lhs_lse, rhs_lse, ftz=True)
    has_mass = frame != neg_inf
    lhs_exp = (
        Float32(cute.math.exp2(lhs_lse - frame, fastmath=True))
        if has_mass
        else Float32(0.0)
    )
    rhs_exp = (
        Float32(cute.math.exp2(rhs_lse - frame, fastmath=True))
        if has_mass
        else Float32(0.0)
    )
    exp_sum = lhs_exp + rhs_exp
    merged_lse = finalize_log2_sum_exp(frame, exp_sum)
    inv_sum = Float32(1.0) / exp_sum if has_mass else Float32(0.0)
    return merged_lse, lhs_exp * inv_sum, rhs_exp * inv_sum


@cute.jit
def unpack_normalized_vec8(
    regs_i32: cutlass.Array,
    use_bf16_partial: cutlass.Constexpr[bool],
) -> cutlass.Array:
    """Decode one packed 16-byte normalized partial-O vector to FP32."""

    regs_vec = cutlass.Vector.from_elements(
        (regs_i32[0], regs_i32[1], regs_i32[2], regs_i32[3]),
        Int32,
    )
    if cutlass.const_expr(use_bf16_partial):
        return regs_vec.bitcast(BFloat16).to(Float32)
    return regs_vec.bitcast(Float16).to(Float32)
