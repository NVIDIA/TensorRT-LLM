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
from cutlass import BFloat16, Float16, Float32, Int32, Int64

from .mla_decode.helpers.constants import LN_2, LOG2_E


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


@cute.jit
def attention_sink_scale(log2_lse, sink):
    """Denominator-only sink mass; keep the KV LSE independent of the sink."""
    scale = Float32(0)
    if log2_lse != Float32(-Float32.inf) and sink != Float32(Float32.inf):
        delta = sink * Float32(LOG2_E) - log2_lse
        z = cute.math.exp2(-cute.math.abs(delta), fastmath=True)
        inverse = cute.math.rcp(Float32(1) + z, approx=True)
        scale = z * inverse if delta > Float32(0) else inverse
    return scale


class MergeMlaSourceStates:
    def __init__(self, independent_sources: bool, output_quant=None):
        self.independent_sources = independent_sources
        self.output_quant = output_quant

    @cute.jit
    def __call__(
        self,
        part_s,
        part_c,
        lse_s,
        lse_c,
        count_s,
        count_c,
        sinks,
        out,
        lse,
        stream,
        output_quant_args=None,
    ):
        self.finish(
            part_s,
            part_c,
            lse_s,
            lse_c,
            count_s,
            count_c,
            sinks,
            out,
            lse,
            output_quant_args,
        ).launch(
            grid=((out.shape[0] * out.shape[1] + 3) // 4, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def finish(
        self,
        part_s,
        part_c,
        lse_s,
        lse_c,
        count_s,
        count_c,
        sinks,
        out,
        lse,
        output_quant_args=None,
    ):
        block, _, _ = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        # One warp handles one output head. Group four heads per CTA to
        # amortize launch overhead and avoid repeating normalization 128 times.
        row = Int64(block) * 4 + Int64(tid // 32)
        q = row // out.shape[1]
        h = row % out.shape[1]
        if row < out.shape[0] * out.shape[1]:
            have_s = Int32(count_s[q]) > 0
            have_c = (Int32(count_c[q]) > 0) if self.independent_sources else False
            l0 = Float32(lse_s[q, h]) if have_s else Float32(-Float32.inf)
            l1 = Float32(lse_c[q, h]) if have_c else Float32(-Float32.inf)
            ws = Float32(0)
            wc = Float32(0)
            public_lse = Float32(-Float32.inf)
            if have_s or have_c:
                kv_lse, ws, wc = merge_log2_lse(l0, l1)
                public_lse = kv_lse * Float32(LN_2)
                sink_scale = attention_sink_scale(kv_lse, Float32(sinks[h]))
                ws *= sink_scale
                wc *= sink_scale
            width = cutlass.const_expr(4 if self.output_quant is not None else 2)
            for chunk in cutlass.range_constexpr(512 // (32 * width)):
                column = lane * width + chunk * (32 * width)
                values = cutlass.Array(Float32, width, space=cutlass.AddressSpace.rmem)
                for j in cutlass.range_constexpr(width):
                    values[j] = Float32(0)
                if have_s:
                    offset = q * part_s.stride[0] + h * part_s.stride[1] + column
                    pair = cutlass.Pointer(
                        part_s.iterator + offset, dtype=part_s.element_type
                    ).load(count=width, alignment=width * 2)
                    for j in cutlass.range_constexpr(width):
                        values[j] = Float32(pair[j]) * ws
                if have_c:
                    offset = q * part_c.stride[0] + h * part_c.stride[1] + column
                    pair = cutlass.Pointer(
                        part_c.iterator + offset, dtype=part_c.element_type
                    ).load(count=width, alignment=width * 2)
                    for j in cutlass.range_constexpr(width):
                        values[j] += Float32(pair[j]) * wc
                if cutlass.const_expr(self.output_quant is not None):
                    self.output_quant.store(output_quant_args, values, row, column)
                else:
                    packed = cutlass.Array(
                        out.element_type, 2, space=cutlass.AddressSpace.rmem
                    )
                    for j in cutlass.range_constexpr(2):
                        packed[j] = values[j].to(out.element_type)
                    offset = q * out.stride[0] + h * out.stride[1] + column
                    cutlass.Pointer(
                        out.iterator + offset, dtype=out.element_type
                    ).store(packed.load(0, 2), alignment=4)
            if lane == 0:
                lse[q, h] = public_lse
