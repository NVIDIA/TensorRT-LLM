# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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
"""FP16 packed ordering and output-centric near-K complement primitives."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def fp16_ordered_key_x2(packed, *, loc=None, ip=None):
    """Return the original raw ordered-key transform of two FP16 halfwords."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [packed.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs;\n\t"
            "prmt.b32 signs, $1, 0, 0xbb99;\n\t"
            "lop3.b32 $0, $1, signs, 0x80008000, 0x1e;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp16_ordered_key_x2_nan_max(packed, *, loc=None, ip=None):
    """Order two valid FP16 halfwords, mapping either sign of NaN to 0xffff.

    Padding must be masked to ordered key zero after this transform, not before.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [packed.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs, clean, canonical;\n\t"
            "mov.b32 canonical, 0x7fff7fff;\n\t"
            "max.f16x2 clean, $1, canonical;\n\t"
            "prmt.b32 signs, clean, 0, 0xbb99;\n\t"
            "lop3.b32 $0, clean, signs, 0x80008000, 0x1e;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sort_three_indices(first, second, third, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        [
            first.ir_value(loc=loc, ip=ip),
            second.ir_value(loc=loc, ip=ip),
            third.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 lower, upper, middle;\n\t"
        "min.s32 lower, $3, $4;\n\t"
        "max.s32 upper, $3, $4;\n\t"
        "min.s32 middle, upper, $5;\n\t"
        "max.s32 $2, upper, $5;\n\t"
        "min.s32 $0, lower, middle;\n\t"
        "max.s32 $1, lower, middle;\n}",
        "=r,=r,=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), result, [0])),
        cutlass.Int32(llvm.extractvalue(T.i32(), result, [1])),
        cutlass.Int32(llvm.extractvalue(T.i32(), result, [2])),
    )


@cute.jit
def emit_complement_fp16(
    excluded_ids,
    exclude_count: cutlass.Int32,
    output_row: cute.Tensor,
    tid: cutlass.Int32,
    selected_k: cutlass.Constexpr,
    envelope: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
):
    """Emit K indices after removing the selected active prefix of excluded IDs.

    The active IDs must be distinct and lie in [0, selected_k + exclude_count).
    The count must lie in [0, envelope - selected_k], with at most three IDs.
    Call once per thread in [0, block_threads) with the same IDs and count.
    The Int32 output row must have unit stride and at least selected_k elements.
    The caller retains its short-row path and exclusion-publication barrier.
    """
    assert selected_k > 0
    assert 0 <= envelope - selected_k <= 3
    assert envelope < 65536
    assert 1 <= block_threads <= 1024
    assert len(excluded_ids) >= envelope - selected_k

    inactive = cutlass.Int32(selected_k + 3)
    removed_first = inactive
    removed_second = inactive
    removed_third = inactive
    if cutlass.const_expr(envelope - selected_k >= 1):
        if exclude_count > cutlass.Int32(0):
            removed_first = cutlass.Int32(excluded_ids[0])
    if cutlass.const_expr(envelope - selected_k >= 2):
        if exclude_count > cutlass.Int32(1):
            removed_second = cutlass.Int32(excluded_ids[1])
    if cutlass.const_expr(envelope - selected_k >= 3):
        if exclude_count > cutlass.Int32(2):
            removed_third = cutlass.Int32(excluded_ids[2])

    removed_first, removed_second, removed_third = _sort_three_indices(
        removed_first, removed_second, removed_third
    )
    threshold_first = removed_first
    threshold_second = removed_second - cutlass.Int32(1)
    threshold_third = removed_third - cutlass.Int32(2)

    for output_block in cutlass.range_constexpr((selected_k + block_threads - 1) // block_threads):
        output_index = tid + cutlass.Int32(output_block * block_threads)
        if output_index < cutlass.Int32(selected_k):
            source_index = (
                output_index
                + cutlass.Int32(output_index >= threshold_first)
                + cutlass.Int32(output_index >= threshold_second)
                + cutlass.Int32(output_index >= threshold_third)
            )
            output_row[output_index] = source_index


__all__ = [
    "fp16_ordered_key_x2",
    "fp16_ordered_key_x2_nan_max",
    "emit_complement_fp16",
]
