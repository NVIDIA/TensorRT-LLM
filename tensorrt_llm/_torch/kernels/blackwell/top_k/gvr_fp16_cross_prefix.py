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
"""FP16 code-radix crossing-bin search: one warp scans a 256-bin (8-bit digit) histogram from the top."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

if __package__:
    from . import gvr_topk_decode_self_sampling as gvr
else:
    import gvr_topk_decode_self_sampling as gvr


@dsl_user_op
def _select_prefix_eight(
    cursor0,
    cursor1,
    cursor2,
    cursor3,
    cursor4,
    cursor5,
    cursor6,
    cursor7,
    limit,
    *,
    need_above,
    loc=None,
    ip=None,
):
    assembly = (
        "{\n\t.reg .pred high4, high2, high1;\n\t"
        ".reg .b32 middle, low_border, high_border, border, ordinal, bit, preceding;\n\t"
        "setp.le.s32 high4, $10, $5;\n\t"
        "selp.b32 middle, $7, $3, high4;\n\t"
        "setp.le.s32 high2, $10, middle;\n\t"
        "selp.b32 low_border, $4, $2, high2;\n\t"
        "selp.b32 high_border, $8, $6, high2;\n\t"
        "selp.b32 border, high_border, low_border, high4;\n\t"
        "setp.le.s32 high1, $10, border;\n\t"
        "selp.b32 ordinal, 4, 0, high4;\n\t"
        "selp.b32 bit, 2, 0, high2;\n\tor.b32 ordinal, ordinal, bit;\n\t"
        "@high1 or.b32 ordinal, ordinal, 1;\n\t"
    )
    if need_above:
        assembly += (
            "selp.b32 low_border, $5, $3, high2;\n\t"
            "selp.b32 high_border, $9, $7, high2;\n\t"
            "selp.b32 middle, high_border, low_border, high4;\n\t"
            "selp.b32 preceding, middle, border, high1;\n\t"
        )
    else:
        assembly += "mov.b32 preceding, 0;\n\t"
    assembly += "mov.b32 $0, ordinal;\n\tmov.b32 $1, preceding;\n}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            value.ir_value(loc=loc, ip=ip)
            for value in (
                cursor0,
                cursor1,
                cursor2,
                cursor3,
                cursor4,
                cursor5,
                cursor6,
                cursor7,
                limit,
            )
        ],
        assembly,
        "=r,=r,r,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(cutlass.Int32(llvm.extractvalue(T.i32(), result, [index])) for index in range(2))


@cute.jit
def find_cross_native_fp16(
    address, target, tid, result, bias, fix_hist, fix_pad, first_pass: cutlass.Constexpr
):
    """Find the bin that holds the target-th largest element of a 256-bin shared-memory histogram.

    Warp 0 reads 8 bins per lane, suffix-scans the lane totals, picks the crossing lane by ballot and the bin
    inside it with a branch-free binary search. Writes the bin to `result[RES_B]`; the first pass also writes
    the count above it (`RES_ABOVE`), the second pass replaces the counts with emit cursors.
    """
    if tid < cutlass.Int32(32):
        row_address = address + tid * cutlass.Int32(32)
        count0, count1, count2, count3 = gvr._lds_v4_u32(row_address)
        count4, count5, count6, count7 = gvr._lds_v4_u32(row_address + cutlass.Int32(16))
        subtotal = ((count0 + count1) + (count2 + count3)) + ((count4 + count5) + (count6 + count7))
        suffix = gvr.warp_suffix_scan_add(subtotal, tid)
        crossing_lane = gvr.hi_bit_or_zero(gvr.ballot(suffix >= target))
        cursor7 = bias + suffix - subtotal
        cursor6 = cursor7 + count7
        cursor5 = cursor6 + count6
        cursor4 = cursor5 + count5
        cursor3 = cursor4 + count4
        cursor2 = cursor3 + count3
        cursor1 = cursor2 + count2
        cursor0 = cursor1 + count1
        if cutlass.const_expr(not first_pass):
            gvr._sts_v4_u32(row_address, cursor0, cursor1, cursor2, cursor3)
            gvr._sts_v4_u32(row_address + cutlass.Int32(16), cursor4, cursor5, cursor6, cursor7)
        local_bin, selected_above = _select_prefix_eight(
            cursor0,
            cursor1,
            cursor2,
            cursor3,
            cursor4,
            cursor5,
            cursor6,
            cursor7,
            target + bias,
            need_above=first_pass,
        )
        if tid == crossing_lane:
            selected_bin = tid * cutlass.Int32(8) + local_bin
            result[gvr.RES_B] = selected_bin
            if cutlass.const_expr(first_pass):
                result[gvr.RES_ABOVE] = selected_above
                if selected_bin == cutlass.Int32(0):
                    fix_hist[0] = cutlass.Int32(0) - fix_pad
