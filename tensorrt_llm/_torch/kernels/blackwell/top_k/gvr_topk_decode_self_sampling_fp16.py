# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP16-only kernels and device helpers of the self-sampling GVR top-K decode.

The device module gvr_topk_decode_self_sampling imports this module as ``_fp16`` and calls
its public names from FP16-only arms; names with a leading underscore are used only here.
Shared device helpers are reached through ``gvr``. Selection works on ordered 16-bit keys
(``ordered_key16x2``): code radix finds the K-th key from a 256-bin histogram of the high
key byte, then from the low bytes of the keys inside the crossing bin.

- Crossing search: ``find_cross256_select`` and its eight-bin step ``crossing_bin8_above``,
  which the cluster merges of the device module also call.
- Mask emit of ``GvrTopkRegKernel.kern_cr2``: ``histogram_high_bytes``,
  ``histogram_crossing_low_bytes``, ``threshold_masks``, ``threshold_mask_ge``,
  ``emit_masked_indices`` and the PTX wrappers ``warp_inclusive_sum``, ``atom_shared_add``
  and ``st_global_v4_b32``.
- Counter emit: ``GvrCodeRadixCounterKernel`` (one CTA per row) and, for
  ``GvrTopkRegKernel.kern_crc``, ``classify_key_pair_cluster``, ``merge_cursors256_cluster``
  and ``scan_above_lane_cursors_cluster``. Both emit through ``emit_counter_pair``.
- Near-K complement, FP16 arm of ``ComplementKernel``: ``ordered_key16x2_nan_max``,
  ``emit_complement`` and ``repair_complement``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass import Int32 as I32
from cutlass import Uint32 as U32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from . import gvr_topk_decode_self_sampling as gvr

# ---- near-K complement repair ----


@cute.jit
def repair_complement(
    removed_ids,
    exclude_count: cutlass.Int32,
    output_row: cute.Tensor,
    lane: cutlass.Int32,
    selected_k: cutlass.Constexpr,
):
    """Patch an identity-filled ``output_row[:selected_k]`` into the complement (one warp).

    ``removed_ids`` holds three Int32 indices; the first ``exclude_count`` (1 to 3) are distinct
    and below selected_k + exclude_count. Lane j < exclude_count overwrites slot
    ``removed_ids[j]``, if it is below selected_k, with a distinct non-excluded index from
    [selected_k, selected_k + exclude_count), so the row holds the complement in no fixed order.
    """
    if lane < exclude_count:
        removed_index = removed_ids[0]
        for pick in cutlass.range_constexpr(1, 3):
            if lane == cutlass.Int32(pick):
                removed_index = removed_ids[pick]
        if removed_index < cutlass.Int32(selected_k):
            replacement = cutlass.Int32(selected_k) + lane
            for step in cutlass.range_constexpr(2):
                previous = replacement
                for pick in cutlass.range_constexpr(3):
                    if cutlass.Int32(pick) < exclude_count:
                        if previous == removed_ids[pick]:
                            replacement = cutlass.Int32(selected_k + pick)
            output_row[removed_index] = replacement


# ---- code-radix mask emit (kern_cr2) ----


def _asm(results, operands, body, constraints, side_effects, loc, ip):
    """Emit PTX lines as one brace-scoped inline-asm op and return its result.

    ``operands`` are DSL scalars lowered with ``ir_value``; ``results`` is the LLVM result type,
    or None when the asm returns nothing.
    """
    return llvm.inline_asm(
        results,
        [x.ir_value(loc=loc, ip=ip) for x in operands],
        "{\n\t" + "\n\t".join(body) + "\n}",
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def histogram_high_bytes(hist, words, *, loc=None, ip=None):
    """Add the high byte of both keys of every packed word to a 256-bin histogram.

    ``hist`` is the shared byte address of 256 u32 bins and ``words`` a list of Uint32 values,
    each holding two ordered 16-bit keys. One straight-line asm block issues all the
    ``red.shared.add`` updates. Pass 1 of ``GvrTopkRegKernel.kern_cr2``.
    """
    body = [".reg .b32 t, a, b;"]
    for i in range(len(words)):
        w = f"${i + 1}"
        body += [
            f"and.b32 t, {w}, 0xFF00FF00;",
            "shl.b32 a, t, 16;",
            "shr.u32 a, a, 22;",
            "add.u32 a, a, $0;",
            "shr.u32 b, t, 22;",
            "add.u32 b, b, $0;",
            "red.relaxed.cta.shared.add.u32 [a], 1;",
            "red.relaxed.cta.shared.add.u32 [b], 1;",
        ]
    _asm(None, [hist] + list(words), body, ",".join(["r"] * (len(words) + 1)), True, loc, ip)


@dsl_user_op
def histogram_crossing_low_bytes(hist, pattern, words, *, loc=None, ip=None):
    """Histogram the low bytes of the keys inside the crossing bin (256 bins plus overflow).

    ``pattern`` holds the crossing high byte at bits 8-15 and 24-31. A key with that high byte
    increments bin ``key & 0xFF``; every other key increments bin 256, so ``hist`` needs 257
    u32 bins and the last one is never read. Pass 2 of ``GvrTopkRegKernel.kern_cr2``.
    """
    body = [".reg .b32 t, m, a, b, c;", "mov.b32 c, 0x01000100;"]
    for i in range(len(words)):
        w = f"${i + 2}"
        body += [
            f"xor.b32 t, {w}, $1;",
            "min.u16x2 m, t, c;",
            "shl.b32 a, m, 16;",
            "shr.u32 a, a, 14;",
            "add.u32 a, a, $0;",
            "shr.u32 b, m, 14;",
            "add.u32 b, b, $0;",
            "red.relaxed.cta.shared.add.u32 [a], 1;",
            "red.relaxed.cta.shared.add.u32 [b], 1;",
        ]
    _asm(
        None, [hist, pattern] + list(words), body, ",".join(["r"] * (len(words) + 2)), True, loc, ip
    )


@dsl_user_op
def threshold_masks(words, above, at_least, *, loc=None, ip=None):
    """Return the bitmasks of the keys above and at least the K-th key.

    ``above`` is ``kth << 16 | 0xFFFF`` and ``at_least`` is ``kth << 16``. Bit 2i (2i + 1) stands
    for the low (high) key of ``words[i]``, so ``words`` holds at most 16 packed key pairs.
    Returns ``(greater, at_least)`` as Uint32 masks.
    """
    body = [".reg .pred p, q;", ".reg .b32 x, g0, g1, g2, g3, e0, e1, e2, e3;"]
    body += [f"mov.b32 g{c}, 0;" for c in range(4)] + [f"mov.b32 e{c}, 0;" for c in range(4)]
    for i in range(len(words)):
        w = f"${i + 4}"
        c = i % 4
        body += [
            f"shl.b32 x, {w}, 16;",
            "setp.gt.u32 p, x, $2;",
            f"@p add.u32 g{c}, g{c}, 0x{1 << (2 * i):08X};",
            "setp.ge.u32 q, x, $3;",
            f"@q add.u32 e{c}, e{c}, 0x{1 << (2 * i):08X};",
            f"setp.gt.u32 p, {w}, $2;",
            f"@p add.u32 g{c}, g{c}, 0x{1 << (2 * i + 1):08X};",
            f"setp.ge.u32 q, {w}, $3;",
            f"@q add.u32 e{c}, e{c}, 0x{1 << (2 * i + 1):08X};",
        ]
    body += [
        "or.b32 g0, g0, g1;",
        "or.b32 g2, g2, g3;",
        "or.b32 $0, g0, g2;",
        "or.b32 e0, e0, e1;",
        "or.b32 e2, e2, e3;",
        "or.b32 $1, e0, e2;",
    ]
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [above, at_least] + list(words),
        body,
        "=r,=r," + ",".join(["r"] * (len(words) + 2)),
        False,
        loc,
        ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0])),
        Uint32(llvm.extractvalue(T.i32(), result, [1])),
    )


@dsl_user_op
def emit_masked_indices(dst, thread_base, mask, offsets, *, loc=None, ip=None):
    """Store ``thread_base + offsets[j]`` for every set bit j of ``mask`` in consecutive words.

    ``dst`` is the shared byte address of the first slot. Bits are taken in ascending order and
    ``offsets`` holds one Python int per bit (at most 32), so popc(mask) words are written.
    """
    body = [".reg .pred p;", ".reg .b32 a, t, v;", "mov.b32 a, $0;"]
    for j, off in enumerate(offsets):
        body += [
            f"and.b32 t, $2, 0x{1 << j:08X};",
            "setp.ne.u32 p, t, 0;",
            f"add.u32 v, $1, {off};",
            "@p st.shared.u32 [a], v;",
            "@p add.u32 a, a, 4;",
        ]
    _asm(None, [dst, thread_base, mask], body, "r,r,r", True, loc, ip)


@dsl_user_op
def atom_shared_add(addr, value, *, loc=None, ip=None):
    """Return the old value of ``atom.relaxed.cta.shared.add.u32 [addr], value``."""
    return Int32(
        _asm(
            T.i32(),
            [addr, value],
            ["atom.relaxed.cta.shared.add.u32 $0, [$1], $2;"],
            "=r,r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def warp_inclusive_sum(value, *, loc=None, ip=None):
    """Return the inclusive warp prefix sum of ``value``; all 32 lanes must call it.

    Each step is ``shfl.sync.up`` plus an add predicated on the shuffle's in-range flag.
    """
    body = [".reg .pred p;", ".reg .b32 x, y;", "mov.b32 x, $1;"]
    for offset in (1, 2, 4, 8, 16):
        body += [f"shfl.sync.up.b32 y|p, x, {offset}, 0, 0xffffffff;", "@p add.u32 x, x, y;"]
    body.append("mov.b32 $0, x;")
    return Int32(_asm(T.i32(), [value], body, "=r,r", True, loc, ip))


@dsl_user_op
def st_global_v4_b32(gaddr, a, b, c, d, *, loc=None, ip=None):
    """Store four 32-bit words at the 16-byte aligned 64-bit global address ``gaddr``."""
    _asm(
        None,
        [gaddr, a, b, c, d],
        ["st.global.v4.b32 [$0], {$1, $2, $3, $4};"],
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def threshold_mask_ge(words, at_least, *, loc=None, ip=None):
    """Return the bitmask of the keys at least the K-th key (``at_least`` is ``kth << 16``).

    Same bit layout as ``threshold_masks``. Used when the keys equal to the K-th fill the row
    exactly, so every key at least the K-th is selected.
    """
    body = [".reg .pred selected;", ".reg .b32 shifted, mask0, mask1, mask2, mask3;"]
    body += [f"mov.b32 mask{chain}, 0;" for chain in range(4)]
    for word_index in range(len(words)):
        word_operand = f"${word_index + 2}"
        chain = word_index % 4
        body += [
            f"shl.b32 shifted, {word_operand}, 16;",
            "setp.ge.u32 selected, shifted, $1;",
            f"@selected add.u32 mask{chain}, mask{chain}, 0x{1 << (2 * word_index):08X};",
            f"setp.ge.u32 selected, {word_operand}, $1;",
            f"@selected add.u32 mask{chain}, mask{chain}, 0x{1 << (2 * word_index + 1):08X};",
        ]
    body += [
        "or.b32 mask0, mask0, mask1;",
        "or.b32 mask2, mask2, mask3;",
        "or.b32 $0, mask0, mask2;",
    ]
    return cutlass.Uint32(
        _asm(
            T.i32(),
            [at_least] + list(words),
            body,
            "=r," + ",".join(["r"] * (len(words) + 1)),
            False,
            loc,
            ip,
        )
    )


# ---- crossing search ----


@dsl_user_op
def crossing_bin8_above(
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
    """Return the crossing bin among one lane's eight bins with a branch-free binary search.

    ``cursor0`` .. ``cursor7`` are the non-increasing biased counts of keys above each bin and
    ``limit`` is target + bias; compares are signed. Returns ``(bin, above)`` with
    bin = #{i < 7 : limit <= cursor_i}; ``above`` is cursor_bin with ``need_above``, else 0.
    """
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
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [index])) for index in range(2))


@cute.jit
def find_cross256_select(
    address, target, tid, result, bias, next_hist, pad_count, first_pass: cutlass.Constexpr
):
    """Find the crossing bin of a 256-bin ascending-key histogram with warp 0.

    Lane l owns bins 8l..8l+7 at ``address`` + 32l. The crossing lane writes ``result[RES_B]``,
    the highest bin whose suffix count reaches ``target``. With ``first_pass`` it also writes
    ``result[RES_ABOVE]`` (``bias`` + count above) and, for bin 0, ``next_hist[0] = -pad_count``;
    without it the bins become cursors (``bias`` + count above each bin). No RES_M, no tie counter.

    In its second call ``GvrTopkRegKernel.kern_cr2`` passes the scanned histogram as ``next_hist``;
    a crossing bin of 0 then overwrites the bin-0 count that its exact-fit test reads, so the test
    fails and the general emit, which is also exact, runs instead.
    """
    if tid < Int32(32):
        row_address = address + tid * Int32(32)
        count0, count1, count2, count3 = gvr._lds_v4_u32(row_address)
        count4, count5, count6, count7 = gvr._lds_v4_u32(row_address + Int32(16))
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
            gvr._sts_v4_u32(row_address + Int32(16), cursor4, cursor5, cursor6, cursor7)
        local_bin, selected_above = crossing_bin8_above(
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
            selected_bin = tid * Int32(8) + local_bin
            result[gvr.RES_B] = selected_bin
            if cutlass.const_expr(first_pass):
                result[gvr.RES_ABOVE] = selected_above
                if selected_bin == Int32(0):
                    next_hist[0] = Int32(0) - pad_count


# ---- ordered keys and near-K complement emit ----


@dsl_user_op
def ordered_key16x2(packed, *, loc=None, ip=None):
    """Map both 16-bit codes of a packed word to monotone 16-bit keys (PRMT + LOP3).

    Per half, a set sign bit flips all 16 bits and a clear one flips only the sign bit. The map
    is format-agnostic and equals ``gvr.fkey16x2_of_pair``: +NaN sorts above +inf, -NaN below
    -inf, and the 0xFFFF padding code maps to key 0.
    """
    return U32(
        llvm.inline_asm(
            T.i32(),
            [packed.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs;\n\tprmt.b32 signs, $1, 0, 0xbb99;\n\tlop3.b32 $0, $1, signs, 0x80008000, 0x1e;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ordered_key16x2_nan_max(packed, *, loc=None, ip=None):
    """Like ``ordered_key16x2``, but an FP16 NaN of either sign maps to key 0xFFFF.

    NaNs then rank above +inf, as in torch.topk. The FP16 arm of ``ComplementKernel`` keys
    its row vectors with it.
    """
    return U32(
        llvm.inline_asm(
            T.i32(),
            [packed.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs, ordered, nan_mask;\n\tset.nan.u32.f16x2 nan_mask, $1, $1;"
            "\n\tprmt.b32 signs, $1, 0, 0xbb99;\n\tlop3.b32 ordered, $1, signs, 0x80008000, 0x1e;"
            "\n\tor.b32 $0, ordered, nan_mask;\n}",
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
    """Return three Int32 values in ascending order (signed min/max network)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        [
            first.ir_value(loc=loc, ip=ip),
            second.ir_value(loc=loc, ip=ip),
            third.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 lower, upper, middle;\n\tmin.s32 lower, $3, $4;\n\tmax.s32 upper, $3, $4;"
        "\n\tmin.s32 middle, upper, $5;\n\tmax.s32 $2, upper, $5;\n\tmin.s32 $0, lower, middle;"
        "\n\tmax.s32 $1, lower, middle;\n}",
        "=r,=r,=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        I32(llvm.extractvalue(T.i32(), result, [0])),
        I32(llvm.extractvalue(T.i32(), result, [1])),
        I32(llvm.extractvalue(T.i32(), result, [2])),
    )


@cute.jit
def emit_complement(
    excluded_ids,
    exclude_count: I32,
    output_row: cute.Tensor,
    tid: I32,
    selected_k: cutlass.Constexpr,
    envelope: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
):
    """Write the K indices left after removing up to three excluded ones, in ascending order.

    Slot o of ``output_row[:selected_k]`` receives the o-th index of
    [0, selected_k + exclude_count) not among the first ``exclude_count`` (distinct)
    ``excluded_ids``. All ``block_threads`` threads call it with the same ids. Requires
    envelope - selected_k <= 3 and envelope < 65536 (the ids come from a 16-bit index field).
    """
    assert selected_k > 0
    assert 0 <= envelope - selected_k <= 3
    assert envelope < 65536
    assert 1 <= block_threads <= 1024
    assert len(excluded_ids) >= envelope - selected_k
    inactive = I32(selected_k + 3)
    removed_first = inactive
    removed_second = inactive
    removed_third = inactive
    if cutlass.const_expr(envelope - selected_k >= 1):
        if exclude_count > I32(0):
            removed_first = I32(excluded_ids[0])
    if cutlass.const_expr(envelope - selected_k >= 2):
        if exclude_count > I32(1):
            removed_second = I32(excluded_ids[1])
    if cutlass.const_expr(envelope - selected_k >= 3):
        if exclude_count > I32(2):
            removed_third = I32(excluded_ids[2])
    removed_first, removed_second, removed_third = _sort_three_indices(
        removed_first, removed_second, removed_third
    )
    threshold_first = removed_first
    threshold_second = removed_second - I32(1)
    threshold_third = removed_third - I32(2)
    for output_block in cutlass.range_constexpr((selected_k + block_threads - 1) // block_threads):
        output_index = tid + I32(output_block * block_threads)
        if output_index < I32(selected_k):
            source_index = (
                output_index
                + I32(output_index >= threshold_first)
                + I32(output_index >= threshold_second)
                + I32(output_index >= threshold_third)
            )
            output_row[output_index] = source_index


# ---- code-radix counter emit ----


@dsl_user_op
def _classify_key_pair(
    packed,
    cross_high,
    valid,
    table_addr,
    discard,
    above_lane,
    *,
    key_count: cutlass.Constexpr = 2,
    loc=None,
    ip=None,
):
    """Count each key of a packed pair on the counter its emit will take a ticket from.

    A valid key whose high byte equals ``cross_high`` counts on low-byte bin ``key & 0xFF``, a
    valid higher key on this lane's ``above_lane`` counter and any other key on this lane's
    ``discard`` counter; offsets are bytes from ``table_addr``. Returns both offsets as 16-bit
    fields, low key first; ``key_count=1`` handles only the low key.
    """
    ptx = [
        "{",
        ".reg .b32 high, offset, prefix, address, encoded;",
        ".reg .pred active, above, member;",
        "mov.b32 encoded, 0;",
        "setp.ne.u32 active, $3, 0;",
    ]
    for half in range(key_count):
        ptx.extend(
            [
                f"bfe.u32 high, $1, {half * 16 + 8}, 8;",
                f"bfe.u32 offset, $1, {half * 16}, 8;",
                "shl.b32 offset, offset, 2;",
                "setp.gt.u32 above, high, $2;",
                "setp.ge.and.u32 member, high, $2, active;",
            ]
        )
        ptx.append("selp.b32 offset, $6, offset, above;")
        ptx.extend(
            [
                "selp.b32 offset, offset, $5, member;",
                "add.u32 address, $4, offset;",
                "red.relaxed.cta.shared.add.u32 [address], 1;",
            ]
        )
        if half:
            ptx.append("shl.b32 offset, offset, 16;")
        ptx.append("or.b32 encoded, encoded, offset;")
    ptx.extend(["mov.b32 $0, encoded;", "}"])
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                value.ir_value(loc=loc, ip=ip)
                for value in (packed, cross_high, valid, table_addr, discard, above_lane)
            ],
            "\n\t".join(ptx),
            "=r,r,r,r,r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def emit_counter_pair(
    encoded,
    table_addr,
    src_index,
    top_k,
    out_addr,
    *,
    staged: cutlass.Constexpr,
    key_count: cutlass.Constexpr = 2,
    loc=None,
    ip=None,
):
    """Give each key a slot from its encoded counter and write its index if the slot is below K.

    ``encoded`` holds two 16-bit byte offsets from ``table_addr`` (made by a classify step) whose
    counters already hold cursors; each key takes a ticket with ``atom.shared.add``. A ticket below
    ``top_k`` stores ``src_index`` + half at word ``ticket`` of ``out_addr``, a shared address when
    ``staged``, else a 64-bit global one. ``key_count=1`` emits only the low key.
    """
    ptx = ["{"]
    for half in range(key_count):
        ptx.extend(
            [
                f".reg .b32 offset{half}, address{half}, position{half}, index{half};",
                f".reg .pred store{half};",
                f".reg .{('b32' if staged else 'b64')} output{half};",
            ]
        )
    for half in range(key_count):
        ptx.extend(
            [
                f"bfe.u32 offset{half}, $0, {half * 16}, 16;",
                f"add.u32 address{half}, $1, offset{half};",
            ]
        )
    for half in range(key_count):
        ptx.append(f"atom.relaxed.cta.shared.add.u32 position{half}, [address{half}], 1;")
    for half in range(key_count):
        ptx.extend(
            [f"setp.lt.u32 store{half}, position{half}, $3;", f"add.u32 index{half}, $2, {half};"]
        )
        if staged:
            ptx.extend(
                [
                    f"@store{half} mad.lo.u32 output{half}, position{half}, 4, $4;",
                    f"@store{half} st.shared.u32 [output{half}], index{half};",
                ]
            )
        else:
            ptx.extend(
                [
                    f"@store{half} mad.wide.u32 output{half}, position{half}, 4, $4;",
                    f"@store{half} st.global.u32 [output{half}], index{half};",
                ]
            )
    ptx.append("}")
    llvm.inline_asm(
        res=None,
        operands_=[
            value.ir_value(loc=loc, ip=ip)
            for value in (encoded, table_addr, src_index, top_k, out_addr)
        ],
        asm_string="\n\t".join(ptx),
        constraints="r,r,r,r," + ("r" if staged else "l") + ",~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _scan_cursors256(address, bias, lane):
    """Rewrite the 256 low-byte counters at ``address`` as descending cursors (one warp).

    Lane l owns bins 8l..8l+7 at ``address`` + 32l. Bin i becomes ``bias`` + count in the bins
    above i, so larger keys take the first slots after the ``bias`` keys above the crossing bin.
    """
    row_addr = address + lane * Int32(32)
    cnt0, cnt1, cnt2, cnt3 = gvr._lds_v4_u32(row_addr)
    cnt4, cnt5, cnt6, cnt7 = gvr._lds_v4_u32(row_addr + Int32(16))
    subtotal = cnt0 + cnt1 + (cnt2 + cnt3) + (cnt4 + cnt5 + (cnt6 + cnt7))
    suffix = gvr.warp_suffix_scan_add(subtotal, lane)
    pos7 = bias + suffix - subtotal
    pos6 = pos7 + cnt7
    pos5 = pos6 + cnt6
    pos4 = pos5 + cnt5
    pos3 = pos4 + cnt4
    pos2 = pos3 + cnt3
    pos1 = pos2 + cnt2
    pos0 = pos1 + cnt1
    gvr._sts_v4_u32(row_addr, pos0, pos1, pos2, pos3)
    gvr._sts_v4_u32(row_addr + Int32(16), pos4, pos5, pos6, pos7)


@cute.jit
def _scan_above_lane_cursors(table, tid):
    """Rewrite the 32 per-lane above counters ``table[256:288]`` as exclusive lane prefixes.

    Called by one full warp (lane = ``tid`` & 31). Lane l's keys above the crossing bin then
    take slots [prefix, prefix + count) of [0, count above).
    """
    lane = tid & Int32(31)
    count = table[Int32(256) + lane]
    table[Int32(256) + lane] = gvr.warp_incl_scan_add(count, lane) - count


class GvrCodeRadixCounterKernel:
    """FP16 code-radix register row with a counter emit, one CTA per row.

    ``get_compiled__reg`` compiles this kernel in place of the ``GvrTopkRegKernel`` it
    configured when called with ``fp16_radix_impl=2``. The row stays in registers as packed
    ordered keys. Pass 1 histograms the high key byte, and ``find_cross256_select`` gives the
    crossing byte B and the count above it. Each key is then classified once
    (``_classify_key_pair``): valid keys inside B count on 256 low-byte counters, valid keys
    above B on 32 per-lane counters, and all other keys on 32 per-lane discard counters.
    Warp 0 turns the low-byte counters into descending cursors that start after the count
    above B, warp 1 turns the per-lane counters into slot bases below that count, and warp 2
    resets the discard counters to K. The emit takes one ticket per key from its encoded
    counter and keeps tickets below K, so no second crossing search is needed; ties at the
    K-th key are cut in atomic order. Rows whose valid length is at most K get the identity
    and a -1 tail.

    Shared memory in 32-bit words: result 0-5, high-byte histogram 16-271, low-byte counters
    272-527, above counters 528-559, discard counters 784-815 (``fixed_smem_bytes`` = 3264).
    With ``stage_output`` the K-word output row starts at byte 3264.

    Args:
        register_kernel: The ``GvrTopkRegKernel`` this kernel replaces. It must be FP16 with
            ``code_radix``, without ``det`` or ``pdl``, with ``parts`` == 1, ``blk`` in
            {256, 512, 1024} and 1 <= ``vpt`` <= 8. Its ``blk``, ``vpt``, ``minb``,
            ``varlen``, ``next_n``, ``cr_shift`` and ``stage_output`` are copied.
    """

    def __init__(self, register_kernel):
        assert register_kernel.dtype == cutlass.Float16
        assert register_kernel.code_radix and (not register_kernel.det)
        assert not register_kernel.pdl and register_kernel.parts == 1
        assert register_kernel.blk in (256, 512, 1024)
        assert 1 <= register_kernel.vpt <= 8
        self.dtype = cutlass.Float16
        self.blk = register_kernel.blk
        self.vpt = register_kernel.vpt
        self.minb = register_kernel.minb
        self.varlen = register_kernel.varlen
        self.next_n = register_kernel.next_n
        self.cr_shift = register_kernel.cr_shift
        self.stage_output = register_kernel.stage_output
        self.fixed_smem_bytes = 3264

    def _vector4(self, thread, slot: int):
        """Return the 8-byte vector index (four halfwords) that ``thread`` holds in ``slot``.

        With an even ``vpt`` slot pairs are adjacent vectors read by one 16-byte load; the map
        equals ``GvrTopkRegKernel._vector4_cr`` for FP16.
        """
        if self.vpt % 2 == 0:
            return (thread << Int32(1)) + Int32(slot // 2 * self.blk * 2 + slot % 2)
        return thread + Int32(slot * self.blk)

    @cute.kernel
    def kern_cr(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: Int32,
        cmp_: Int32,
        qc: Int32,
        smem_bytes: Int32,
    ):
        """Select the top-K indices of one row per CTA, as described in the class docstring.

        ``logits`` is [rows, npad] float16, ``pre_idx`` [rows, K] int32 (only K is read),
        ``kv_lens`` [requests] int32 (read when ``varlen``) and ``out`` [rows, K] int32; ``n`` is
        the envelope length in elements. ``cmp_``, ``qc`` and ``smem_bytes`` are unused and keep
        the ``GvrTopkRegKernel`` launch signature.
        """
        block = cutlass.const_expr(self.blk)
        slots = cutlass.const_expr(self.vpt)
        clear_n = cutlass.const_expr(800)
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        vector_id = Int32(0)
        out_index = tid
        load_vector = Int32(0)
        index = Int32(0)
        src_index = Int32(0)
        valid_pair = Int32(0)
        clear_id = Int32(0)
        clear_val = Int32(0)
        word = Uint32(0)
        input_addr = logits[row, None].iterator.toint()
        packed = [cute.make_rmem_tensor((2,), Uint32) for _ in range(slots)]
        env4 = n >> Int32(2)
        if env4 > Int32(block * slots):
            env4 = Int32(block * slots)
        if env4 == Int32(block * slots):
            if cutlass.const_expr(self.vpt % 2 == 0):
                for pair_step in cutlass.range_constexpr(slots // 2):
                    load_vector = self._vector4(tid, pair_step * 2)
                    (
                        packed[pair_step * 2][0],
                        packed[pair_step * 2][1],
                        packed[pair_step * 2 + 1][0],
                        packed[pair_step * 2 + 1][1],
                    ) = gvr._ld_g_nc_v4_b32(input_addr + Int64(load_vector) * Int64(8))
            else:
                for slot in cutlass.range_constexpr(slots):
                    packed[slot][0], packed[slot][1] = gvr._ld_g_nc_v2_b32(
                        input_addr + Int64(self._vector4(tid, slot)) * Int64(8)
                    )
        else:
            for slot in cutlass.range_constexpr(slots):
                vector_id = self._vector4(tid, slot)
                if vector_id < env4:
                    packed[slot][0], packed[slot][1] = gvr._ld_g_nc_v2_b32(
                        input_addr + Int64(vector_id) * Int64(8)
                    )
                else:
                    packed[slot][0] = Uint32(4294967295)
                    packed[slot][1] = Uint32(4294967295)
        env_tail = n - (env4 << Int32(2))
        tail_raw = Uint32(65535)
        if tid < env_tail:
            tail_raw = gvr._ld_g_nc_u16(input_addr + Int64((env4 << Int32(2)) + tid) * Int64(2))
        top_k = Int32(pre_idx.shape[1])
        valid_n = n
        if cutlass.const_expr(self.varlen):
            request = row // Int32(self.next_n)
            request_row = row % Int32(self.next_n)
            valid_n = kv_lens[request] - Int32(self.next_n) + request_row + Int32(1) >> Int32(
                self.cr_shift
            )
            if valid_n < Int32(0):
                valid_n = Int32(0)
            if valid_n > n:
                valid_n = n
        if valid_n <= top_k:
            out_index = tid
            while out_index < top_k:
                index = Int32(-1)
                if out_index < valid_n:
                    index = out_index
                out[row, out_index] = index
                out_index = out_index + Int32(block)
        else:
            shared_ptr = cute.arch.get_dyn_smem(Int32, alignment=16)
            smem_base = shared_ptr.toint()
            result = gvr._smem_view(Int32, smem_base, 0, 6)
            histograms = gvr._smem_view(Int32, smem_base, 16, clear_n)
            table = gvr._smem_view(Int32, smem_base, 272, 544)
            high_addr = gvr._smem_addr_reg__reg(smem_base + Int32(64))
            table_addr = high_addr + Int32(1024)
            discard = Int32(2048) + ((tid & Int32(31)) << Int32(2))
            above_lane = discard - Int32(1024)
            if cutlass.const_expr(self.stage_output):
                out_addr = gvr._smem_addr_reg__reg(smem_base + Int32(3264))
                stage_row = cute.make_tensor(
                    cute.make_ptr(Int32, out_addr, cute.AddressSpace.smem, assumed_align=16),
                    cute.make_layout((pre_idx.shape[1],)),
                )
            else:
                out_addr = out[row, None].iterator.toint()
            valid4 = valid_n >> Int32(2)
            if valid4 > Int32(block * slots):
                valid4 = Int32(block * slots)
            tail_n = valid_n - (valid4 << Int32(2))
            tail_id = (valid4 << Int32(2)) + tid
            pad = Int32(block * slots) - valid4 << Int32(2)
            for clear_step in cutlass.range_constexpr((clear_n + block - 1) // block):
                clear_id = tid + Int32(clear_step * block)
                clear_val = Int32(0)
                if cutlass.const_expr((clear_step + 1) * block <= clear_n):
                    histograms[clear_id] = clear_val
                elif clear_id < Int32(clear_n):
                    histograms[clear_id] = clear_val
            if tid == Int32(0):
                histograms[0] = Int32(0) - pad
            cute.arch.barrier()
            for slot in cutlass.range_constexpr(slots):
                if self._vector4(tid, slot) >= valid4:
                    packed[slot][0] = Uint32(4294967295)
                    packed[slot][1] = Uint32(4294967295)
                packed[slot][0] = ordered_key16x2(packed[slot][0])
                packed[slot][1] = ordered_key16x2(packed[slot][1])
            tail_key = Uint32(0)
            if tid < tail_n:
                if valid4 == env4:
                    tail_key = gvr.fkey16_of_code16(tail_raw)
                else:
                    tail_key = gvr.fkey16_of_code16(
                        gvr._ld_g_nc_u16(input_addr + Int64(tail_id) * Int64(2))
                    )
            for slot in cutlass.range_constexpr(slots):
                for pair_id in cutlass.range_constexpr(2):
                    word = packed[slot][pair_id]
                    gvr._red_shared_add1__reg(high_addr + Int32(word >> Uint32(6) & Uint32(1020)))
                    gvr._red_shared_add1__reg(high_addr + Int32(word >> Uint32(22) & Uint32(1020)))
            if tid < tail_n:
                gvr._red_shared_add1__reg(high_addr + Int32(tail_key >> Uint32(6) & Uint32(1020)))
            cute.arch.barrier()
            find_cross256_select(
                high_addr, top_k, tid, result, Int32(0), table, Int32(0), first_pass=True
            )
            cute.arch.barrier()
            cross_high = Uint32(result[gvr.RES_B])
            high_above = result[gvr.RES_ABOVE]
            for slot in cutlass.range_constexpr(slots):
                vector_id = self._vector4(tid, slot)
                valid_pair = Int32(vector_id < valid4)
                for pair_id in cutlass.range_constexpr(2):
                    packed[slot][pair_id] = _classify_key_pair(
                        packed[slot][pair_id],
                        cross_high,
                        valid_pair,
                        table_addr,
                        discard,
                        above_lane,
                    )
            if tid < tail_n:
                tail_key = _classify_key_pair(
                    tail_key, cross_high, Int32(1), table_addr, discard, above_lane, key_count=1
                )
            cute.arch.barrier()
            if tid < Int32(32):
                _scan_cursors256(table_addr, high_above, tid)
            if tid >= Int32(32) and tid < Int32(64):
                _scan_above_lane_cursors(table, tid)
            if tid >= Int32(64) and tid < Int32(96):
                table[Int32(512) + (tid & Int32(31))] = top_k
            cute.arch.barrier()
            for slot in cutlass.range_constexpr(slots):
                vector_id = self._vector4(tid, slot)
                for pair_id in cutlass.range_constexpr(2):
                    src_index = (vector_id << Int32(2)) + Int32(pair_id * 2)
                    emit_counter_pair(
                        packed[slot][pair_id],
                        table_addr,
                        src_index,
                        top_k,
                        out_addr,
                        staged=self.stage_output,
                    )
            if tid < tail_n:
                emit_counter_pair(
                    tail_key,
                    table_addr,
                    tail_id,
                    top_k,
                    out_addr,
                    staged=self.stage_output,
                    key_count=1,
                )
            if cutlass.const_expr(self.stage_output):
                cute.arch.barrier()
                if cutlass.const_expr(cute.is_static(out.shape[1])):
                    for output_step in cutlass.range_constexpr(out.shape[1] // block):
                        out_index = tid + Int32(output_step * block)
                        out[row, out_index] = stage_row[out_index]
                    if cutlass.const_expr(out.shape[1] % block != 0):
                        out_index = tid + Int32(out.shape[1] // block * block)
                        if out_index < top_k:
                            out[row, out_index] = stage_row[out_index]
                else:
                    out_index = tid
                    while out_index < top_k:
                        out[row, out_index] = stage_row[out_index]
                        out_index = out_index + Int32(block)

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: Int32,
        cmp_: Int32,
        qc: Int32,
        smem_bytes: Int32,
        stream,
    ):
        """Launch ``kern_cr`` with one ``blk``-thread CTA per logits row on ``stream``.

        Arguments match ``GvrTopkRegKernel.__call__``. ``smem_bytes`` is ignored: the dynamic
        shared memory is ``fixed_smem_bytes``, plus 4 * K bytes with ``stage_output``.
        """
        smem = self.fixed_smem_bytes
        if cutlass.const_expr(self.stage_output):
            smem = smem + pre_idx.shape[1] * 4
        self.kern_cr(logits, pre_idx, kv_lens, out, n, cmp_, qc, smem_bytes).launch(
            grid=(logits.shape[0], 1, 1),
            block=(self.blk, 1, 1),
            stream=stream,
            smem=smem,
            min_blocks_per_mp=self.minb,
        )


# ---- cluster counter emit (kern_crc) ----


@dsl_user_op
def classify_key_pair_cluster(
    packed,
    cross_high,
    valid,
    table_addr,
    discard,
    above_lane,
    *,
    key_count: cutlass.Constexpr = 2,
    loc=None,
    ip=None,
):
    """Cluster form of ``_classify_key_pair`` for ``GvrTopkRegKernel.kern_crc``.

    A crossing-bin key counts on the pass-2 histogram at ``table_addr`` + 1024 + 4 * low byte but
    encodes 4 * low byte, where ``merge_cursors256_cluster`` writes the merged cursors. High bytes
    are compared as FP16x2 subnormals (exact only without ``.ftz``). Invalid keys encode ``discard``
    but count by value, so they must be key 0 (``gvr._crc_merge1`` sets bin 0 to -padding).
    """
    assembly = [
        "{",
        ".reg .b32 high, pattern, above, member, lower, shifted, accepted, rejected, counts, encoded, offset, address;",
        ".reg .pred active;",
        "setp.ne.u32 active, $3, 0;",
        "shr.u32 high, $1, 8;",
        "and.b32 high, high, 0x00ff00ff;",
        "mul.lo.u32 pattern, $2, 0x00010001;",
        "set.gt.u32.f16x2 above, high, pattern;",
        "set.ge.u32.f16x2 member, high, pattern;",
        "and.b32 lower, $1, 0x00ff00ff;",
        "shl.b32 lower, lower, 2;",
        "add.u32 shifted, lower, 0x04000400;",
        "mul.lo.u32 accepted, $6, 0x00010001;",
        "mul.lo.u32 rejected, $5, 0x00010001;",
        "lop3.b32 counts, above, accepted, shifted, 0xca;",
        "lop3.b32 counts, member, counts, rejected, 0xca;",
        "lop3.b32 encoded, above, accepted, lower, 0xca;",
        "lop3.b32 encoded, member, encoded, rejected, 0xca;",
        "@!active mov.b32 encoded, rejected;",
    ]
    for half in range(key_count):
        assembly.extend(
            [
                f"bfe.u32 offset, counts, {16 * half}, 16;",
                "add.u32 address, $4, offset;",
                "red.relaxed.cta.shared.add.u32 [address], 1;",
            ]
        )
    assembly.extend(["mov.b32 $0, encoded;", "}"])
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                value.ir_value(loc=loc, ip=ip)
                for value in (packed, cross_high, valid, table_addr, discard, above_lane)
            ],
            "\n\t".join(assembly),
            "=r,r,r,r,r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def merge_cursors256_cluster(recv, hcur, rank, bias, lane, cs: cutlass.Constexpr):
    """Merge the cluster's pass-2 histograms into this CTA's descending cursors (one warp).

    ``recv`` holds every peer's 256-bin histogram as pushed by ``gvr._crc_push_hist``. Bin i of
    ``hcur`` becomes ``bias`` + cluster count above bin i + the count of bin i on lower-ranked
    peers, so each CTA owns a disjoint part of every bin. Unlike ``gvr._crc_merge2`` it does no
    crossing search.
    """
    total = cute.make_rmem_tensor((8,), Int32)
    lower = cute.make_rmem_tensor((8,), Int32)
    gvr._crc_sum_peers(recv, lane, rank, total, lower, cs, True)
    subtotal = ((total[0] + total[1]) + (total[2] + total[3])) + (
        (total[4] + total[5]) + (total[6] + total[7])
    )
    suffix = gvr.warp_suffix_scan_add(subtotal, lane)
    cursor7 = bias + suffix - subtotal
    cursor6 = cursor7 + total[7]
    cursor5 = cursor6 + total[6]
    cursor4 = cursor5 + total[5]
    cursor3 = cursor4 + total[4]
    cursor2 = cursor3 + total[3]
    cursor1 = cursor2 + total[2]
    cursor0 = cursor1 + total[1]
    address = hcur + (lane << Int32(5))
    gvr._sts_v4_u32(
        address, cursor0 + lower[0], cursor1 + lower[1], cursor2 + lower[2], cursor3 + lower[3]
    )
    gvr._sts_v4_u32(
        address + Int32(16),
        cursor4 + lower[4],
        cursor5 + lower[5],
        cursor6 + lower[6],
        cursor7 + lower[7],
    )


@cute.jit
def scan_above_lane_cursors_cluster(hcur, rank_above, rank, lane):
    """Rewrite this CTA's 32 per-lane above counters at ``hcur`` as cluster-wide slot bases.

    ``rank_above`` holds each peer's local count above the crossing bin, one word per rank.
    Lane l's base is the sum over lower-ranked peers plus the counts of lanes below l. One warp.
    """
    peer_above = Int32(0)
    if lane < rank:
        peer_above = gvr._lds_i32(rank_above + (lane << Int32(2)))
    bias = gvr.warp_add_i32(peer_above)
    address = hcur + (lane << Int32(2))
    count = gvr._lds_i32(address)
    cursor = bias + gvr.warp_incl_scan_add(count, lane) - count
    cursors = gvr._smem_view(Int32, hcur, 0, 32)
    cursors[lane] = cursor


__all__ = [
    "GvrCodeRadixCounterKernel",
    "atom_shared_add",
    "classify_key_pair_cluster",
    "crossing_bin8_above",
    "emit_complement",
    "emit_counter_pair",
    "emit_masked_indices",
    "find_cross256_select",
    "histogram_crossing_low_bytes",
    "histogram_high_bytes",
    "merge_cursors256_cluster",
    "ordered_key16x2",
    "ordered_key16x2_nan_max",
    "repair_complement",
    "scan_above_lane_cursors_cluster",
    "st_global_v4_b32",
    "threshold_mask_ge",
    "threshold_masks",
    "warp_inclusive_sum",
]
