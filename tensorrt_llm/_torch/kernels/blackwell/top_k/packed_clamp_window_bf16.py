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
"""BF16 packed clamp window register kernel for the 16-bit decode register plans.

A row is held in registers as packed BF16 pairs, by one CTA or by `parts` CTAs (partitioned emit). A sampled
anchor places a window of `nbh` raw BF16 codes on the side of zero that holds the K-th value; codes outside the
window clamp into its edge bins. When the crossing bin is an interior bin (a single code) every element is
emitted by window cursor with no refinement; otherwise the row takes an exact radix select over the 16-bit
codes. The host routes a row here only when its row stride is a multiple of 64 elements.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as dsl_types
from cutlass.cutlass_dsl import dsl_user_op

if __package__:
    from . import gvr_topk_decode_self_sampling as _seed
    from .warp_histogram_scan_bf16 import scan_cross_small_warp_parts_u32, scan_cross_small_warp_u32
else:
    import gvr_topk_decode_self_sampling as _seed
    from warp_histogram_scan_bf16 import scan_cross_small_warp_parts_u32, scan_cross_small_warp_u32


@dsl_user_op
def _normalize_pair(packed_pair: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 nan_mask;\n\t"
            "set.nan.u32.bf16x2 nan_mask, $1, $1;\n\t"
            "lop3.b32 $0, $1, nan_mask, 0x80008000, 0x70;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sign_quad(
    lower_pair: cutlass.Uint32, upper_pair: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [lower_pair.ir_value(loc=loc, ip=ip), upper_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs;\n\t"
            "prmt.b32 signs, $1, $2, 0x7531;\n\t"
            "and.b32 $0, signs, 0x80808080;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ordered_pair(packed_pair: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 sign_mask;\n\t"
            "prmt.b32 sign_mask, $1, 0, 0xbb99;\n\t"
            "lop3.b32 $0, $1, 0x80008000, sign_mask, 0x1e;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half_code(
    packed_pair: cutlass.Uint32,
    upper: bool,
    signed: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    if signed:
        selector = "0xbb32" if upper else "0x9910"
    else:
        selector = "0x4432" if upper else "0x4410"
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            f"prmt.b32 $0, $1, 0, {selector};",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _histogram_raw_code(
    histogram_address: cutlass.Int32,
    raw_code: cutlass.Int32,
    window_base: cutlass.Int32,
    bins: int,
    positive: bool,
    *,
    loc=None,
    ip=None,
) -> None:
    difference = "sub.s32 offset, $1, $2;\n\t"
    if not positive:
        difference = "sub.s32 offset, $2, $1;\n\t"
    llvm.inline_asm(
        res=None,
        operands_=[
            histogram_address.ir_value(loc=loc, ip=ip),
            raw_code.ir_value(loc=loc, ip=ip),
            window_base.ir_value(loc=loc, ip=ip),
        ],
        asm_string="{\n\t.reg .b32 offset, bin_index, address;\n\t"
        ".reg .pred included;\n\t"
        + difference
        + "setp.ge.s32 included, offset, 0;\n\t"
        + f"min.u32 bin_index, offset, {bins - 1};\n\t"
        "mad.lo.u32 address, bin_index, 4, $0;\n\t"
        "@included red.relaxed.cta.shared.add.u32 [address], 1;\n}",
        constraints="r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _raw_window_cursor(
    histogram_address: cutlass.Int32,
    raw_code: cutlass.Int32,
    window_base: cutlass.Int32,
    raw_cut: cutlass.Int32,
    bins: int,
    positive: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    difference = "sub.s32 bin_index, $2, $3;\n\t"
    comparison = "setp.ge.s32 selected, $2, $4;\n\t"
    if not positive:
        difference = "sub.s32 bin_index, $3, $2;\n\t"
        comparison = "setp.le.u32 selected, $2, $4;\n\t"
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [
                histogram_address.ir_value(loc=loc, ip=ip),
                raw_code.ir_value(loc=loc, ip=ip),
                window_base.ir_value(loc=loc, ip=ip),
                raw_cut.ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 bin_index, address;\n\t"
            ".reg .pred selected;\n\t"
            + difference
            + f"min.u32 bin_index, bin_index, {bins - 1};\n\t"
            "mad.lo.u32 address, bin_index, 4, $1;\n\t" + comparison + "mov.u32 $0, 0x40000000;\n\t"
            "@selected atom.relaxed.cta.shared.add.u32 $0, [address], 1;\n}",
            "=r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_output_quad(
    destination_address: cutlass.Int64,
    first_index: cutlass.Int32,
    second_index: cutlass.Int32,
    third_index: cutlass.Int32,
    fourth_index: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        res=None,
        operands_=[
            destination_address.ir_value(loc=loc, ip=ip),
            first_index.ir_value(loc=loc, ip=ip),
            second_index.ir_value(loc=loc, ip=ip),
            third_index.ir_value(loc=loc, ip=ip),
            fourth_index.ir_value(loc=loc, ip=ip),
        ],
        asm_string="st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        constraints="l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _copy_output_quad(source_address, destination_address, vector_index):
    byte_offset = vector_index * cutlass.Int32(16)
    first_index, second_index, third_index, fourth_index = _seed._lds_v4_u32(
        source_address + byte_offset
    )
    _store_output_quad(
        destination_address + cutlass.Int64(byte_offset),
        first_index,
        second_index,
        third_index,
        fourth_index,
    )


@cute.jit
def _radix_cursor(key, crossing_high, threshold, high_address, low_address, tie_address):
    high_byte = key >> cutlass.Uint32(8)
    cursor_address = high_address + cutlass.Int32(high_byte << cutlass.Uint32(2))
    if high_byte == crossing_high:
        cursor_address = low_address + cutlass.Int32(
            (key & cutlass.Uint32(255)) << cutlass.Uint32(2)
        )
    if key == threshold:
        cursor_address = tie_address
    return _seed._atom_shared_add1_if_ge(cursor_address, key, threshold)


@dsl_user_op
def _packed_clamp_bins(
    packed_pair: cutlass.Uint32,
    flip: cutlass.Uint32,
    lower: cutlass.Uint32,
    upper: cutlass.Uint32,
    pivot: cutlass.Uint32,
    positive: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    operands = (packed_pair, flip, lower, upper, pivot)
    instructions = (
        "xor.b32 code_pair, $1, $2;\n\t"
        "max.u16x2 code_pair, code_pair, $3;\n\t"
        "min.u16x2 code_pair, code_pair, $4;\n\t"
        "sub.u32 $0, code_pair, $5;\n}"
    )
    constraints = "=r,r,r,r,r,r"
    if positive:
        operands = (packed_pair, lower, upper, pivot)
        instructions = (
            "max.s16x2 code_pair, $1, $2;\n\t"
            "min.s16x2 code_pair, code_pair, $3;\n\t"
            "sub.u32 $0, code_pair, $4;\n}"
        )
        constraints = "=r,r,r,r,r"
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [value.ir_value(loc=loc, ip=ip) for value in operands],
            "{\n\t.reg .b32 code_pair;\n\t" + instructions,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _packed_clamp_cursor(
    histogram_address: cutlass.Int32,
    bin_index: cutlass.Int32,
    crossing_bin: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [
                value.ir_value(loc=loc, ip=ip)
                for value in (histogram_address, bin_index, crossing_bin)
            ],
            "{\n\t.reg .b32 address;\n\t.reg .pred selected;\n\t"
            "setp.ge.u32 selected, $2, $3;\n\t"
            "mad.lo.u32 address, $2, 4, $1;\n\t"
            "mov.u32 $0, 0x40000000;\n\t"
            "@selected atom.relaxed.cta.shared.add.u32 $0, [address], 1;\n}",
            "=r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class GvrTopkPackedClampWindowBF16Kernel:
    """Decode top-K for BF16 rows: `blk` threads, `vpt` 8-element vectors per thread, `nbh` window bins,
    `parts` CTAs per row, optional staged output."""

    def __init__(
        self,
        blk: int,
        vpt: int,
        nbh: int = 1024,
        next_n: int = 1,
        cr_shift: int = 0,
        *,
        minb: int = 1,
        varlen: bool = True,
        parts: int = 1,
        stage_output: bool = False,
        dtype=cutlass.BFloat16,
    ) -> None:
        assert dtype == cutlass.BFloat16
        assert blk in (256, 512, 1024) and 1 <= vpt <= 10
        assert nbh in (256, 512, 1024, 2048)
        assert next_n >= 1 and 0 <= cr_shift < 31
        assert minb >= 1 and minb * blk <= 2048
        assert parts in (1, 2, 4)
        assert not (stage_output and parts > 1)
        self.blk = int(blk)
        self.vpt = int(vpt)
        self.nbh = int(nbh)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        self.minb = int(minb)
        self.varlen = bool(varlen)
        self.parts = int(parts)
        self.part_shift = self.parts.bit_length() - 1
        self.stage_output = bool(stage_output)
        self.resident_sample = self.blk * 4 * self.vpt >= 16384
        self.sample_warp = self.blk // 128
        self.dtype = dtype
        self.capacity = self.blk * (4 * self.vpt + 1)
        self.histogram_words = max(512, self.parts * self.nbh)
        self.output_offset_words = _seed.STATIC_WORDS + self.histogram_words
        self.required_smem_bytes = self.output_offset_words * 4

    def _histogram_offset(self, scalar_fragment: int) -> int:
        return (scalar_fragment * self.parts // (4 * self.vpt)) * self.nbh * 4

    @cute.jit
    def _window_controls(self, window_base, thread_index, positive: cutlass.Constexpr):
        code_base = window_base
        flip = cutlass.Uint32(0x80008000)
        maximum_base = cutlass.Int32(32768 - self.nbh)
        if cutlass.const_expr(not positive):
            code_base = cutlass.Int32(0xFFFF) - window_base
            flip = cutlass.Uint32(0xFFFFFFFF)
            maximum_base = cutlass.Int32(65536 - self.nbh)
        if code_base < cutlass.Int32(0):
            code_base = cutlass.Int32(0)
        if code_base > maximum_base:
            code_base = maximum_base
        lane_index = thread_index & cutlass.Int32(31)
        pivot = cutlass.Uint32(code_base) * cutlass.Uint32(65537)
        lower = cutlass.Uint32(code_base + lane_index) * cutlass.Uint32(65537)
        upper = cutlass.Uint32(
            code_base + cutlass.Int32(self.nbh - 32) + lane_index
        ) * cutlass.Uint32(65537)
        return flip, lower, upper, pivot

    @cute.jit
    def _fill_window_histogram(
        self,
        packed_fragments,
        raw_tail,
        histogram_address,
        window_base,
        thread_index,
        body_vectors,
        tail_count,
        positive: cutlass.Constexpr,
    ):
        flip, lower, upper, pivot = self._window_controls(window_base, thread_index, positive)
        for fragment_index in cutlass.range_constexpr(self.vpt):
            vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
            if vector_index < body_vectors:
                for pair_index in cutlass.range_constexpr(2):
                    bins = _packed_clamp_bins(
                        packed_fragments[fragment_index][pair_index],
                        flip,
                        lower,
                        upper,
                        pivot,
                        positive,
                    )
                    packed_fragments[fragment_index][pair_index] = bins
                    for half_index in cutlass.range_constexpr(2):
                        bin_index = _half_code(bins, half_index == 1, False)
                        _seed._red_shared_add1__reg(
                            histogram_address
                            + cutlass.Int32(
                                self._histogram_offset(
                                    fragment_index * 4 + pair_index * 2 + half_index
                                )
                            )
                            + bin_index * cutlass.Int32(4)
                        )
        tail_bin = cutlass.Int32(0)
        if thread_index < tail_count:
            tail_bins = _packed_clamp_bins(raw_tail, flip, lower, upper, pivot, positive)
            tail_bin = _half_code(tail_bins, False, False)
            _seed._red_shared_add1__reg(
                histogram_address
                + cutlass.Int32((self.parts - 1) * self.nbh * 4)
                + tail_bin * cutlass.Int32(4)
            )
        return tail_bin

    @cute.jit
    def _emit_window(
        self,
        packed_fragments,
        tail_bin,
        output_row,
        thread_index,
        part_index,
        tail_index,
        histogram_address,
        window_base,
        crossing_bin,
        top_k,
        body_vectors,
        tail_count,
        positive: cutlass.Constexpr,
    ):
        for group_index in cutlass.range_constexpr(self.parts):
            if part_index == cutlass.Int32(group_index):
                for scalar_fragment in cutlass.range_constexpr(
                    group_index * 4 * self.vpt // self.parts,
                    (group_index + 1) * 4 * self.vpt // self.parts,
                ):
                    vector_index = thread_index + cutlass.Int32((scalar_fragment // 4) * self.blk)
                    if vector_index < body_vectors:
                        bins = packed_fragments[scalar_fragment // 4][(scalar_fragment % 4) // 2]
                        bin_index = _half_code(bins, scalar_fragment % 2 == 1, False)
                        output_position = _packed_clamp_cursor(
                            histogram_address, bin_index, crossing_bin
                        )
                        if output_position < top_k:
                            selected_index = vector_index * cutlass.Int32(4) + cutlass.Int32(
                                scalar_fragment % 4
                            )
                            output_row[output_position] = selected_index
                if cutlass.const_expr(group_index == self.parts - 1):
                    if thread_index < tail_count:
                        output_position = _packed_clamp_cursor(
                            histogram_address, tail_bin, crossing_bin
                        )
                        if output_position < top_k:
                            output_row[output_position] = tail_index

    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        envelope: cutlass.Int32,
    ):
        thread_index, _, _ = cute.arch.thread_idx()
        block_index, _, _ = cute.arch.block_idx()
        row_index = block_index >> cutlass.Int32(self.part_shift)
        part_index = block_index & cutlass.Int32(self.parts - 1)
        lane_index = thread_index & cutlass.Int32(31)
        warp_index = thread_index >> cutlass.Int32(5)
        top_k = cutlass.Int32(pre_idx.shape[1])
        shared_pointer = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
        shared_base = shared_pointer.toint()
        if cutlass.const_expr(self.stage_output):
            output_row = _seed._smem_view(
                cutlass.Int32, shared_base, self.output_offset_words, pre_idx.shape[1]
            )
        else:
            output_row = out[row_index, None]
        envelope_body_vectors = envelope >> cutlass.Int32(2)
        if envelope_body_vectors > cutlass.Int32(self.blk * self.vpt):
            envelope_body_vectors = cutlass.Int32(self.blk * self.vpt)
        envelope_tail_begin = envelope_body_vectors << cutlass.Int32(2)
        envelope_tail_count = envelope - envelope_tail_begin
        envelope_tail_index = envelope_tail_begin + thread_index
        row_address = logits[row_index, None].iterator.toint()
        packed_fragments = [cute.make_rmem_tensor((2,), cutlass.Uint32) for _ in range(self.vpt)]
        for fragment_index in cutlass.range_constexpr(self.vpt):
            packed_fragments[fragment_index][0] = cutlass.Uint32(0xFFFFFFFF)
            packed_fragments[fragment_index][1] = cutlass.Uint32(0xFFFFFFFF)
            vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
            if vector_index < envelope_body_vectors:
                lower_word, upper_word = _seed._ld_g_nc_v2_b32(
                    row_address + cutlass.Int64(vector_index) * cutlass.Int64(8)
                )
                packed_fragments[fragment_index][0] = lower_word
                packed_fragments[fragment_index][1] = upper_word
        envelope_raw_tail = cutlass.Uint32(0xFFFF)
        if thread_index < envelope_tail_count:
            envelope_raw_tail = _seed._ld_g_nc_u16(
                row_address + cutlass.Int64(envelope_tail_index) * cutlass.Int64(2)
            )
        shared_histogram = _seed._smem_view(
            cutlass.Int32, shared_base, _seed.STATIC_WORDS, self.histogram_words
        )
        histogram_address = _seed._smem_addr_reg__reg(
            shared_base + cutlass.Int32(_seed.STATIC_WORDS * 4)
        )
        for histogram_batch in cutlass.range_constexpr(
            (self.parts * self.nbh + self.blk - 1) // self.blk
        ):
            histogram_index = thread_index + cutlass.Int32(histogram_batch * self.blk)
            if histogram_index < cutlass.Int32(self.parts * self.nbh):
                shared_histogram[histogram_index] = cutlass.Int32(0)
        row_length = envelope
        if cutlass.const_expr(self.varlen):
            request_index = row_index // cutlass.Int32(self.next_n)
            query_position = row_index % cutlass.Int32(self.next_n)
            row_length = (
                kv_lens[request_index]
                - cutlass.Int32(self.next_n)
                + query_position
                + cutlass.Int32(1)
            ) >> cutlass.Int32(self.cr_shift)
        if row_length < cutlass.Int32(0):
            row_length = cutlass.Int32(0)
        if row_length > envelope:
            row_length = envelope
        if row_length <= top_k:
            if part_index == cutlass.Int32(0):
                output_index = thread_index
                while output_index < top_k:
                    selected_index = cutlass.Int32(-1)
                    if output_index < row_length:
                        selected_index = output_index
                    output_row[output_index] = selected_index
                    output_index = output_index + cutlass.Int32(self.blk)
        else:
            body_vectors = row_length >> cutlass.Int32(2)
            if body_vectors > cutlass.Int32(self.blk * self.vpt):
                body_vectors = cutlass.Int32(self.blk * self.vpt)
            tail_begin = body_vectors << cutlass.Int32(2)
            tail_count = row_length - tail_begin
            tail_index = tail_begin + thread_index
            local_minimum = cutlass.Uint32(0xFFFF)
            local_maximum = cutlass.Uint32(0)
            positive_local = cutlass.Int32(0)
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if vector_index < body_vectors:
                    packed_fragments[fragment_index][0] = _normalize_pair(
                        packed_fragments[fragment_index][0]
                    )
                    packed_fragments[fragment_index][1] = _normalize_pair(
                        packed_fragments[fragment_index][1]
                    )
                    positive_local = (
                        positive_local
                        + cutlass.Int32(4)
                        - cutlass.Int32(
                            cute.arch.popc(
                                cutlass.Int32(
                                    _sign_quad(
                                        packed_fragments[fragment_index][0],
                                        packed_fragments[fragment_index][1],
                                    )
                                )
                            )
                        )
                    )
                    if cutlass.const_expr(not self.resident_sample):
                        for pair_index in cutlass.range_constexpr(2):
                            raw_pair = packed_fragments[fragment_index][pair_index]
                            sample_begin = vector_index * cutlass.Int32(4) + cutlass.Int32(
                                pair_index * 2
                            )
                            if sample_begin < top_k:
                                sample_pair = _ordered_pair(raw_pair)
                                for half_index in cutlass.range_constexpr(2):
                                    sample_index = sample_begin + cutlass.Int32(half_index)
                                    if sample_index < top_k:
                                        sample_key = cutlass.Uint32(
                                            _half_code(sample_pair, half_index == 1, False)
                                        )
                                        if sample_key < local_minimum:
                                            local_minimum = sample_key
                                        if sample_key > local_maximum:
                                            local_maximum = sample_key
                else:
                    packed_fragments[fragment_index][0] = cutlass.Uint32(0xFFFFFFFF)
                    packed_fragments[fragment_index][1] = cutlass.Uint32(0xFFFFFFFF)
            raw_tail = cutlass.Uint32(0xFFFF)
            if thread_index < tail_count:
                if body_vectors == envelope_body_vectors:
                    raw_tail = envelope_raw_tail
                else:
                    raw_tail = _seed._ld_g_nc_u16(
                        row_address + cutlass.Int64(tail_index) * cutlass.Int64(2)
                    )
                raw_tail = _normalize_pair(raw_tail)
                positive_local = positive_local + cutlass.Int32(
                    (raw_tail & cutlass.Uint32(0x8000)) == cutlass.Uint32(0)
                )
                if cutlass.const_expr(not self.resident_sample):
                    if tail_index < top_k:
                        sample_key = cutlass.Uint32(_seed.fkey16_of_code16(raw_tail))
                        if sample_key < local_minimum:
                            local_minimum = sample_key
                        if sample_key > local_maximum:
                            local_maximum = sample_key

            shared_result = _seed._smem_view(cutlass.Int32, shared_base, 0, 6)
            shared_count = _seed._smem_view(cutlass.Int32, shared_base, 6, 2, align=4)
            shared_work = _seed._smem_view(cutlass.Int32, shared_base, 16, 32)
            positive_local = cutlass.Int32(cute.arch.warp_redux_sync(positive_local, "add"))
            if lane_index == cutlass.Int32(0):
                shared_work[warp_index] = positive_local
            if cutlass.const_expr(self.resident_sample):
                shared_samples = _seed._smem_view(cutlass.Uint32, shared_base, 48, 3)
                if warp_index == cutlass.Int32(self.sample_warp):
                    local_minimum = cutlass.Uint32(0xFFFF)
                    local_maximum = cutlass.Uint32(0)
                    for fragment_index in cutlass.range_constexpr(self.vpt):
                        vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                        if vector_index < body_vectors:
                            sample_pair = _ordered_pair(packed_fragments[fragment_index][1])
                            sample_key = cutlass.Uint32(_half_code(sample_pair, True, False))
                            if sample_key < local_minimum:
                                local_minimum = sample_key
                            if sample_key > local_maximum:
                                local_maximum = sample_key
                    local_minimum = cute.arch.warp_redux_sync(local_minimum, "min")
                    local_maximum = cute.arch.warp_redux_sync(local_maximum, "max")
                    if lane_index == cutlass.Int32(0):
                        shared_samples[0] = local_minimum
                        shared_samples[1] = local_maximum
                cute.arch.barrier()
                sample_minimum = cutlass.Uint32(shared_samples[0])
                sample_maximum = cutlass.Uint32(shared_samples[1])
                positive_count = cutlass.Int32(0)
                if lane_index < cutlass.Int32(self.blk // 32):
                    positive_count = shared_work[lane_index]
                positive_count = cutlass.Int32(cute.arch.warp_redux_sync(positive_count, "add"))
                negative_cut = positive_count < top_k
            else:
                shared_minimum = _seed._smem_view(cutlass.Uint32, shared_base, 48, 32)
                shared_maximum = _seed._smem_view(cutlass.Uint32, shared_base, 80, 32)
                local_minimum = cute.arch.warp_redux_sync(local_minimum, "min")
                local_maximum = cute.arch.warp_redux_sync(local_maximum, "max")
                if lane_index == cutlass.Int32(0):
                    shared_minimum[warp_index] = local_minimum
                    shared_maximum[warp_index] = local_maximum
                cute.arch.barrier()
                sample_minimum = cutlass.Uint32(0xFFFF)
                sample_maximum = cutlass.Uint32(0)
                positive_count = cutlass.Int32(0)
                if lane_index < cutlass.Int32(self.blk // 32):
                    sample_minimum = cutlass.Uint32(shared_minimum[lane_index])
                    sample_maximum = cutlass.Uint32(shared_maximum[lane_index])
                    positive_count = cutlass.Int32(shared_work[lane_index])
                sample_minimum = cute.arch.warp_redux_sync(sample_minimum, "min")
                sample_maximum = cute.arch.warp_redux_sync(sample_maximum, "max")
                positive_count = cutlass.Int32(cute.arch.warp_redux_sync(positive_count, "add"))
                negative_cut = positive_count < top_k
            window_base = (
                cutlass.Int32(sample_maximum) - cutlass.Int32(0x8000) - cutlass.Int32(self.nbh - 33)
            )
            if negative_cut:
                window_base = cutlass.Int32(
                    sample_minimum ^ cutlass.Uint32(0xFFFF)
                ) + cutlass.Int32(1)
            elif window_base < cutlass.Int32(0):
                window_base = cutlass.Int32(0)
            tail_bin = cutlass.Int32(0)
            if negative_cut:
                tail_bin = self._fill_window_histogram(
                    packed_fragments,
                    raw_tail,
                    histogram_address,
                    window_base,
                    thread_index,
                    body_vectors,
                    tail_count,
                    positive=False,
                )
            else:
                tail_bin = self._fill_window_histogram(
                    packed_fragments,
                    raw_tail,
                    histogram_address,
                    window_base,
                    thread_index,
                    body_vectors,
                    tail_count,
                    positive=True,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.nbh <= 512 and self.parts > 1):
                scan_cross_small_warp_parts_u32(
                    shared_histogram,
                    top_k,
                    thread_index,
                    shared_result,
                    part_index,
                    nb=self.nbh,
                    parts=self.parts,
                )
            elif cutlass.const_expr(self.nbh <= 512):
                scan_cross_small_warp_u32(
                    shared_histogram,
                    top_k,
                    thread_index,
                    shared_result,
                    nb=self.nbh,
                )
            elif cutlass.const_expr(self.parts > 1):
                _seed.scan_cross_w_parts(
                    shared_histogram,
                    shared_work,
                    top_k,
                    thread_index,
                    shared_result,
                    part_index,
                    blk=self.blk,
                    nb=self.nbh,
                    parts=self.parts,
                )
            else:
                _seed.scan_cross_w(
                    shared_histogram,
                    shared_work,
                    top_k,
                    thread_index,
                    shared_result,
                    blk=self.blk,
                    nb=self.nbh,
                )
            cute.arch.barrier()
            crossing_bin = shared_result[_seed.RES_B]
            if (
                shared_result[_seed.RES_TOT] >= top_k
                and shared_result[_seed.RES_M] > cutlass.Int32(0)
                and crossing_bin >= cutlass.Int32(32)
                and crossing_bin < cutlass.Int32(self.nbh - 32)
                and (not negative_cut or window_base - crossing_bin >= cutlass.Int32(0x8000))
            ):
                if negative_cut:
                    self._emit_window(
                        packed_fragments,
                        tail_bin,
                        output_row,
                        thread_index,
                        part_index,
                        tail_index,
                        histogram_address,
                        window_base,
                        crossing_bin,
                        top_k,
                        body_vectors,
                        tail_count,
                        positive=False,
                    )
                else:
                    self._emit_window(
                        packed_fragments,
                        tail_bin,
                        output_row,
                        thread_index,
                        part_index,
                        tail_index,
                        histogram_address,
                        window_base,
                        crossing_bin,
                        top_k,
                        body_vectors,
                        tail_count,
                        positive=True,
                    )
            elif part_index == cutlass.Int32(0):
                for fragment_index in cutlass.range_constexpr(self.vpt):
                    vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                    if vector_index < body_vectors:
                        fallback_lower, fallback_upper = _seed._ld_g_nc_v2_b32(
                            row_address + cutlass.Int64(vector_index) * cutlass.Int64(8)
                        )
                        packed_fragments[fragment_index][0] = _normalize_pair(fallback_lower)
                        packed_fragments[fragment_index][1] = _normalize_pair(fallback_upper)
                for fragment_index in cutlass.range_constexpr(self.vpt):
                    for pair_index in cutlass.range_constexpr(2):
                        packed_fragments[fragment_index][pair_index] = _ordered_pair(
                            packed_fragments[fragment_index][pair_index]
                        )
                tail_key = cutlass.Uint32(0)
                if thread_index < tail_count:
                    tail_key = cutlass.Uint32(_seed.fkey16_of_code16(raw_tail))
                shared_low = _seed._smem_view(
                    cutlass.Int32, shared_base, _seed.STATIC_WORDS + 256, 256
                )
                low_address = histogram_address + cutlass.Int32(1024)
                for histogram_batch in cutlass.range_constexpr((512 + self.blk - 1) // self.blk):
                    histogram_index = thread_index + cutlass.Int32(histogram_batch * self.blk)
                    if histogram_index < cutlass.Int32(512):
                        shared_histogram[histogram_index] = cutlass.Int32(0)
                cute.arch.barrier()
                for fragment_index in cutlass.range_constexpr(self.vpt):
                    vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                    if vector_index < body_vectors:
                        for pair_index in cutlass.range_constexpr(2):
                            key_pair = packed_fragments[fragment_index][pair_index]
                            for half_index in cutlass.range_constexpr(2):
                                key = cutlass.Uint32(_half_code(key_pair, half_index == 1, False))
                                _seed._red_shared_add1__reg(
                                    histogram_address
                                    + cutlass.Int32((key >> cutlass.Uint32(8)) << cutlass.Uint32(2))
                                )
                if thread_index < tail_count:
                    _seed._red_shared_add1__reg(
                        histogram_address
                        + cutlass.Int32((tail_key >> cutlass.Uint32(8)) << cutlass.Uint32(2))
                    )
                cute.arch.barrier()
                _seed.find_cross_b6(
                    histogram_address,
                    top_k,
                    thread_index,
                    shared_result,
                    cutlass.Int32(0),
                    shared_count,
                    shared_low,
                    cutlass.Int32(0),
                    fix=False,
                )
                cute.arch.barrier()
                crossing_high = cutlass.Uint32(shared_result[_seed.RES_B])
                above_high = shared_result[_seed.RES_ABOVE]
                for fragment_index in cutlass.range_constexpr(self.vpt):
                    vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                    if vector_index < body_vectors:
                        for pair_index in cutlass.range_constexpr(2):
                            key_pair = packed_fragments[fragment_index][pair_index]
                            for half_index in cutlass.range_constexpr(2):
                                key = cutlass.Uint32(_half_code(key_pair, half_index == 1, False))
                                _seed._red_shared_add1_if_eq(
                                    low_address
                                    + cutlass.Int32(
                                        (key & cutlass.Uint32(255)) << cutlass.Uint32(2)
                                    ),
                                    key >> cutlass.Uint32(8),
                                    crossing_high,
                                )
                if thread_index < tail_count:
                    _seed._red_shared_add1_if_eq(
                        low_address
                        + cutlass.Int32((tail_key & cutlass.Uint32(255)) << cutlass.Uint32(2)),
                        tail_key >> cutlass.Uint32(8),
                        crossing_high,
                    )
                cute.arch.barrier()
                _seed.find_cross_b6(
                    low_address,
                    top_k - above_high,
                    thread_index,
                    shared_result,
                    above_high,
                    shared_count,
                    shared_low,
                    cutlass.Int32(0),
                    fix=False,
                )
                cute.arch.barrier()
                threshold = (crossing_high << cutlass.Uint32(8)) | cutlass.Uint32(
                    shared_result[_seed.RES_B]
                )
                tie_address = shared_base + cutlass.Int32(7 * 4)
                for fragment_index in cutlass.range_constexpr(self.vpt):
                    for pair_index in cutlass.range_constexpr(2):
                        key_pair = packed_fragments[fragment_index][pair_index]
                        for half_index in cutlass.range_constexpr(2):
                            key = cutlass.Uint32(_half_code(key_pair, half_index == 1, False))
                            output_position = _radix_cursor(
                                key,
                                crossing_high,
                                threshold,
                                histogram_address,
                                low_address,
                                tie_address,
                            )
                            if output_position < top_k:
                                selected_index = (
                                    thread_index + cutlass.Int32(fragment_index * self.blk)
                                ) * cutlass.Int32(4) + cutlass.Int32(pair_index * 2 + half_index)
                                output_row[output_position] = selected_index
                output_position = _radix_cursor(
                    tail_key, crossing_high, threshold, histogram_address, low_address, tie_address
                )
                if output_position < top_k:
                    output_row[output_position] = tail_index

        if cutlass.const_expr(self.stage_output):
            cute.arch.barrier()
            source_address = _seed._smem_addr_reg__reg(
                shared_base + cutlass.Int32(self.output_offset_words * 4)
            )
            destination_address = out[row_index, None].iterator.toint()
            scalar_begin = cutlass.Int32(0)
            if (destination_address & cutlass.Int64(15)) == cutlass.Int64(0) and out.stride[1] == 1:
                vector_count = top_k >> cutlass.Int32(2)
                if cutlass.const_expr(cute.is_static(pre_idx.shape[1])):
                    for copy_step in cutlass.range_constexpr(
                        (pre_idx.shape[1] // 4 + self.blk - 1) // self.blk
                    ):
                        vector_index = thread_index + cutlass.Int32(copy_step * self.blk)
                        if vector_index < vector_count:
                            _copy_output_quad(source_address, destination_address, vector_index)
                else:
                    vector_index = thread_index
                    while vector_index < vector_count:
                        _copy_output_quad(source_address, destination_address, vector_index)
                        vector_index = vector_index + cutlass.Int32(self.blk)
                scalar_begin = vector_count << cutlass.Int32(2)
            output_index = scalar_begin + thread_index
            while output_index < top_k:
                out[row_index, output_index] = output_row[output_index]
                output_index = output_index + cutlass.Int32(self.blk)

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        cmp_: cutlass.Int32,
        qc: cutlass.Int32,
        smem_bytes: cutlass.Int32,
        stream,
    ):
        assert logits.element_type == cutlass.BFloat16
        assert pre_idx.element_type == cutlass.Int32
        assert kv_lens.element_type == cutlass.Int32
        assert out.element_type == cutlass.Int32
        if cutlass.const_expr(cute.is_static(logits.shape[1])):
            assert logits.shape[1] % 64 == 0
        shared_bytes = self.required_smem_bytes
        if cutlass.const_expr(self.stage_output):
            shared_bytes = shared_bytes + pre_idx.shape[1] * 4
        self.kern(logits, pre_idx, kv_lens, out, n).launch(
            grid=(logits.shape[0] * self.parts, 1, 1),
            block=(self.blk, 1, 1),
            stream=stream,
            smem=shared_bytes,
            min_blocks_per_mp=self.minb,
        )


__all__ = ["GvrTopkPackedClampWindowBF16Kernel"]
