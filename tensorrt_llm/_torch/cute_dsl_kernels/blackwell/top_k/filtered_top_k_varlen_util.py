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


import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.utils.distributed import atomicAdd

from .block_scan import block_prefix_sum_kernel, fence_acq_rel_cta

"""
top-k varlen utils. could be used by prefill and decode phase.
"""


def half_as_ushort(half_val):
    """Interpret FP16 value as uint16 bit pattern"""
    return llvm.bitcast(cutlass.Uint16.mlir_type, half_val.ir_value())


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern"""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


class FilteredTopKKernelVarlen:
    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        return_val: bool = True,
        enable_multi_cta: bool = False,
        chunk_size_per_cta: int = 16384,
        num_ctas_per_row: int = 1,
        merge_blocks: bool = False,
        overflow_policy: str = "REREAD",
        num_threads_override: int = 0,
    ):
        """
        Args:
            overflow_policy: Controls behavior when threshold-bin candidates exceed the
                SMEM input buffer (filtered_topk_smem_input_size).  Only takes effect
                when max_num_cols > filtered_topk_smem_input_size; otherwise all policies
                are equivalent and no extra cost is incurred.

                "GMEM_SPILL"    -- Spill excess candidates to a pre-allocated GMEM
                                   extra_buffer.  Exact result.  Requires caller to
                                   allocate extra_buffer proportional to batch size;
                                   may OOM at large batch.
                "TRUNCATE"      -- Discard candidates that overflow SMEM.  Only
                                   retained candidates contribute to the refinement
                                   histogram, so refinement operates consistently on
                                   the stored set.  Non-exact (may output fewer than
                                   top_k indices when the threshold bin is dense).  No
                                   extra_buffer needed.
                "REREAD_ALWAYS" -- Skip SMEM collection entirely in the coarse pass;
                                   always perform a second GMEM scan to collect
                                   threshold-bin candidates.  Exact result.  No
                                   extra_buffer needed; costs one extra GMEM read per
                                   row unconditionally.
                "REREAD"        -- Optimistic: attempt SMEM collection first.  If
                                   overflow is detected at runtime (s_overflow_flag),
                                   fall back to a REREAD_ALWAYS-style second GMEM scan.
                                   Exact result.  No extra_buffer needed; pays the
                                   extra GMEM read only when overflow actually occurs.
        """
        self.dtype = dtype
        self.max_num_cols = max_num_cols
        self.top_k = top_k
        self.num_copy_bits = num_copy_bits
        self.enable_multi_cta = enable_multi_cta
        self.chunk_size_per_cta = chunk_size_per_cta
        self.num_ctas_per_row = num_ctas_per_row
        self.merge_blocks = merge_blocks
        self.overflow_policy = overflow_policy
        assert overflow_policy in ("GMEM_SPILL", "TRUNCATE", "REREAD_ALWAYS", "REREAD"), (
            f"Unknown overflow_policy: {overflow_policy}"
        )

        # Tested with top_k in {512, 1024, 2048}. Other values may work but
        # have not been validated and may require minor changes.
        assert top_k <= 2048, f"top_k must be <= 2048, but got {top_k}"
        # s_indices only needs top_k slots; size to top_k to save SMEM.
        self.filtered_topk_max_k = top_k
        # 8 bits for radix-based filter.
        self.radix = 256

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            self.num_buffer_smem_input_idx = 2
        else:
            self.num_buffer_smem_input_idx = 1

        # 65536 is the max index value for uint16.
        if cutlass.const_expr(enable_multi_cta):
            self.per_row_max_num_cols = chunk_size_per_cta * num_ctas_per_row
        else:
            self.per_row_max_num_cols = self.max_num_cols

        if cutlass.const_expr(self.per_row_max_num_cols <= 65536):
            self.index_type = cutlass.Uint16
            # TODO: remove after perf validation (replaced by _compute_smem_input_size_for_occupancy)
            # if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
            #     self.max_smem_input_size = 32 * 1024
            # else:
            #     self.max_smem_input_size = 64 * 1024
        else:
            self.index_type = cutlass.Uint32
            # TODO: remove after perf validation (replaced by _compute_smem_input_size_for_occupancy)
            # if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
            #     self.max_smem_input_size = 16 * 1024
            # else:
            #     self.max_smem_input_size = 32 * 1024

        self.vec_size = num_copy_bits // dtype.width
        if cutlass.const_expr(dtype not in [cutlass.Float32, cute.BFloat16, cutlass.Float16]):
            raise ValueError(f"Unsupported dtype: {dtype}")

        if num_threads_override > 0:
            self.num_threads_per_cta = num_threads_override
        elif cutlass.const_expr(dtype == cutlass.Float32):
            if self.max_num_cols >= self.vec_size * 1024:
                self.num_threads_per_cta = 1024
            else:
                if cutlass.const_expr(self.max_num_cols > 2048 and self.max_num_cols < 8192):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256
        else:
            if self.max_num_cols >= 43008:
                self.num_threads_per_cta = 1024
            else:
                if cutlass.const_expr(self.max_num_cols > 4096 and self.max_num_cols < 43008):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256

        # num_threads_per_cta must be set before _compute_smem_input_size() since
        # _compute_smem_input_size_for_occupancy() uses it to derive num_warps.
        self.filtered_topk_smem_input_size = self._compute_smem_input_size()

        _needs_extra = self.max_num_cols > self.filtered_topk_smem_input_size
        self.enable_gmem_store = (overflow_policy == "GMEM_SPILL") and _needs_extra
        self.enable_truncate = (overflow_policy == "TRUNCATE") and _needs_extra
        self.enable_reread_always = (overflow_policy == "REREAD_ALWAYS") and _needs_extra
        self.enable_reread = (overflow_policy == "REREAD") and _needs_extra

        self.return_val = return_val
        # Subclasses set to True to subtract row_start from absolute indices before
        # writing output (used in prefill where row_start may be non-zero).
        self.subtract_row_start_on_output = False

        # radix-based filter parameters.
        if cutlass.const_expr(dtype == cutlass.Float32):
            self.ordered_type = cute.Uint32
            self.first_refine_shift = 24
            self.num_refine_rounds = 4
        elif cutlass.const_expr(dtype in [cutlass.Float16, cute.BFloat16]):
            self.ordered_type = cute.Uint16
            self.first_refine_shift = 0
            self.num_refine_rounds = 1

    def _compute_smem_input_size(self) -> int:
        return self._compute_smem_input_size_for_occupancy(target_blocks_per_sm=1)

    def _compute_smem_input_size_for_occupancy(self, target_blocks_per_sm: int) -> int:
        """Compute max candidate-buffer size (S) for a given occupancy target.

            input_idx_budget = 128 KB // target_blocks_per_sm
            S = input_idx_budget // (num_buffer * idx_sz)

        This keeps total SMEM ≈ 38 KB/block at 4 blocks/SM, preserving ~104 KB
        of unified L1 per SM for LDG caching. Using the full per-block budget
        (256 KB / 4 = 64 KB) shrinks L1 to ~32 KB and causes a ~5-10% regression.

        Resulting S values (matches old hardcoded values):
          4 blocks/SM: Uint16/nb=2→8192  Uint16/nb=1→16384
                       Uint32/nb=2→4096  Uint32/nb=1→8192
          1 block/SM:  Uint16/nb=2→32768 Uint16/nb=1→65536
                       Uint32/nb=2→16384 Uint32/nb=1→32768
        """
        INPUT_IDX_BUDGET_BASE = 128 * 1024  # 128 KB at 1 block/SM
        input_idx_budget = INPUT_IDX_BUDGET_BASE // target_blocks_per_sm
        idx_sz = 2 if self.index_type == cutlass.Uint16 else 4
        max_S = input_idx_budget // (self.num_buffer_smem_input_idx * idx_sz)
        return min(max_S, self.max_num_cols)

    @cute.jit
    def to_coarse_key(self, x):
        """Convert to coarse 8-bit key for histogram"""

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            # Convert to FP16 and extract high 8 bits
            h = x.to(cutlass.Float16)
            bits = half_as_ushort(h)

            key = cutlass.Uint16(0)

            # extract the sign bit
            # key = (bits & 0x8000) ? bits : ~bits & 0x7fff;
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)

            # high 8 bits
            return cute.Uint8((key >> 8) & 0xFF)
        else:
            # For half/bfloat16, extract high 8 bits directly
            bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            # high 8 bits
            return cute.Uint8((key >> 8) & 0xFF)

    @cute.jit
    def to_ordered(self, x):
        """Convert to ordered integer for comparison"""
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            bits = float_as_uint32(x)

            key = cutlass.Uint32(0)
            if bits & 0x80000000:
                key = cutlass.Uint32(bits)
            else:
                key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
            return cute.Uint32(key)
        else:
            bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cute.Uint16(0xFFFF)) & cute.Uint16(0x7FFF)
            return cute.Uint16(key)

    @cute.jit
    def to_ordered_and_coarse(self, x):
        """Return (ordered, coarse_key) for x.
        For bf16/fp16, shares the half_as_ushort + sign-flip computation.
        For fp32, the two transforms differ (fp32->fp16 truncation vs full 32-bit
        sign-flip), so both are computed independently.
        """
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            return self.to_ordered(x), self.to_coarse_key(x)
        else:
            ordered = self.to_ordered(x)
            coarse_shift = cutlass.const_expr(self.ordered_type.width - int(math.log2(self.radix)))
            coarse = cute.Uint8(
                (ordered >> self.ordered_type(coarse_shift)) & self.ordered_type(0xFF)
            )
            return ordered, coarse

    @cute.jit
    def _collect_below_threshold_coarse(
        self,
        tidx,
        threshold_bin,
        s_counter,
        s_indices,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        score,
        row_start,
        prologue_elems,
        left_start,
        left_size,
    ):
        """Collect all indices with coarse bin < threshold_bin from GMEM, then barrier."""
        val_one = cutlass.Int32(1)
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        vec_size = self.vec_size
        ic = tidx * cutlass.Int32(vec_size)
        while ic + cutlass.Int32(vec_size - 1) < aligned_size:
            cute.copy(
                _copy_atom,
                cute.make_tensor(
                    cute.make_ptr(
                        self.dtype,
                        _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=_align_bytes,
                    ),
                    cute.make_layout((vec_size,)),
                ),
                scan_frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                bin_val = self.to_coarse_key(scan_frag[j])
                if bin_val < threshold_bin:
                    pos = atomicAdd(s_counter.iterator, val_one)
                    s_indices[pos] = self.index_type(vec_start + ic + cutlass.Int32(j))
            ic = ic + cutlass.Int32(_step_vec)

        for j in range(tidx, prologue_elems, self.num_threads_per_cta):
            col_idx = cutlass.Int32(row_start + j)
            raw = score[col_idx]
            bin_val = self.to_coarse_key(raw)
            if bin_val < threshold_bin:
                pos = atomicAdd(s_counter.iterator, val_one)
                s_indices[pos] = self.index_type(col_idx)

        for j in range(tidx, left_size, self.num_threads_per_cta):
            col_idx = cutlass.Int32(left_start + j)
            raw = score[col_idx]
            bin_val = self.to_coarse_key(raw)
            if bin_val < threshold_bin:
                pos = atomicAdd(s_counter.iterator, val_one)
                s_indices[pos] = self.index_type(col_idx)

        cute.arch.barrier()

    @cute.jit
    def _collect_below_threshold_refine(
        self,
        tidx,
        threshold,
        offset,
        num_input,
        r_idx,
        s_input_idx,
        score,
        s_counter,
        s_indices,
        cur_g_num_input,
        buffer,
    ):
        """Collect all indices with refined bin < threshold from SMEM (and GMEM buffer), then barrier."""
        val_one = cutlass.Int32(1)
        for i in range(tidx, num_input, self.num_threads_per_cta):
            idx = s_input_idx[r_idx, i]
            idx = cutlass.Int32(cutlass.Uint32(idx))
            bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
            if bin_val < threshold:
                pos = atomicAdd(s_counter.iterator, val_one)
                s_indices[pos] = self.index_type(idx)
        if cutlass.const_expr(self.enable_gmem_store):
            for i in range(tidx, cur_g_num_input, self.num_threads_per_cta):
                idx = buffer[r_idx, i]
                bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
                if bin_val < threshold:
                    pos = atomicAdd(s_counter.iterator, val_one)
                    s_indices[pos] = self.index_type(idx)
        cute.arch.barrier()

    @cute.jit
    def _filter_and_histogram_per_elem_coarse(
        self,
        bin_val,
        threshold_bin,
        idx,
        raw_input,
        s_counter,
        s_indices,
        s_input_idx,
        s_num_input,
        s_histogram,
        g_num_input,
        buffer,
        s_overflow_flag,
    ):
        """Per-element if/elif handler for the coarse filter pass.

        bin_val < threshold_bin  → write to s_indices.
        bin_val == threshold_bin → store to s_input_idx (+ optional buffer) and
                                   update s_histogram for the next refinement round.
        """
        val_one = cutlass.Int32(1)
        if bin_val < threshold_bin:
            pos = atomicAdd(s_counter.iterator, val_one)
            s_indices[pos] = idx
        elif bin_val == threshold_bin:
            if cutlass.const_expr(self.enable_gmem_store):
                pos = atomicAdd(s_num_input.iterator, val_one)
                if pos < self.filtered_topk_smem_input_size:
                    s_input_idx[0, pos] = idx
                else:
                    buffer_pos = atomicAdd(g_num_input.iterator, val_one)
                    buffer[0, buffer_pos] = cutlass.Int32(cutlass.Uint32(idx))
                ordered = self.to_ordered(raw_input)
                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                atomicAdd(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
            elif cutlass.const_expr(self.enable_truncate):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    ordered = cutlass.Uint32(0)
                    sub_bin = cutlass.Uint32(0)
                else:
                    ordered = cutlass.Uint16(0)
                    sub_bin = cutlass.Int32(0)
                pos = atomicAdd(s_num_input.iterator, val_one)
                if pos < self.filtered_topk_smem_input_size:
                    s_input_idx[0, pos] = idx
                    ordered = self.to_ordered(raw_input)
                    sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                    atomicAdd(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
            elif cutlass.const_expr(self.enable_reread):
                pos = atomicAdd(s_num_input.iterator, val_one)
                if pos < self.filtered_topk_smem_input_size:
                    s_input_idx[0, pos] = idx
                else:
                    # Use atomicAdd (not plain store) to avoid concurrent non-atomic writes
                    # from multiple threads to the same SMEM address.  Any non-zero value
                    # means overflow; the did_overflow check uses != 0.
                    atomicAdd(s_overflow_flag.iterator, val_one)
                ordered = self.to_ordered(raw_input)
                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                atomicAdd(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
            else:
                if cutlass.const_expr(not self.enable_reread_always):
                    pos = atomicAdd(s_num_input.iterator, val_one)
                    if pos < self.filtered_topk_smem_input_size:
                        s_input_idx[0, pos] = idx
                ordered = self.to_ordered(raw_input)
                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                atomicAdd(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)

    @cute.jit
    def _filter_and_histogram_per_elem_refine(
        self,
        bin_val,
        threshold,
        idx_int32,
        ordered_val,
        offset,
        r_idx,
        is_last_round,
        s_counter,
        s_indices,
        s_input_idx,
        s_num_input,
        s_histogram,
        s_last_remain,
        g_num_input,
        buffer,
    ):
        """Per-element if/elif handler for refinement rounds.

        idx_int32   – Int32 column index, used for score lookup and buffer writes.
        ordered_val – pre-computed self.to_ordered(raw_input); avoids recomputing
                      it for sub_bin extraction when bin_val == threshold.

        bin_val < threshold  → write to s_indices.
        bin_val == threshold → last round: s_last_remain countdown;
                               otherwise: store to s_input_idx[r_idx^1] (+ optional
                               buffer) and update s_histogram for the next round.
        """
        val_one = cutlass.Int32(1)
        val_one_negative = cutlass.Int32(-1)
        idx = self.index_type(idx_int32)
        if bin_val < threshold:
            pos = atomicAdd(s_counter.iterator, val_one)
            s_indices[pos] = idx
        elif bin_val == threshold:
            if is_last_round:
                cur_pos = atomicAdd(s_last_remain.iterator, val_one_negative)
                if cur_pos > 0:
                    s_indices[self.top_k - cur_pos] = idx
            else:
                cur_pos = atomicAdd(s_num_input.iterator + (r_idx ^ 1), val_one)
                if cutlass.const_expr(self.enable_gmem_store):
                    if cur_pos < self.filtered_topk_smem_input_size:
                        s_input_idx[r_idx ^ 1, cur_pos] = idx
                    else:
                        buffer_pos = atomicAdd(g_num_input.iterator + (r_idx ^ 1), val_one)
                        buffer[r_idx ^ 1, buffer_pos] = idx_int32
                    sub_bin = (ordered_val >> (offset - 8)) & 0xFF
                    atomicAdd(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
                else:
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        sub_bin = cutlass.Uint32(0)
                    else:
                        sub_bin = cutlass.Int32(0)
                    if cur_pos < self.filtered_topk_smem_input_size:
                        s_input_idx[r_idx ^ 1, cur_pos] = idx
                        sub_bin = (ordered_val >> (offset - 8)) & 0xFF
                        atomicAdd(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)

    @cute.jit
    def _filter_and_histogram_coarse(
        self,
        tidx,
        threshold_bin,
        s_counter,
        s_indices,
        s_input_idx,
        s_num_input,
        s_histogram,
        g_num_input,
        buffer,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        score,
        row_start,
        prologue_elems,
        left_start,
        left_size,
        s_overflow_flag,
    ):
        """Reset histogram, filter all input elements through three loops, then barrier.

        Covers vec-aligned GMEM, prologue scalar, and left scalar segments.
        """
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        cute.arch.barrier()
        for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
            s_histogram[_hi] = 0
        cute.arch.barrier()

        vec_size = self.vec_size
        ic = tidx * cutlass.Int32(vec_size)
        while ic + cutlass.Int32(vec_size - 1) < aligned_size:
            cute.copy(
                _copy_atom,
                cute.make_tensor(
                    cute.make_ptr(
                        self.dtype,
                        _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=_align_bytes,
                    ),
                    cute.make_layout((vec_size,)),
                ),
                scan_frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                raw_input = scan_frag[j]
                bin_val = self.to_coarse_key(raw_input)
                idx = self.index_type(vec_start + ic + cutlass.Int32(j))
                self._filter_and_histogram_per_elem_coarse(
                    bin_val,
                    threshold_bin,
                    idx,
                    raw_input,
                    s_counter,
                    s_indices,
                    s_input_idx,
                    s_num_input,
                    s_histogram,
                    g_num_input,
                    buffer,
                    s_overflow_flag,
                )
            ic = ic + cutlass.Int32(_step_vec)

        for j in range(tidx, prologue_elems, self.num_threads_per_cta):
            col_idx = cutlass.Int32(row_start + j)
            raw = score[col_idx]
            bin_val = self.to_coarse_key(raw)
            idx = self.index_type(col_idx)
            self._filter_and_histogram_per_elem_coarse(
                bin_val,
                threshold_bin,
                idx,
                raw,
                s_counter,
                s_indices,
                s_input_idx,
                s_num_input,
                s_histogram,
                g_num_input,
                buffer,
                s_overflow_flag,
            )

        for j in range(tidx, left_size, self.num_threads_per_cta):
            col_idx = cutlass.Int32(left_start + j)
            raw = score[col_idx]
            bin_val = self.to_coarse_key(raw)
            idx = self.index_type(col_idx)
            self._filter_and_histogram_per_elem_coarse(
                bin_val,
                threshold_bin,
                idx,
                raw,
                s_counter,
                s_indices,
                s_input_idx,
                s_num_input,
                s_histogram,
                g_num_input,
                buffer,
                s_overflow_flag,
            )
        fence_acq_rel_cta()
        cute.arch.barrier()

    @cute.jit
    def _reread_always_per_elem_output(
        self,
        include_threshold,
        raw,
        col_idx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        s_counter,
        s_indices,
        s_last_remain,
    ):
        """Per-element handler for REREAD_ALWAYS output scan.
        include_threshold is a compile-time bool.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value. Both carry accumulated prior-round constraints.
        When chain_mask == 0 (round 0), ordered & 0 == 0 is always True.
        """
        ordered, coarse = self.to_ordered_and_coarse(raw)
        if coarse == threshold_bin:
            passes_chain = (ordered & self.ordered_type(chain_mask)) == chain_prefix
            if passes_chain:
                bin_val = (ordered >> offset) & 0xFF
                idx = self.index_type(col_idx)
                val_one = cutlass.Int32(1)
                if bin_val < T2:
                    pos = atomicAdd(s_counter.iterator, val_one)
                    s_indices[pos] = idx
                elif cutlass.const_expr(include_threshold):
                    if bin_val == T2:
                        cur_pos = atomicAdd(s_last_remain.iterator, cutlass.Int32(-1))
                        if cur_pos > 0:
                            s_indices[self.top_k - cur_pos] = idx

    @cute.jit
    def _reread_always_per_elem_combined(
        self,
        raw,
        col_idx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        s_counter,
        s_indices,
        s_histogram,
    ):
        """Per-element handler for REREAD_ALWAYS non-last-round combined scan.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value. Both carry accumulated prior-round constraints.
        For elements passing coarse + chain filters:
          bin_val < T2  → write col_idx to s_indices (definitely top-K).
          bin_val == T2 → histogram at (ordered >> (offset - 8)) & 0xFF.
        """
        ordered, coarse = self.to_ordered_and_coarse(raw)
        if coarse == threshold_bin:
            passes_chain = (ordered & self.ordered_type(chain_mask)) == chain_prefix
            if passes_chain:
                bin_val = (ordered >> offset) & 0xFF
                val_one = cutlass.Int32(1)
                if bin_val < T2:
                    pos = atomicAdd(s_counter.iterator, val_one)
                    s_indices[pos] = self.index_type(col_idx)
                elif bin_val == T2:
                    next_sub_bin = (ordered >> (offset - 8)) & 0xFF
                    atomicAdd(s_histogram.iterator + cutlass.Int32(next_sub_bin), val_one)

    @cute.jit
    def _reread_always_gmem_output_scan(
        self,
        include_threshold,
        tidx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        score,
        s_counter,
        s_indices,
        s_last_remain,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        row_start,
        prologue_elems,
        left_start,
        left_size,
    ):
        """GMEM scan for REREAD_ALWAYS output phase.
        include_threshold is a compile-time bool.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value — both carry prior-round constraints.
        Scans all three GMEM segments and writes qualifying indices to s_indices.
        Ends with cute.arch.barrier() to sync all writes before Phase 3.
        """
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        vec_size = self.vec_size

        ic = tidx * cutlass.Int32(vec_size)
        while ic + cutlass.Int32(vec_size - 1) < aligned_size:
            cute.copy(
                _copy_atom,
                cute.make_tensor(
                    cute.make_ptr(
                        self.dtype,
                        _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=_align_bytes,
                    ),
                    cute.make_layout((vec_size,)),
                ),
                scan_frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                col_idx = cutlass.Int32(vec_start + ic + cutlass.Int32(j))
                self._reread_always_per_elem_output(
                    include_threshold,
                    scan_frag[j],
                    col_idx,
                    threshold_bin,
                    T2,
                    offset,
                    chain_mask,
                    chain_prefix,
                    s_counter,
                    s_indices,
                    s_last_remain,
                )
            ic = ic + cutlass.Int32(_step_vec)

        for j in range(tidx, prologue_elems, self.num_threads_per_cta):
            col_idx = cutlass.Int32(row_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_output(
                include_threshold,
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_last_remain,
            )

        for j in range(tidx, left_size, self.num_threads_per_cta):
            col_idx = cutlass.Int32(left_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_output(
                include_threshold,
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_last_remain,
            )

        cute.arch.barrier()

    @cute.jit
    def _reread_always_gmem_combined_scan(
        self,
        tidx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        score,
        s_counter,
        s_indices,
        s_histogram,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        row_start,
        prologue_elems,
        left_start,
        left_size,
    ):
        """GMEM scan for REREAD_ALWAYS non-last rounds: reset histogram, output < T2
        elements, and build histogram for the next round.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value — both carry prior-round constraints.
        Ends with fence_acq_rel_cta() + cute.arch.barrier().
        Returns updated chain_prefix (runtime DSL value); caller updates chain_mask
        via | (cutlass.Int32(0xFF) << offset).
        """
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        vec_size = self.vec_size

        # Barrier before clearing s_histogram: ensures all threads have already
        # read s_histogram[threshold-1] to update topk_remaining in the caller
        # before any thread starts zeroing it here.
        cute.arch.barrier()
        for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
            s_histogram[_hi] = 0
        cute.arch.barrier()

        ic = tidx * cutlass.Int32(vec_size)
        while ic + cutlass.Int32(vec_size - 1) < aligned_size:
            cute.copy(
                _copy_atom,
                cute.make_tensor(
                    cute.make_ptr(
                        self.dtype,
                        _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=_align_bytes,
                    ),
                    cute.make_layout((vec_size,)),
                ),
                scan_frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                col_idx = cutlass.Int32(vec_start + ic + cutlass.Int32(j))
                self._reread_always_per_elem_combined(
                    scan_frag[j],
                    col_idx,
                    threshold_bin,
                    T2,
                    offset,
                    chain_mask,
                    chain_prefix,
                    s_counter,
                    s_indices,
                    s_histogram,
                )
            ic = ic + cutlass.Int32(_step_vec)

        for j in range(tidx, prologue_elems, self.num_threads_per_cta):
            col_idx = cutlass.Int32(row_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_combined(
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_histogram,
            )

        for j in range(tidx, left_size, self.num_threads_per_cta):
            col_idx = cutlass.Int32(left_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_combined(
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_histogram,
            )

        fence_acq_rel_cta()
        cute.arch.barrier()

        # Return updated chain_prefix (runtime DSL value).
        # Caller updates chain_mask via | (cutlass.Int32(0xFF) << offset).
        return chain_prefix | self.ordered_type(self.ordered_type(T2) << self.ordered_type(offset))

    @cute.jit
    def _reread_gmem_rescan(
        self,
        topk_remaining,
        is_last_round,
        tidx,
        threshold_bin,
        threshold,
        offset,
        chain_mask,
        chain_prefix,
        score,
        s_counter,
        s_indices,
        s_last_remain,
        s_histogram,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        row_start,
        prologue_elems,
        left_start,
        left_size,
    ):
        """GMEM re-scan phase shared by REREAD_ALWAYS and REREAD-overflow paths.

        Returns (run_next_round, chain_mask, chain_prefix).
        """
        run_next_round = True
        if topk_remaining == 0:
            self._reread_always_gmem_output_scan(
                False,
                tidx,
                threshold_bin,
                threshold,
                offset,
                chain_mask,
                chain_prefix,
                score,
                s_counter,
                s_indices,
                s_last_remain,
                _copy_atom,
                scan_frag,
                _aligned_base,
                vec_start,
                aligned_size,
                row_start,
                prologue_elems,
                left_start,
                left_size,
            )
            run_next_round = False
        else:
            if is_last_round:
                self._reread_always_gmem_output_scan(
                    True,
                    tidx,
                    threshold_bin,
                    threshold,
                    offset,
                    chain_mask,
                    chain_prefix,
                    score,
                    s_counter,
                    s_indices,
                    s_last_remain,
                    _copy_atom,
                    scan_frag,
                    _aligned_base,
                    vec_start,
                    aligned_size,
                    row_start,
                    prologue_elems,
                    left_start,
                    left_size,
                )
            else:
                chain_prefix = self._reread_always_gmem_combined_scan(
                    tidx,
                    threshold_bin,
                    threshold,
                    offset,
                    chain_mask,
                    chain_prefix,
                    score,
                    s_counter,
                    s_indices,
                    s_histogram,
                    _copy_atom,
                    scan_frag,
                    _aligned_base,
                    vec_start,
                    aligned_size,
                    row_start,
                    prologue_elems,
                    left_start,
                    left_size,
                )
                chain_mask = chain_mask | (cutlass.Int32(0xFF) << cutlass.Int32(offset))
        return run_next_round, chain_mask, chain_prefix

    @cute.jit
    def _filter_and_histogram_refine(
        self,
        tidx,
        threshold,
        offset,
        r_idx,
        is_last_round,
        num_input,
        cur_g_num_input,
        score,
        s_counter,
        s_indices,
        s_input_idx,
        s_num_input,
        s_histogram,
        s_last_remain,
        g_num_input,
        buffer,
    ):
        """Reset histogram, filter all threshold-bucket elements, then barrier.

        Covers SMEM s_input_idx loop and optional GMEM buffer loop.
        """
        cute.arch.barrier()
        for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
            s_histogram[_hi] = 0
        cute.arch.barrier()

        for i in range(tidx, num_input, self.num_threads_per_cta):
            idx_tmp = s_input_idx[r_idx, i]
            idx_int32 = cutlass.Int32(cutlass.Uint32(idx_tmp))
            raw_input = score[idx_int32]
            ordered_val = self.to_ordered(raw_input)
            bin_val = (ordered_val >> offset) & 0xFF
            self._filter_and_histogram_per_elem_refine(
                bin_val,
                threshold,
                idx_int32,
                ordered_val,
                offset,
                r_idx,
                is_last_round,
                s_counter,
                s_indices,
                s_input_idx,
                s_num_input,
                s_histogram,
                s_last_remain,
                g_num_input,
                buffer,
            )

        if cutlass.const_expr(self.enable_gmem_store):
            cute.arch.barrier()
            for i in range(tidx, cur_g_num_input, self.num_threads_per_cta):
                idx_int32 = buffer[r_idx, i]
                raw_input = score[idx_int32]
                ordered_val = self.to_ordered(raw_input)
                bin_val = (ordered_val >> offset) & 0xFF
                self._filter_and_histogram_per_elem_refine(
                    bin_val,
                    threshold,
                    idx_int32,
                    ordered_val,
                    offset,
                    r_idx,
                    is_last_round,
                    s_counter,
                    s_indices,
                    s_input_idx,
                    s_num_input,
                    s_histogram,
                    s_last_remain,
                    g_num_input,
                    buffer,
                )
            fence_acq_rel_cta()
        cute.arch.barrier()

    @cute.jit
    def prefix_sum_and_find_threshold_coarse(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        g_num_input,
        s_num_input_idx=0,
    ):
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(
                    val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1
                )
                s_histogram[tidx] = val
                # sync among self.radix threads
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store):
                        g_num_input[s_num_input_idx] = 0
                    # TODO: the difference between 1 and 2.
                    s_counter[0] = 0
            # sync among all threads in a cta.
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # sync among all threads in a cta.
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store):
                    g_num_input[s_num_input_idx] = 0
                # the difference between coarse and fine-grained.
                s_counter[0] = 0
                run_loop = False

            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if previous <= topk_remaining and s_histogram[i] > topk_remaining:
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            if cutlass.const_expr(self.enable_gmem_store):
                                g_num_input[s_num_input_idx] = 0
                            # the difference between coarse and fine-grained.
                            s_counter[0] = 0
                            run_next_loop = False
            # sync among all threads in a cta.
            cute.arch.barrier()

    @cute.jit
    def prefix_sum_and_find_threshold_fine_grained(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        g_num_input,
        s_num_input_idx=0,
    ):
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(
                    val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1
                )
                s_histogram[tidx] = val
                # sync
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store):
                        g_num_input[s_num_input_idx] = 0
                    # the first difference between coarse and fine-grained.
                    s_last_remain[0] = topk_remaining - previous
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # sync among all threads in a cta.
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store):
                    g_num_input[s_num_input_idx] = 0
                # the difference between coarse and fine-grained.
                s_last_remain[0] = topk_remaining - previous
                run_loop = False
            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if previous <= topk_remaining and s_histogram[i] > topk_remaining:
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            if cutlass.const_expr(self.enable_gmem_store):
                                g_num_input[s_num_input_idx] = 0
                            # the difference between coarse and fine-grained.
                            s_last_remain[0] = topk_remaining - previous
                            run_next_loop = False
            # sync among all threads in a cta.
            cute.arch.barrier()

    @cute.jit
    def filtered_topk_kernel_per_row(
        self,
        input: cute.Tensor,
        # gmem, used for the merge blocks kernel.
        input_indices: cute.Tensor,
        extra_buffer: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        row_start: int,
        length: int,
        bidx: int,
        s_histogram,
        s_counter,
        s_threshold_bin_id,
        s_num_input,
        g_num_input,
        s_indices,
        s_input_idx,
        s_last_remain,
        num_warps,
        s_warp_sums,
        s_overflow_flag,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm."""
        # # Thread and block indexing
        tidx, _, _ = cute.arch.thread_idx()

        score = input[bidx, None]
        if cutlass.const_expr(self.merge_blocks):
            indices = input_indices[bidx, None]
        if cutlass.const_expr(self.enable_multi_cta):
            dst = output_indices
            if cutlass.const_expr(self.return_val):
                dst_values = output_values
        else:
            dst = output_indices[bidx, None]
            if cutlass.const_expr(self.return_val):
                dst_values = output_values[bidx, None]
        # Note, for multi-cta version, each ctas must have its own extra_buffer.
        buffer = None
        if cutlass.const_expr(self.enable_gmem_store):
            if cutlass.const_expr(self.enable_multi_cta):
                grid_dim_x, grid_dim_y, _ = cute.arch.grid_dim()
                bidx_val, bidy_val, _ = cute.arch.block_idx()
                buffer_row_id = bidx_val * grid_dim_y + bidy_val
                buffer = extra_buffer[buffer_row_id, None, None]
            else:
                buffer = extra_buffer[bidx, None, None]

        # for initial scalar load part.
        row_ptr = score.iterator + row_start
        row_addr_u64 = row_ptr.toint()

        # 256/8 = 32bytes
        align_bytes = self.num_copy_bits // 8
        # fp32: 4bytes
        elem_bytes = self.dtype.width // 8

        misalign = row_addr_u64 % align_bytes
        fix_bytes = cutlass.Int64(0)
        if misalign != 0:
            fix_bytes = align_bytes - misalign

        prologue_elems = cutlass.Int32(fix_bytes // elem_bytes)

        remaining = length - prologue_elems
        aligned_size = (remaining // self.vec_size) * self.vec_size
        left_size = remaining - aligned_size

        vec_start = row_start + prologue_elems
        left_start = vec_start + aligned_size

        # GVR-style direct GMEM load constants (all Python ints, compile-time).
        # Loop bounds computed from runtime aligned_size so threads past the
        # actual row end execute zero iterations — no OOB waste for short rows.
        vec_size = self.vec_size
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        # Byte address of the aligned portion start for this row (score[vec_start]).
        _aligned_base = (score.iterator + vec_start).toint()
        # TODO: add invariant=True for .CONSTANT cache hint once validated
        _copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.dtype,
            num_bits_per_copy=self.num_copy_bits,
        )

        scan_frag = cute.make_fragment((vec_size,), self.dtype)

        # Trivial case: length <= top_k
        if length <= self.top_k:
            for i in range(tidx, self.top_k, self.num_threads_per_cta):
                # TODO: add multi-cta version support here.
                if i < length:
                    if cutlass.const_expr(self.enable_multi_cta):
                        dst[i] = i + row_start
                    elif cutlass.const_expr(self.merge_blocks):
                        dst[i] = indices[i]
                    else:
                        dst[i] = i
                    if cutlass.const_expr(self.return_val):
                        if cutlass.const_expr(self.enable_multi_cta):
                            dst_values[i] = score[i + row_start]
                        else:
                            dst_values[i] = score[i]
                else:
                    dst[i] = -1
                    if cutlass.const_expr(self.return_val):
                        dst_values[i] = dst_values.element_type(
                            dst_values.element_type.inf * dst_values.element_type(-1.0)
                        )
        else:
            topk_remaining = self.top_k

            val_one = cutlass.Int32(1)

            # Stage 1: Coarse histogram.
            # Use a strided loop so every bin is cleared even when
            # num_threads_per_cta < radix (e.g. 128 < 256).
            for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
                s_histogram[_hi] = 0
            if cutlass.const_expr(self.enable_reread):
                if tidx == 0:
                    s_overflow_flag[0] = 0
            cute.arch.barrier()

            # 1.1 Build histogram
            ic = tidx * cutlass.Int32(vec_size)
            while ic + cutlass.Int32(vec_size - 1) < aligned_size:
                cute.copy(
                    _copy_atom,
                    cute.make_tensor(
                        cute.make_ptr(
                            self.dtype,
                            _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                            cute.AddressSpace.gmem,
                            assumed_align=_align_bytes,
                        ),
                        cute.make_layout((vec_size,)),
                    ),
                    scan_frag,
                )
                for j in cutlass.range_constexpr(vec_size):
                    bin_val = self.to_coarse_key(scan_frag[j])
                    atomicAdd(s_histogram.iterator + cutlass.Int32(bin_val), val_one)
                ic = ic + cutlass.Int32(_step_vec)

            # for initial scalar load part.
            for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                col_idx = cutlass.Int32(row_start + j)
                raw = score[col_idx]
                bin_val = self.to_coarse_key(raw)
                atomicAdd(
                    s_histogram.iterator + cutlass.Int32(bin_val),
                    val_one,
                )

            # for left part (left_size)
            for j in range(tidx, left_size, self.num_threads_per_cta):
                col_idx = cutlass.Int32(left_start + j)
                raw = score[col_idx]
                bin_val = self.to_coarse_key(raw)
                atomicAdd(
                    s_histogram.iterator + cutlass.Int32(bin_val),
                    val_one,
                )

            cute.arch.barrier()

            # 1.2 and 1.3  Suffix sum to find threshold and find threshold bin
            self.prefix_sum_and_find_threshold_coarse(
                tidx,
                s_histogram,
                s_warp_sums,
                num_warps,
                s_threshold_bin_id,
                s_num_input,
                s_counter,
                s_last_remain,
                topk_remaining,
                g_num_input,
                s_num_input_idx=0,
            )

            threshold_bin = s_threshold_bin_id[0]
            if threshold_bin > 0:
                topk_remaining -= s_histogram[threshold_bin - 1]

            # 1.4 Collect indices
            if topk_remaining == 0:
                self._collect_below_threshold_coarse(
                    tidx,
                    threshold_bin,
                    s_counter,
                    s_indices,
                    _copy_atom,
                    scan_frag,
                    _aligned_base,
                    vec_start,
                    aligned_size,
                    score,
                    row_start,
                    prologue_elems,
                    left_start,
                    left_size,
                )
            else:
                self._filter_and_histogram_coarse(
                    tidx,
                    threshold_bin,
                    s_counter,
                    s_indices,
                    s_input_idx,
                    s_num_input,
                    s_histogram,
                    g_num_input,
                    buffer,
                    _copy_atom,
                    scan_frag,
                    _aligned_base,
                    vec_start,
                    aligned_size,
                    score,
                    row_start,
                    prologue_elems,
                    left_start,
                    left_size,
                    s_overflow_flag,
                )

                # Phase 2: Refinement rounds
                # chain_mask (DSL Int32) and chain_prefix (runtime DSL ordered_type)
                # accumulate prior-round constraints for REREAD_ALWAYS / REREAD overflow
                # fallback. chain_mask is Int32 so it survives DSL phi-merge across the
                # dynamic loop.
                chain_mask = cutlass.Int32(0)
                chain_prefix = self.ordered_type(0)
                # REREAD: read overflow flag once before the loop; runtime bool that
                # selects SMEM refinement (no overflow) vs GMEM re-scan (overflow).
                # Visibility of s_overflow_flag[0] is guaranteed by the fence_acq_rel_cta()
                # + barrier() at the end of _filter_and_histogram_coarse above; no additional
                # barrier is needed here.  If that function's terminal barrier is ever moved
                # to the call site, a barrier must be inserted before this read.
                if cutlass.const_expr(self.enable_reread):
                    did_overflow = s_overflow_flag[0] != 0
                run_next_round = True
                for round in range(self.num_refine_rounds):
                    if run_next_round:
                        r_idx = round % 2

                        self.prefix_sum_and_find_threshold_fine_grained(
                            tidx,
                            s_histogram,
                            s_warp_sums,
                            num_warps,
                            s_threshold_bin_id,
                            s_num_input,
                            s_counter,
                            s_last_remain,
                            topk_remaining,
                            g_num_input,
                            s_num_input_idx=r_idx ^ 1,
                        )
                        threshold = s_threshold_bin_id[0]
                        if threshold > 0:
                            topk_remaining -= s_histogram[threshold - 1]
                        offset = self.first_refine_shift - round * 8
                        is_last_round = round == self.num_refine_rounds - 1

                        if cutlass.const_expr(self.enable_reread_always):
                            run_next_round, chain_mask, chain_prefix = self._reread_gmem_rescan(
                                topk_remaining,
                                is_last_round,
                                tidx,
                                threshold_bin,
                                threshold,
                                offset,
                                chain_mask,
                                chain_prefix,
                                score,
                                s_counter,
                                s_indices,
                                s_last_remain,
                                s_histogram,
                                _copy_atom,
                                scan_frag,
                                _aligned_base,
                                vec_start,
                                aligned_size,
                                row_start,
                                prologue_elems,
                                left_start,
                                left_size,
                            )
                        elif cutlass.const_expr(self.enable_reread):
                            if did_overflow:
                                # Overflow fallback: REREAD_ALWAYS-style GMEM re-scan.
                                run_next_round, chain_mask, chain_prefix = self._reread_gmem_rescan(
                                    topk_remaining,
                                    is_last_round,
                                    tidx,
                                    threshold_bin,
                                    threshold,
                                    offset,
                                    chain_mask,
                                    chain_prefix,
                                    score,
                                    s_counter,
                                    s_indices,
                                    s_last_remain,
                                    s_histogram,
                                    _copy_atom,
                                    scan_frag,
                                    _aligned_base,
                                    vec_start,
                                    aligned_size,
                                    row_start,
                                    prologue_elems,
                                    left_start,
                                    left_size,
                                )
                            else:
                                # No overflow: SMEM-based refinement (same as GMEM_SPILL).
                                num_input = min(
                                    s_num_input[r_idx], self.filtered_topk_smem_input_size
                                )
                                cur_g_num_input = cutlass.Int32(0)
                                if topk_remaining == 0:
                                    self._collect_below_threshold_refine(
                                        tidx,
                                        threshold,
                                        offset,
                                        num_input,
                                        r_idx,
                                        s_input_idx,
                                        score,
                                        s_counter,
                                        s_indices,
                                        cur_g_num_input,
                                        None,
                                    )
                                    run_next_round = False
                                else:
                                    self._filter_and_histogram_refine(
                                        tidx,
                                        threshold,
                                        offset,
                                        r_idx,
                                        is_last_round,
                                        num_input,
                                        cur_g_num_input,
                                        score,
                                        s_counter,
                                        s_indices,
                                        s_input_idx,
                                        s_num_input,
                                        s_histogram,
                                        s_last_remain,
                                        None,
                                        None,
                                    )
                        else:
                            num_input = min(s_num_input[r_idx], self.filtered_topk_smem_input_size)
                            cur_g_num_input = cutlass.Int32(0)
                            if cutlass.const_expr(self.enable_gmem_store):
                                cur_g_num_input = g_num_input[r_idx]

                            if topk_remaining == 0:
                                self._collect_below_threshold_refine(
                                    tidx,
                                    threshold,
                                    offset,
                                    num_input,
                                    r_idx,
                                    s_input_idx,
                                    score,
                                    s_counter,
                                    s_indices,
                                    cur_g_num_input,
                                    buffer,
                                )
                                run_next_round = False
                            else:
                                self._filter_and_histogram_refine(
                                    tidx,
                                    threshold,
                                    offset,
                                    r_idx,
                                    is_last_round,
                                    num_input,
                                    cur_g_num_input,
                                    score,
                                    s_counter,
                                    s_indices,
                                    s_input_idx,
                                    s_num_input,
                                    s_histogram,
                                    s_last_remain,
                                    g_num_input,
                                    buffer,
                                )

            # Phase 3: Output phase
            vecsize_out = cutlass.const_expr(
                min(
                    self.top_k,
                    cute.ceil_div(self.top_k, self.num_threads_per_cta),
                    self.num_copy_bits // self.dtype.width,
                    # TODO: only tested for float32. need to check for other dtypes.
                    2,
                )
            )
            assert self.top_k % vecsize_out == 0

            nvec_per_thread = cutlass.const_expr(
                cute.ceil_div(self.top_k, vecsize_out * self.num_threads_per_cta)
            )
            topk_vals = cute.make_fragment((vecsize_out, nvec_per_thread), self.dtype)
            topk_indices = cute.make_fragment((vecsize_out, nvec_per_thread), cutlass.Int32)

            stride = self.num_threads_per_cta * vecsize_out
            for i in cutlass.range(nvec_per_thread, unroll_full=True):
                idx = i * stride + tidx % self.num_threads_per_cta * vecsize_out
                if idx < self.top_k:
                    for v in cutlass.range(vecsize_out, unroll_full=True):
                        index_raw = s_indices[idx + v]
                        index = cutlass.Int32(cutlass.Uint32(index_raw))
                        if cutlass.const_expr(self.return_val):
                            topk_vals[v, i] = score[index]
                        if cutlass.const_expr(self.merge_blocks):
                            topk_indices[v, i] = indices[index]
                        elif cutlass.const_expr(self.subtract_row_start_on_output):
                            topk_indices[v, i] = index - cutlass.Int32(row_start)
                        else:
                            topk_indices[v, i] = index
            # [atom, rest_vec]
            mIndices_store = cute.tiled_divide(dst, (vecsize_out,))
            if cutlass.const_expr(self.return_val):
                mValues_store = cute.tiled_divide(dst_values, (vecsize_out,))
            # i represents the index of the vector in the output.
            for i in cutlass.range(cute.size(topk_vals.shape, [1]), unroll_full=True):
                col = i * self.num_threads_per_cta + tidx % self.num_threads_per_cta
                if col < self.top_k // vecsize_out:
                    cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])
                    if cutlass.const_expr(self.return_val):
                        cute.autovec_copy(topk_vals[None, i], mValues_store[None, col])


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    pad_to_vec_size: bool = False,
    vec_size: int = 8,
) -> torch.Tensor:
    """Create random logits tensor for testing.

    Args:
        row_starts: Tensor of shape (num_rows,) indicating the start position of each row
        row_ends: Tensor of shape (num_rows,) indicating the end position (exclusive) of each row
        dtype: Data type for the logits tensor
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (num_rows, max_row_length) with random values and -inf padding
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_rows = row_starts.shape[0]
    max_len = int(row_ends.max().item())
    if pad_to_vec_size:
        max_len = (max_len + vec_size - 1) // vec_size * vec_size

    # Generate random logits
    logits = torch.randn(num_rows, max_len, dtype=dtype, device="cuda")

    # Vectorized masking: set positions outside [row_start, row_end) to -inf
    col_indices = torch.arange(max_len, device="cuda").unsqueeze(0)  # (1, max_len)
    mask_lo = col_indices < row_starts.unsqueeze(1)  # positions before row_start
    mask_hi = col_indices >= row_ends.unsqueeze(1)  # positions at or after row_end
    mask = mask_lo | mask_hi  # positions outside valid range
    logits[mask] = float("-inf")

    return logits


def run_reference_top_k(logits, row_starts, row_ends, index_topk):
    # Run reference implementation
    torch_indices = logits.topk(min(index_topk, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    return torch_indices


def compare_top_k_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Handles different shapes and -1 placeholders in cuda_indices.

    Args:
        logits: Input logits tensor [num_rows, vocab_size]
        cuda_indices: CUDA implementation output [num_rows, cuda_k], may contain -1
        torch_indices: PyTorch reference output [num_rows, torch_k], may contain -1
        row_starts: Start positions for each row [num_rows]
        row_ends: End positions for each row [num_rows]
        top_k: Target top-k value
        tolerance: Tolerance for floating point comparison

    Returns:
        True if results match within tolerance, False otherwise
    """
    num_rows = cuda_indices.shape[0]

    # Calculate valid lengths for each row (vectorized)
    row_lengths = row_ends - row_starts

    # For each row, compare only the valid indices (non -1)
    for row_idx in range(num_rows):
        row_len = row_lengths[row_idx].item()
        expected_valid = min(row_len, top_k)

        # Get valid indices from both implementations (filter out -1)
        cuda_row = cuda_indices[row_idx]
        torch_row = torch_indices[row_idx]

        # Filter out -1 (invalid) indices
        cuda_valid_mask = cuda_row != -1
        torch_valid_mask = torch_row != -1

        cuda_valid = cuda_row[cuda_valid_mask]
        torch_valid = torch_row[torch_valid_mask]

        # Check if the number of valid indices matches
        if cuda_valid.shape[0] != torch_valid.shape[0]:
            print(
                f"Row {row_idx}: Different number of valid indices - "
                f"CUDA: {cuda_valid.shape[0]}, PyTorch: {torch_valid.shape[0]}"
            )
            return False

        if cuda_valid.shape[0] != expected_valid:
            print(
                f"Row {row_idx}: Expected {expected_valid} valid indices, got {cuda_valid.shape[0]}"
            )
            return False

        # If no valid indices, continue
        if cuda_valid.shape[0] == 0:
            continue

        # Gather the corresponding logit values
        row_start = row_starts[row_idx].item()
        logits_row = logits[row_idx]

        # Adjust indices to absolute positions (add row_start offset)
        cuda_abs_indices = cuda_valid + row_start
        torch_abs_indices = torch_valid + row_start

        # Get logit values for the selected indices
        cuda_values = logits_row[cuda_abs_indices]
        torch_values = logits_row[torch_abs_indices]

        # Sort both value arrays in descending order
        cuda_values_sorted, _ = torch.sort(cuda_values, descending=True)
        torch_values_sorted, _ = torch.sort(torch_values, descending=True)

        # Compare sorted values
        if not torch.allclose(
            cuda_values_sorted, torch_values_sorted, rtol=tolerance, atol=tolerance
        ):
            # Additional debug: check if sets are identical
            cuda_set = set(cuda_valid.cpu().tolist())
            torch_set = set(torch_valid.cpu().tolist())
            print(f"row_idx: {row_idx}, row_len: {row_len}, expected_valid: {expected_valid}")
            print(f"cuda_values_sorted: {cuda_values_sorted}")
            print(f"torch_values_sorted: {torch_values_sorted}")
            if cuda_set != torch_set:
                print("  Different indices selected:")
                print(f"    Only in CUDA: {cuda_set - torch_set}")
                print(f"    Only in Torch: {torch_set - cuda_set}")

            return False

    return True
