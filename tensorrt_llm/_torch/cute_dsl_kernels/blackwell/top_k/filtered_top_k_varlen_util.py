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
    ):
        self.dtype = dtype
        self.max_num_cols = max_num_cols
        self.top_k = top_k
        self.num_copy_bits = num_copy_bits
        self.enable_multi_cta = enable_multi_cta
        self.chunk_size_per_cta = chunk_size_per_cta
        self.num_ctas_per_row = num_ctas_per_row
        self.merge_blocks = merge_blocks

        # Note: now we only support top_k <= 2048, we could change the code here to support larger top_k.
        self.filtered_topk_max_k = 2048
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
            if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
                self.max_smem_input_size = 32 * 1024
            else:
                self.max_smem_input_size = 64 * 1024
        else:
            self.index_type = cutlass.Uint32
            if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
                self.max_smem_input_size = 16 * 1024
            else:
                self.max_smem_input_size = 32 * 1024

        self.filtered_topk_smem_input_size = min(self.max_smem_input_size, self.max_num_cols)

        if cutlass.const_expr(self.max_num_cols > self.filtered_topk_smem_input_size):
            self.enable_gmem_store = True
        else:
            self.enable_gmem_store = False

        self.return_val = return_val

        self.vec_size = num_copy_bits // dtype.width
        if cutlass.const_expr(dtype not in [cutlass.Float32, cute.BFloat16, cutlass.Float16]):
            raise ValueError(f"Unsupported dtype: {dtype}")

        if cutlass.const_expr(dtype == cutlass.Float32):
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

        # radix-based filter parameters.
        if cutlass.const_expr(dtype == cutlass.Float32):
            self.ordered_type = cute.Uint32
            self.first_refine_shift = 24
            self.num_refine_rounds = 4
        elif cutlass.const_expr(dtype in [cutlass.Float16, cute.BFloat16]):
            self.ordered_type = cute.Uint16
            self.first_refine_shift = 0
            self.num_refine_rounds = 1

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
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:  # BFloat16
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
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:  # BFloat16
                bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cute.Uint16(0xFFFF)) & cute.Uint16(0x7FFF)
            return cute.Uint16(key)

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
        tiler_mn: cute.Shape,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
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

        shape = input.shape

        idX = cute.make_identity_tensor((shape[0], aligned_size))
        input_ptr = input.iterator + vec_start
        input_addr_u64 = input_ptr.toint()
        input_ptr_aligned = cute.make_ptr(self.dtype, input_addr_u64, assumed_align=align_bytes)

        input_tensor = cute.make_tensor(
            input_ptr_aligned,
            cute.make_layout((shape[0], aligned_size), stride=input.stride),
        )

        # slice for CTAs
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, None)) for mT in (input_tensor, idX)]
        # Note, we use gX_aligned here to avoid the alignment issue when the input is not aligned.
        gX_aligned_ptr = cute.make_ptr(self.dtype, gX.iterator.toint(), assumed_align=align_bytes)
        gX_aligned = cute.make_tensor(gX_aligned_ptr, cute.make_layout(gX.shape, stride=gX.stride))

        self.num_sub_tiles = gX.shape[2]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX_aligned)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None, None]
        tXrX = cute.make_fragment_like(tXgX[None, None, None, 0])

        tXcX_tile = thr_copy.partition_S(cX)

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
            val_one_negative = cutlass.Int32(-1)

            # Stage 1: Coarse histogram.
            if tidx < self.radix + 1:
                s_histogram[tidx] = 0
            cute.arch.barrier()

            # 1.1 Build histogram with vectorized loads
            vec_size = self.vec_size

            for tile_idx in range(self.num_sub_tiles):
                tXpX_tile = self.predicate_tile(
                    tXcX_tile[None, None, None, tile_idx],
                    cutlass.Int32(aligned_size),
                )
                cute.copy(
                    copy_atom,
                    tXgX[None, None, None, tile_idx],
                    tXrX,
                    pred=tXpX_tile[None, None, None],
                )
                self._fill_oob(
                    tXrX,
                    tXpX_tile[None, None, None],
                    -tXrX.element_type.inf,
                )

                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    bin_val = self.to_coarse_key(tXrX[i])
                    atomicAdd(
                        s_histogram.iterator + cutlass.Int32(bin_val),
                        val_one,
                    )

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
                # Collect indices where bin > threshold
                for tile_idx in range(self.num_sub_tiles):
                    tXpX_tile = self.predicate_tile(
                        tXcX_tile[None, None, None, tile_idx],
                        cutlass.Int32(aligned_size),
                    )
                    cute.copy(
                        copy_atom,
                        tXgX[None, None, None, tile_idx],
                        tXrX,
                        pred=tXpX_tile[None, None, None],
                    )
                    self._fill_oob(
                        tXrX,
                        tXpX_tile[None, None, None],
                        -tXrX.element_type.inf,
                    )
                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        bin_val = self.to_coarse_key(tXrX[i])
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one)
                            idx = self.index_type(
                                cur_tXcX[i // vec_size][1] + i % vec_size + vec_start
                            )
                            s_indices[pos] = idx

                # for initial scalar load part.
                for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(row_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx

                # for left part (left_size)
                for j in range(tidx, left_size, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(left_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx

                cute.arch.barrier()

            else:
                # Reset histogram for refinement
                cute.arch.barrier()
                if tidx < self.radix + 1:
                    s_histogram[tidx] = 0
                cute.arch.barrier()

                # Filter and build refinement histogram
                for tile_idx in range(self.num_sub_tiles):
                    tXpX_tile = self.predicate_tile(
                        tXcX_tile[None, None, None, tile_idx],
                        cutlass.Int32(aligned_size),
                    )
                    cute.copy(
                        copy_atom,
                        tXgX[None, None, None, tile_idx],
                        tXrX,
                        pred=tXpX_tile[None, None, None],
                    )
                    self._fill_oob(
                        tXrX,
                        tXpX_tile[None, None, None],
                        -tXrX.element_type.inf,
                    )

                    for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                        raw_input = tXrX[i]
                        bin_val = self.to_coarse_key(raw_input)
                        cur_tXcX = tXcX[None, None, None, tile_idx]
                        idx = self.index_type(cur_tXcX[i // vec_size][1] + i % vec_size + vec_start)
                        if bin_val < threshold_bin:
                            pos = atomicAdd(s_counter.iterator, val_one)
                            s_indices[pos] = idx
                        elif bin_val == threshold_bin:
                            # pos = atomicAdd(s_num_input[0], 1)
                            pos = atomicAdd(s_num_input.iterator, val_one)
                            if cutlass.const_expr(self.enable_gmem_store):
                                if pos < self.filtered_topk_smem_input_size:
                                    s_input_idx[0, pos] = idx
                                else:
                                    buffer_pos = atomicAdd(
                                        g_num_input.iterator,
                                        val_one,
                                    )
                                    buffer[0, buffer_pos] = cutlass.Int32(cutlass.Uint32(idx))
                                ordered = self.to_ordered(raw_input)
                                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                # atomicAdd(s_histogram[sub_bin], 1)
                                atomicAdd(
                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                    val_one,
                                )
                            else:
                                if pos < self.filtered_topk_smem_input_size:
                                    s_input_idx[0, pos] = idx
                                ordered = self.to_ordered(raw_input)
                                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                # atomicAdd(s_histogram[sub_bin], 1)
                                atomicAdd(
                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                    val_one,
                                )

                # for initial scalar load part.
                for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(row_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx
                    elif bin_val == threshold_bin:
                        pos = atomicAdd(
                            s_num_input.iterator,
                            val_one,
                        )
                        # TODO: add gmem buffer here.
                        if cutlass.const_expr(self.enable_gmem_store):
                            if pos < self.filtered_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                            else:
                                buffer_pos = atomicAdd(
                                    g_num_input.iterator,
                                    val_one,
                                )
                                buffer[0, buffer_pos] = cutlass.Int32(col_idx)
                            ordered = self.to_ordered(raw)
                            sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                            atomicAdd(
                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                val_one,
                            )
                        else:
                            # TODO: how to handle the type of sub_bin and ordered?
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                ordered = cutlass.Uint32(0)
                                sub_bin = cutlass.Uint32(0)
                            else:
                                ordered = cutlass.Uint16(0)
                                sub_bin = cutlass.Int32(0)
                            if pos < self.filtered_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                                ordered = self.to_ordered(raw)
                                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                atomicAdd(
                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                    val_one,
                                )

                # for left part
                for j in range(tidx, left_size, self.num_threads_per_cta):
                    col_idx = cutlass.Int32(left_start + j)
                    raw = score[col_idx]
                    bin_val = self.to_coarse_key(raw)
                    if bin_val < threshold_bin:
                        pos = atomicAdd(s_counter.iterator, val_one)
                        idx = self.index_type(col_idx)
                        s_indices[pos] = idx
                    elif bin_val == threshold_bin:
                        pos = atomicAdd(
                            s_num_input.iterator,
                            val_one,
                        )
                        # TODO: add gmem buffer here.
                        if cutlass.const_expr(self.enable_gmem_store):
                            if pos < self.filtered_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                            else:
                                buffer_pos = atomicAdd(
                                    g_num_input.iterator,
                                    val_one,
                                )
                                buffer[0, buffer_pos] = cutlass.Int32(col_idx)
                            ordered = self.to_ordered(raw)
                            sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                            atomicAdd(
                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                val_one,
                            )
                        else:
                            # TODO: how to handle the type of sub_bin and ordered?
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                ordered = cutlass.Uint32(0)
                                sub_bin = cutlass.Uint32(0)
                            else:
                                ordered = cutlass.Uint16(0)
                                sub_bin = cutlass.Int32(0)
                            if pos < self.filtered_topk_smem_input_size:
                                s_input_idx[0, pos] = self.index_type(col_idx)
                                ordered = self.to_ordered(raw)
                                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                                atomicAdd(
                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                    val_one,
                                )
                fence_acq_rel_cta()
                cute.arch.barrier()

                # Phase 2: Refinement rounds
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
                        num_input = min(s_num_input[r_idx], self.filtered_topk_smem_input_size)
                        if cutlass.const_expr(self.enable_gmem_store):
                            cur_g_num_input = g_num_input[r_idx]

                        threshold = s_threshold_bin_id[0]
                        if threshold > 0:
                            topk_remaining -= s_histogram[threshold - 1]
                        offset = self.first_refine_shift - round * 8
                        is_last_round = round == self.num_refine_rounds - 1

                        if topk_remaining == 0:
                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                idx = cutlass.Int32(cutlass.Uint32(idx))
                                bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(s_counter.iterator, val_one)
                                    s_indices[pos] = self.index_type(idx)
                            if cutlass.const_expr(self.enable_gmem_store):
                                for i in range(
                                    tidx,
                                    cur_g_num_input,
                                    self.num_threads_per_cta,
                                ):
                                    idx = buffer[r_idx, i]
                                    bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
                                    if bin_val < threshold:
                                        pos = atomicAdd(s_counter.iterator, val_one)
                                        s_indices[pos] = self.index_type(idx)
                            cute.arch.barrier()
                            # break
                            run_next_round = False
                        else:
                            # Reset histogram
                            cute.arch.barrier()
                            if tidx < self.radix + 1:
                                s_histogram[tidx] = 0
                            cute.arch.barrier()

                            for i in range(tidx, num_input, self.num_threads_per_cta):
                                idx = s_input_idx[r_idx, i]
                                idx_int32 = cutlass.Int32(cutlass.Uint32(idx))
                                raw_input = score[idx_int32]
                                idx = self.index_type(idx_int32)
                                bin_val = (self.to_ordered(raw_input) >> offset) & 0xFF
                                if bin_val < threshold:
                                    pos = atomicAdd(s_counter.iterator, val_one)
                                    s_indices[pos] = idx
                                elif bin_val == threshold:
                                    if is_last_round:
                                        cur_pos = atomicAdd(
                                            s_last_remain.iterator,
                                            val_one_negative,
                                        )
                                        if cur_pos > 0:
                                            s_indices[self.top_k - cur_pos] = idx
                                    else:
                                        # pos = atomicAdd(s_num_input[r_idx ^ 1], 1)
                                        cur_pos = atomicAdd(
                                            s_num_input.iterator + (r_idx ^ 1),
                                            val_one,
                                        )
                                        # TODO: remove this if logic for gmem store?
                                        # num_input < filter_topk_smem_input_size
                                        if cutlass.const_expr(self.enable_gmem_store):
                                            if cur_pos < self.filtered_topk_smem_input_size:
                                                s_input_idx[r_idx ^ 1, cur_pos] = idx
                                            else:
                                                buffer_pos = atomicAdd(
                                                    g_num_input.iterator + (r_idx ^ 1),
                                                    val_one,
                                                )
                                                buffer[r_idx ^ 1, buffer_pos] = idx_int32
                                            bin32 = self.to_ordered(raw_input)
                                            sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                            # atomicAdd(s_histogram[sub_bin], 1)
                                            atomicAdd(
                                                s_histogram.iterator + cutlass.Int32(sub_bin),
                                                val_one,
                                            )
                                        else:
                                            # TODO: how to handle the type of sub_bin and bin32?
                                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                                bin32 = cutlass.Uint32(0)
                                                sub_bin = cutlass.Uint32(0)
                                            else:
                                                bin32 = cutlass.Uint16(0)
                                                sub_bin = cutlass.Int32(0)
                                            if cur_pos < self.filtered_topk_smem_input_size:
                                                s_input_idx[r_idx ^ 1, cur_pos] = idx
                                                bin32 = self.to_ordered(raw_input)
                                                sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                                # atomicAdd(s_histogram[sub_bin], 1)
                                                atomicAdd(
                                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                                    val_one,
                                                )

                            cute.arch.barrier()
                            if cutlass.const_expr(self.enable_gmem_store):
                                for i in range(
                                    tidx,
                                    cur_g_num_input,
                                    self.num_threads_per_cta,
                                ):
                                    # int32
                                    idx = buffer[r_idx, i]
                                    raw_input = score[idx]
                                    bin_val = (self.to_ordered(raw_input) >> offset) & 0xFF
                                    if bin_val < threshold:
                                        pos = atomicAdd(
                                            s_counter.iterator,
                                            val_one,
                                        )
                                        s_indices[pos] = self.index_type(idx)
                                    elif bin_val == threshold:
                                        if is_last_round:
                                            cur_pos = atomicAdd(
                                                s_last_remain.iterator,
                                                val_one_negative,
                                            )
                                            if cur_pos > 0:
                                                s_indices[self.top_k - cur_pos] = self.index_type(
                                                    idx
                                                )
                                        else:
                                            # pos = atomicAdd(s_num_input[r_idx ^ 1], 1)
                                            cur_pos = atomicAdd(
                                                s_num_input.iterator + (r_idx ^ 1),
                                                val_one,
                                            )
                                            if cutlass.const_expr(self.enable_gmem_store):
                                                if cur_pos < self.filtered_topk_smem_input_size:
                                                    s_input_idx[r_idx ^ 1, cur_pos] = (
                                                        self.index_type(idx)
                                                    )
                                                else:
                                                    buffer_pos = atomicAdd(
                                                        g_num_input.iterator + (r_idx ^ 1),
                                                        val_one,
                                                    )
                                                    buffer[r_idx ^ 1, buffer_pos] = idx
                                                bin32 = self.to_ordered(raw_input)
                                                sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                                # atomicAdd(s_histogram[sub_bin], 1)
                                                atomicAdd(
                                                    s_histogram.iterator + cutlass.Int32(sub_bin),
                                                    val_one,
                                                )
                                            else:
                                                if cur_pos < self.filtered_topk_smem_input_size:
                                                    s_input_idx[r_idx ^ 1, cur_pos] = idx
                                                    bin32 = self.to_ordered(raw_input)
                                                    sub_bin = (bin32 >> (offset - 8)) & 0xFF
                                                    # atomicAdd(s_histogram[sub_bin], 1)
                                                    atomicAdd(
                                                        s_histogram.iterator
                                                        + cutlass.Int32(sub_bin),
                                                        val_one,
                                                    )
                            fence_acq_rel_cta()
                            cute.arch.barrier()

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

    def _get_tiled_copy(self):
        threads_per_row = self.num_threads_per_cta
        tiler_mn = (
            1,
            self.vec_size * threads_per_row,
        )

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=self.num_copy_bits,
        )

        thr_layout = cute.make_ordered_layout(
            (1, threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, self.vec_size))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        return (
            copy_atom,
            tiled_copy,
            tiler_mn,
        )

    @cute.jit
    def predicate_tile(self, tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    cute.size(tAcA, mode=[0, 1]),
                    cute.size(tAcA, mode=[1]),
                    cute.size(tAcA, mode=[2]),
                ),
                stride=(cute.size(tAcA, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for rest_k in range(tApA.shape[2]):
                tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
        return tApA

    @cute.jit
    def _fill_oob(self, tXrX: cute.Tensor, tXpX: cute.Tensor, fill_value: cute.Numeric) -> None:
        """Fill out-of-bounds values in register tensor.

        Args:
            tXrX: Register tensor to fill
            tXpX: Predicate tensor indicating valid elements
            fill_value: Value to fill OOB locations with
        """
        tXrX_fill = cute.make_fragment_like(tXrX[(None, 0), None, 0])
        tXrX_fill.fill(fill_value)
        for rest_v in range(tXrX.shape[0][1]):
            for rest_k in range(tXrX.shape[2]):
                if cutlass.const_expr(tXpX is not None):
                    if not tXpX[0, rest_v, rest_k]:
                        cute.autovec_copy(tXrX_fill, tXrX[(None, rest_v), None, rest_k])


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
