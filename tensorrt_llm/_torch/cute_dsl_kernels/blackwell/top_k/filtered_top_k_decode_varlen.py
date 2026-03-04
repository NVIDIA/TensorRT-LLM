# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


import argparse
import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.torch import dtype as torch_dtype
from cutlass.utils.distributed import atomicAdd

from .filtered_top_k_varlen_util import (
    FilteredTopKKernelVarlen,
    compare_top_k_results,
    create_random_logits,
    run_reference_top_k,
)

"""
A high-performance topk kernel example based on radix-based filter algorithm for
the NVIDIA Blackwell SM100 architecture based on CuTe DSL.

The radix-based filter top-k algorithm mainly includes two phases: coarse filter and multi-round fine-grained filter.
For each phase:
1. histogram: Build a histogram of the input values using vectorized loads.
2. prefix sum: Find the threshold bin using prefix sum.
3. find target bin id: Find the target bin id using multiple rounds.
Finally, write the top-k values and indices to the output tensor.

Supported data types:
- Float32
- Float16
- BFloat16

To run this example:
.. code-block:: bash
    python examples/blackwell/sort/filter_top_k_decode_varlen.py  \
      --dtype Float32 --batch_size 1 --max_num_cols 4096 --next_n 3 \
      --top_k 2048 --do_ref_check --return_val --do_benchmark

Constraints for this example:
* The problem size of top_k <= 2048.
* The input tensor has data contiguous on the n dimension (row-major).
* The supported input data types are Float32, Float16, or BFloat16.
"""


class FilteredTopKKernelVarlenDecode(FilteredTopKKernelVarlen):
    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        return_val: bool = True,
        large_occupancy: bool = False,
        # for multi-cta version.
        enable_multi_cta: bool = False,
        chunk_size_per_cta: int = 16384,
        num_ctas_per_row: int = 1,
        merge_blocks: bool = False,
    ):
        super().__init__(
            dtype,
            max_num_cols,
            top_k,
            num_copy_bits,
            return_val,
            enable_multi_cta,
            chunk_size_per_cta,
            num_ctas_per_row,
            merge_blocks,
        )
        self.next_n = next_n
        self.enable_multi_cta = enable_multi_cta
        self.chunk_size_per_cta = chunk_size_per_cta
        self.merge_blocks = merge_blocks
        self.num_ctas_per_row = num_ctas_per_row

        print(f"large_occupancy: {large_occupancy}")
        if cutlass.const_expr(large_occupancy):
            # reduce the smem usage and improve occupancy.
            if self.max_num_cols >= 262144:
                self.filtered_topk_smem_input_size = 4096
            elif self.max_num_cols >= 131072:
                self.filtered_topk_smem_input_size = 3072
            elif self.max_num_cols >= 65536:
                self.filtered_topk_smem_input_size = 2048
            elif self.max_num_cols >= 32768:
                self.filtered_topk_smem_input_size = 1024
            elif self.max_num_cols >= 16384:
                self.filtered_topk_smem_input_size = 1024
            elif self.max_num_cols >= 8192:
                self.filtered_topk_smem_input_size = 512
            else:
                self.filtered_topk_smem_input_size = 256

            if cutlass.const_expr(self.max_num_cols > self.filtered_topk_smem_input_size):
                self.enable_gmem_store = True
            else:
                self.enable_gmem_store = False

            # set the number of threads per cta to 512.
            self.num_threads_per_cta = 512

            print(f"return_val: {return_val}")
            print(f"limin: max_num_cols: {self.max_num_cols}")
            print(f"limin: num_threads_per_cta: {self.num_threads_per_cta}")
            print(
                f"limin: filtered_topk_smem_input_size: {self.filtered_topk_smem_input_size}",
            )
            print(f"limin: enable_gmem_store: {self.enable_gmem_store}")

    @cute.jit
    def run_kernel(
        self,
        input,
        indices,
        extra_buffer,
        output_indices,
        output_values,
        tiler_mn,
        copy_atom,
        tiled_copy,
        seqlen,
        task_id,
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
        # TODO: update row_start to align with multi-cta version.
        row_start = 0
        seq_len = seqlen[task_id // self.next_n]
        row_end = seq_len - self.next_n + (task_id % self.next_n) + 1

        length = row_end - row_start

        self.filtered_topk_kernel_per_row(
            input,
            indices,
            extra_buffer,
            output_indices,
            output_values,
            tiler_mn,
            copy_atom,
            tiled_copy,
            row_start,
            length,
            task_id,
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
        )

    @cute.kernel
    def filtered_topk_kernel(
        self,
        input: cute.Tensor,
        indices: cute.Tensor,
        extra_buffer: cute.Tensor,
        g_global_counter: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        tiler_mn: cute.Shape,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        enable_persistent_dymamic_scheduling: cutlass.Constexpr[bool] = False,
        min_blocks_per_mp: cutlass.Constexpr[int] = 1,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm."""
        smem = utils.SmemAllocator()
        # TODO: how to simplify the smem allocate codes?
        s_histogram_buf_layout = cute.make_ordered_layout((self.radix + 1), order=(0))
        s_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=s_histogram_buf_layout,
            byte_alignment=128,
        )
        s_counter = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        s_threshold_bin_id = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        s_num_input = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2,), order=(0)),
            byte_alignment=128,
        )
        if cutlass.const_expr(self.enable_gmem_store):
            g_num_input = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((2), order=(0)),
                byte_alignment=128,
            )
        else:
            g_num_input = None
        s_indices = smem.allocate_tensor(
            element_type=self.index_type,
            layout=cute.make_ordered_layout((self.filtered_topk_max_k,), order=(0)),
            byte_alignment=128,
        )
        s_input_idx = smem.allocate_tensor(
            element_type=self.index_type,
            layout=cute.make_ordered_layout(
                (
                    self.num_buffer_smem_input_idx,
                    self.filtered_topk_smem_input_size,
                ),
                order=(1, 0),
            ),
            byte_alignment=128,
        )
        s_last_remain = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        num_warps = cutlass.const_expr(
            min(self.radix, self.num_threads_per_cta) // cutlass.Int32(32)
        )
        s_warp_sums = smem.allocate_tensor(
            element_type=cute.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=128,
        )

        if cutlass.const_expr(not enable_persistent_dymamic_scheduling):
            # Thread and block indexing
            bidx, bidy, _ = cute.arch.block_idx()

            row_start = 0
            row_end = 0
            length = 0
            seq_len = 0

            if not cutlass.const_expr(self.merge_blocks):
                seq_len = seqlen[bidx // self.next_n]
                row_end = seq_len - self.next_n + (bidx % self.next_n) + 1
                length = row_end - row_start

            if cutlass.const_expr(self.enable_multi_cta):
                # update row_start and row_end.
                row_start = self.chunk_size_per_cta * bidy
                row_end = min(row_end, row_start + self.chunk_size_per_cta)
                length = row_end - row_start
                output_indices = cute.flat_divide(output_indices, (1, self.top_k))[
                    0, None, bidx, bidy
                ]
                output_values = cute.flat_divide(output_values, (1, self.top_k))[
                    0, None, bidx, bidy
                ]

            if cutlass.const_expr(self.merge_blocks):
                # Note, after 1st kernel, the output is fix-lenght.
                row_end = self.max_num_cols
                length = self.max_num_cols

            self.filtered_topk_kernel_per_row(
                input,
                indices,
                extra_buffer,
                output_indices,
                output_values,
                tiler_mn,
                copy_atom,
                tiled_copy,
                row_start,
                length,
                bidx,
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
            )
        else:
            num_rows = input.shape[0]
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            row_start = cutlass.Int32(0)
            row_end = cutlass.Int32(0)
            length = cutlass.Int32(0)
            seq_len = cutlass.Int32(0)

            # dynamic scheduler.
            work_remaining = True

            s_row_id = smem.allocate_tensor(
                element_type=cute.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )

            # version-1: dynamic scheduler.
            while work_remaining:
                # let tidx 0 dynamic get the next task
                if tidx == 0:
                    s_row_id[0] = atomicAdd(g_global_counter.iterator, 1)
                cute.arch.barrier()

                row_id = s_row_id[0]
                has_work = row_id < num_rows

                if has_work:
                    task_id = row_id

                    row_start = 0
                    seq_len = seqlen[task_id // self.next_n]
                    row_end = seq_len - self.next_n + (task_id % self.next_n) + 1

                    length = row_end - row_start

                    self.filtered_topk_kernel_per_row(
                        input,
                        indices,
                        extra_buffer,
                        output_indices,
                        output_values,
                        tiler_mn,
                        copy_atom,
                        tiled_copy,
                        row_start,
                        length,
                        task_id,
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
                    )
                # use the result of the task to update the while loop condition
                work_remaining = has_work

    @cute.jit
    def __call__(
        self,
        input_values,
        indices,
        extra_buffer,
        g_global_counter,
        seqlen,
        output_indices,
        output_values,
        stream: cuda.CUstream,
        enable_persistent_dymamic_scheduling: cutlass.Constexpr[bool] = False,
        min_blocks_per_mp: cutlass.Constexpr[int] = 1,
    ):
        """Host function for the filtered topk kernel"""
        # now we don't support it.
        assert not (self.enable_multi_cta and enable_persistent_dymamic_scheduling), (
            "enable_multi_cta and enable_persistent_dymamic_scheduling cannot both be True"
        )

        num_rows = input_values.shape[0]
        # each cta processes one row of input.
        if cutlass.const_expr(not enable_persistent_dymamic_scheduling):
            blocks = (num_rows, self.num_ctas_per_row, 1)
        else:
            blocks = (min(148 * min_blocks_per_mp, num_rows), self.num_ctas_per_row, 1)

        (
            copy_atom,
            tiled_copy,
            tiler_mn,
        ) = self._get_tiled_copy()
        self.filtered_topk_kernel(
            input_values,
            indices,
            extra_buffer,
            g_global_counter,
            seqlen,
            output_indices,
            output_values,
            tiler_mn,
            copy_atom,
            tiled_copy,
            enable_persistent_dymamic_scheduling,
            min_blocks_per_mp,
        ).launch(
            grid=blocks,
            block=(tiled_copy.size, 1, 1),
            stream=stream,
        )
        return


# This function is used for integration of framework, e.g. trtllm.
compiled_filter_topk_dict = {}


def cute_dsl_topk_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n,
    return_val=True,
    load_balance=False,
    num_copy_bits=256,
):
    torch_dtype = input_values.dtype
    torch_dtype_to_cutlass_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    dtype = torch_dtype_to_cutlass_dtype[torch_dtype]
    num_rows, num_cols = input_values.shape

    large_occupancy = num_rows > 148
    print(f"large_occupancy: {large_occupancy}, return_val: {return_val}")

    # Note: don't forget num_cols, which means the maximum columns.
    key = (
        dtype,
        num_cols,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        load_balance,
        large_occupancy,
    )
    if key not in compiled_filter_topk_dict:
        # Create fake tensors for compilation
        n = cute.sym_int()
        n_div_32 = cute.sym_int(divisibility=32)
        input_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (n, n_div_32), stride_order=(1, 0), assumed_align=32
        )
        # used for large num_cols
        buffer_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (cute.sym_int(), cute.sym_int(), cute.sym_int()),
            stride_order=(2, 1, 0),
            assumed_align=32,
        )
        seqlen_fake = cute.runtime.make_fake_compact_tensor(
            cute.Int32,
            (n,),
            stride_order=(0,),
        )
        output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n, top_k),
            stride_order=(1, 0),
        )
        if return_val:
            output_values_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n, top_k),
                stride_order=(1, 0),
            )
        else:
            output_values_fake = None
        fake_stream = cute.runtime.make_fake_stream()

        filtered_topk_func = FilteredTopKKernelVarlenDecode(
            dtype,
            num_cols,
            top_k,
            next_n,
            num_copy_bits=num_copy_bits,
            return_val=return_val,
            large_occupancy=large_occupancy,
        )

        # Compile the kernel
        compiled_kernel = cute.compile(
            filtered_topk_func,
            input_fake,
            None,  # indices_fake,
            buffer_fake,
            None,  # g_global_counter_fake,
            seqlen_fake,
            output_indices_fake,
            output_values_fake,
            stream=fake_stream,
            enable_persistent_dymamic_scheduling=load_balance,
            min_blocks_per_mp=1,  # TODO: do we need this one?
        )
        compiled_filter_topk_dict[key] = compiled_kernel
    else:
        compiled_kernel = compiled_filter_topk_dict[key]

    output_indices_torch = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    if return_val:
        output_values_torch = torch.empty(num_rows, top_k, dtype=torch_dtype, device="cuda")
    else:
        output_values_torch = None

    if dtype == cutlass.Float32:
        buffer_numbers = 2
    else:
        buffer_numbers = 1
    # Note: zeros will trigger an elementwise_add kernel.
    buffer_torch = torch.empty(num_rows, buffer_numbers, num_cols, dtype=torch.int32, device="cuda")
    g_global_counter_torch = None

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_kernel(
        input_values,
        None,  # indices, used for merge blocks kernel of the multi-cta.
        buffer_torch,
        g_global_counter_torch,
        seq_lens,
        output_indices_torch,
        output_values_torch,
        current_stream,
    )
    return output_indices_torch, output_values_torch


def cute_dsl_topk_multi_cta_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n,
    return_val=True,
    load_balance=False,
    num_copy_bits=256,
    chunk_size_per_cta=16384,
):
    torch_dtype = input_values.dtype
    torch_dtype_to_cutlass_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    dtype = torch_dtype_to_cutlass_dtype[torch_dtype]
    num_rows, num_cols = input_values.shape

    large_occupancy = num_rows > 148
    print(f"large_occupancy: {large_occupancy}, return_val: {return_val}")

    # Note: don't forget num_cols, which means the maximum columns.
    enable_multi_cta = True
    num_ctas_per_row = math.ceil(num_cols / chunk_size_per_cta)
    print(f"num_ctas_per_row: {num_ctas_per_row}")
    key = (
        dtype,
        num_cols,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        load_balance,
        large_occupancy,
        enable_multi_cta,
        chunk_size_per_cta,
    )
    if key not in compiled_filter_topk_dict:
        # Create fake tensors for compilation
        n = cute.sym_int()
        n_div_32 = cute.sym_int(divisibility=32)
        input_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (n, n_div_32), stride_order=(1, 0), assumed_align=32
        )
        # used for large num_cols
        buffer_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (cute.sym_int(), cute.sym_int(), cute.sym_int()),
            stride_order=(2, 1, 0),
            assumed_align=32,
        )
        seqlen_fake = cute.runtime.make_fake_compact_tensor(
            cute.Int32,
            (n,),
            stride_order=(0,),
        )
        # used for load-balance, now we don't support it.
        # TODO: used for first kernel output.
        first_kernel_output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n, n_div_32),
            stride_order=(1, 0),
        )
        first_kernel_output_values_fake = cute.runtime.make_fake_compact_tensor(
            dtype,
            (n, n_div_32),
            stride_order=(1, 0),
            assumed_align=32,
        )
        fake_stream = cute.runtime.make_fake_stream()

        filtered_topk_func_first = FilteredTopKKernelVarlenDecode(
            dtype,
            chunk_size_per_cta,  # num_cols
            top_k,
            next_n,
            num_copy_bits=num_copy_bits,
            # for the first kernel, it must return values.
            return_val=True,
            large_occupancy=large_occupancy,
            enable_multi_cta=True,
            chunk_size_per_cta=chunk_size_per_cta,
            num_ctas_per_row=num_ctas_per_row,
            merge_blocks=False,
        )
        # Compile the kernel
        compiled_kernel_first = cute.compile(
            filtered_topk_func_first,
            input_fake,
            None,  # indices_fake,
            buffer_fake,
            None,  # g_global_counter_fake,
            seqlen_fake,
            # output_indices_fake,
            # output_values_fake,
            first_kernel_output_indices_fake,
            first_kernel_output_values_fake,
            stream=fake_stream,
            enable_persistent_dymamic_scheduling=load_balance,
            min_blocks_per_mp=1,
        )

        # TODO: 2nd kernel: use the output of the first kernel as the input.
        indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n, n_div_32),
            stride_order=(1, 0),
        )
        output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n, top_k),
            stride_order=(1, 0),
        )
        output_values_fake = cute.runtime.make_fake_compact_tensor(
            dtype,
            (n, top_k),
            stride_order=(1, 0),
        )
        filtered_topk_func_second = FilteredTopKKernelVarlenDecode(
            dtype,
            num_ctas_per_row * top_k,  # num_cols
            top_k,
            next_n,
            num_copy_bits=num_copy_bits,
            return_val=return_val,
            large_occupancy=large_occupancy,
            enable_multi_cta=False,
            # chunk_size_per_cta=chunk_size_per_cta, # no use
            # num_ctas_per_row=1, # no use
            merge_blocks=True,
        )
        # Compile the kernel
        compiled_kernel_second = cute.compile(
            filtered_topk_func_second,
            input_fake,
            indices_fake,
            buffer_fake,
            None,  # g_global_counter_fake,
            seqlen_fake,
            output_indices_fake,
            output_values_fake,
            stream=fake_stream,
            enable_persistent_dymamic_scheduling=load_balance,
            min_blocks_per_mp=1,
        )

        compiled_filter_topk_dict[key] = (compiled_kernel_first, compiled_kernel_second)
    else:
        compiled_kernel_first, compiled_kernel_second = compiled_filter_topk_dict[key]

    first_kernel_output_indices_torch = torch.empty(
        num_rows, num_ctas_per_row * top_k, dtype=torch.int32, device="cuda"
    )
    first_kernel_output_values_torch = torch.empty(
        num_rows, num_ctas_per_row * top_k, dtype=torch_dtype, device="cuda"
    )
    output_indices_torch = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    if return_val:
        output_values_torch = torch.empty(num_rows, top_k, dtype=torch_dtype, device="cuda")
    else:
        output_values_torch = None

    if dtype == cutlass.Float32:
        buffer_numbers = 2
    else:
        buffer_numbers = 1
    buffer_torch = torch.empty(
        num_rows * num_ctas_per_row,
        buffer_numbers,
        max(chunk_size_per_cta, num_ctas_per_row * top_k),
        dtype=torch.int32,
        device="cuda",
    )
    g_global_counter_torch = None

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_kernel_first(
        input_values,
        None,  # indices, used for merge blocks kernel of the multi-cta.
        buffer_torch,
        g_global_counter_torch,
        seq_lens,
        first_kernel_output_indices_torch,
        first_kernel_output_values_torch,
        current_stream,
    )

    compiled_kernel_second(
        first_kernel_output_values_torch,
        first_kernel_output_indices_torch,
        buffer_torch,
        g_global_counter_torch,
        seq_lens,
        output_indices_torch,
        output_values_torch,
        current_stream,
    )
    return output_indices_torch, output_values_torch


def generate_seq_lens(batch_size, min_long_seq, num_tokens):
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    is_long = torch.rand(batch_size, device="cuda") < 0.9
    num_long = is_long.sum().item()
    if num_long > 0:
        seq_lens[is_long] = torch.randint(
            min_long_seq, num_tokens, (num_long,), dtype=torch.int32, device="cuda"
        )

    num_short = (~is_long).sum().item()
    if num_short > 0:
        seq_lens[~is_long] = torch.randint(
            1, min_long_seq, (num_short,), dtype=torch.int32, device="cuda"
        )
    return seq_lens


def run_filtered_topk_decode(
    dtype: Type[cutlass.Numeric],
    batch_size,
    max_num_cols,
    top_k,
    next_n,
    load_balance: bool = False,
    num_copy_bits=256,
    return_val=True,
    large_occupancy=False,
    do_ref_check=True,
    do_benchmark=False,
    warmup_iterations=10,
    iterations=100,
    use_cold_l2=True,
    print_verbose=True,
):
    """
    Prepare input tensors, launch GPU kernel, and reference checking.
    """
    if print_verbose:
        print("=" * 60)
        print("Launching Blackwell Filtered TopK Test")
        print("-" * 60)
        print(f"Data Types & Precision: {dtype}")
        print(f"    Input matrix: {dtype}")
        print(f"    Output indices: {cutlass.Int32}")
        print(f"    Output values: {dtype}")
        print(
            f"Input dimensions (batch_size, max_num_cols, top_k): {batch_size, max_num_cols, top_k}"
        )
        print(f"    batch_size: {batch_size}")
        print(f"    next_n: {next_n}")
        print(f"    max_num_cols: {max_num_cols}")
        print(f"    top_k: {top_k}")
        print(f"    load_balance: {load_balance}")
        print(f"    num_copy_bits: {num_copy_bits}")
        print(f"    return_val: {return_val}")
        print(f"    large_occupancy: {large_occupancy}")
        print(f"Do reference checking: {do_ref_check}")
        print(f"Do benchmark: {do_benchmark}")
        print(f"Warmup iterations: {warmup_iterations}")
        print(f"Iterations: {iterations}")
        print(f"Use cold L2: {use_cold_l2}")
        print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(1111)

    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # Create fake tensors for compilation
    n = cute.sym_int()
    n_div_32 = cute.sym_int(divisibility=32)

    # # We need to pad the input tensor so that each row can be aligned to vec_size and could use vectorized copy.
    input_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (n, n_div_32), stride_order=(1, 0), assumed_align=32
    )
    # TODO
    if dtype == cutlass.Float32:
        buffer_numbers = 2
    else:
        buffer_numbers = 1
    buffer_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n, n, n_div_32),
        stride_order=(2, 1, 0),
        assumed_align=32,
    )
    g_global_counter_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), stride_order=(0,)
    )
    seqlen_fake = cute.runtime.make_fake_compact_tensor(
        cute.Int32,
        (n,),
        stride_order=(0,),
    )
    print("input_fake: ", input_fake)
    print("seqlen_fake: ", seqlen_fake)
    output_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n, top_k),
        stride_order=(1, 0),
    )
    if return_val:
        output_values_fake = cute.runtime.make_fake_compact_tensor(
            dtype,
            (n, top_k),
            stride_order=(1, 0),
        )
    else:
        output_values_fake = None
    fake_stream = cute.runtime.make_fake_stream()

    filtered_topk_func = FilteredTopKKernelVarlenDecode(
        dtype,
        max_num_cols,
        top_k,
        next_n,
        num_copy_bits=num_copy_bits,
        return_val=return_val,
        large_occupancy=large_occupancy,
    )

    # Compile the kernel
    compiled_kernel = cute.compile(
        filtered_topk_func,
        input_fake,
        None,  # indices, used for merge blocks kernel of the multi-cta.
        buffer_fake,
        g_global_counter_fake,
        seqlen_fake,
        output_indices_fake,
        output_values_fake,
        stream=fake_stream,
        enable_persistent_dymamic_scheduling=load_balance,
        # TODO: confirm this parameter.
        min_blocks_per_mp=4 if large_occupancy else 1,
    )

    # Set input data
    # num_gen_tokens is the number of rows in the input tensor
    g_global_counter_torch = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    print("g_global_counter_torch: ", g_global_counter_torch)
    num_gen_tokens = batch_size * next_n  # Use the same variable name as dsa.py
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    # max_num_cols is the maximum col length in the input tensor
    seq_lens = generate_seq_lens(batch_size, top_k, max_num_cols)
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    row_ends = row_ends.to(torch.int32)
    # print("row_ends: ", row_ends)
    print("row_ends.dtype: ", row_ends.dtype)

    input_torch = create_random_logits(
        row_starts,
        row_ends,
        torch_dtype(dtype),
        seed,
    )
    print("input_torch.shape: ", input_torch.shape)

    output_indices_torch = torch.empty(num_gen_tokens, top_k, dtype=torch.int32, device="cuda")
    if return_val:
        output_values_torch = torch.empty(
            num_gen_tokens, top_k, dtype=torch_dtype(dtype), device="cuda"
        )
    else:
        output_values_torch = None
    buffer_torch = torch.zeros(
        num_gen_tokens,
        buffer_numbers,
        max_num_cols,
        dtype=torch.int32,
        device="cuda",
    )

    compiled_kernel(
        input_torch,
        None,  # indices, used for merge blocks kernel of the multi-cta.
        buffer_torch,
        g_global_counter_torch,
        seq_lens,
        output_indices_torch,
        output_values_torch,
        current_stream,
    )

    if do_ref_check and top_k <= max_num_cols and return_val:
        torch.cuda.synchronize()

        # Compare results
        torch_indices = run_reference_top_k(input_torch, row_starts, row_ends, top_k)
        assert compare_top_k_results(
            input_torch,
            output_indices_torch,
            torch_indices,
            row_starts,
            row_ends,
            top_k,
        ), "CUDA top_k_per_row results don't match torch.topk"
        if print_verbose:
            print("PASSED")

        if not load_balance:
            wrapper_output_indices, wrapper_output_values = cute_dsl_topk_wrapper(
                input_torch,
                seq_lens,
                top_k,
                next_n,
                return_val,
                load_balance,
                num_copy_bits,
            )
            wrapper_output_val_sorted = torch.sort(
                wrapper_output_values.cpu(), dim=1, descending=True
            ).values
            output_val_ref_sorted = torch.sort(
                output_values_torch.cpu(), dim=1, descending=True
            ).values

            assert torch.allclose(wrapper_output_val_sorted, output_val_ref_sorted, atol=1e-5), (
                "CUDA top_k_per_row results don't match wrapper"
            )
            if print_verbose:
                print("Wrapper: PASSED")

            # test multi-cta version.
            wrapper_output_indices_multi_cta, wrapper_output_values_multi_cta = (
                cute_dsl_topk_multi_cta_wrapper(
                    input_torch,
                    seq_lens,
                    top_k,
                    next_n,
                    return_val,
                    load_balance,
                    num_copy_bits,
                    chunk_size_per_cta=8192,
                )
            )
            wrapper_output_val_sorted_multi_cta = torch.sort(
                wrapper_output_values_multi_cta.cpu(), dim=1, descending=True
            ).values
            output_val_ref_sorted = torch.sort(
                output_values_torch.cpu(), dim=1, descending=True
            ).values

            for i in range(num_gen_tokens):
                if not torch.allclose(
                    wrapper_output_val_sorted_multi_cta[i, :],
                    output_val_ref_sorted[i, :],
                    atol=1e-5,
                ):
                    print(f"FAILED for row_id: {i}")
                    print(
                        f"wrapper_output_val_sorted_multi_cta: {wrapper_output_val_sorted_multi_cta[i]}"
                    )
                    print(f"output_values_torch: {output_val_ref_sorted[i]}")
                    break
            assert torch.allclose(
                wrapper_output_val_sorted_multi_cta,
                output_val_ref_sorted.cpu(),
                atol=1e-5,
            ), "CUDA top_k_per_row results don't match wrapper multi-cta"
            if print_verbose:
                print("Wrapper multi-cta: PASSED")

    if do_benchmark:

        def generate_inputs():
            g_global_counter_torch = torch.zeros(1, dtype=torch.int32, device="cuda")
            torch.cuda.synchronize()
            input_tensor = create_random_logits(
                row_starts,
                row_ends,
                torch_dtype(dtype),
                seed,
            )

            output_indices_tensor = torch.empty(
                num_gen_tokens, top_k, dtype=torch.int32, device="cuda"
            )
            if return_val:
                output_values_tensor = torch.empty(
                    num_gen_tokens, top_k, dtype=torch_dtype(dtype), device="cuda"
                )
            else:
                output_values_tensor = None
            return cute.testing.JitArguments(
                input_tensor,
                None,  # indices, used for merge blocks kernel of the multi-cta.
                buffer_torch,
                g_global_counter_torch,
                seq_lens,
                output_indices_tensor,
                output_values_tensor,
                current_stream,
            )

        workspace_count = 1
        if use_cold_l2:
            one_workspace_bytes = (
                input_torch.numel() * input_torch.element_size()
                + row_starts.numel() * row_starts.element_size()
                + row_ends.numel() * row_ends.element_size()
                + seq_lens.numel() * seq_lens.element_size()
                + output_indices_torch.numel() * output_indices_torch.element_size()
                + (
                    output_values_torch.numel() * output_values_torch.element_size()
                    if return_val
                    else 0
                )
            )
            workspace_count = cute.testing.get_workspace_count(
                one_workspace_bytes, warmup_iterations, iterations
            )
            # Note: when load-balance is enabled, we need to memset g_global_counter_torch to 0 for each iteration.
            # without this, the kernel will accumulate the global counter from previous iterations.
            # Here, we war the memset by setting the workspace_count to the sum of warmup_iterations and iterations.
            workspace_count = iterations + warmup_iterations
            print("workspace_count: ", workspace_count)
        time = cute.testing.benchmark(
            compiled_kernel,
            workspace_generator=generate_inputs,
            workspace_count=workspace_count,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            use_cuda_graphs=True,
            stream=current_stream,
        )
        if print_verbose:
            print(f"Time: {time} us")
        print(f"{dtype}-{batch_size}-{max_num_cols}-{top_k} {time}")
    torch.cuda.synchronize()


def run_topk_decode(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    max_num_cols: int,
    top_k: int,
    next_n: int,
    load_balance: bool = False,
    num_copy_bits: int = 256,
    return_val: bool = True,
    large_occupancy: bool = False,
    do_ref_check: bool = True,
    do_benchmark: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 10,
    use_cold_l2: bool = True,
):
    run_filtered_topk_decode(
        dtype,
        batch_size,
        max_num_cols,
        top_k,
        next_n,
        load_balance,
        num_copy_bits,
        return_val,
        large_occupancy,
        do_ref_check,
        do_benchmark,
        warmup_iterations,
        iterations,
        use_cold_l2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of Sm100 Dense BlockScaled GEMM.")
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        choices=[cutlass.Float32, cutlass.Float16, cutlass.BFloat16],
        help="Data type of the input matrix",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch_size",
    )
    parser.add_argument("--max_num_cols", type=int, default=4096, help="max_num_cols")
    parser.add_argument("--next_n", type=int, default=3, help="next_n")
    parser.add_argument("--top_k", type=int, default=2048, help="top_k")
    parser.add_argument(
        "--load_balance",
        action="store_true",
        default=False,
        help="Use load balance for varlen optimization",
    )
    parser.add_argument(
        "--num_copy_bits",
        type=int,
        default=256,
        help="num_copy_bits, used for vectorization",
    )
    parser.add_argument(
        "--return_val",
        action="store_true",
        default=False,
        help="Return values",
    )
    parser.add_argument(
        "--large_occupancy",
        action="store_true",
        default=False,
        help="Use large occupancy",
    )
    parser.add_argument(
        "--do_ref_check",
        action="store_true",
        default=False,
        help="Do reference checking",
    )
    parser.add_argument(
        "--do_benchmark", action="store_true", default=False, help="Do benchmark test"
    )
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument("--use_cold_l2", action="store_true", default=True, help="Use cold L2")

    args = parser.parse_args()
    if args.top_k % 2 != 0:
        parser.error("top_k must be a multiple of 2 (got top_k={})".format(args.top_k))

    run_topk_decode(
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_num_cols=args.max_num_cols,
        top_k=args.top_k,
        next_n=args.next_n,
        load_balance=args.load_balance,
        num_copy_bits=args.num_copy_bits,
        return_val=args.return_val,
        large_occupancy=args.large_occupancy,
        do_ref_check=args.do_ref_check,
        do_benchmark=args.do_benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        use_cold_l2=args.use_cold_l2,
    )
