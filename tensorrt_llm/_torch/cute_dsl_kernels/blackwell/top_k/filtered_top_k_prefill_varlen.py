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

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.torch import dtype as torch_dtype

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait
from .filtered_top_k_varlen_util import (
    FilteredTopKKernelVarlen,
    compare_top_k_results,
    create_random_logits,
    run_reference_top_k,
)

"""
Prefill top-k kernel using the radix-based filter algorithm.

Differences from the decode variant:
- Row extents come from row_starts / row_ends tensors rather than seq_lens,
  so row_start may be non-zero.
- Always single-CTA per row (no multi-CTA / merge path).
- Always large_occupancy=True (num_rows >> num_sms in prefill).
- Outputs LOCAL indices relative to row_start, matching the CUDA kernel
  contract in IndexerTopKOp.cpp / indexerTopK.cu.
"""


class FilteredTopKKernelVarlenPrefill(FilteredTopKKernelVarlen):
    """Single-CTA large-occupancy top-k kernel for the prefill phase.

    Key differences vs FilteredTopKKernelVarlenDecode:
    - Takes row_starts / row_ends per-row tensors; row_start may be non-zero.
    - Always single-CTA (no multi-CTA / merge blocks).
    - Always 512 threads (large_occupancy path) with reduced SMEM for high
      occupancy.
    - Output indices are LOCAL (0-indexed within [row_start, row_end)), matching
      the CUDA indexer_topk_prefill convention.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        return_val: bool = False,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
    ):
        super().__init__(
            dtype,
            max_num_cols,
            top_k,
            num_copy_bits,
            return_val,
            enable_multi_cta=False,
            chunk_size_per_cta=16384,
            num_ctas_per_row=1,
            merge_blocks=False,
            overflow_policy=overflow_policy,
            num_threads_override=512,  # always 512 threads for large-occupancy path
            cache_smem_values=cache_smem_values,
        )

        # Output local indices: subtract row_start before writing to output.
        self.subtract_row_start_on_output = True

    def _compute_smem_input_size(self) -> int:
        return self._compute_smem_input_size_for_occupancy(target_blocks_per_sm=4)

    @cute.kernel
    def filtered_topk_kernel(
        self,
        input: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        extra_buffer: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
    ):
        """CuTe DSL top-k kernel for the prefill phase."""
        griddepcontrol_wait()

        smem = utils.SmemAllocator()
        s_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.radix + 1), order=(0)),
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
        if cutlass.const_expr(not self.enable_reread_always):
            s_input_idx = smem.allocate_tensor(
                element_type=self.index_type,
                layout=cute.make_ordered_layout(
                    (self.num_buffer_smem_input_idx, self.filtered_topk_smem_input_size),
                    order=(1, 0),
                ),
                byte_alignment=128,
            )
        else:
            s_input_idx = None
        if cutlass.const_expr(self.cache_smem_values and not self.enable_reread_always):
            s_input_val = smem.allocate_tensor(
                element_type=self.ordered_type,
                layout=cute.make_ordered_layout(
                    (
                        self.num_buffer_smem_input_idx,
                        self.filtered_topk_smem_input_size,
                    ),
                    order=(1, 0),
                ),
                byte_alignment=128,
            )
        else:
            s_input_val = None
        if cutlass.const_expr(self.enable_reread):
            s_overflow_flag = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
        else:
            s_overflow_flag = None
        s_last_remain = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
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

        bidx, _, _ = cute.arch.block_idx()
        row_start = cutlass.Int32(row_starts[bidx])
        row_end = cutlass.Int32(row_ends[bidx])
        length = row_end - row_start

        self.filtered_topk_kernel_per_row(
            input,
            output_indices,  # dummy: input_indices unused when merge_blocks=False
            extra_buffer,
            output_indices,
            output_values,
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
            s_input_val,
            s_last_remain,
            num_warps,
            s_warp_sums,
            s_overflow_flag,
        )

        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(
        self,
        input_values,
        row_starts,
        row_ends,
        extra_buffer,
        output_indices,
        output_values,
        stream: cuda.CUstream,
        min_blocks_per_mp: cutlass.Constexpr[int] = 4,
    ):
        """Host function: launch one CTA per row."""
        num_rows = input_values.shape[0]
        self.filtered_topk_kernel(
            input_values,
            row_starts,
            row_ends,
            extra_buffer,
            output_indices,
            output_values,
        ).launch(
            grid=(num_rows, 1, 1),
            block=(self.num_threads_per_cta, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=min_blocks_per_mp,
        )


def _next_positive_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _bucket_num_cols(num_cols: int) -> int:
    return _next_positive_power_of_2(num_cols)


compiled_filter_topk_prefill_dict = {}


def cute_dsl_topk_prefill_wrapper(
    input_values,
    row_starts,
    row_ends,
    top_k,
    return_val=False,
    num_copy_bits=256,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool = False,
):
    torch_dtype_ = input_values.dtype
    dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype_]
    num_rows, num_cols = input_values.shape
    bucketed_num_cols = _bucket_num_cols(num_cols)

    key = (
        dtype,
        bucketed_num_cols,
        top_k,
        return_val,
        num_copy_bits,
        overflow_policy,
        cache_smem_values,
    )
    if key not in compiled_filter_topk_prefill_dict:
        n_rows = cute.sym_int()
        n_cols = cute.sym_int()
        input_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=32
        )
        row_starts_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_rows,), stride_order=(0,)
        )
        row_ends_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_rows,), stride_order=(0,)
        )
        if overflow_policy == "GMEM_SPILL":
            buffer_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, cute.sym_int(), n_cols),
                stride_order=(2, 1, 0),
                assumed_align=32,
            )
        else:
            buffer_fake = None
        output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, top_k), stride_order=(1, 0)
        )
        if return_val:
            output_values_fake = cute.runtime.make_fake_compact_tensor(
                dtype, (n_rows, top_k), stride_order=(1, 0)
            )
        else:
            output_values_fake = None
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        filtered_topk_func = FilteredTopKKernelVarlenPrefill(
            dtype,
            bucketed_num_cols,
            top_k,
            num_copy_bits=num_copy_bits,
            return_val=return_val,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
        )
        compiled_kernel = cute.compile(
            filtered_topk_func,
            input_fake,
            row_starts_fake,
            row_ends_fake,
            buffer_fake,
            output_indices_fake,
            output_values_fake,
            stream=fake_stream,
            min_blocks_per_mp=4,
            options="--enable-tvm-ffi",
        )
        compiled_filter_topk_prefill_dict[key] = compiled_kernel
    else:
        compiled_kernel = compiled_filter_topk_prefill_dict[key]

    output_indices_torch = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    if return_val:
        output_values_torch = torch.empty(num_rows, top_k, dtype=torch_dtype_, device="cuda")
    else:
        output_values_torch = None

    if overflow_policy == "GMEM_SPILL":
        buffer_numbers = 2 if dtype == cutlass.Float32 else 1
        buffer_torch = torch.empty(
            num_rows, buffer_numbers, num_cols, dtype=torch.int32, device="cuda"
        )
    else:
        buffer_torch = None

    compiled_kernel(
        input_values,
        row_starts,
        row_ends,
        buffer_torch,
        output_indices_torch,
        output_values_torch,
    )
    return output_indices_torch, output_values_torch


def run_filtered_topk_prefill(
    dtype: Type[cutlass.Numeric],
    num_rows: int,
    max_num_cols: int,
    top_k: int,
    num_copy_bits: int = 256,
    return_val: bool = False,
    do_ref_check: bool = True,
    do_benchmark: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
    use_cold_l2: bool = True,
    print_verbose: bool = True,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool = False,
):
    """Prepare input tensors, launch prefill top-k kernel, and check reference."""
    if print_verbose:
        print("=" * 60)
        print("Launching Blackwell Filtered TopK Prefill Test")
        print("-" * 60)
        print(f"dtype: {dtype}")
        print(f"num_rows: {num_rows}")
        print(f"max_num_cols: {max_num_cols}")
        print(f"top_k: {top_k}")
        print(f"num_copy_bits: {num_copy_bits}")
        print(f"return_val: {return_val}")
        print(f"overflow_policy: {overflow_policy}")
        print(f"Do reference checking: {do_ref_check}")
        print(f"Do benchmark: {do_benchmark}")
        print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Each row spans [0, row_end); row lengths drawn uniformly from [top_k, max_num_cols].
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.randint(top_k, max_num_cols + 1, (num_rows,), dtype=torch.int32, device="cuda")

    input_torch = create_random_logits(row_starts, row_ends, torch_dtype(dtype), seed)

    n_rows, n_cols = input_torch.shape

    n_rows_fake = cute.sym_int()
    n_cols_fake = cute.sym_int()
    input_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (n_rows_fake, n_cols_fake), stride_order=(1, 0), assumed_align=32
    )
    row_starts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_rows_fake,), stride_order=(0,)
    )
    row_ends_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_rows_fake,), stride_order=(0,)
    )
    if overflow_policy == "GMEM_SPILL":
        buffer_numbers = 2 if dtype == cutlass.Float32 else 1
        buffer_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_rows_fake, cute.sym_int(), n_cols_fake),
            stride_order=(2, 1, 0),
            assumed_align=32,
        )
    else:
        buffer_fake = None
    output_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_rows_fake, top_k), stride_order=(1, 0)
    )
    if return_val:
        output_values_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (n_rows_fake, top_k), stride_order=(1, 0)
        )
    else:
        output_values_fake = None
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    filtered_topk_func = FilteredTopKKernelVarlenPrefill(
        dtype,
        max_num_cols,
        top_k,
        num_copy_bits=num_copy_bits,
        return_val=return_val,
        overflow_policy=overflow_policy,
        cache_smem_values=cache_smem_values,
    )
    compiled_kernel = cute.compile(
        filtered_topk_func,
        input_fake,
        row_starts_fake,
        row_ends_fake,
        buffer_fake,
        output_indices_fake,
        output_values_fake,
        stream=fake_stream,
        min_blocks_per_mp=4,
        options="--enable-tvm-ffi",
    )

    output_indices_torch = torch.empty(n_rows, top_k, dtype=torch.int32, device="cuda")
    if return_val:
        output_values_torch = torch.empty(n_rows, top_k, dtype=torch_dtype(dtype), device="cuda")
    else:
        output_values_torch = None
    if overflow_policy == "GMEM_SPILL":
        buffer_torch = torch.empty(n_rows, buffer_numbers, n_cols, dtype=torch.int32, device="cuda")
    else:
        buffer_torch = None

    compiled_kernel(
        input_torch,
        row_starts,
        row_ends,
        buffer_torch,
        output_indices_torch,
        output_values_torch,
    )

    if do_ref_check and top_k <= max_num_cols:
        torch.cuda.synchronize()
        torch_indices = run_reference_top_k(input_torch, row_starts, row_ends, top_k)
        assert compare_top_k_results(
            input_torch, output_indices_torch, torch_indices, row_starts, row_ends, top_k
        ), "prefill top-k results don't match torch.topk"
        if print_verbose:
            print("PASSED")

        wrapper_indices, wrapper_values = cute_dsl_topk_prefill_wrapper(
            input_torch,
            row_starts,
            row_ends,
            top_k,
            return_val,
            num_copy_bits,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
        )
        assert compare_top_k_results(
            input_torch, wrapper_indices, torch_indices, row_starts, row_ends, top_k
        ), "prefill wrapper results don't match torch.topk"
        if print_verbose:
            print("Wrapper: PASSED")

    if do_benchmark:

        def generate_inputs():
            torch.cuda.synchronize()
            input_tensor = create_random_logits(row_starts, row_ends, torch_dtype(dtype), seed)
            output_indices_tensor = torch.empty(n_rows, top_k, dtype=torch.int32, device="cuda")
            if return_val:
                output_values_tensor = torch.empty(
                    n_rows, top_k, dtype=torch_dtype(dtype), device="cuda"
                )
            else:
                output_values_tensor = None
            return cute.testing.JitArguments(
                input_tensor,
                row_starts,
                row_ends,
                buffer_torch,
                output_indices_tensor,
                output_values_tensor,
            )

        workspace_count = iterations + warmup_iterations if use_cold_l2 else 1
        print("workspace_count: ", workspace_count)
        torch_stream = torch.cuda.Stream()
        benchmark_stream = cuda.CUstream(torch_stream.cuda_stream)
        with torch.cuda.stream(torch_stream):
            time = cute.testing.benchmark(
                compiled_kernel,
                workspace_generator=generate_inputs,
                workspace_count=workspace_count,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
                use_cuda_graphs=True,
                stream=benchmark_stream,
            )
        if print_verbose:
            print(f"Time: {time} us")
        print(f"{dtype}-{num_rows}-{max_num_cols}-{top_k} {time}")
    torch.cuda.synchronize()


def run_topk_prefill(
    dtype: Type[cutlass.Numeric],
    num_rows: int,
    max_num_cols: int,
    top_k: int,
    num_copy_bits: int = 256,
    return_val: bool = False,
    do_ref_check: bool = True,
    do_benchmark: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 10,
    use_cold_l2: bool = True,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool = False,
):
    run_filtered_topk_prefill(
        dtype,
        num_rows,
        max_num_cols,
        top_k,
        num_copy_bits,
        return_val,
        do_ref_check,
        do_benchmark,
        warmup_iterations,
        iterations,
        use_cold_l2,
        overflow_policy=overflow_policy,
        cache_smem_values=cache_smem_values,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Blackwell CuTE DSL filtered top-k prefill benchmark."
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        choices=[cutlass.Float32, cutlass.Float16, cutlass.BFloat16],
        help="Data type of the input matrix",
    )
    parser.add_argument("--num_rows", type=int, default=256, help="Number of rows (sequences)")
    parser.add_argument("--max_num_cols", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--top_k", type=int, default=1024, help="top_k")
    parser.add_argument(
        "--num_copy_bits", type=int, default=256, help="num_copy_bits, used for vectorization"
    )
    parser.add_argument("--return_val", action="store_true", default=False, help="Return values")
    parser.add_argument(
        "--do_ref_check", action="store_true", default=False, help="Do reference checking"
    )
    parser.add_argument(
        "--do_benchmark", action="store_true", default=False, help="Do benchmark test"
    )
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument("--use_cold_l2", action="store_true", default=True, help="Use cold L2")
    parser.add_argument(
        "--overflow_policy",
        type=str,
        default="REREAD",
        choices=["GMEM_SPILL", "TRUNCATE", "REREAD", "REREAD_ALWAYS"],
        help="Overflow policy when candidates exceed SMEM capacity",
    )
    parser.add_argument(
        "--cache_smem_values",
        action="store_true",
        default=False,
        help="Cache ordered values alongside indices in SMEM to avoid re-reading from GMEM in refinement rounds",
    )

    args = parser.parse_args()
    if args.top_k % 2 != 0:
        parser.error("top_k must be a multiple of 2 (got top_k={})".format(args.top_k))

    run_topk_prefill(
        dtype=args.dtype,
        num_rows=args.num_rows,
        max_num_cols=args.max_num_cols,
        top_k=args.top_k,
        num_copy_bits=args.num_copy_bits,
        return_val=args.return_val,
        do_ref_check=args.do_ref_check,
        do_benchmark=args.do_benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        use_cold_l2=args.use_cold_l2,
        overflow_policy=args.overflow_policy,
        cache_smem_values=args.cache_smem_values,
    )
