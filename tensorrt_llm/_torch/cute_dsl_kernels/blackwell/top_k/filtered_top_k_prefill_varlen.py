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

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait
from .filtered_top_k_varlen_util import FilteredTopKKernelVarlen

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
        debug: bool = False,
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
        )

        # Output local indices: subtract row_start before writing to output.
        self.subtract_row_start_on_output = True

        # Clamp filtered_topk_smem_input_size so the kernel fits 4 blocks/SM on
        # B200 (256 KB/SM ÷ 4 = 64 KB/block budget).
        #
        # SMEM layout per block (worst-case top_k=2048, 128-B alignment):
        #   fixed overhead  ~1920 B  (histogram + scalars + warp_sums)
        #   s_indices        top_k × sizeof(index_type)
        #   s_input_idx      num_buffer × S × sizeof(index_type)
        #
        # Solving 1920 + 2048*idx_sz + num_buffer*S*idx_sz <= 65536:
        #   fp32 / Uint16  (num_buffer=2, idx=2B): S <= 7440  → pow2 cap 8192
        #   fp32 / Uint32  (num_buffer=2, idx=4B): S <= 6928  → pow2 cap 4096
        #   bf16 / Uint16  (num_buffer=1, idx=2B): S <= 29760 → pow2 cap 16384
        #   bf16 / Uint32  (num_buffer=1, idx=4B): S <= 13856 → pow2 cap 8192
        #
        # Verification fp32/Uint16 S=8192: 1920+4096+2*8192*2 = 38784 B < 64 KB ✓
        if self.index_type == cutlass.Uint16:
            max_S = 8192 if self.num_buffer_smem_input_idx == 2 else 16384
        else:  # Uint32 (max_num_cols > 65536)
            max_S = 4096 if self.num_buffer_smem_input_idx == 2 else 8192

        self.filtered_topk_smem_input_size = min(max_S, self.max_num_cols)
        self.enable_gmem_store = self.max_num_cols > self.filtered_topk_smem_input_size

        # Always 512 threads for large-occupancy path.
        self.num_threads_per_cta = 512

        if debug:
            print(f"dtype: {self.dtype}, vec_size: {self.vec_size}")
            print(
                f"max_num_cols: {self.max_num_cols}, "
                f"num_threads_per_cta: {self.num_threads_per_cta}"
            )
            print(f"filtered_topk_smem_input_size: {self.filtered_topk_smem_input_size}")
            print(f"enable_gmem_store: {self.enable_gmem_store}")
            print(f"subtract_row_start_on_output: {self.subtract_row_start_on_output}")

    @cute.kernel
    def filtered_topk_kernel(
        self,
        input: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        extra_buffer: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        tiler_mn: cute.Shape,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
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
        s_input_idx = smem.allocate_tensor(
            element_type=self.index_type,
            layout=cute.make_ordered_layout(
                (self.num_buffer_smem_input_idx, self.filtered_topk_smem_input_size),
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
        copy_atom, tiled_copy, tiler_mn = self._get_tiled_copy()
        self.filtered_topk_kernel(
            input_values,
            row_starts,
            row_ends,
            extra_buffer,
            output_indices,
            output_values,
            tiler_mn,
            copy_atom,
            tiled_copy,
        ).launch(
            grid=(num_rows, 1, 1),
            block=(tiled_copy.size, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=min_blocks_per_mp,
        )
