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
"""Cluster-accelerated single-pass multi-CTA radix top-k kernel for Blackwell.

Inherits from ``SinglePassMultiCTARadixTopKKernel`` (the distributed variant)
and replaces global memory atomics + arrival counter polling with:
  - Cluster barriers (cluster_arrive_relaxed + cluster_wait) for inter-CTA sync
  - DSMEM (distributed shared memory) for histogram merging across CTAs

This eliminates the triple-buffered global histogram, the arrival counter, and
all GPU-scope acquire/release PTX for barriers.  Only the output counter
(1 int32 in GMEM) is retained for atomicAdd during output collection.
"""

import functools

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int32 as CuteInt32
from cutlass.cute.typing import Pointer as CutePointer
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.hardware_info import HardwareInfo
from cutlass.utils.smem_allocator import SmemAllocator

from .single_pass_multi_cta_radix_topk import SinglePassMultiCTARadixTopKKernel, st_release_gpu


@functools.lru_cache(maxsize=1)
def _query_max_cluster_size() -> int:
    """Query the hardware max cluster size using an empty kernel.

    Uses ``cuOccupancyMaxPotentialClusterSize`` with a lightweight dummy
    kernel compiled via CuTE DSL.  The result reflects the GPC topology
    (number of SMs in the largest GPC) of the current device.
    """
    hw = HardwareInfo()
    func = hw._get_device_function()

    # Allow non-portable cluster sizes for accurate detection.
    hw._checkCudaErrors(
        driver.cuFuncSetAttribute(
            func,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            1,
        )
    )

    config = driver.CUlaunchConfig()
    config.blockDimX = 1
    config.blockDimY = 1
    config.blockDimZ = 1
    config.gridDimX = 1
    config.gridDimY = 1
    config.gridDimZ = 1
    config.sharedMemBytes = 0
    config.numAttrs = 0
    config.attrs = []

    cluster_size = hw._checkCudaErrors(driver.cuOccupancyMaxPotentialClusterSize(func, config))
    return cluster_size


# ---------------------------------------------------------------------------
# Global-state layout constants (cluster variant)
# ---------------------------------------------------------------------------
# row_states: (num_groups, STATE_SIZE) int32 tensor
#   [0]  output_counter
STATE_SIZE = 1


# ---------------------------------------------------------------------------
# DSMEM primitives (inline PTX)
# ---------------------------------------------------------------------------
@dsl_user_op
def _mapa_shared_cluster(
    smem_ptr: CutePointer, peer_rank: CuteInt32, *, loc=None, ip=None
) -> CuteInt32:
    """Map local SMEM address to peer CTA's SMEM in cluster address space.

    PTX: mapa.shared::cluster.u32 $0, $1, $2;
    """
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def mapa_shared_cluster(smem_ptr, peer_rank):
    """Map local SMEM address to peer CTA's SMEM in cluster address space."""
    return _mapa_shared_cluster(smem_ptr, peer_rank)


@dsl_user_op
def _ld_shared_cluster_i32(mapped_addr: CuteInt32, *, loc=None, ip=None) -> CuteInt32:
    """Load int32 from cluster SMEM address.

    PTX: ld.shared::cluster.u32 $0, [$1];
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def ld_shared_cluster_i32(mapped_addr):
    """Load int32 from cluster SMEM address."""
    return _ld_shared_cluster_i32(mapped_addr)


# ---------------------------------------------------------------------------
# SinglePassMultiCTARadixTopKClusterKernel
# ---------------------------------------------------------------------------
class SinglePassMultiCTARadixTopKClusterKernel(SinglePassMultiCTARadixTopKKernel):
    """Cluster-accelerated single-pass multi-CTA radix top-k kernel.

    Inherits shared logic from ``SinglePassMultiCTARadixTopKKernel``:
      - ``to_ordered`` / ``from_ordered`` (bit-pattern helpers)
      - ``load_chunk_to_smem`` (vectorized chunk loading)
      - ``_radix_round_single_cta`` (single-CTA radix round)
      - ``prefix_sum_and_find_threshold`` (prefix sum + threshold)
      - ``compute_local_gt_count`` (count > pivot elements)
      - ``collect_output_single_cta`` (single-CTA output collection)

    Overrides / adds cluster-specific methods:
      - ``build_local_histogram`` (SMEM-only histogram, no global merge)
      - ``merge_histogram_dsmem`` (DSMEM-based histogram merging)
      - ``_radix_round_cluster`` (cluster barrier + DSMEM merge)
      - ``collect_output_cluster`` (cluster barrier between passes)
      - ``single_pass_multi_cta_topk_kernel`` (different multi-CTA path)
      - ``__call__`` (cluster launch parameter)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        chunk_size: int,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        ctas_per_group: int = 1,
        num_sms: int = 148,
    ):
        super().__init__(dtype, chunk_size, top_k, next_n, num_copy_bits, ctas_per_group, num_sms)
        # Clamp to hardware max cluster size (= max SMs per GPC).
        hw_max = _query_max_cluster_size()
        self.ctas_per_group = min(self.ctas_per_group, hw_max)

    # ------------------------------------------------------------------
    # Step 2a-cluster: Merge histograms via DSMEM
    # ------------------------------------------------------------------
    @cute.jit
    def merge_histogram_dsmem(self, local_histogram, prefix_buf, tidx):
        """Read all peers' local_histogram via DSMEM and sum into prefix_buf.

        Writes to prefix_buf (not local_histogram) to avoid a write-read race:
        peer CTAs may still be reading our local_histogram via DSMEM.
        """
        for i in range(tidx, self.radix, self.num_threads):
            total = cutlass.Int32(0)
            local_ptr = local_histogram.iterator + cutlass.Int32(i)
            for peer in cutlass.range_constexpr(self.ctas_per_group):
                remote_addr = mapa_shared_cluster(local_ptr, cutlass.Int32(peer))
                total = total + ld_shared_cluster_i32(remote_addr)
            prefix_buf[i] = total
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Step 2c-cluster: Cluster radix round (DSMEM merge + cluster barrier)
    # ------------------------------------------------------------------
    @cute.jit
    def _radix_round_cluster(
        self,
        round_idx: cutlass.Constexpr,
        shift: cutlass.Constexpr,
        shared_ordered,
        actual_chunk_size,
        prefix,
        remaining_k,
        local_histogram,
        prefix_buf,
        s_scalars,
        s_warp_sums,
        tidx,
    ):
        """Execute one radix select round with cluster-based inter-CTA sync."""
        prefix_mask = self._compute_prefix_mask(round_idx)

        # 1. Build local histogram in SMEM
        self.build_local_histogram(
            shared_ordered,
            actual_chunk_size,
            prefix,
            prefix_mask,
            shift,
            local_histogram,
            tidx,
        )

        # 2. Cluster barrier: publish all local histograms
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        # 3. Merge histograms via DSMEM (local_histogram -> prefix_buf)
        self.merge_histogram_dsmem(local_histogram, prefix_buf, tidx)

        # 4. Prefix sum and find threshold bucket on merged histogram
        self.prefix_sum_and_find_threshold(
            prefix_buf, prefix_buf, s_scalars, remaining_k, s_warp_sums, tidx
        )

        # Update prefix and remaining_k
        found_bucket = s_scalars[0]
        found_remaining_k = s_scalars[1]

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            prefix = prefix | cutlass.Uint32(cutlass.Uint32(found_bucket) << cutlass.Uint32(shift))
        else:
            prefix = prefix | cutlass.Uint16(cutlass.Uint16(found_bucket) << cutlass.Uint16(shift))
        remaining_k = found_remaining_k

        # Cluster barrier: ensure all CTAs finish DSMEM reads before any
        # CTA clears its local_histogram in the next round.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        return prefix, remaining_k

    # ------------------------------------------------------------------
    # Step 3: Collect output indices and values (cluster variant)
    # ------------------------------------------------------------------
    @cute.jit
    def collect_output_cluster(
        self,
        shared_ordered,
        actual_chunk_size,
        chunk_start,
        prologue_elems,
        aligned_size,
        left_size,
        ordered_pivot,
        top_k,
        local_gt_count,
        output_counter_ptr,
        local_histogram,
        output_indices_row,
        output_values_row,
        tidx,
    ):
        """Collect elements into output: first > pivot, then == pivot.

        With the descending ordered mapping, float > pivot <-> ordered < pivot.

        Uses FlashInfer-style batch atomicAdd for > pivot elements:
        one global atomicAdd per CTA (instead of per element) to get
        a contiguous allocation, then local atomicAdd within the CTA.

        A cluster barrier separates the two passes to ensure all > pivot
        elements from all CTAs are counted before == pivot elements start
        filling the remaining slots.
        """
        # Reuse local_histogram for counters (FlashInfer convention):
        #   [0] = local_offset_gt  (local position within CTA allocation)
        #   [1] = global_base_gt   (global base from batch atomicAdd)
        if tidx == 0:
            local_histogram[0] = cutlass.Int32(0)  # local_offset_gt
            local_histogram[1] = cutlass.Int32(0)  # global_base_gt
            if local_gt_count > 0:
                local_histogram[1] = atomicAdd(output_counter_ptr, local_gt_count)
        cute.arch.barrier()

        # Pass 1: float strictly greater than pivot (ordered < ordered_pivot)
        self._collect_pass_gt(
            shared_ordered,
            chunk_start,
            prologue_elems,
            aligned_size,
            left_size,
            ordered_pivot,
            local_histogram,
            output_indices_row,
            output_values_row,
            tidx,
        )

        # Cluster barrier between pass 1 and pass 2
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        # Pass 2: equal to pivot
        self._collect_pass_eq(
            shared_ordered,
            chunk_start,
            prologue_elems,
            aligned_size,
            left_size,
            ordered_pivot,
            top_k,
            output_counter_ptr,
            output_indices_row,
            output_values_row,
            tidx,
        )

    # ------------------------------------------------------------------
    # Main kernel (override: cluster-specific multi-CTA path)
    # ------------------------------------------------------------------
    @cute.kernel
    def single_pass_multi_cta_topk_kernel(
        self,
        input_data: cute.Tensor,
        row_states: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
    ):
        """Main single-pass multi-CTA radix top-k kernel (cluster variant).

        Grid: (total_ctas, 1, 1) where total_ctas = num_groups * ctas_per_group
        Each group processes one row at a time in persistent round-robin fashion.
        Cluster launch groups ctas_per_group CTAs together for DSMEM access.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()

        ctas_per_group = cutlass.const_expr(self.ctas_per_group)
        chunk_size = cutlass.const_expr(self.chunk_size)
        top_k = cutlass.const_expr(self.top_k)
        next_n = cutlass.const_expr(self.next_n)
        num_threads = cutlass.const_expr(self.num_threads)

        if cutlass.const_expr(ctas_per_group > 1):
            # Cluster API: CTA rank within the cluster
            cta_in_group = cute.arch.block_idx_in_cluster()
            group_id = bidx // ctas_per_group
        else:
            group_id = bidx
            cta_in_group = cutlass.Int32(0)
        num_groups = grid_size // ctas_per_group
        num_rows = input_data.shape[0]

        # ---- Shared memory allocation ----
        smem = SmemAllocator()

        # local histogram [256] int32
        local_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.radix,), order=(0,)),
            byte_alignment=128,
        )
        # prefix sum buffer [256] int32
        prefix_buf = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.radix,), order=(0,)),
            byte_alignment=128,
        )
        # scalars: [0]=found_bucket, [1]=found_remaining_k,
        #          [2]=prefix_lo (lower 32 bits), [3]=remaining_k
        s_scalars = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=16,
        )
        # warp sums for prefix sum (need num_warps entries, but radix scan uses
        # min(radix, 256)//32 = 8 warps)
        num_warps_scan = cutlass.const_expr(min(self.radix, 256) // 32)
        s_warp_sums = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_warps_scan,), order=(0,)),
            byte_alignment=128,
        )
        # shared ordered data [chunk_size] ordered_type
        shared_ordered = smem.allocate_tensor(
            element_type=self.ordered_type,
            layout=cute.make_ordered_layout((self.chunk_size,), order=(0,)),
            byte_alignment=128,
        )

        # ---- Global state pointer (output counter only) ----
        if cutlass.const_expr(ctas_per_group > 1):
            state_size = cutlass.const_expr(STATE_SIZE)
            output_counter_ptr = row_states.iterator + cutlass.Int32(group_id * state_size)
            # Defensive init: ensure output_counter is 0 before the first
            # row, even if row_states was allocated with torch.empty.
            # No barrier needed — the round-0 cluster barrier in the first
            # row will make this store visible to all CTAs.
            if cta_in_group == 0 and tidx == 0:
                st_release_gpu(output_counter_ptr, cutlass.Int32(0))

        # ---- Persistent loop: round-robin over rows ----
        row_idx = group_id

        while row_idx < num_rows:
            # Compute effective length from seqlen
            seq_len = seqlen[row_idx // next_n]
            length = seq_len - next_n + (row_idx % next_n) + 1

            # My chunk boundaries
            chunk_start = cta_in_group * chunk_size
            actual_chunk_size = cutlass.Int32(0)
            if chunk_start < length:
                remaining_len = length - chunk_start
                if remaining_len > chunk_size:
                    actual_chunk_size = cutlass.Int32(chunk_size)
                else:
                    actual_chunk_size = cutlass.Int32(remaining_len)

            # Early exit: k >= length -> return all indices directly
            input_row = input_data[row_idx, None]
            output_indices_row = output_indices[row_idx, None]
            if cutlass.const_expr(output_values is not None):
                output_values_row = output_values[row_idx, None]
            else:
                output_values_row = None

            if top_k >= length:
                for i in range(tidx, actual_chunk_size, num_threads):
                    if chunk_start + i < top_k:
                        output_indices_row[chunk_start + i] = cutlass.Int32(chunk_start + i)
                        if cutlass.const_expr(output_values is not None):
                            output_values_row[chunk_start + i] = input_row[chunk_start + i]
                # Fill remaining slots with -1 (only CTA 0 / single-CTA)
                if cta_in_group == 0:
                    for i in range(tidx + length, top_k, num_threads):
                        output_indices_row[i] = cutlass.Int32(-1)
                # No histogram buffer clearing needed (no global histogram buffers)
            else:
                # Step 1: Load chunk to smem as ordered
                prologue_elems, aligned_size, left_size = self.load_chunk_to_smem(
                    input_row, shared_ordered, chunk_start, actual_chunk_size, tidx
                )

                # Step 2: Multi-round radix select
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    prefix = cutlass.Uint32(0)
                else:
                    prefix = cutlass.Uint16(0)
                remaining_k = cutlass.Int32(top_k)

                if cutlass.const_expr(ctas_per_group == 1):
                    # ---- Single-CTA path: no global state, no barriers ----
                    self._single_cta_radix_select_and_collect(
                        shared_ordered,
                        actual_chunk_size,
                        chunk_start,
                        prologue_elems,
                        aligned_size,
                        left_size,
                        local_histogram,
                        prefix_buf,
                        s_scalars,
                        s_warp_sums,
                        top_k,
                        num_threads,
                        output_indices_row,
                        output_values_row,
                        tidx,
                    )

                else:
                    # ---- Multi-CTA (cluster) path ----
                    for r in cutlass.range_constexpr(self.num_rounds):
                        shift = cutlass.const_expr(self.ordered_bits - (r + 1) * self.radix_bits)
                        prefix, remaining_k = self._radix_round_cluster(
                            r,
                            shift,
                            shared_ordered,
                            actual_chunk_size,
                            prefix,
                            remaining_k,
                            local_histogram,
                            prefix_buf,
                            s_scalars,
                            s_warp_sums,
                            tidx,
                        )

                    # Count > pivot elements per CTA (for batch atomicAdd)
                    local_gt_count = self.compute_local_gt_count(
                        shared_ordered, actual_chunk_size, prefix, local_histogram, tidx
                    )

                    # Step 3: Collect output (cluster barriers)
                    self.collect_output_cluster(
                        shared_ordered,
                        actual_chunk_size,
                        chunk_start,
                        prologue_elems,
                        aligned_size,
                        left_size,
                        prefix,
                        top_k,
                        local_gt_count,
                        output_counter_ptr,
                        local_histogram,
                        output_indices_row,
                        output_values_row,
                        tidx,
                    )

                    # Final cluster barrier: ensure all CTAs finish before
                    # moving to the next row.
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()

                    # Reset output counter for next row.  All CTAs are
                    # synchronized by the barrier above.  The reset will
                    # be visible to every CTA by the time they reach
                    # collect_output in the next row, because at least
                    # 2 (fp16/bf16) or 4 (fp32) radix-round cluster
                    # barriers intervene.
                    if cta_in_group == 0 and tidx == 0:
                        st_release_gpu(output_counter_ptr, cutlass.Int32(0))

            # Advance to next row (round-robin).
            row_idx = row_idx + num_groups

        # End-of-kernel cleanup: CTA 0 resets output counter so the
        # next kernel call can use torch.empty instead of torch.zeros.
        if cutlass.const_expr(ctas_per_group > 1):
            if cta_in_group == 0:
                if tidx == 0:
                    st_release_gpu(output_counter_ptr, cutlass.Int32(0))

    # ------------------------------------------------------------------
    # Host-side launcher (override: adds cluster= launch parameter)
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        input_data: cute.Tensor,
        row_states: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        stream,
    ):
        num_rows = input_data.shape[0]
        ctas_per_group = cutlass.const_expr(self.ctas_per_group)
        num_groups = min(self.num_sms // ctas_per_group, num_rows)
        total_ctas = num_groups * ctas_per_group

        self.single_pass_multi_cta_topk_kernel(
            input_data,
            row_states,
            seqlen,
            output_indices,
            output_values,
        ).launch(
            grid=(total_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(ctas_per_group, 1, 1) if cutlass.const_expr(ctas_per_group > 1) else None,
            stream=stream,
        )
