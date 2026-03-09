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
"""Distributed radix top-k kernel using FlashInfer-style fused multi-CTA approach.

All CTAs in a group cooperatively find the global pivot via multi-round radix
select with global histogram merging, then each CTA collects results from its
own chunk.  Single kernel launch, no intermediate buffer, no merge kernel.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

from .block_scan import block_prefix_sum_kernel
from .filtered_top_k_varlen_util import float_as_uint32, half_as_ushort

# ---------------------------------------------------------------------------
# Global-state layout constants
# ---------------------------------------------------------------------------
# row_states: (num_groups, STATE_SIZE) int32 tensor
#   [0..255]     histogram buffer 0
#   [256..511]   histogram buffer 1
#   [512..767]   histogram buffer 2
#   [768]        arrival_counter
#   [769]        output_counter
_HIST_SIZE = 256
_HIST_BUF_0 = 0
_HIST_BUF_1 = _HIST_SIZE
_HIST_BUF_2 = 2 * _HIST_SIZE
_ARRIVAL_COUNTER = 3 * _HIST_SIZE  # 768
_OUTPUT_COUNTER = 3 * _HIST_SIZE + 1  # 769
STATE_SIZE = 3 * _HIST_SIZE + 2  # 770


# ---------------------------------------------------------------------------
# GPU-scope synchronisation primitives (inline PTX)
# ---------------------------------------------------------------------------
@cute.jit
def fence_acq_rel_gpu(*, loc=None, ip=None):
    """GPU-scope acquire-release fence."""
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="fence.acq_rel.gpu;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def red_release_gpu(ptr, val):
    """Atomic add with release semantics (GPU scope)."""
    fence_acq_rel_gpu()
    atomicAdd(ptr, val)


@cute.jit
def ld_acquire_gpu(ptr):
    """Acquire-semantic read via atomicAdd(ptr, 0)."""
    return atomicAdd(ptr, cutlass.Int32(0))


@cute.jit
def barrier_inter_cta(arrival_counter_ptr, target, tidx):
    """Inter-CTA barrier: thread 0 spins on arrival counter, then syncthreads."""
    if tidx == 0:
        while ld_acquire_gpu(arrival_counter_ptr) < target:
            pass
    cute.arch.barrier()


# ---------------------------------------------------------------------------
# DistributedRadixTopKKernel
# ---------------------------------------------------------------------------
class DistributedRadixTopKKernel:
    """FlashInfer-style distributed radix top-k kernel.

    All CTAs in a *group* cooperatively process one row at a time.  The groups
    are persistent and iterate over rows in round-robin fashion.

    Algorithm:
      1. Each CTA loads its chunk into shared memory as *ordered* integers.
      2. Multi-round (2 or 4) radix select:
         a. Each CTA builds a local 256-bin histogram on its smem data.
         b. atomicAdd to a global histogram (triple-buffered in ``row_states``).
         c. Inter-CTA barrier.
         d. All CTAs read the merged histogram, compute suffix sum, find the
            threshold bucket, update prefix and remaining_k.
      3. Collect output:
         a. Pass 1: elements strictly greater than pivot → atomicAdd to get
            position in output.
         b. Inter-CTA barrier.
         c. Pass 2: elements equal to pivot → per-element atomicAdd, write
            only while pos < k.
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
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.next_n = next_n
        self.ctas_per_group = ctas_per_group
        self.num_sms = num_sms
        self.num_copy_bits = num_copy_bits

        # Radix config
        self.radix = 256
        self.radix_bits = 8
        if cutlass.const_expr(dtype == cutlass.Float32):
            self.ordered_type = cutlass.Uint32
            self.ordered_bits = 32
            self.num_rounds = 4
        else:
            self.ordered_type = cutlass.Uint16
            self.ordered_bits = 16
            self.num_rounds = 2

        # Thread config — fixed at 1024 (FlashInfer convention)
        self.num_threads = 1024
        self.num_warps = self.num_threads // 32

        # Vec size for loading
        self.vec_size = num_copy_bits // dtype.width

    # ------------------------------------------------------------------
    # Bit-pattern helpers
    # ------------------------------------------------------------------
    @cute.jit
    def to_ordered(self, x):
        """Convert float to an ordered unsigned integer (descending order).

        Sign-flip mapping so that larger float → larger ordered integer:
          negative (sign=1) → flip ALL bits  (~bits)
          positive (sign=0) → flip sign bit only  (bits ^ 0x80000000)
        This matches FlashInfer's RadixTopKTraits::ToOrdered.
        """
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            bits = float_as_uint32(x)
            key = cutlass.Uint32(0)
            if bits & cutlass.Uint32(0x80000000):
                key = bits ^ cutlass.Uint32(0xFFFFFFFF)
            else:
                key = bits ^ cutlass.Uint32(0x80000000)
            return cutlass.Uint32(key)
        else:
            bits = half_as_ushort(x)
            key = cutlass.Uint16(0)
            if bits & cutlass.Uint16(0x8000):
                key = bits ^ cutlass.Uint16(0xFFFF)
            else:
                key = bits ^ cutlass.Uint16(0x8000)
            return cutlass.Uint16(key)

    @cute.jit
    def from_ordered(self, ordered):
        """Inverse of ``to_ordered``: ordered integer → float.

        Matches FlashInfer's RadixTopKTraits::FromOrdered:
          sign bit set (was positive) → flip sign bit  (ordered ^ 0x80000000)
          sign bit clear (was negative) → flip ALL bits  (~ordered)
        """
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            # Initialize before dynamic branch to satisfy DSL scoping.
            bits = cutlass.Uint32(0)
            if ordered & cutlass.Uint32(0x80000000):
                bits = ordered ^ cutlass.Uint32(0x80000000)
            else:
                bits = ordered ^ cutlass.Uint32(0xFFFFFFFF)
            return llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        else:
            bits = cutlass.Uint16(0)
            if ordered & cutlass.Uint16(0x8000):
                bits = ordered ^ cutlass.Uint16(0x8000)
            else:
                bits = ordered ^ cutlass.Uint16(0xFFFF)
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                return llvm.bitcast(cutlass.Float16.mlir_type, bits.ir_value())
            else:
                return llvm.bitcast(cute.BFloat16.mlir_type, bits.ir_value())

    # ------------------------------------------------------------------
    # Step 1: Load chunk → smem (convert to ordered)
    # ------------------------------------------------------------------
    @cute.jit
    def load_chunk_to_smem(self, input_row_ptr, shared_ordered, chunk_start,
                           actual_chunk_size, tidx):
        """Load one chunk of input into shared memory as ordered integers.

        OOB elements are filled with 0 (smallest ordered value, i.e. most
        negative float, which won't affect descending top-k).
        """
        for i in range(tidx, self.chunk_size, self.num_threads):
            if i < actual_chunk_size:
                val = input_row_ptr[chunk_start + i]
                shared_ordered[i] = self.to_ordered(val)
            else:
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    shared_ordered[i] = cutlass.Uint32(0)
                else:
                    shared_ordered[i] = cutlass.Uint16(0)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Step 2a: Build local histogram and merge to global
    # ------------------------------------------------------------------
    @cute.jit
    def build_and_merge_histogram(self, shared_ordered, actual_chunk_size,
                                  prefix, prefix_mask, shift,
                                  local_histogram,
                                  global_histogram_ptr, tidx):
        """Build per-CTA histogram on smem data and atomicAdd to global."""
        # Clear local histogram
        for i in range(tidx, self.radix, self.num_threads):
            local_histogram[i] = cutlass.Int32(0)
        cute.arch.barrier()

        # Build local histogram
        val_one = cutlass.Int32(1)
        for i in range(tidx, actual_chunk_size, self.num_threads):
            ordered = shared_ordered[i]
            if (ordered & prefix_mask) == prefix:
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    bucket = cutlass.Int32(
                        (ordered >> cutlass.Uint32(shift)) & cutlass.Uint32(
                            0xFF))
                else:
                    bucket = cutlass.Int32(
                        (ordered >> cutlass.Uint16(shift)) & cutlass.Uint16(
                            0xFF))
                atomicAdd(local_histogram.iterator + bucket, val_one)
        cute.arch.barrier()

        # Merge to global histogram
        for i in range(tidx, self.radix, self.num_threads):
            count = local_histogram[i]
            if count > 0:
                atomicAdd(global_histogram_ptr + cutlass.Int32(i), count)

    # ------------------------------------------------------------------
    # Step 2b: Suffix sum + find threshold bucket
    # ------------------------------------------------------------------
    @cute.jit
    def suffix_sum_and_find_threshold(self, local_histogram, suffix_buf,
                                      s_scalars, remaining_k, s_warp_sums,
                                      tidx):
        """Compute suffix sum over 256-bin histogram and find threshold bucket.

        The threshold bucket is the one where:
          suffix_sum[bucket] >= remaining_k  AND  suffix_sum[bucket+1] < remaining_k

        Results written to s_scalars:
          [0] = found_bucket
          [1] = found_remaining_k (remaining_k - suffix_sum[bucket+1])
        """
        # Read global histogram into suffix_buf (already done by caller or here)
        # We do a *reverse* prefix sum: suffix_sum[i] = sum(histogram[i..255])
        # Implemented as: reverse the array → prefix sum → reverse back

        # Step 1: Reverse copy into suffix_buf
        if tidx < self.radix:
            suffix_buf[tidx] = local_histogram[self.radix - 1 - tidx]
        cute.arch.barrier()

        # Step 2: Inclusive prefix sum on reversed data
        if tidx < self.radix:
            val = suffix_buf[tidx]
            num_warps_scan = cutlass.const_expr(min(self.radix, 256) // 32)
            val, _ = block_prefix_sum_kernel(val, s_warp_sums, tidx,
                                             self.radix, num_warps_scan,
                                             barrier_id=1)
            suffix_buf[tidx] = val
        cute.arch.barrier()

        # Now suffix_buf[i] = sum of histogram[(255-i)..255]
        # So suffix_sum for original bucket b = suffix_buf[255 - b]

        # Initialize fallback values (FlashInfer: found_bucket=0,
        # found_remaining_k=remaining_k).  Without this, degenerate cases
        # where no thread satisfies the threshold condition would leave
        # s_scalars with stale values from a previous round.
        if tidx == 0:
            s_scalars[0] = cutlass.Int32(0)  # found_bucket
            s_scalars[1] = remaining_k  # found_remaining_k
        cute.arch.barrier()

        # Step 3: Find threshold bucket
        if tidx < self.radix:
            b = tidx  # original bucket index
            count_ge = suffix_buf[self.radix - 1 - b]  # count >= bucket b
            # For last bucket (b==255), count_gt=0 (nothing strictly above).
            # Define count_gt unconditionally to avoid DSL scoping issues.
            count_gt = cutlass.Int32(0)
            if b < self.radix - 1:
                count_gt = suffix_buf[self.radix - 2 - b]  # count > bucket b

            if count_ge >= remaining_k and count_gt < remaining_k:
                s_scalars[0] = cutlass.Int32(b)  # found_bucket
                s_scalars[1] = remaining_k - count_gt  # found_remaining_k
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Step 2c: Single radix round (called explicitly per round)
    # ------------------------------------------------------------------
    @cute.jit
    def _radix_round(self, round_idx: cutlass.Constexpr, hist_offset: cutlass.Constexpr,
                     next_hist_offset: cutlass.Constexpr, shift: cutlass.Constexpr,
                     shared_ordered, actual_chunk_size, prefix, remaining_k,
                     local_histogram, suffix_buf, s_scalars, s_warp_sums,
                     state_base_ptr, state_row, cta_in_group, barrier_phase,
                     ctas_per_group, num_threads, tidx):
        """Execute one radix select round with inter-CTA synchronisation."""
        global_hist_ptr = state_base_ptr + cutlass.Int32(hist_offset)

        # Compute prefix_mask for this round (top round_idx*8 bits)
        prefix_mask_bits = cutlass.const_expr(round_idx * self.radix_bits)
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            if cutlass.const_expr(prefix_mask_bits == 0):
                prefix_mask = cutlass.Uint32(0)
            else:
                prefix_mask = cutlass.Uint32(
                    ((1 << prefix_mask_bits) - 1) << (32 - prefix_mask_bits))
        else:
            if cutlass.const_expr(prefix_mask_bits == 0):
                prefix_mask = cutlass.Uint16(0)
            else:
                prefix_mask = cutlass.Uint16(
                    ((1 << prefix_mask_bits) - 1) << (16 - prefix_mask_bits))

        # Build local histogram and merge to global
        self.build_and_merge_histogram(shared_ordered, actual_chunk_size,
                                       prefix, prefix_mask, shift,
                                       local_histogram,
                                       global_hist_ptr, tidx)

        # CTA 0 clears the *next* round's histogram buffer
        if cta_in_group == 0:
            for i in range(tidx, self.radix, num_threads):
                state_row[next_hist_offset + i] = cutlass.Int32(0)

        # Inter-CTA barrier (only thread 0 signals arrival)
        if tidx == 0:
            red_release_gpu(
                state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
                cutlass.Int32(1))
        barrier_phase = barrier_phase + 1
        barrier_inter_cta(
            state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
            barrier_phase * ctas_per_group, tidx)

        # Read merged histogram into local_histogram (smem)
        for i in range(tidx, self.radix, num_threads):
            local_histogram[i] = state_row[hist_offset + i]
        cute.arch.barrier()

        # Suffix sum and find threshold bucket
        self.suffix_sum_and_find_threshold(local_histogram, suffix_buf,
                                           s_scalars, remaining_k,
                                           s_warp_sums, tidx)

        # Update prefix and remaining_k
        found_bucket = s_scalars[0]
        found_remaining_k = s_scalars[1]

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            prefix = prefix | cutlass.Uint32(
                cutlass.Uint32(found_bucket) << cutlass.Uint32(shift))
        else:
            prefix = prefix | cutlass.Uint16(
                cutlass.Uint16(found_bucket) << cutlass.Uint16(shift))
        remaining_k = found_remaining_k

        # Barrier before next round (only thread 0 signals arrival)
        if tidx == 0:
            red_release_gpu(
                state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
                cutlass.Int32(1))
        barrier_phase = barrier_phase + 1
        barrier_inter_cta(
            state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
            barrier_phase * ctas_per_group, tidx)

        return prefix, remaining_k, barrier_phase

    # ------------------------------------------------------------------
    # Step 2d: Count elements > pivot per CTA (for batch atomicAdd)
    # ------------------------------------------------------------------
    @cute.jit
    def compute_local_gt_count(self, shared_ordered, actual_chunk_size,
                               ordered_pivot, local_histogram, tidx):
        """Count elements strictly greater than ordered_pivot in this
        CTA's shared_ordered.  Uses local_histogram[0] as shared counter.

        Matches FlashInfer's local_gt_count computation at the end of
        RadixSelectFromSharedMemory.
        """
        if tidx == 0:
            local_histogram[0] = cutlass.Int32(0)
        cute.arch.barrier()

        my_count = cutlass.Int32(0)
        for i in range(tidx, actual_chunk_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered > ordered_pivot:
                my_count = my_count + 1
        if my_count > 0:
            atomicAdd(local_histogram.iterator, my_count)
        cute.arch.barrier()

        return local_histogram[0]

    # ------------------------------------------------------------------
    # Step 3: Collect output indices and values
    # ------------------------------------------------------------------
    @cute.jit
    def collect_output(self, shared_ordered, actual_chunk_size, chunk_start,
                       ordered_pivot, top_k, local_gt_count,
                       output_counter_ptr,
                       arrival_counter_ptr, barrier_phase, ctas_per_group,
                       local_histogram,
                       output_indices_row, output_values_row, tidx):
        """Collect elements into output: first > pivot, then == pivot.

        Uses FlashInfer-style batch atomicAdd for > pivot elements:
        one global atomicAdd per CTA (instead of per element) to get
        a contiguous allocation, then local atomicAdd within the CTA.

        An inter-CTA barrier separates the two passes to ensure all > pivot
        elements from all CTAs are counted before == pivot elements start
        filling the remaining slots.
        """
        val_one = cutlass.Int32(1)

        # Reuse local_histogram for counters (FlashInfer convention):
        #   [0] = local_offset_gt  (local position within CTA allocation)
        #   [1] = global_base_gt   (global base from batch atomicAdd)
        if tidx == 0:
            local_histogram[0] = cutlass.Int32(0)  # local_offset_gt
            local_histogram[1] = cutlass.Int32(0)  # global_base_gt
            if local_gt_count > 0:
                local_histogram[1] = atomicAdd(output_counter_ptr,
                                               local_gt_count)
        cute.arch.barrier()

        # Pass 1: strictly greater than pivot
        # All > pivot elements are guaranteed to be in top-k, so no
        # pos < top_k check needed.  Use local atomicAdd for position.
        for i in range(tidx, actual_chunk_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered > ordered_pivot:
                local_pos = atomicAdd(local_histogram.iterator, val_one)
                pos = local_histogram[1] + local_pos
                output_indices_row[pos] = cutlass.Int32(
                    chunk_start + i)
                if cutlass.const_expr(output_values_row is not None):
                    output_values_row[pos] = self.from_ordered(ordered)

        # Inter-CTA barrier between pass 1 and pass 2 (only thread 0 signals)
        if tidx == 0:
            red_release_gpu(arrival_counter_ptr, cutlass.Int32(1))
        barrier_phase = barrier_phase + 1
        barrier_inter_cta(arrival_counter_ptr,
                          barrier_phase * ctas_per_group, tidx)

        # Pass 2: equal to pivot (fill remaining slots)
        # Use per-element global atomicAdd since cross-CTA coordination
        # is needed to respect the k limit.
        for i in range(tidx, actual_chunk_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered == ordered_pivot:
                pos = atomicAdd(output_counter_ptr, val_one)
                if pos < top_k:
                    output_indices_row[pos] = cutlass.Int32(
                        chunk_start + i)
                    if cutlass.const_expr(output_values_row is not None):
                        output_values_row[pos] = self.from_ordered(
                            ordered_pivot)

        return barrier_phase

    # ------------------------------------------------------------------
    # Main kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def distributed_topk_kernel(
        self,
        input_data: cute.Tensor,
        row_states: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
    ):
        """Main distributed radix top-k kernel.

        Grid: (total_ctas, 1, 1) where total_ctas = num_groups * ctas_per_group
        Each group processes one row at a time in persistent round-robin fashion.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()

        ctas_per_group = cutlass.const_expr(self.ctas_per_group)
        chunk_size = cutlass.const_expr(self.chunk_size)
        top_k = cutlass.const_expr(self.top_k)
        next_n = cutlass.const_expr(self.next_n)
        num_threads = cutlass.const_expr(self.num_threads)

        group_id = bidx // ctas_per_group
        cta_in_group = bidx % ctas_per_group
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
        # suffix sum buffer [256] int32
        suffix_buf = smem.allocate_tensor(
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

        # ---- Persistent loop: round-robin over rows ----
        row_idx = group_id
        barrier_phase = cutlass.Int32(0)

        state_size = cutlass.const_expr(STATE_SIZE)
        state_base_ptr = row_states.iterator + cutlass.Int32(
            group_id * state_size)
        state_row = row_states[group_id, None]

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

            # Early exit: k >= length → return all indices directly
            # (FlashInfer: "k >= vocab_size: return all indices")
            input_row = input_data[row_idx, None]
            output_indices_row = output_indices[row_idx, None]
            if cutlass.const_expr(output_values is not None):
                output_values_row = output_values[row_idx, None]
            else:
                output_values_row = None

            if top_k >= length:
                for i in range(tidx, actual_chunk_size, num_threads):
                    if chunk_start + i < top_k:
                        output_indices_row[chunk_start + i] = cutlass.Int32(
                            chunk_start + i)
                        if cutlass.const_expr(output_values is not None):
                            output_values_row[chunk_start + i] = input_row[
                                chunk_start + i]
                # Fill remaining slots with -1 (only CTA 0)
                if cta_in_group == 0:
                    for i in range(tidx + length, top_k, num_threads):
                        output_indices_row[i] = cutlass.Int32(-1)
                # Inter-CTA barrier to keep barrier_phase in sync.
                # Unlike FlashInfer (which uses `continue` + triple buffer),
                # we need an explicit barrier here.
                if tidx == 0:
                    red_release_gpu(
                        state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
                        cutlass.Int32(1))
                barrier_phase = barrier_phase + 1
                barrier_inter_cta(
                    state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
                    barrier_phase * ctas_per_group, tidx)
            else:
                # Step 1: Load chunk to smem as ordered
                self.load_chunk_to_smem(
                    input_row, shared_ordered, chunk_start,
                    actual_chunk_size, tidx)

                # All CTAs cooperatively clear histogram buffers
                # BEFORE the initial barrier.  arrival_counter is NEVER
                # zeroed (it accumulates across rows with barrier_phase).
                hist_total = cutlass.const_expr(3 * self.radix)
                for i in range(tidx, hist_total, num_threads):
                    state_row[i] = cutlass.Int32(0)
                cute.arch.barrier()

                # Initial inter-CTA barrier for this row.
                if tidx == 0:
                    red_release_gpu(
                        state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
                        cutlass.Int32(1))
                barrier_phase = barrier_phase + 1
                barrier_inter_cta(
                    state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
                    barrier_phase * ctas_per_group, tidx)

                # CTA 0 clears output counter AFTER barrier
                # (FlashInfer: st_release(&state->output_counter, 0))
                if cta_in_group == 0:
                    if tidx == 0:
                        fence_acq_rel_gpu()
                        state_row[_OUTPUT_COUNTER] = cutlass.Int32(0)

                # Step 2: Multi-round radix select
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    prefix = cutlass.Uint32(0)
                else:
                    prefix = cutlass.Uint16(0)
                remaining_k = cutlass.Int32(top_k)

                # Round 0 (bits [ordered_bits-8 .. ordered_bits-1])
                prefix, remaining_k, barrier_phase = self._radix_round(
                    0, _HIST_BUF_0, _HIST_BUF_1,
                    self.ordered_bits - 1 * self.radix_bits,
                    shared_ordered, actual_chunk_size, prefix, remaining_k,
                    local_histogram, suffix_buf, s_scalars, s_warp_sums,
                    state_base_ptr, state_row, cta_in_group, barrier_phase,
                    ctas_per_group, num_threads, tidx)

                # Round 1 (bits [ordered_bits-16 .. ordered_bits-9])
                prefix, remaining_k, barrier_phase = self._radix_round(
                    1, _HIST_BUF_1, _HIST_BUF_2,
                    self.ordered_bits - 2 * self.radix_bits,
                    shared_ordered, actual_chunk_size, prefix, remaining_k,
                    local_histogram, suffix_buf, s_scalars, s_warp_sums,
                    state_base_ptr, state_row, cta_in_group, barrier_phase,
                    ctas_per_group, num_threads, tidx)

                if cutlass.const_expr(self.num_rounds > 2):
                    # Round 2 (fp32 only)
                    prefix, remaining_k, barrier_phase = self._radix_round(
                        2, _HIST_BUF_2, _HIST_BUF_0,
                        self.ordered_bits - 3 * self.radix_bits,
                        shared_ordered, actual_chunk_size, prefix,
                        remaining_k, local_histogram, suffix_buf,
                        s_scalars, s_warp_sums, state_base_ptr,
                        state_row, cta_in_group, barrier_phase,
                        ctas_per_group, num_threads, tidx)

                    # Round 3 (fp32 only)
                    prefix, remaining_k, barrier_phase = self._radix_round(
                        3, _HIST_BUF_0, _HIST_BUF_1,
                        self.ordered_bits - 4 * self.radix_bits,
                        shared_ordered, actual_chunk_size, prefix,
                        remaining_k, local_histogram, suffix_buf,
                        s_scalars, s_warp_sums, state_base_ptr,
                        state_row, cta_in_group, barrier_phase,
                        ctas_per_group, num_threads, tidx)

                # prefix is now the ordered pivot
                ordered_pivot = prefix

                # Count > pivot elements per CTA (for batch atomicAdd)
                local_gt_count = self.compute_local_gt_count(
                    shared_ordered, actual_chunk_size, ordered_pivot,
                    local_histogram, tidx)

                # Step 3: Collect output
                output_counter_ptr = state_base_ptr + cutlass.Int32(
                    _OUTPUT_COUNTER)
                arrival_counter_ptr = state_base_ptr + cutlass.Int32(
                    _ARRIVAL_COUNTER)

                barrier_phase = self.collect_output(
                    shared_ordered, actual_chunk_size, chunk_start,
                    ordered_pivot, top_k, local_gt_count,
                    output_counter_ptr,
                    arrival_counter_ptr, barrier_phase, ctas_per_group,
                    local_histogram,
                    output_indices_row, output_values_row, tidx)

                # Final inter-CTA barrier: ensure all CTAs finish output
                # collection before any CTA starts clearing state for
                # the next row.
                if tidx == 0:
                    red_release_gpu(arrival_counter_ptr,
                                    cutlass.Int32(1))
                barrier_phase = barrier_phase + 1
                barrier_inter_cta(arrival_counter_ptr,
                                  barrier_phase * ctas_per_group, tidx)

            # Advance to next row (round-robin).
            row_idx = row_idx + num_groups

    # ------------------------------------------------------------------
    # Host-side launcher
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

        self.distributed_topk_kernel(
            input_data,
            row_states,
            seqlen,
            output_indices,
            output_values,
        ).launch(
            grid=(total_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )
