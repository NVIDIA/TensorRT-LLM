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
"""Single-pass multi-CTA radix top-k kernel using FlashInfer-style fused multi-CTA approach.

All CTAs in a group cooperatively find the global pivot via multi-round radix
select with global histogram merging, then each CTA collects results from its
own chunk.  Single kernel launch, no intermediate buffer, no merge kernel.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int32 as CuteInt32
from cutlass.cute.typing import Pointer as CutePointer
from cutlass.cutlass_dsl import dsl_user_op
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


@dsl_user_op
def _red_relaxed_gpu(ptr: CutePointer, val: CuteInt32, loc=None, ip=None) -> None:
    """fence.acq_rel.gpu + red.relaxed.gpu.global.add.s32 (no read-back)."""
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "fence.acq_rel.gpu;\nred.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_release_gpu(ptr: CutePointer, val: CuteInt32, loc=None, ip=None) -> None:
    """fence.acq_rel.gpu + st.release.gpu.global.b32 (release store, no RMW)."""
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "fence.acq_rel.gpu;\nst.release.gpu.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def red_release_gpu(ptr, val):
    """fence + red.relaxed.gpu (no read-back, lighter than atomicAdd)."""
    _red_relaxed_gpu(ptr, val)


@cute.jit
def st_release_gpu(ptr, val):
    """fence + st.release.gpu (release store for counter resets)."""
    _st_release_gpu(ptr, val)


@dsl_user_op
def _ld_acquire_gpu(ptr: CutePointer, loc=None, ip=None) -> CuteInt32:
    """GPU-scope acquire load: ld.global.acquire.gpu.b32 (plain load, not atomic).

    Much cheaper than atomicAdd(ptr, 0) in the spin loop — multiple CTAs
    can read simultaneously without serialization.
    """
    from cutlass.cutlass_dsl import T

    return llvm.inline_asm(
        T.i32(),
        [ptr.toint().ir_value(loc=loc, ip=ip)],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def ld_acquire_gpu(ptr):
    """GPU-scope acquire load (plain load, not atomic)."""
    return _ld_acquire_gpu(ptr)


@cute.jit
def barrier_inter_cta(arrival_counter_ptr, target, tidx):
    """Inter-CTA barrier: thread 0 spins on arrival counter, then syncthreads."""
    if tidx == 0:
        while ld_acquire_gpu(arrival_counter_ptr) < target:
            pass
    cute.arch.barrier()


# ---------------------------------------------------------------------------
# SinglePassMultiCTARadixTopKKernel
# ---------------------------------------------------------------------------
class SinglePassMultiCTARadixTopKKernel:
    """FlashInfer-style single-pass multi-CTA radix top-k kernel.

    All CTAs in a *group* cooperatively process one row at a time.  The groups
    are persistent and iterate over rows in round-robin fashion.

    Algorithm:
      1. Each CTA loads its chunk into shared memory as *ordered* integers.
      2. Multi-round (2 or 4) radix select:
         a. Each CTA builds a local 256-bin histogram on its smem data.
         b. atomicAdd to a global histogram (triple-buffered in ``row_states``).
         c. Inter-CTA barrier.
         d. All CTAs read the merged histogram, compute prefix sum, find the
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
        """Convert float to an ordered unsigned integer (descending mapping).

        Descending sign-flip so that larger float → smaller ordered integer:
          negative (sign=1) → keep bits unchanged  (bits)
          positive (sign=0) → flip all bits and clear sign bit  (~bits & 0x7FFFFFFF)
        This allows using a natural prefix sum to locate the top-k boundary.
        """
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            bits = float_as_uint32(x)
            key = cutlass.Uint32(0)
            if bits & cutlass.Uint32(0x80000000):
                key = cutlass.Uint32(bits)
            else:
                key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
            return cutlass.Uint32(key)
        else:
            bits = half_as_ushort(x)
            key = cutlass.Uint16(0)
            if bits & cutlass.Uint16(0x8000):
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            return cutlass.Uint16(key)

    @cute.jit
    def from_ordered(self, ordered):
        """Inverse of ``to_ordered``: ordered integer → float.

        Inverse of the descending mapping:
          sign bit set (was negative) → bits unchanged  (ordered)
          sign bit clear (was positive) → flip all and clear sign  (~ordered & 0x7FFFFFFF)
        """
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            # Initialize before dynamic branch to satisfy DSL scoping.
            bits = cutlass.Uint32(0)
            if ordered & cutlass.Uint32(0x80000000):
                bits = ordered
            else:
                bits = (ordered ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
            return llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        else:
            bits = cutlass.Uint16(0)
            if ordered & cutlass.Uint16(0x8000):
                bits = ordered
            else:
                bits = (ordered ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            if cutlass.const_expr(self.dtype == cutlass.Float16):
                return llvm.bitcast(cutlass.Float16.mlir_type, bits.ir_value())
            else:
                return llvm.bitcast(cute.BFloat16.mlir_type, bits.ir_value())

    # ------------------------------------------------------------------
    # Step 1: Load chunk → smem (convert to ordered)
    # ------------------------------------------------------------------
    @cute.jit
    def load_chunk_to_smem(self, input_row, shared_ordered, chunk_start, actual_chunk_size, tidx):
        """Load valid chunk elements into smem as ordered integers.

        Follows the prologue/aligned/tail pattern from filtered_top_k_varlen_util:
          - Scalar prologue: handle misaligned prefix at chunk_start
          - Vectorized main: num_copy_bits-wide loads for the aligned region
          - Scalar tail: remaining elements after the last aligned vector
        Thread layout: thread t handles every (num_threads)-th vector in the
        aligned region (coalesced warp access).
        """
        vec_size = cutlass.const_expr(self.vec_size)
        num_threads = cutlass.const_expr(self.num_threads)
        # align_bytes and elem_bytes are compile-time Python ints
        align_bytes = self.num_copy_bits // 8
        elem_bytes = self.dtype.width // 8

        # --- Compute prologue / aligned / tail region sizes ---
        # Pointer to chunk[0]; .toint() gives byte address as Int64.
        row_ptr = input_row.iterator + chunk_start
        row_addr_u64 = row_ptr.toint()

        misalign = row_addr_u64 % align_bytes
        fix_bytes = cutlass.Int64(0)
        if misalign != 0:
            fix_bytes = align_bytes - misalign

        prologue_elems = cutlass.Int32(fix_bytes // elem_bytes)
        remaining = actual_chunk_size - prologue_elems
        aligned_size = (remaining // vec_size) * vec_size
        left_size = remaining - aligned_size
        # Byte address of first element in the aligned region
        aligned_addr = row_addr_u64 + fix_bytes

        # --- Part 2: Vectorized aligned region ---
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=cutlass.const_expr(self.num_copy_bits),
        )
        frag = cute.make_fragment((vec_size,), self.dtype)
        stride = cutlass.const_expr(num_threads * vec_size)

        # Thread t covers [t*vs, t*vs+vs), [t*vs+stride, ...) in the aligned region.
        # All start offsets i = t*vs + k*stride are multiples of vec_size, so
        # aligned_addr + i*elem_bytes is always align_bytes-aligned.
        # Assert this to the MLIR verifier via assumed_align on each make_ptr.
        i = tidx * vec_size
        while i < aligned_size:
            src_ptr = cute.make_ptr(
                self.dtype,
                aligned_addr + cutlass.Int64(i) * elem_bytes,
                assumed_align=align_bytes,
            )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_size,)))

            cute.copy(copy_atom, src, frag)
            # Apply to_ordered into a correctly-typed uint32 fragment so
            # autovec_copy sees matching types (uint32 → uint32) and emits
            # STS.128 instead of 8 scalar STS instructions.
            ordered_frag = cute.make_fragment((vec_size,), self.ordered_type)
            for j in cutlass.range(vec_size, unroll_full=True):
                ordered_frag[j] = self.to_ordered(frag[j])

            # Smem layout reordering: aligned region at smem[0..aligned_size-1],
            # prologue at smem[aligned_size..+prologue-1], tail after that.
            # smem_base is 128B-aligned and i is always a multiple of vec_size*elem_bytes
            # → smem[i] is always align_bytes-aligned → STS.128 is valid.
            shared_addr_u64 = (shared_ordered.iterator + i).toint()
            shared_ptr = cute.make_ptr(
                self.ordered_type,
                shared_addr_u64,
                cute.AddressSpace.smem,
                assumed_align=align_bytes,
            )
            shared_tensor = cute.make_tensor(shared_ptr, cute.make_layout((vec_size,)))
            cute.autovec_copy(ordered_frag, shared_tensor)

            i = i + stride

        # --- Part 1: Scalar prologue (before alignment boundary) ---
        for j in range(tidx, prologue_elems, num_threads):
            shared_ordered[aligned_size + j] = self.to_ordered(input_row[chunk_start + j])

        # --- Part 3: Scalar tail (after last aligned vector) ---
        for j in range(tidx, left_size, num_threads):
            k = prologue_elems + aligned_size + j
            shared_ordered[k] = self.to_ordered(input_row[chunk_start + k])

        cute.arch.barrier()
        return prologue_elems, aligned_size, left_size

    # ------------------------------------------------------------------
    # Step 2a: Build local histogram and merge to global
    # ------------------------------------------------------------------
    @cute.jit
    def build_and_merge_histogram(
        self,
        shared_ordered,
        actual_chunk_size,
        prefix,
        prefix_mask,
        shift,
        local_histogram,
        global_histogram_ptr,
        tidx,
    ):
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
                        (ordered >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                else:
                    bucket = cutlass.Int32(
                        (ordered >> cutlass.Uint16(shift)) & cutlass.Uint16(0xFF)
                    )
                atomicAdd(local_histogram.iterator + bucket, val_one)
        cute.arch.barrier()

        # Merge to global histogram
        for i in range(tidx, self.radix, self.num_threads):
            count = local_histogram[i]
            if count > 0:
                atomicAdd(global_histogram_ptr + cutlass.Int32(i), count)

    # ------------------------------------------------------------------
    # Step 2b: Prefix sum + find threshold bucket
    # ------------------------------------------------------------------
    @cute.jit
    def prefix_sum_and_find_threshold(
        self, local_histogram, prefix_buf, s_scalars, remaining_k, s_warp_sums, tidx
    ):
        """Compute prefix sum over 256-bin histogram and find threshold bucket.

        With the descending ordered mapping (larger float → smaller bucket),
        prefix_sum[b] = count of elements in buckets 0..b = count of the
        largest elements.  The threshold bucket is the first one where:
          prefix_sum[b] >= remaining_k  AND  prefix_sum[b-1] < remaining_k

        Results written to s_scalars:
          [0] = found_bucket
          [1] = found_remaining_k (remaining_k - prefix_sum[bucket-1])
        """
        # Step 1: Inclusive prefix sum directly on the histogram
        if tidx < self.radix:
            val = local_histogram[tidx]
            num_warps_scan = cutlass.const_expr(min(self.radix, 256) // 32)
            val, _ = block_prefix_sum_kernel(
                val, s_warp_sums, tidx, self.radix, num_warps_scan, barrier_id=1
            )
            prefix_buf[tidx] = val
        cute.arch.barrier()

        # Initialize fallback values: found_bucket=0,
        # found_remaining_k=remaining_k.  Without this, degenerate cases
        # where no thread satisfies the threshold condition would leave
        # s_scalars with stale values from a previous round.
        if tidx == 0:
            s_scalars[0] = cutlass.Int32(0)  # found_bucket
            s_scalars[1] = remaining_k  # found_remaining_k
        cute.arch.barrier()

        # Step 2: Find threshold bucket
        if tidx < self.radix:
            b = tidx  # bucket index
            current = prefix_buf[b]  # count of elements in buckets 0..b
            # For first bucket (b==0), previous=0.
            # Define previous unconditionally to avoid DSL scoping issues.
            previous = cutlass.Int32(0)
            if b > 0:
                previous = prefix_buf[b - 1]  # count of elements in buckets 0..b-1

            if current >= remaining_k and previous < remaining_k:
                s_scalars[0] = cutlass.Int32(b)  # found_bucket
                s_scalars[1] = remaining_k - previous  # found_remaining_k
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Step 2c-single: Single-CTA radix round (no global state, no barriers)
    # ------------------------------------------------------------------
    @cute.jit
    def _radix_round_single_cta(
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
        num_threads,
        tidx,
    ):
        """Execute one radix select round for single-CTA mode.

        No global memory histogram merging and no inter-CTA barriers needed:
        the CTA owns all data, so ``local_histogram`` (smem) is already the
        complete histogram after the local build pass.
        """
        # Compute prefix_mask for this round (top round_idx*8 bits)
        prefix_mask_bits = cutlass.const_expr(round_idx * self.radix_bits)
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            if cutlass.const_expr(prefix_mask_bits == 0):
                prefix_mask = cutlass.Uint32(0)
            else:
                prefix_mask = cutlass.Uint32(
                    ((1 << prefix_mask_bits) - 1) << (32 - prefix_mask_bits)
                )
        else:
            if cutlass.const_expr(prefix_mask_bits == 0):
                prefix_mask = cutlass.Uint16(0)
            else:
                prefix_mask = cutlass.Uint16(
                    ((1 << prefix_mask_bits) - 1) << (16 - prefix_mask_bits)
                )

        # Clear local histogram
        for i in range(tidx, self.radix, num_threads):
            local_histogram[i] = cutlass.Int32(0)
        cute.arch.barrier()

        # Build histogram directly in smem (no global merge)
        val_one = cutlass.Int32(1)
        for i in range(tidx, actual_chunk_size, num_threads):
            ordered = shared_ordered[i]
            if (ordered & prefix_mask) == prefix:
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    bucket = cutlass.Int32(
                        (ordered >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                else:
                    bucket = cutlass.Int32(
                        (ordered >> cutlass.Uint16(shift)) & cutlass.Uint16(0xFF)
                    )
                atomicAdd(local_histogram.iterator + bucket, val_one)
        cute.arch.barrier()

        # local_histogram is already the complete histogram — compute
        # prefix sum and find the threshold bucket directly.
        self.prefix_sum_and_find_threshold(
            local_histogram, prefix_buf, s_scalars, remaining_k, s_warp_sums, tidx
        )

        found_bucket = s_scalars[0]
        found_remaining_k = s_scalars[1]

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            prefix = prefix | cutlass.Uint32(cutlass.Uint32(found_bucket) << cutlass.Uint32(shift))
        else:
            prefix = prefix | cutlass.Uint16(cutlass.Uint16(found_bucket) << cutlass.Uint16(shift))
        remaining_k = found_remaining_k

        return prefix, remaining_k

    # ------------------------------------------------------------------
    # Step 2c: Multi-CTA radix round (called explicitly per round)
    # ------------------------------------------------------------------
    @cute.jit
    def _radix_round(
        self,
        round_idx: cutlass.Constexpr,
        num_rounds: cutlass.Constexpr,
        iter,
        shift: cutlass.Constexpr,
        shared_ordered,
        actual_chunk_size,
        prefix,
        remaining_k,
        local_histogram,
        prefix_buf,
        s_scalars,
        s_warp_sums,
        state_base_ptr,
        state_row,
        cta_in_group,
        barrier_phase,
        ctas_per_group,
        num_threads,
        tidx,
    ):
        """Execute one radix select round with inter-CTA synchronisation."""
        # FlashInfer-style triple-buffer rotation
        hist_buf_idx = (
            iter * cutlass.Int32(num_rounds) + cutlass.Int32(round_idx)
        ) % cutlass.Int32(3)
        next_hist_buf_idx = (hist_buf_idx + cutlass.Int32(1)) % cutlass.Int32(3)
        hist_offset = hist_buf_idx * cutlass.Int32(self.radix)
        next_hist_offset = next_hist_buf_idx * cutlass.Int32(self.radix)
        global_hist_ptr = state_base_ptr + hist_offset

        # Compute prefix_mask for this round (top round_idx*8 bits)
        prefix_mask_bits = cutlass.const_expr(round_idx * self.radix_bits)
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            if cutlass.const_expr(prefix_mask_bits == 0):
                prefix_mask = cutlass.Uint32(0)
            else:
                prefix_mask = cutlass.Uint32(
                    ((1 << prefix_mask_bits) - 1) << (32 - prefix_mask_bits)
                )
        else:
            if cutlass.const_expr(prefix_mask_bits == 0):
                prefix_mask = cutlass.Uint16(0)
            else:
                prefix_mask = cutlass.Uint16(
                    ((1 << prefix_mask_bits) - 1) << (16 - prefix_mask_bits)
                )

        # Build local histogram and merge to global
        self.build_and_merge_histogram(
            shared_ordered,
            actual_chunk_size,
            prefix,
            prefix_mask,
            shift,
            local_histogram,
            global_hist_ptr,
            tidx,
        )

        # CTA 0 clears the *next* round's histogram buffer
        if cta_in_group == 0:
            for i in range(tidx, self.radix, num_threads):
                state_row[next_hist_offset + cutlass.Int32(i)] = cutlass.Int32(0)

        # Inter-CTA barrier (only thread 0 signals arrival)
        if tidx == 0:
            red_release_gpu(state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER), cutlass.Int32(1))
        barrier_phase = barrier_phase + 1
        barrier_inter_cta(
            state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
            barrier_phase * ctas_per_group,
            tidx,
        )

        # Read merged histogram into local_histogram (smem)
        for i in range(tidx, self.radix, num_threads):
            local_histogram[i] = state_row[hist_offset + cutlass.Int32(i)]
        cute.arch.barrier()

        # Prefix sum and find threshold bucket
        self.prefix_sum_and_find_threshold(
            local_histogram, prefix_buf, s_scalars, remaining_k, s_warp_sums, tidx
        )

        # Update prefix and remaining_k
        found_bucket = s_scalars[0]
        found_remaining_k = s_scalars[1]

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            prefix = prefix | cutlass.Uint32(cutlass.Uint32(found_bucket) << cutlass.Uint32(shift))
        else:
            prefix = prefix | cutlass.Uint16(cutlass.Uint16(found_bucket) << cutlass.Uint16(shift))
        remaining_k = found_remaining_k

        return prefix, remaining_k, barrier_phase

    # ------------------------------------------------------------------
    # Step 2d: Count elements > pivot per CTA (for batch atomicAdd)
    # ------------------------------------------------------------------
    @cute.jit
    def compute_local_gt_count(
        self, shared_ordered, actual_chunk_size, ordered_pivot, local_histogram, tidx
    ):
        """Count elements whose float value is strictly greater than pivot.

        With the descending ordered mapping, float > pivot ↔ ordered < pivot.
        Uses local_histogram[0] as shared counter.

        Uses warp reduction (FlashInfer pattern): each thread counts its
        elements, then a butterfly warp-sum reduces 32 threads to one
        atomicAdd per warp, cutting smem contention by 32x.
        """
        if tidx == 0:
            local_histogram[0] = cutlass.Int32(0)
        cute.arch.barrier()

        my_count = cutlass.Int32(0)
        for i in range(tidx, actual_chunk_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered < ordered_pivot:
                my_count = my_count + 1

        # Warp-level reduction: butterfly sum across 32 threads.
        warp_sum = cute.arch.warp_reduction_sum(my_count)
        # Only lane 0 of each warp atomics the warp's total to smem.
        if cute.arch.lane_idx() == 0:
            if warp_sum > 0:
                atomicAdd(local_histogram.iterator, warp_sum)
        cute.arch.barrier()

        return local_histogram[0]

    # ------------------------------------------------------------------
    # Step 3: Collect output indices and values
    # ------------------------------------------------------------------
    @cute.jit
    def collect_output(
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
        arrival_counter_ptr,
        barrier_phase,
        ctas_per_group,
        local_histogram,
        output_indices_row,
        output_values_row,
        tidx,
    ):
        """Collect elements into output: first > pivot, then == pivot.

        With the descending ordered mapping, float > pivot ↔ ordered < pivot.

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
                local_histogram[1] = atomicAdd(output_counter_ptr, local_gt_count)
        cute.arch.barrier()

        # Pass 1: float strictly greater than pivot (ordered < ordered_pivot)
        # 3-region structure mirrors the reordered smem layout from
        # load_chunk_to_smem:
        #   aligned:  smem[i]                       → gmem[chunk_start + prologue_elems + i]
        #   prologue: smem[aligned_size + i]         → gmem[chunk_start + i]
        #   tail:     smem[aligned_size+prologue+i]  → gmem[chunk_start + prologue_elems + aligned_size + i]
        for i in range(tidx, aligned_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered < ordered_pivot:
                local_pos = atomicAdd(local_histogram.iterator, val_one)
                pos = local_histogram[1] + local_pos
                output_indices_row[pos] = cutlass.Int32(chunk_start + i + prologue_elems)
                if cutlass.const_expr(output_values_row is not None):
                    output_values_row[pos] = self.from_ordered(ordered)
        for i in range(tidx, prologue_elems, self.num_threads):
            ordered = shared_ordered[i + aligned_size]
            if ordered < ordered_pivot:
                local_pos = atomicAdd(local_histogram.iterator, val_one)
                pos = local_histogram[1] + local_pos
                output_indices_row[pos] = cutlass.Int32(chunk_start + i)
                if cutlass.const_expr(output_values_row is not None):
                    output_values_row[pos] = self.from_ordered(ordered)
        for i in range(tidx, left_size, self.num_threads):
            ordered = shared_ordered[i + aligned_size + prologue_elems]
            if ordered < ordered_pivot:
                local_pos = atomicAdd(local_histogram.iterator, val_one)
                pos = local_histogram[1] + local_pos
                output_indices_row[pos] = cutlass.Int32(
                    chunk_start + prologue_elems + aligned_size + i
                )
                if cutlass.const_expr(output_values_row is not None):
                    output_values_row[pos] = self.from_ordered(ordered)

        # Inter-CTA barrier between pass 1 and pass 2 (only thread 0 signals)
        if tidx == 0:
            red_release_gpu(arrival_counter_ptr, cutlass.Int32(1))
        barrier_phase = barrier_phase + 1
        barrier_inter_cta(arrival_counter_ptr, barrier_phase * ctas_per_group, tidx)

        # Pass 2: equal to pivot — same 3-region structure
        for i in range(tidx, aligned_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered == ordered_pivot:
                pos = atomicAdd(output_counter_ptr, val_one)
                if pos < top_k:
                    output_indices_row[pos] = cutlass.Int32(chunk_start + i + prologue_elems)
                    if cutlass.const_expr(output_values_row is not None):
                        output_values_row[pos] = self.from_ordered(ordered_pivot)
        for i in range(tidx, prologue_elems, self.num_threads):
            ordered = shared_ordered[i + aligned_size]
            if ordered == ordered_pivot:
                pos = atomicAdd(output_counter_ptr, val_one)
                if pos < top_k:
                    output_indices_row[pos] = cutlass.Int32(chunk_start + i)
                    if cutlass.const_expr(output_values_row is not None):
                        output_values_row[pos] = self.from_ordered(ordered_pivot)
        for i in range(tidx, left_size, self.num_threads):
            ordered = shared_ordered[i + aligned_size + prologue_elems]
            if ordered == ordered_pivot:
                pos = atomicAdd(output_counter_ptr, val_one)
                if pos < top_k:
                    output_indices_row[pos] = cutlass.Int32(
                        chunk_start + prologue_elems + aligned_size + i
                    )
                    if cutlass.const_expr(output_values_row is not None):
                        output_values_row[pos] = self.from_ordered(ordered_pivot)

        return barrier_phase

    # ------------------------------------------------------------------
    # Step 3-single: Single-CTA output collection (no inter-CTA barrier)
    # ------------------------------------------------------------------
    @cute.jit
    def collect_output_single_cta(
        self,
        shared_ordered,
        actual_chunk_size,
        chunk_start,
        prologue_elems,
        aligned_size,
        left_size,
        ordered_pivot,
        top_k,
        local_histogram,
        output_indices_row,
        output_values_row,
        tidx,
    ):
        """Single-CTA output collection using a smem counter.

        No inter-CTA barrier is needed.  ``local_histogram[2]`` is reused as
        a smem output counter (safe because the histogram phase is complete).
        With the descending ordered mapping, float > pivot ↔ ordered < pivot.
        """
        val_one = cutlass.Int32(1)

        # Reuse local_histogram[2] as smem output counter
        if tidx == 0:
            local_histogram[2] = cutlass.Int32(0)
        cute.arch.barrier()

        # Pass 1: float strictly greater than pivot (ordered < ordered_pivot)
        # 3-region structure
        for i in range(tidx, aligned_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered < ordered_pivot:
                pos = atomicAdd(local_histogram.iterator + cutlass.Int32(2), val_one)
                output_indices_row[pos] = cutlass.Int32(chunk_start + i + prologue_elems)
                if cutlass.const_expr(output_values_row is not None):
                    output_values_row[pos] = self.from_ordered(ordered)
        for i in range(tidx, prologue_elems, self.num_threads):
            ordered = shared_ordered[i + aligned_size]
            if ordered < ordered_pivot:
                pos = atomicAdd(local_histogram.iterator + cutlass.Int32(2), val_one)
                output_indices_row[pos] = cutlass.Int32(chunk_start + i)
                if cutlass.const_expr(output_values_row is not None):
                    output_values_row[pos] = self.from_ordered(ordered)
        for i in range(tidx, left_size, self.num_threads):
            ordered = shared_ordered[i + aligned_size + prologue_elems]
            if ordered < ordered_pivot:
                pos = atomicAdd(local_histogram.iterator + cutlass.Int32(2), val_one)
                output_indices_row[pos] = cutlass.Int32(
                    chunk_start + prologue_elems + aligned_size + i
                )
                if cutlass.const_expr(output_values_row is not None):
                    output_values_row[pos] = self.from_ordered(ordered)
        cute.arch.barrier()

        # Pass 2: equal to pivot (fill remaining slots up to k)
        for i in range(tidx, aligned_size, self.num_threads):
            ordered = shared_ordered[i]
            if ordered == ordered_pivot:
                pos = atomicAdd(local_histogram.iterator + cutlass.Int32(2), val_one)
                if pos < top_k:
                    output_indices_row[pos] = cutlass.Int32(chunk_start + i + prologue_elems)
                    if cutlass.const_expr(output_values_row is not None):
                        output_values_row[pos] = self.from_ordered(ordered_pivot)
        for i in range(tidx, prologue_elems, self.num_threads):
            ordered = shared_ordered[i + aligned_size]
            if ordered == ordered_pivot:
                pos = atomicAdd(local_histogram.iterator + cutlass.Int32(2), val_one)
                if pos < top_k:
                    output_indices_row[pos] = cutlass.Int32(chunk_start + i)
                    if cutlass.const_expr(output_values_row is not None):
                        output_values_row[pos] = self.from_ordered(ordered_pivot)
        for i in range(tidx, left_size, self.num_threads):
            ordered = shared_ordered[i + aligned_size + prologue_elems]
            if ordered == ordered_pivot:
                pos = atomicAdd(local_histogram.iterator + cutlass.Int32(2), val_one)
                if pos < top_k:
                    output_indices_row[pos] = cutlass.Int32(
                        chunk_start + prologue_elems + aligned_size + i
                    )
                    if cutlass.const_expr(output_values_row is not None):
                        output_values_row[pos] = self.from_ordered(ordered_pivot)

    # ------------------------------------------------------------------
    # Main kernel
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
        """Main single-pass multi-CTA radix top-k kernel.

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

        # ---- Persistent loop: round-robin over rows ----
        row_idx = group_id
        barrier_phase = cutlass.Int32(0)

        # Multi-CTA only: global state pointer (compiled away for single-CTA)
        if cutlass.const_expr(ctas_per_group > 1):
            state_size = cutlass.const_expr(STATE_SIZE)
            state_base_ptr = row_states.iterator + cutlass.Int32(group_id * state_size)
            state_row = row_states[group_id, None]

            # Per-call iteration counter (row_states zeroed by caller).
            iter_var = cutlass.Int32(0)

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
                # Multi-CTA: clear next iter's first histogram buffer.
                # No inter-CTA barrier needed here (FlashInfer pattern):
                # early-exit writes no global histogram, so there is nothing
                # to synchronize.  barrier_phase is intentionally NOT
                # incremented so the next normal row's barrier targets remain
                # consistent.
                if cutlass.const_expr(ctas_per_group > 1):
                    # CTA 0 clears the first buffer used by the next iteration
                    # (FlashInfer pattern: histogram[(iter+1)*num_rounds % 3]).
                    if cta_in_group == 0:
                        num_rounds_const = cutlass.const_expr(self.num_rounds)
                        next_first_offset = (
                            ((iter_var + cutlass.Int32(1)) * cutlass.Int32(num_rounds_const))
                            % cutlass.Int32(3)
                            * cutlass.Int32(self.radix)
                        )
                        for i in range(tidx, self.radix, num_threads):
                            state_row[next_first_offset + cutlass.Int32(i)] = cutlass.Int32(0)
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
                    # Round 0
                    prefix, remaining_k = self._radix_round_single_cta(
                        0,
                        self.ordered_bits - 1 * self.radix_bits,
                        shared_ordered,
                        actual_chunk_size,
                        prefix,
                        remaining_k,
                        local_histogram,
                        prefix_buf,
                        s_scalars,
                        s_warp_sums,
                        num_threads,
                        tidx,
                    )
                    # Round 1
                    prefix, remaining_k = self._radix_round_single_cta(
                        1,
                        self.ordered_bits - 2 * self.radix_bits,
                        shared_ordered,
                        actual_chunk_size,
                        prefix,
                        remaining_k,
                        local_histogram,
                        prefix_buf,
                        s_scalars,
                        s_warp_sums,
                        num_threads,
                        tidx,
                    )
                    if cutlass.const_expr(self.num_rounds > 2):
                        # Round 2 (fp32 only)
                        prefix, remaining_k = self._radix_round_single_cta(
                            2,
                            self.ordered_bits - 3 * self.radix_bits,
                            shared_ordered,
                            actual_chunk_size,
                            prefix,
                            remaining_k,
                            local_histogram,
                            prefix_buf,
                            s_scalars,
                            s_warp_sums,
                            num_threads,
                            tidx,
                        )
                        # Round 3 (fp32 only)
                        prefix, remaining_k = self._radix_round_single_cta(
                            3,
                            self.ordered_bits - 4 * self.radix_bits,
                            shared_ordered,
                            actual_chunk_size,
                            prefix,
                            remaining_k,
                            local_histogram,
                            prefix_buf,
                            s_scalars,
                            s_warp_sums,
                            num_threads,
                            tidx,
                        )

                    # Step 3: Collect output (smem counter, no inter-CTA sync)
                    self.collect_output_single_cta(
                        shared_ordered,
                        actual_chunk_size,
                        chunk_start,
                        prologue_elems,
                        aligned_size,
                        left_size,
                        prefix,
                        top_k,
                        local_histogram,
                        output_indices_row,
                        output_values_row,
                        tidx,
                    )

                else:
                    # ---- Multi-CTA path ----
                    # Initial inter-CTA barrier for this row.
                    if tidx == 0:
                        red_release_gpu(
                            state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER), cutlass.Int32(1)
                        )
                    barrier_phase = barrier_phase + 1
                    barrier_inter_cta(
                        state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER),
                        barrier_phase * ctas_per_group,
                        tidx,
                    )

                    # CTA 0 resets output counter AFTER barrier (release store)
                    if cta_in_group == 0:
                        if tidx == 0:
                            st_release_gpu(
                                state_base_ptr + cutlass.Int32(_OUTPUT_COUNTER),
                                cutlass.Int32(0),
                            )

                    num_rounds_const = cutlass.const_expr(self.num_rounds)
                    # Round 0
                    prefix, remaining_k, barrier_phase = self._radix_round(
                        0,
                        num_rounds_const,
                        iter_var,
                        self.ordered_bits - 1 * self.radix_bits,
                        shared_ordered,
                        actual_chunk_size,
                        prefix,
                        remaining_k,
                        local_histogram,
                        prefix_buf,
                        s_scalars,
                        s_warp_sums,
                        state_base_ptr,
                        state_row,
                        cta_in_group,
                        barrier_phase,
                        ctas_per_group,
                        num_threads,
                        tidx,
                    )
                    # Round 1
                    prefix, remaining_k, barrier_phase = self._radix_round(
                        1,
                        num_rounds_const,
                        iter_var,
                        self.ordered_bits - 2 * self.radix_bits,
                        shared_ordered,
                        actual_chunk_size,
                        prefix,
                        remaining_k,
                        local_histogram,
                        prefix_buf,
                        s_scalars,
                        s_warp_sums,
                        state_base_ptr,
                        state_row,
                        cta_in_group,
                        barrier_phase,
                        ctas_per_group,
                        num_threads,
                        tidx,
                    )

                    if cutlass.const_expr(self.num_rounds > 2):
                        # Round 2 (fp32 only)
                        prefix, remaining_k, barrier_phase = self._radix_round(
                            2,
                            num_rounds_const,
                            iter_var,
                            self.ordered_bits - 3 * self.radix_bits,
                            shared_ordered,
                            actual_chunk_size,
                            prefix,
                            remaining_k,
                            local_histogram,
                            prefix_buf,
                            s_scalars,
                            s_warp_sums,
                            state_base_ptr,
                            state_row,
                            cta_in_group,
                            barrier_phase,
                            ctas_per_group,
                            num_threads,
                            tidx,
                        )
                        # Round 3 (fp32 only)
                        prefix, remaining_k, barrier_phase = self._radix_round(
                            3,
                            num_rounds_const,
                            iter_var,
                            self.ordered_bits - 4 * self.radix_bits,
                            shared_ordered,
                            actual_chunk_size,
                            prefix,
                            remaining_k,
                            local_histogram,
                            prefix_buf,
                            s_scalars,
                            s_warp_sums,
                            state_base_ptr,
                            state_row,
                            cta_in_group,
                            barrier_phase,
                            ctas_per_group,
                            num_threads,
                            tidx,
                        )

                    # Count > pivot elements per CTA (for batch atomicAdd)
                    local_gt_count = self.compute_local_gt_count(
                        shared_ordered, actual_chunk_size, prefix, local_histogram, tidx
                    )

                    # Step 3: Collect output
                    output_counter_ptr = state_base_ptr + cutlass.Int32(_OUTPUT_COUNTER)
                    arrival_counter_ptr = state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER)

                    barrier_phase = self.collect_output(
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
                        arrival_counter_ptr,
                        barrier_phase,
                        ctas_per_group,
                        local_histogram,
                        output_indices_row,
                        output_values_row,
                        tidx,
                    )

                    # Final inter-CTA barrier: ensure all CTAs finish before
                    # moving to the next row.
                    if tidx == 0:
                        red_release_gpu(arrival_counter_ptr, cutlass.Int32(1))
                    barrier_phase = barrier_phase + 1
                    barrier_inter_cta(arrival_counter_ptr, barrier_phase * ctas_per_group, tidx)

            # Advance to next row (round-robin).
            row_idx = row_idx + num_groups
            if cutlass.const_expr(ctas_per_group > 1):
                iter_var = iter_var + cutlass.Int32(1)

        # End-of-kernel cleanup (FlashInfer-style): CTA 0 resets state so the
        # next kernel call can use torch.empty instead of torch.zeros.
        if cutlass.const_expr(ctas_per_group > 1):
            if cta_in_group == 0:
                hist_total = cutlass.const_expr(3 * self.radix)
                for i in range(tidx, hist_total, num_threads):
                    state_row[i] = cutlass.Int32(0)
                if tidx == 0:
                    st_release_gpu(
                        state_base_ptr + cutlass.Int32(_ARRIVAL_COUNTER), cutlass.Int32(0)
                    )

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

        self.single_pass_multi_cta_topk_kernel(
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
