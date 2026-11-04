# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM indexer top-k: 512-thread / 2048-bin histogram radix select.

The prefill paths and the decode wrapper (self_attn_indexer_topk_decode.py) share it.

Prefill contract (TRT-LLM ``indexer_topk_prefill``): logits [rows, cols], visibility
windows row_starts / row_ends [rows], output_indices [rows, top_k] row-local and
``-1``-filled. Short windows (row_len <= top_k, the ISL 1024 / top_k 2048 target)
write local indices without reading logits; long windows compute a real top-k over
the visible range and convert absolute indices back to the row-local contract.

Histogram schedule (``_make_prefill_histogram_topk_class``, also the body of the
decode kernel): the exact top-k set under the ordering (ordered fp32 bits
descending, column ascending) -- fp16 coarse histogram, one-round threshold-bin
scan, rank sort of the threshold bin up to ``_REFINE_CUTOFF`` candidates, bounded
9-bit radix rounds over a shared-memory candidate list above it. The emitted slot
order is not deterministic; the set is.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator

_THREADS_PER_BLOCK = 512
_DEFAULT_TOP_K = 2048
_HISTOGRAM_NUM_BINS = 2048
# SMEM (value, column) pairs kept for the bounded final select; must be a
# multiple of the CTA size. 4096 keeps the whole working set under 50 KiB.
_CANDIDATE_CAPACITY = 4096
# Threshold bins with more candidates than this leave the rank sort for the
# bounded radix rounds. The sort costs ceil(n/512)*n dependent SMEM reads per
# thread, so its cost doubles just above 512 candidates; 512 is the measured
# break-even against the refine path.
_REFINE_CUTOFF = 512


def _make_prefill_histogram_topk_class():
    from .block_scan import block_prefix_sum_kernel
    from .filtered_top_k_varlen_util import float_as_uint32, half_as_ushort

    class IndexerTopKPrefillHistogram:
        """Single-CTA GLM top-k: histogram fast path + bounded SMEM radix select.

        Hybrid of the original rank-sort schedule and a bounded radix select:

        * fast path (unchanged schedule): step-0 fp16 coarse histogram over the row, one
          2048-bin threshold scan, direct emission of the lower bins, and the
          rank sort over the threshold-bin candidates when there are at most
          ``refine_cutoff`` of them (n <= 512: at most 512 dependent SMEM reads
          per thread). The rank sort breaks
          ties on the ordered key then the smaller column, so the selected set
          is the exact reference set on every row;
        * slow path (threshold bin > ``refine_cutoff`` candidates, the layer-1 /
          layer-2 regime): candidates go to a 4096-entry SMEM list and a
          **dynamic** loop of 9-bit radix rounds (512 bins = one block scan of
          the 512 threads) resolves the exact remaining set: value digits
          (bits 23..31, 14..22, 5..13, 0..4 of the ordered fp32 key) then column
          digits (bits 6..14, 0..5). One round is ~8 shared-memory loads + <= 8 warp-aggregated
          SMEM atomics + one block scan + <= 8 appends per thread;
        * threshold bin > 4096 (more than the list holds): the fp32 row passes
          of the original schedule (steps 1..3), with the threshold computed for ``top_k - found``
          so the set stays exact, then the list; > 4096 bit-identical scores at
          the cut fall to a column-order tail scan.

        The refine loop is not unrolled and there is a single copy of every
        pass, which keeps the code near the original size (a fully unrolled
        variant was much larger and slower).
        """

        def __init__(
            self,
            top_k: int,
            num_threads_per_cta: int = _THREADS_PER_BLOCK,
            candidate_capacity: int = _CANDIDATE_CAPACITY,
            refine_cutoff: int = _REFINE_CUTOFF,
        ):
            if top_k != _DEFAULT_TOP_K:
                raise ValueError(f"histogram prefill top-k expects top_k=2048, got {top_k}")
            if num_threads_per_cta != _THREADS_PER_BLOCK:
                raise ValueError(
                    "histogram prefill top-k expects 512 threads per CTA, "
                    f"got {num_threads_per_cta}"
                )
            if candidate_capacity < top_k or candidate_capacity % num_threads_per_cta != 0:
                raise ValueError(
                    "candidate_capacity must be a multiple of the CTA size and at "
                    f"least top_k; got {candidate_capacity}"
                )
            if not 0 <= refine_cutoff <= candidate_capacity:
                raise ValueError(f"refine_cutoff out of range: {refine_cutoff}")
            self.top_k = top_k
            self.num_threads_per_cta = num_threads_per_cta
            self.num_bins = _HISTOGRAM_NUM_BINS
            self.num_warps = num_threads_per_cta // 32
            self.candidate_capacity = candidate_capacity
            self.candidate_slots = candidate_capacity // num_threads_per_cta
            self.refine_cutoff = refine_cutoff
            # Refine rounds use one bin per thread: a single block scan.
            self.refine_bins = num_threads_per_cta
            # The row-step threshold scan gives every thread four consecutive
            # coarse bins (see ``find_threshold_bin``).
            if self.num_bins != 4 * num_threads_per_cta:
                raise ValueError(
                    "the single-round threshold scan needs 4 bins per thread; "
                    f"got {self.num_bins} bins for {num_threads_per_cta} threads"
                )
            # One packed 64-bit key per rank-sort candidate (never empty so the
            # allocation stays well formed when the sort is disabled).
            self.sort_keys_capacity = max(refine_cutoff, 1)

        def smem_bytes(self) -> int:
            """Dynamic shared memory the per-row select allocates (for docs/tests)."""

            scalars = 8 * 128
            return (
                self.num_bins * 4
                + self.top_k * 4
                + 2 * self.candidate_capacity * 4
                + self.num_warps * 4
                + scalars
                + self.sort_keys_capacity * 8
            )

        # ------------------------------------------------------------------
        # Key arithmetic
        # ------------------------------------------------------------------
        @cute.jit
        def to_ordered(self, x):
            """fp32 -> uint32 key; ascending key == descending value (bit order)."""

            bits = float_as_uint32(x)
            key = cutlass.Uint32(0)
            if bits & 0x80000000:
                key = cutlass.Uint32(bits)
            else:
                key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
            return key

        @cute.jit
        def extract_bin(self, x, STEP: cutlass.Constexpr[int]):
            """Row-level digit for global ``STEP`` (0: fp16 coarse, 1..3: fp32)."""

            if cutlass.const_expr(STEP == 0):
                h = x.to(cutlass.Float16)
                bits = half_as_ushort(h)
                key = cutlass.Uint16(0)
                if bits & 0x8000:
                    key = cutlass.Uint16(bits)
                else:
                    key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
                # Original binning: positives in bins 0..1023, negatives 1024..2047.
                # (A ``>> 4`` clamp would merge every negative score into one
                # bin and send rows the fast path handles through the
                # refine path.)
                return cutlass.Int32((key >> 5) & cutlass.Uint16(0x7FF))

            ordered = self.to_ordered(x)
            if cutlass.const_expr(STEP == 1):
                return cutlass.Int32((ordered >> 21) & cutlass.Uint32(0x7FF))
            if cutlass.const_expr(STEP == 2):
                return cutlass.Int32((ordered >> 10) & cutlass.Uint32(0x7FF))
            return cutlass.Int32(ordered & cutlass.Uint32(0x3FF))

        @cute.jit
        def refine_digit_params(self, digit):
            """(use_column, shift, mask) of 9-bit refine digit ``digit`` (uniform).

            0..3: ordered value bits 23..31, 14..22, 5..13, 0..4;
            4..5: column bits 6..14, 0..5 (columns < 32768).
            """

            shift = cutlass.Int32(0)
            mask = cutlass.Uint32(0x1F)
            if digit == cutlass.Int32(0):
                shift = cutlass.Int32(23)
                mask = cutlass.Uint32(0x1FF)
            elif digit == cutlass.Int32(1):
                shift = cutlass.Int32(14)
                mask = cutlass.Uint32(0x1FF)
            elif digit == cutlass.Int32(2):
                shift = cutlass.Int32(5)
                mask = cutlass.Uint32(0x1FF)
            elif digit == cutlass.Int32(4):
                shift = cutlass.Int32(6)
                mask = cutlass.Uint32(0x1FF)
            elif digit == cutlass.Int32(5):
                shift = cutlass.Int32(0)
                mask = cutlass.Uint32(0x3F)
            use_column = digit >= cutlass.Int32(4)
            return use_column, shift, mask

        @cute.jit
        def refine_digit(self, value, column, use_column, shift, mask):
            key = self.to_ordered(value)
            if use_column:
                key = cutlass.Uint32(column)
            return cutlass.Int32((key >> shift) & mask)

        @cute.jit
        def key_range_of_list(self, s_scratch, s_cand_vals, count, tidx):
            """(min, max) ordered key over the SMEM candidate list (one barrier).

            Per-thread over its slots, butterfly-reduced per warp, warp results
            in ``s_scratch[512:528]`` / ``s_scratch[528:544]`` (histogram entries
            the 512-bin refine does not use), then every thread folds the 16
            warp results itself so no second barrier is needed.
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            num_warps = cutlass.const_expr(self.num_warps)
            slots = cutlass.const_expr(self.candidate_slots)

            kmin = cutlass.Uint32(0xFFFFFFFF)
            kmax = cutlass.Uint32(0)
            for s in cutlass.range_constexpr(slots):
                i = tidx + cutlass.Int32(s * num_threads)
                key = self.to_ordered(s_cand_vals[i])
                if i < count:
                    if key < kmin:
                        kmin = key
                    if key > kmax:
                        kmax = key
            for offset in cutlass.range_constexpr(5):
                other_min = cute.arch.shuffle_sync_bfly(kmin, 1 << offset)
                other_max = cute.arch.shuffle_sync_bfly(kmax, 1 << offset)
                if other_min < kmin:
                    kmin = other_min
                if other_max > kmax:
                    kmax = other_max
            lane = cute.arch.lane_idx()
            warp = tidx // cutlass.Int32(32)
            base = cutlass.Int32(self.refine_bins)
            if lane == 0:
                s_scratch[base + warp] = cutlass.Int32(kmin)
                s_scratch[base + cutlass.Int32(num_warps) + warp] = cutlass.Int32(kmax)
            cute.arch.barrier()
            for w in cutlass.range_constexpr(num_warps):
                other_min = cutlass.Uint32(s_scratch[base + w])
                other_max = cutlass.Uint32(s_scratch[base + num_warps + w])
                if other_min < kmin:
                    kmin = other_min
                if other_max > kmax:
                    kmax = other_max
            return kmin, kmax

        @cute.jit
        def first_varying_digit(self, kmin, kmax):
            """Lowest 9-bit refine digit that can differ over keys in [kmin, kmax].

            Digit 0 covers key bits 23..31, 1 -> 14..22, 2 -> 5..13, 3 -> 0..4;
            equal keys (tie group) go straight to the column digits (4).
            """

            diff = kmin ^ kmax
            digit = cutlass.Int32(4)
            if diff >= cutlass.Uint32(1 << 23):
                digit = cutlass.Int32(0)
            elif diff >= cutlass.Uint32(1 << 14):
                digit = cutlass.Int32(1)
            elif diff >= cutlass.Uint32(1 << 5):
                digit = cutlass.Int32(2)
            elif diff != cutlass.Uint32(0):
                digit = cutlass.Int32(3)
            return digit

        @cute.jit
        def is_partial_match(
            self,
            x,
            pattern: cutlass.Uint32,
            SHIFT: cutlass.Constexpr[int],
        ):
            if cutlass.const_expr(SHIFT == 0):
                return True
            ordered = self.to_ordered(x)
            return ((ordered ^ pattern) >> SHIFT) == cutlass.Uint32(0)

        # ------------------------------------------------------------------
        # Warp-aggregated shared-memory counters (refine path only)
        # ------------------------------------------------------------------
        @cute.jit
        def histogram_add(self, s_histogram, bin_idx, contrib):
            """``s_histogram[bin_idx] += 1`` for lanes with ``contrib`` (warp-uniform call).

            Lanes that agree on a bin issue one atomic. DSL pitfall: a
            dynamic ``if`` is lowered through a synthesized function
            whose first parameter is named ``pred``; a closure variable called
            ``pred`` read inside an ``if`` body resolves to the branch condition.
            Nothing here is named ``pred`` and all votes precede the branches.
            """

            lead_bin = cute.arch.shuffle_sync(bin_idx, 0)
            effective_bin = lead_bin
            if contrib:
                effective_bin = bin_idx
            same = cute.arch.vote_all_sync(effective_bin == lead_bin)
            diverged = cute.arch.vote_any_sync(effective_bin != lead_bin)
            active = cute.arch.popc(cute.arch.vote_ballot_sync(contrib))
            lane = cute.arch.lane_idx()
            aggregate_leader = same & (lane == cutlass.Int32(0)) & (active > cutlass.Int32(0))
            per_lane = contrib & diverged
            if aggregate_leader:
                cute.arch.atomic_add(
                    ptr=(s_histogram.iterator + lead_bin).llvm_ptr,
                    val=active,
                    sem="relaxed",
                    scope="sys",
                )
            if per_lane:
                cute.arch.atomic_add(
                    ptr=(s_histogram.iterator + bin_idx).llvm_ptr,
                    val=cutlass.Int32(1),
                    sem="relaxed",
                    scope="sys",
                )

        @cute.jit
        def warp_append(self, s_counter, slot, contrib):
            """Reserve one slot in the list counted by ``s_counter[slot]`` for lanes with ``contrib``."""

            ballot = cute.arch.vote_ballot_sync(contrib)
            lane = cute.arch.lane_idx()
            total = cute.arch.popc(ballot)
            base = cutlass.Int32(0)
            leader = (lane == cutlass.Int32(0)) & (total > cutlass.Int32(0))
            if leader:
                base = cute.arch.atomic_add(
                    ptr=(s_counter.iterator + slot).llvm_ptr,
                    val=total,
                    sem="relaxed",
                    scope="sys",
                )
            base = cute.arch.shuffle_sync(base, 0)
            lane_mask_lt = (cutlass.Int32(1) << lane) - cutlass.Int32(1)
            return base + cute.arch.popc(ballot & lane_mask_lt)

        # ------------------------------------------------------------------
        # Threshold bin of the 2048-bin row histogram in one scan round
        # ------------------------------------------------------------------
        @cute.jit
        def find_threshold_bin(
            self,
            s_histogram,
            s_warp_sums,
            s_threshold_bin,
            s_final_bin_size,
            s_found_threshold,
            need,
            tidx,
        ):
            """Locate the bin with ``prefix < need <= prefix + count`` (2 CTA barriers).

            One scan suffices because the histogram total is never below
            ``need``: on step 0 the row is longer than ``top_k`` (the caller
            takes the identity path otherwise), step 1 recounts the whole row
            for ``need = top_k``, and steps 2/3 recount the previous threshold
            bin, which held at least ``need`` elements by construction. The
            prefix sums are therefore monotone from 0 to at least ``need`` and
            exactly one bin satisfies the test, so nothing has to be searched
            round by round: thread ``t`` owns bins ``4t..4t+3``, a shuffle scan
            over the 16 warps' per-thread sums plus the 16 warp totals gives
            every thread its exclusive prefix, and the one thread whose four
            bins straddle ``need`` publishes the answer. The original schedule
            ran up to four early-exit rounds of a 512-thread block scan (3
            barriers each); this is the same arithmetic in 2 barriers and less
            code. The per-bin
            prefixes are not written back: no later pass reads them (the refine
            rounds reset ``s_histogram`` before reusing it).
            """

            num_warps = cutlass.const_expr(self.num_warps)
            lane = cute.arch.lane_idx()
            warp = tidx // cutlass.Int32(32)
            base = tidx * cutlass.Int32(4)
            c0 = s_histogram[base]
            c1 = s_histogram[base + cutlass.Int32(1)]
            c2 = s_histogram[base + cutlass.Int32(2)]
            c3 = s_histogram[base + cutlass.Int32(3)]
            local = c0 + c1 + c2 + c3
            # Inclusive warp scan of the per-thread sums.
            inclusive = local
            for i in cutlass.range_constexpr(5):
                offset = 1 << i
                up = cute.arch.shuffle_sync_up(inclusive, offset, mask=0xFFFFFFFF)
                if lane >= cutlass.Int32(offset):
                    inclusive = inclusive + up
            if lane == cutlass.Int32(31):
                s_warp_sums[warp] = inclusive
            cute.arch.barrier()
            warp_offset = cutlass.Int32(0)
            for w in cutlass.range_constexpr(num_warps):
                if cutlass.Int32(w) < warp:
                    warp_offset = warp_offset + s_warp_sums[w]
            # Exclusive prefixes of the four owned bins.
            p0 = warp_offset + inclusive - local
            p1 = p0 + c0
            p2 = p1 + c1
            p3 = p2 + c2
            p4 = p3 + c3
            if p0 < need and p1 >= need:
                s_threshold_bin[0] = base
                s_final_bin_size[0] = c0
                s_found_threshold[0] = cutlass.Int32(1)
            if p1 < need and p2 >= need:
                s_threshold_bin[0] = base + cutlass.Int32(1)
                s_final_bin_size[0] = c1
                s_found_threshold[0] = cutlass.Int32(1)
            if p2 < need and p3 >= need:
                s_threshold_bin[0] = base + cutlass.Int32(2)
                s_final_bin_size[0] = c2
                s_found_threshold[0] = cutlass.Int32(1)
            if p3 < need and p4 >= need:
                s_threshold_bin[0] = base + cutlass.Int32(3)
                s_final_bin_size[0] = c3
                s_found_threshold[0] = cutlass.Int32(1)
            cute.arch.barrier()

        # ------------------------------------------------------------------
        # Global-row step (original structure, plain atomics), exact `need` accounting
        # ------------------------------------------------------------------
        @cute.jit
        def process_step(
            self,
            score,
            row_start,
            length,
            s_histogram,
            s_output,
            s_cand_vals,
            s_cand_idx,
            s_threshold_bin,
            s_cand_count,
            s_final_bin_size,
            s_found_topk,
            s_found_threshold,
            s_pattern,
            s_warp_sums,
            tidx,
            STEP: cutlass.Constexpr[int],
        ):
            """One radix step over the visible row (same passes as the original schedule).

            Returns True (uniformly) when the threshold bin holds more than
            ``candidate_capacity`` elements, i.e. the caller must run the next
            global step. Otherwise the threshold bin has been collected into the
            SMEM candidate list. Steps 2 and 3 subtract the columns already
            emitted from ``top_k`` when locating the threshold (the original schedule did not).
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            num_bins = cutlass.const_expr(self.num_bins)
            top_k = cutlass.const_expr(self.top_k)
            cand_cap = cutlass.const_expr(self.candidate_capacity)

            if cutlass.const_expr(STEP == 2):
                if tidx == 0:
                    s_pattern[0] = cutlass.Uint32(s_threshold_bin[0] & 0x7FF) << 21
            elif cutlass.const_expr(STEP == 3):
                if tidx == 0:
                    s_pattern[0] = s_pattern[0] | (cutlass.Uint32(s_threshold_bin[0] & 0x7FF) << 10)
            if tidx == 0:
                s_cand_count[0] = cutlass.Int32(0)
            cute.arch.barrier()
            pattern = s_pattern[0]
            need = cutlass.Int32(top_k) - s_found_topk[0]
            shift = cutlass.const_expr(0 if STEP < 2 else 21 if STEP == 2 else 10)

            hist_idx = tidx
            while hist_idx < cutlass.Int32(num_bins):
                s_histogram[hist_idx] = cutlass.Int32(0)
                hist_idx = hist_idx + cutlass.Int32(num_threads)
            cute.arch.barrier()

            vec_width = cutlass.const_expr(8)
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float32,
                num_bits_per_copy=256,
            )
            tiler_mn = (1, num_threads * vec_width)
            thr_layout = cute.make_ordered_layout((1, num_threads), order=(1, 0))
            val_layout = cute.make_layout((1, vec_width))
            tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

            row_addr = (score.iterator + row_start).toint()
            misaligned_bytes = cutlass.Int32(row_addr % cutlass.Int64(32))
            aligned_prefix = cutlass.Int32(0)
            if misaligned_bytes != cutlass.Int32(0):
                aligned_prefix = (cutlass.Int32(32) - misaligned_bytes) // cutlass.Int32(4)
                if aligned_prefix > length:
                    aligned_prefix = length
            aligned_size = ((length - aligned_prefix) // cutlass.Int32(vec_width)) * cutlass.Int32(
                vec_width
            )
            idX = cute.make_identity_tensor((1, aligned_size))
            row_ptr = cute.make_ptr(
                cutlass.Float32,
                (score.iterator + row_start + aligned_prefix).toint(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            )
            row_tensor = cute.make_tensor(
                row_ptr,
                cute.make_layout((1, aligned_size), stride=(aligned_size, 1)),
            )
            gX, cX = [cute.local_tile(mT, tiler_mn, (0, None)) for mT in (row_tensor, idX)]
            thr_copy = tiled_copy.get_slice(tidx)
            tXgX = thr_copy.partition_S(gX)
            tXcX = thr_copy.partition_S(cX)[(0, None), None, None, None]
            tXcX_tile = thr_copy.partition_S(cX)
            tXrX = cute.make_fragment_like(tXgX[None, None, None, 0])

            # --- histogram -------------------------------------------------
            idx = tidx
            while idx < aligned_prefix:
                raw = score[row_start + idx]
                if self.is_partial_match(raw, pattern, SHIFT=shift):
                    bin_idx = self.extract_bin(raw, STEP=STEP)
                    cute.arch.atomic_add(
                        ptr=(s_histogram.iterator + bin_idx).llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="sys",
                    )
                idx = idx + cutlass.Int32(num_threads)
            for tile_idx in range(gX.shape[2]):
                tXpX_tile = self.predicate_tile(
                    tXcX_tile[None, None, None, tile_idx],
                    aligned_size,
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
                for v in cutlass.range(cute.size(tXrX), unroll_full=True):
                    raw = tXrX[v]
                    # Out-of-range vector lanes hold the -inf fill; the original
                    # schedule counted them, which only matters if the cut reaches -inf.
                    matched = tXpX_tile[0, v // vec_width, 0]
                    if cutlass.const_expr(shift != 0):
                        matched = matched & self.is_partial_match(raw, pattern, SHIFT=shift)
                    if matched:
                        bin_idx = self.extract_bin(raw, STEP=STEP)
                        cute.arch.atomic_add(
                            ptr=(s_histogram.iterator + bin_idx).llvm_ptr,
                            val=cutlass.Int32(1),
                            sem="relaxed",
                            scope="sys",
                        )
            idx = aligned_prefix + aligned_size + tidx
            while idx < length:
                raw = score[row_start + idx]
                if self.is_partial_match(raw, pattern, SHIFT=shift):
                    bin_idx = self.extract_bin(raw, STEP=STEP)
                    cute.arch.atomic_add(
                        ptr=(s_histogram.iterator + bin_idx).llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="sys",
                    )
                idx = idx + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # --- threshold scan (2048 bins, one round) -----------------------
            # The histogram barrier above orders the counts; the two barriers
            # inside publish the warp totals and the threshold. No reset of the
            # threshold scalars is needed: exactly one thread writes them
            # before the second barrier (see ``find_threshold_bin``).
            self.find_threshold_bin(
                s_histogram,
                s_warp_sums,
                s_threshold_bin,
                s_final_bin_size,
                s_found_threshold,
                need,
                tidx,
            )

            threshold_bin = s_threshold_bin[0]
            final_bin_size = s_final_bin_size[0]
            collect = final_bin_size <= cutlass.Int32(cand_cap)

            # --- emit / collect (original structure) ------------------------------
            idx = tidx
            while idx < aligned_prefix:
                raw = score[row_start + idx]
                if self.is_partial_match(raw, pattern, SHIFT=shift):
                    bin_idx = self.extract_bin(raw, STEP=STEP)
                    if bin_idx < threshold_bin:
                        pos = cute.arch.atomic_add(
                            ptr=(s_found_topk.iterator).llvm_ptr,
                            val=cutlass.Int32(1),
                            sem="relaxed",
                            scope="sys",
                        )
                        s_output[pos] = idx
                    elif bin_idx == threshold_bin:
                        if collect:
                            pos = cute.arch.atomic_add(
                                ptr=(s_cand_count.iterator).llvm_ptr,
                                val=cutlass.Int32(1),
                                sem="relaxed",
                                scope="sys",
                            )
                            s_cand_vals[pos] = raw
                            s_cand_idx[pos] = idx
                idx = idx + cutlass.Int32(num_threads)

            for tile_idx in range(gX.shape[2]):
                tXpX_tile = self.predicate_tile(
                    tXcX_tile[None, None, None, tile_idx],
                    aligned_size,
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
                cur_tXcX = tXcX[None, None, None, tile_idx]
                for v in cutlass.range(cute.size(tXrX), unroll_full=True):
                    raw = tXrX[v]
                    idx = cutlass.Int32(
                        aligned_prefix + cur_tXcX[v // vec_width][1] + v % vec_width
                    )
                    matched = tXpX_tile[0, v // vec_width, 0]
                    if cutlass.const_expr(shift != 0):
                        matched = matched & self.is_partial_match(raw, pattern, SHIFT=shift)
                    if matched:
                        bin_idx = self.extract_bin(raw, STEP=STEP)
                        if bin_idx < threshold_bin:
                            pos = cute.arch.atomic_add(
                                ptr=(s_found_topk.iterator).llvm_ptr,
                                val=cutlass.Int32(1),
                                sem="relaxed",
                                scope="sys",
                            )
                            s_output[pos] = idx
                        elif bin_idx == threshold_bin:
                            if collect:
                                pos = cute.arch.atomic_add(
                                    ptr=(s_cand_count.iterator).llvm_ptr,
                                    val=cutlass.Int32(1),
                                    sem="relaxed",
                                    scope="sys",
                                )
                                s_cand_vals[pos] = raw
                                s_cand_idx[pos] = idx

            idx = aligned_prefix + aligned_size + tidx
            while idx < length:
                raw = score[row_start + idx]
                if self.is_partial_match(raw, pattern, SHIFT=shift):
                    bin_idx = self.extract_bin(raw, STEP=STEP)
                    if bin_idx < threshold_bin:
                        pos = cute.arch.atomic_add(
                            ptr=(s_found_topk.iterator).llvm_ptr,
                            val=cutlass.Int32(1),
                            sem="relaxed",
                            scope="sys",
                        )
                        s_output[pos] = idx
                    elif bin_idx == threshold_bin:
                        if collect:
                            pos = cute.arch.atomic_add(
                                ptr=(s_cand_count.iterator).llvm_ptr,
                                val=cutlass.Int32(1),
                                sem="relaxed",
                                scope="sys",
                            )
                            s_cand_vals[pos] = raw
                            s_cand_idx[pos] = idx
                idx = idx + cutlass.Int32(num_threads)
            cute.arch.barrier()

            return final_bin_size > cutlass.Int32(cand_cap)

        @cute.jit
        def predicate_tile(self, tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
            tApA = cute.make_rmem_tensor(
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
                    tApA[rest_v, 0, rest_k] = cute.elem_less(
                        tAcA[(0, rest_v), 0, rest_k][1],
                        limit,
                    )
            return tApA

        @cute.jit
        def _fill_oob(self, tXrX: cute.Tensor, tXpX: cute.Tensor, fill_value) -> None:
            tXrX_fill = cute.make_fragment_like(tXrX[(None, 0), None, 0])
            tXrX_fill.fill(fill_value)
            for rest_v in range(tXrX.shape[0][1]):
                for rest_k in range(tXrX.shape[2]):
                    if not tXpX[0, rest_v, rest_k]:
                        cute.autovec_copy(tXrX_fill, tXrX[(None, rest_v), None, rest_k])

        # ------------------------------------------------------------------
        # Fast path: rank sort over <= refine_cutoff candidates, exact ties
        # ------------------------------------------------------------------
        @cute.jit
        def sort_final_candidates(
            self,
            s_output,
            s_cand_vals,
            s_cand_idx,
            s_cand_count,
            s_sort_keys,
            s_found_topk,
            tidx,
        ):
            """Rank each candidate against the others with the exact tie rule.

            Candidate ``j`` beats ``i`` when its ordered key is smaller (larger
            score, bit order, NaN/-0 included) or equal with a smaller column.
            Each candidate is packed once into ``s_sort_keys`` as
            ``ordered_key << 32 | column`` so that rule is one unsigned 64-bit
            compare and the inner loop is a single shared-memory load per pair.
            (Recomputing the key and reading the column per pair measured
            slower: the loop is issue-bound across the 16 warps, not
            latency-bound.)  Cost is
            ``ceil(n / 512) * n`` loads per thread; the caller only uses this
            for ``n <= refine_cutoff`` (``s_sort_keys`` holds that many).
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            top_k = cutlass.const_expr(self.top_k)

            base_idx = s_found_topk[0]
            final_count = s_cand_count[0]
            i = tidx
            while i < final_count:
                key = self.to_ordered(s_cand_vals[i])
                col = cutlass.Uint32(s_cand_idx[i])
                s_sort_keys[i] = (cutlass.Uint64(key) << 32) | cutlass.Uint64(col)
                i = i + cutlass.Int32(num_threads)
            cute.arch.barrier()
            i = tidx
            while i < final_count:
                out_index = cutlass.Int32(0)
                mine = s_sort_keys[i]
                j = cutlass.Int32(0)
                while j < final_count:
                    if s_sort_keys[j] < mine:
                        out_index = out_index + cutlass.Int32(1)
                    j = j + cutlass.Int32(1)
                dst = base_idx + out_index
                if dst < cutlass.Int32(top_k):
                    s_output[dst] = s_cand_idx[i]
                i = i + cutlass.Int32(num_threads)
            cute.arch.barrier()

        # ------------------------------------------------------------------
        # Slow path: dynamic loop of 9-bit radix rounds over the SMEM list
        # ------------------------------------------------------------------
        @cute.jit
        def refine_candidates(
            self,
            s_histogram,
            s_output,
            s_cand_vals,
            s_cand_idx,
            s_cand_count,
            s_threshold_bin,
            s_final_bin_size,
            s_found_topk,
            s_found_threshold,
            s_warp_sums,
            tidx,
            start_digit,
        ):
            """Resolve the collected threshold bin with 512-bin radix rounds.

            ``digit`` runs from ``start_digit`` to 5 (see ``refine_digit_params``)
            in a dynamic loop; each round: zero 512 bins, histogram of the list
            (<= 8 elements/thread, warp-aggregated atomics), one block scan for
            ``need = top_k - found`` (``s_found_threshold`` carries the exclusive
            prefix of the threshold bin), emit the lower bins, and either emit
            the threshold bin whole (exactly ``need - prefix`` elements) or
            compact it in place as the next list. Column digits guarantee
            termination because columns are unique.
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            num_warps = cutlass.const_expr(self.num_warps)
            top_k = cutlass.const_expr(self.top_k)
            slots = cutlass.const_expr(self.candidate_slots)

            stage_vals = cute.make_rmem_tensor(cute.make_layout((slots,)), cutlass.Float32)
            stage_cols = cute.make_rmem_tensor(cute.make_layout((slots,)), cutlass.Int32)
            stage_bins = cute.make_rmem_tensor(cute.make_layout((slots,)), cutlass.Int32)

            # Leading digits that are uniform over the collected keys are skipped
            # (fp16-coarse bins share their sign/exponent bits; tie groups share
            # every value bit).
            src = cutlass.Int32(0)
            count = s_cand_count[0]
            digit = start_digit
            kmin, kmax = self.key_range_of_list(s_histogram, s_cand_vals, count, tidx)
            varying = self.first_varying_digit(kmin, kmax)
            if varying > digit:
                digit = varying
            # Candidate count lives in a double-buffered counter so the next
            # round's list can be compacted without a reset barrier.
            while digit < cutlass.Int32(6):
                use_column, shift, mask = self.refine_digit_params(digit)
                dst = cutlass.Int32(1) - src
                need = cutlass.Int32(top_k) - s_found_topk[0]

                s_histogram[tidx] = cutlass.Int32(0)
                if tidx == 0:
                    s_cand_count[dst] = cutlass.Int32(0)
                cute.arch.barrier()

                for s in cutlass.range_constexpr(slots):
                    i = tidx + cutlass.Int32(s * num_threads)
                    valid = i < count
                    value = s_cand_vals[i]
                    column = s_cand_idx[i]
                    bin_idx = self.refine_digit(value, column, use_column, shift, mask)
                    stage_vals[s] = value
                    stage_cols[s] = column
                    stage_bins[s] = bin_idx
                    self.histogram_add(s_histogram, bin_idx, valid)
                cute.arch.barrier()

                bin_count = s_histogram[tidx]
                inclusive, total = block_prefix_sum_kernel(
                    bin_count,
                    s_warp_sums,
                    tidx,
                    num_threads,
                    num_warps,
                    barrier_id=1,
                    need_total_sum=False,
                )
                prefix = inclusive - bin_count
                if prefix < need and inclusive >= need:
                    s_threshold_bin[0] = tidx
                    s_final_bin_size[0] = bin_count
                    s_found_threshold[0] = prefix
                cute.arch.barrier()

                threshold_bin = s_threshold_bin[0]
                bin_size = s_final_bin_size[0]
                remaining = need - s_found_threshold[0]
                take_all = bin_size == remaining
                partition = (bin_size != remaining) & (bin_size < count)

                for s in cutlass.range_constexpr(slots):
                    i = tidx + cutlass.Int32(s * num_threads)
                    valid = i < count
                    bin_idx = stage_bins[s]
                    column = stage_cols[s]
                    below = valid & (bin_idx < threshold_bin)
                    in_bin = valid & (bin_idx == threshold_bin)
                    emit = below | (in_bin & take_all)
                    pos = self.warp_append(s_found_topk, cutlass.Int32(0), emit)
                    if emit:
                        s_output[pos] = column
                    keep = in_bin & partition
                    pos = self.warp_append(s_cand_count, dst, keep)
                    if keep:
                        s_cand_vals[pos] = stage_vals[s]
                        s_cand_idx[pos] = column
                cute.arch.barrier()

                if partition:
                    count = s_cand_count[dst]
                    src = dst
                if take_all:
                    digit = cutlass.Int32(6)
                else:
                    digit = digit + cutlass.Int32(1)

        # ------------------------------------------------------------------
        # Rare tail: > candidate_capacity bit-identical scores at the cut
        # ------------------------------------------------------------------
        @cute.jit
        def select_tied_in_column_order(
            self,
            score,
            row_start,
            length,
            key,
            s_output,
            s_found_topk,
            s_warp_sums,
            tidx,
        ):
            """Emit the first ``top_k - found`` columns whose ordered key == ``key``."""

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            num_warps = cutlass.const_expr(self.num_warps)
            top_k = cutlass.const_expr(self.top_k)
            vec = cutlass.const_expr(8)

            found = s_found_topk[0]
            need = cutlass.Int32(top_k) - found
            hits = cute.make_rmem_tensor(cute.make_layout((vec,)), cutlass.Boolean)
            running = cutlass.Int32(0)
            tile_start = cutlass.Int32(0)
            while tile_start < length:
                if running < need:
                    local = cutlass.Int32(0)
                    for v in cutlass.range_constexpr(vec):
                        col = tile_start + tidx * cutlass.Int32(vec) + cutlass.Int32(v)
                        hit = cutlass.Boolean(False)
                        if col < length:
                            hit = self.to_ordered(score[row_start + col]) == key
                        hits[v] = hit
                        if hit:
                            local = local + cutlass.Int32(1)
                    inclusive, total = block_prefix_sum_kernel(
                        local,
                        s_warp_sums,
                        tidx,
                        num_threads,
                        num_warps,
                        barrier_id=1,
                        need_total_sum=True,
                    )
                    pos = running + inclusive - local
                    for v in cutlass.range_constexpr(vec):
                        if hits[v]:
                            if pos < need:
                                s_output[found + pos] = (
                                    tile_start + tidx * cutlass.Int32(vec) + cutlass.Int32(v)
                                )
                            pos = pos + cutlass.Int32(1)
                    running = running + total
                    cute.arch.barrier()
                tile_start = tile_start + cutlass.Int32(num_threads * vec)
            if tidx == 0:
                s_found_topk[0] = cutlass.Int32(top_k)
            cute.arch.barrier()

        # ------------------------------------------------------------------
        # Per-row driver shared by the prefill and decode kernels
        # ------------------------------------------------------------------
        @cute.jit
        def select_topk_row(
            self,
            score,
            row_start,
            length,
            dst,
            s_histogram,
            s_output,
            s_cand_vals,
            s_cand_idx,
            s_threshold_bin,
            s_cand_count,
            s_final_bin_size,
            s_found_topk,
            s_found_threshold,
            s_pattern,
            s_warp_sums,
            s_sort_keys,
            tidx,
        ):
            """Select ``top_k`` columns of ``score[row_start:row_start+length]`` into ``dst``."""

            top_k = cutlass.const_expr(self.top_k)
            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            cutoff = cutlass.const_expr(self.refine_cutoff)

            if length <= cutlass.Int32(top_k):
                i = tidx
                while i < length:
                    dst[i] = i
                    i = i + cutlass.Int32(num_threads)
                i = length + tidx
                while i < cutlass.Int32(top_k):
                    dst[i] = cutlass.Int32(-1)
                    i = i + cutlass.Int32(num_threads)
            else:
                if tidx == 0:
                    s_pattern[0] = cutlass.Uint32(0)
                    s_found_topk[0] = cutlass.Int32(0)
                    s_cand_count[0] = cutlass.Int32(0)
                cute.arch.barrier()

                start_digit = cutlass.Int32(0)
                overflow = self.process_step(
                    score,
                    row_start,
                    length,
                    s_histogram,
                    s_output,
                    s_cand_vals,
                    s_cand_idx,
                    s_threshold_bin,
                    s_cand_count,
                    s_final_bin_size,
                    s_found_topk,
                    s_found_threshold,
                    s_pattern,
                    s_warp_sums,
                    tidx,
                    STEP=0,
                )
                if overflow:
                    # The fp16 coarse bin is not a prefix of the fp32 digits:
                    # restart the partition from the whole row at fp32 bits 21+.
                    if tidx == 0:
                        s_found_topk[0] = cutlass.Int32(0)
                    cute.arch.barrier()
                    start_digit = cutlass.Int32(1)
                    overflow = self.process_step(
                        score,
                        row_start,
                        length,
                        s_histogram,
                        s_output,
                        s_cand_vals,
                        s_cand_idx,
                        s_threshold_bin,
                        s_cand_count,
                        s_final_bin_size,
                        s_found_topk,
                        s_found_threshold,
                        s_pattern,
                        s_warp_sums,
                        tidx,
                        STEP=1,
                    )
                if overflow:
                    start_digit = cutlass.Int32(2)
                    overflow = self.process_step(
                        score,
                        row_start,
                        length,
                        s_histogram,
                        s_output,
                        s_cand_vals,
                        s_cand_idx,
                        s_threshold_bin,
                        s_cand_count,
                        s_final_bin_size,
                        s_found_topk,
                        s_found_threshold,
                        s_pattern,
                        s_warp_sums,
                        tidx,
                        STEP=2,
                    )
                if overflow:
                    start_digit = cutlass.Int32(4)
                    overflow = self.process_step(
                        score,
                        row_start,
                        length,
                        s_histogram,
                        s_output,
                        s_cand_vals,
                        s_cand_idx,
                        s_threshold_bin,
                        s_cand_count,
                        s_final_bin_size,
                        s_found_topk,
                        s_found_threshold,
                        s_pattern,
                        s_warp_sums,
                        tidx,
                        STEP=3,
                    )
                if overflow:
                    tied_key = s_pattern[0] | cutlass.Uint32(s_threshold_bin[0] & 0x3FF)
                    self.select_tied_in_column_order(
                        score,
                        row_start,
                        length,
                        tied_key,
                        s_output,
                        s_found_topk,
                        s_warp_sums,
                        tidx,
                    )
                else:
                    candidates = s_cand_count[0]
                    if candidates <= cutlass.Int32(cutoff):
                        self.sort_final_candidates(
                            s_output,
                            s_cand_vals,
                            s_cand_idx,
                            s_cand_count,
                            s_sort_keys,
                            s_found_topk,
                            tidx,
                        )
                    else:
                        # Poison the slots the refine still has to fill so a slot
                        # it never writes surfaces as -1. Slots [0, found) hold
                        # the step-0 emissions and must be kept.
                        i = s_found_topk[0] + tidx
                        while i < cutlass.Int32(top_k):
                            s_output[i] = cutlass.Int32(-1)
                            i = i + cutlass.Int32(num_threads)
                        cute.arch.barrier()
                        self.refine_candidates(
                            s_histogram,
                            s_output,
                            s_cand_vals,
                            s_cand_idx,
                            s_cand_count,
                            s_threshold_bin,
                            s_final_bin_size,
                            s_found_topk,
                            s_found_threshold,
                            s_warp_sums,
                            tidx,
                            start_digit,
                        )

                i = tidx
                while i < cutlass.Int32(top_k):
                    dst[i] = s_output[i]
                    i = i + cutlass.Int32(num_threads)

        @cute.jit
        def allocate_row_smem(self, smem):
            """Allocate the per-row shared-memory working set (order is fixed)."""

            s_histogram = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((self.num_bins,), order=(0,)),
                byte_alignment=128,
            )
            s_output = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((self.top_k,), order=(0,)),
                byte_alignment=128,
            )
            s_cand_vals = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_ordered_layout((self.candidate_capacity,), order=(0,)),
                byte_alignment=128,
            )
            s_cand_idx = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((self.candidate_capacity,), order=(0,)),
                byte_alignment=128,
            )
            s_threshold_bin = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            s_cand_count = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((2,), order=(0,)),
                byte_alignment=128,
            )
            s_final_bin_size = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            s_found_topk = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            s_found_threshold = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            s_pattern = smem.allocate_tensor(
                element_type=cutlass.Uint32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            s_warp_sums = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((self.num_warps,), order=(0,)),
                byte_alignment=128,
            )
            # Packed rank-sort keys (ordered key << 32 | column), one per
            # candidate of a threshold bin that fits ``refine_cutoff``.
            s_sort_keys = smem.allocate_tensor(
                element_type=cutlass.Uint64,
                layout=cute.make_ordered_layout((self.sort_keys_capacity,), order=(0,)),
                byte_alignment=128,
            )
            return (
                s_histogram,
                s_output,
                s_cand_vals,
                s_cand_idx,
                s_threshold_bin,
                s_cand_count,
                s_final_bin_size,
                s_found_topk,
                s_found_threshold,
                s_pattern,
                s_warp_sums,
                s_sort_keys,
            )

        @cute.kernel
        def histogram_topk_prefill_kernel(
            self,
            logits: cute.Tensor,
            row_starts: cute.Tensor,
            row_ends: cute.Tensor,
            output_indices: cute.Tensor,
        ):
            row, _, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            row_start = row_starts[row]
            row_end = row_ends[row]
            length = row_end - row_start
            score = logits[row, None]
            dst = output_indices[row, None]

            smem = SmemAllocator()
            (
                s_histogram,
                s_output,
                s_cand_vals,
                s_cand_idx,
                s_threshold_bin,
                s_cand_count,
                s_final_bin_size,
                s_found_topk,
                s_found_threshold,
                s_pattern,
                s_warp_sums,
                s_sort_keys,
            ) = self.allocate_row_smem(smem)

            self.select_topk_row(
                score,
                row_start,
                length,
                dst,
                s_histogram,
                s_output,
                s_cand_vals,
                s_cand_idx,
                s_threshold_bin,
                s_cand_count,
                s_final_bin_size,
                s_found_topk,
                s_found_threshold,
                s_pattern,
                s_warp_sums,
                s_sort_keys,
                tidx,
            )

        @cute.jit
        def __call__(
            self,
            logits: cute.Tensor,
            row_starts: cute.Tensor,
            row_ends: cute.Tensor,
            output_indices: cute.Tensor,
            stream: cuda.CUstream,
        ):
            rows = cute.size(logits, mode=[0])
            self.histogram_topk_prefill_kernel(
                logits,
                row_starts,
                row_ends,
                output_indices,
            ).launch(
                grid=(rows, 1, 1),
                block=(self.num_threads_per_cta, 1, 1),
                stream=stream,
            )

    return IndexerTopKPrefillHistogram
