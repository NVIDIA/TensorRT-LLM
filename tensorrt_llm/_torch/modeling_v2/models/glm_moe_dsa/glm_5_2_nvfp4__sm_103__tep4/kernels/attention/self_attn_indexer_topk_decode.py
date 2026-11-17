# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL GLM decode indexer top-k.

TRT-LLM's ``indexer_topk_decode`` selects top-k sparse-indexer rows from
decode logits:

    logits         float32 [num_rows, num_columns]
    seq_lens       int32   [num_rows / next_n]
    output_indices int32   [num_rows, top_k]

For speculative decode, each request owns ``next_n`` rows. Row ``r`` scans the
visible prefix
``seq_lens[r // next_n] - next_n + (r % next_n) + 1``. The GLM5.2 steady
decode target is ``num_rows=64``, ``num_columns=8192``, ``next_n=2``, and
``top_k=2048``.

The histogram/threshold/select stage and the final sort remain two logical
stages, but this implementation runs them inside one TopK kernel. It reads the
logits with the row pitch of the preceding MQA logits kernel, so no copy kernel
runs between the two.

The selected index set per row is the exact top-k under the ordering (ordered
fp32 bits descending, column ascending). Threshold bins with more than 512
candidates are resolved by a bounded loop of 9-bit radix rounds in shared
memory instead of the O(n^2) rank sort on large bins; smaller
bins keep the original rank sort with the same tie rule.

Long rows (cluster split, ``_make_decode_cluster_topk_class``): one CTA per row
scans a 160K-column row in ~60 us. Engines wider than ``CLUSTER_MIN_COLUMNS``
also get cluster launchers (4..16 CTAs per row, chosen per call by the row count,
``_CLUSTER_BY_ROWS``); inside them each row picks its path on the device from its
actual length: up to ``_CLUSTER_SPLIT_MIN`` columns cluster rank 0 runs the one-CTA
schedule (the other ranks exit), longer rows take a sampled-threshold single
collect pass across the cluster (~13-18 us at 160K), with the histogram split as
the exact fallback. Same selected set and tie rule as the one-CTA kernel. The two
common paths (a short row's fp16 step, a long row's guess) lead the cluster kernel's
code and the rare ones follow: with cold instruction caches, code deep into the
~300 KB kernel stalls on instruction fetch (``select_topk_row_head``).

Programmatic Dependent Launch (opt-in, ``pdl=True`` on
``compile_indexer_topk_decode``): the launch carries the programmatic stream
serialization attribute so the grid may start while the predecessor (the fp8
paged MQA logits kernel) drains.  Inside the kernel ``griddepcontrol.wait`` is issued
after the ``seq_lens`` read and the shared-memory carve-up and immediately
before the first ``logits`` read / first ``output_indices`` store; ``seq_lens``
is host metadata that predates that kernel, so reading it pre-wait is legal and hides
its latency.  ``griddepcontrol.launch_dependents`` is issued by thread 0 after
a CTA barrier that follows the last global store.  The default (``pdl=False``)
emits neither instruction and launches without the attribute, i.e. the
existing kernel.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.memory import SmemAllocator

from . import self_attn_indexer_topk_histogram as _histogram

_DEFAULT_TOP_K = 2048
_DEFAULT_NEXT_N = 2
RawTopKLauncher = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, cuda.CUstream], object]

# Cluster split (see ``_make_decode_cluster_topk_class``): engines whose logits rows are
# at most this wide keep the one-CTA kernel only.
CLUSTER_MIN_COLUMNS = 16384
# Rows at most this long take the one-CTA select inside the cluster kernel (cluster
# rank 0 alone, the other ranks exit), so short contexts on a long engine run today's
# schedule; longer rows are split across the cluster.
_CLUSTER_SPLIT_MIN = 16384
# Guess path of the cluster kernel: evenly spaced scores sampled per row, and the
# expected number of row columns at or above the sampled threshold (1.5 x top_k: a
# miss -- fewer than top_k or more than the candidate capacity -- is ~3 sigma away
# at 4096 samples, and falls back to the exact histogram split).
_GUESS_SAMPLES = 4096
_GUESS_TARGET = 3072
# (largest row count, cluster size): the first entry whose row count covers the call
# is launched; more rows than the last entry keep the one-CTA kernel. Measured on B300
# (width 172032): rows x cluster above ~96 CTAs costs short rows (all on rank 0)
# +3 us of cluster scheduling, and from 48 rows the one-CTA kernel's row parallelism
# beats the split.
_CLUSTER_BY_ROWS: tuple[tuple[int, int], ...] = ((6, 16), (12, 8), (24, 4))


@dataclass(frozen=True)
class CompiledIndexerTopKDecode:
    """A compiled TopK launcher bound to its serving specialization.

    ``cluster_variants`` holds ``(max_rows, cluster, raw)`` launchers of the cluster
    kernel for wide engines (empty otherwise); ``_raw`` is the one-CTA kernel.
    """

    _raw: RawTopKLauncher
    device_id: int
    logits_dtype: torch.dtype
    max_num_columns: int
    top_k: int
    next_n: int
    pdl: bool = False
    cluster_variants: tuple[tuple[int, int, RawTopKLauncher], ...] = ()

    def launcher_for(self, rows: int) -> tuple[int, RawTopKLauncher]:
        """``(cluster size, launcher)`` for ``rows`` logits rows; cluster 1 is the one-CTA kernel."""
        for max_rows, cluster, raw in self.cluster_variants:
            if rows <= max_rows:
                return cluster, raw
        return 1, self._raw


_COMPILE_CACHE: dict[tuple[int, str, int, int, int, bool, int], CompiledIndexerTopKDecode] = {}
_CLUSTER_COMPILE_CACHE: dict[tuple[int, int, int, bool, int], RawTopKLauncher] = {}


@dsl_user_op
def _ld_peer_u32(smem_addr, rank, *, loc=None, ip=None):
    """32-bit load of cluster CTA ``rank``'s shared memory at the local address ``smem_addr``.

    ``mapa.shared::cluster`` + ``ld.shared::cluster``; the caller orders it with
    cluster barriers (a release arrive by the writer, an acquire wait by the reader).
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(rank).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .u32 a;\nmapa.shared::cluster.u32 a, $1, $2;\n"
            "ld.shared::cluster.u32 $0, [a];\n}",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atom_add_peer_u32(smem_addr, rank, value, *, loc=None, ip=None):
    """Atomic add of ``value`` at cluster CTA ``rank``'s shared address ``smem_addr``; the old value."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(rank).ir_value(loc=loc, ip=ip),
                cutlass.Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .u32 a;\nmapa.shared::cluster.u32 a, $1, $2;\n"
            "atom.shared::cluster.add.u32 $0, [a], $3;\n}",
            "=r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_peer_u32(smem_addr, rank, value, *, loc=None, ip=None):
    """32-bit store to cluster CTA ``rank``'s shared memory at ``smem_addr`` (published by a
    release cluster arrive)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(rank).ir_value(loc=loc, ip=ip),
            cutlass.Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .u32 a;\nmapa.shared::cluster.u32 a, $0, $1;\nst.shared::cluster.u32 [a], $2;\n}",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _make_decode_histogram_topk_class():
    prefill_cls = _histogram._make_prefill_histogram_topk_class()

    class IndexerTopKDecodeHistogram(prefill_cls):
        """Single-CTA GLM decode top-k: histogram fast path + bounded SMEM refine."""

        # Programmatic Dependent Launch.  Set once by the compile entry; it is a
        # compile-time constant inside the kernel and the launch.
        pdl: bool = False

        @cute.kernel
        def histogram_topk_decode_kernel(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            output_indices: cute.Tensor,
        ):
            row, _, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            next_n = cutlass.const_expr(self.next_n)

            req = row // cutlass.Int32(next_n)
            nn = row % cutlass.Int32(next_n)
            seq_len = seq_lens[req]
            length = seq_len - cutlass.Int32(next_n) + nn + cutlass.Int32(1)
            num_columns = cute.size(logits, mode=[1])
            if length > num_columns:
                length = num_columns
            if length < cutlass.Int32(0):
                length = cutlass.Int32(0)

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

            if cutlass.const_expr(self.pdl):
                # The predecessor grid (MQA logits) writes ``logits``.  ``seq_lens``
                # above is host metadata that predates it and the SMEM
                # carve-up is CTA-local, so this is the latest legal point:
                # before the first ``logits`` read and the first ``dst`` store,
                # both inside select_topk_row.
                cute.arch.griddepcontrol_wait()

            self.select_topk_row(
                score,
                cutlass.Int32(0),
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

            if cutlass.const_expr(self.pdl):
                # Every ``dst`` store of this CTA precedes the barrier; the
                # dependent's own wait provides global visibility.
                cute.arch.barrier()
                if tidx == 0:
                    cute.arch.griddepcontrol_launch_dependents()

        @cute.jit
        def __call__(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            output_indices: cute.Tensor,
            stream: cuda.CUstream,
        ):
            rows = cute.size(logits, mode=[0])
            self.histogram_topk_decode_kernel(
                logits,
                seq_lens,
                output_indices,
            ).launch(
                grid=(rows, 1, 1),
                block=(self.num_threads_per_cta, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )

    return IndexerTopKDecodeHistogram


def _make_decode_cluster_topk_class():
    """Cluster-split decode top-k: one cluster of ``cluster`` CTAs per row.

    A row longer than ``split_min`` first takes the guess path
    (``select_topk_row_guess``: sampled threshold, one collect pass, exact resolve on
    rank 0). When the sampled threshold misses (fewer than ``top_k`` or more than
    ``candidate_capacity`` columns collected), the histogram split below runs:

    A long row is cut into ``cluster`` contiguous chunks, one per CTA. Each radix
    step of the one-CTA schedule (fp16 coarse step 0, fp32 steps 1..3) runs on the
    chunks: every CTA builds the step's histogram of its chunk, the cluster sums
    the histograms over DSMEM (each CTA keeps the merged copy), every CTA finds the
    same threshold bin for ``top_k - found`` (found summed the same way), then each
    CTA emits its chunk's columns below the bin into its own ``s_output`` and
    collects the bin's columns into its own candidate list. A step whose global
    threshold bin overflows the candidate capacity continues with the next step,
    exactly as the one-CTA schedule does. Once a bin fits, rank 0 gathers every
    CTA's candidates (the global bin has at most ``candidate_capacity``), resolves
    them with the one-CTA rank sort / radix refine, and every CTA writes its
    emitted columns to a disjoint range of the output row. The selected set is the
    one-CTA kernel's: the same ordering, the same thresholds, the same tie rule.

    Rows no longer than ``split_min`` (and rows at most ``top_k``) are selected by
    rank 0 alone with the one-CTA schedule (``select_topk_row_head``, rerun whole
    when its step 0 overflows); the other ranks exit at once. A row whose fp32 steps
    all overflow (more than ``candidate_capacity`` bit-identical scores at the cut)
    is also left to rank 0's one-CTA schedule.
    """

    prefill_cls = _histogram._make_prefill_histogram_topk_class()

    class IndexerTopKDecodeCluster(prefill_cls):
        """See ``_make_decode_cluster_topk_class``."""

        pdl: bool = False
        cluster: int = 2
        split_min: int = _CLUSTER_SPLIT_MIN

        # --------------------------------------------------------------
        # Short rows: the one-CTA schedule's common case, laid out first
        # --------------------------------------------------------------
        @cute.jit
        def select_topk_row_head(
            self,
            score,
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
            """Step 0 of ``select_topk_row`` (the common case of a short row): rows at most
            ``top_k`` written whole, longer ones left as the fp16 bin's candidate list for
            ``finish_candidates``.

            Returns True (CTA-uniformly) when step 0 overflows the candidate capacity;
            then nothing has been written and the caller reruns ``select_topk_row``. Kept
            apart from the steps 1..3 / tied-scan code, the code both common paths run
            (this, the guess path and their shared tail) leads the kernel: with cold
            instruction caches (a served layer between calls) the one-CTA schedule placed
            ~200 KB into the kernel stalled on instruction fetch, +3.7 us per call.
            """

            top_k = cutlass.const_expr(self.top_k)
            num_threads = cutlass.const_expr(self.num_threads_per_cta)

            overflow = cutlass.Boolean(False)
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
                overflow = self.process_step(
                    score,
                    cutlass.Int32(0),
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
            return overflow

        @cute.jit
        def finish_candidates(
            self,
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
            s_warp_sums,
            s_sort_keys,
            tidx,
        ):
            """The tail both common paths share (one copy of its code): the exact top
            ``top_k - found`` of the SMEM candidate list into ``s_output[found:]`` (rank
            sort, or the radix refine above ``refine_cutoff`` candidates), then
            ``s_output`` to ``dst``. The list is ``select_topk_row_head``'s fp16 bin or
            ``select_topk_row_guess``'s narrowed one (rank 0)."""

            top_k = cutlass.const_expr(self.top_k)
            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            if s_cand_count[0] <= cutlass.Int32(self.refine_cutoff):
                self.sort_final_candidates(
                    s_output, s_cand_vals, s_cand_idx, s_cand_count, s_sort_keys, s_found_topk,
                    tidx,
                )  # fmt: skip
            else:
                # Poison the slots the refine still has to fill (a slot it never
                # writes surfaces as -1); [0, found) holds emitted columns.
                i = s_found_topk[0] + tidx
                while i < cutlass.Int32(top_k):
                    s_output[i] = cutlass.Int32(-1)
                    i = i + cutlass.Int32(num_threads)
                cute.arch.barrier()
                self.refine_candidates(
                    s_histogram, s_output, s_cand_vals, s_cand_idx, s_cand_count,
                    s_threshold_bin, s_final_bin_size, s_found_topk, s_found_threshold,
                    s_warp_sums, tidx, cutlass.Int32(0),
                )  # fmt: skip
            i = tidx
            while i < cutlass.Int32(top_k):
                dst[i] = s_output[i]
                i = i + cutlass.Int32(num_threads)

        # --------------------------------------------------------------
        # One radix step over this CTA's chunk
        # --------------------------------------------------------------
        @cute.jit
        def chunk_pass(
            self,
            score,
            chunk_start,
            chunk_len,
            pattern,
            threshold_bin,
            collect,
            s_histogram,
            s_output,
            s_cand_vals,
            s_cand_idx,
            s_cand_count,
            s_found_topk,
            tidx,
            STEP: cutlass.Constexpr[int],
            EMIT: cutlass.Constexpr[bool],
        ):
            """Histogram (``EMIT=False``) or emit / collect (``EMIT=True``) over the chunk.

            The loads, bins and predicates are ``process_step``'s; emitted and
            collected columns are row columns (``chunk_start`` added).
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            shift = cutlass.const_expr(0 if STEP < 2 else 21 if STEP == 2 else 10)
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

            row_addr = (score.iterator + chunk_start).toint()
            misaligned_bytes = cutlass.Int32(row_addr % cutlass.Int64(32))
            aligned_prefix = cutlass.Int32(0)
            if misaligned_bytes != cutlass.Int32(0):
                aligned_prefix = (cutlass.Int32(32) - misaligned_bytes) // cutlass.Int32(4)
                if aligned_prefix > chunk_len:
                    aligned_prefix = chunk_len
            aligned_size = (
                (chunk_len - aligned_prefix) // cutlass.Int32(vec_width)
            ) * cutlass.Int32(vec_width)
            idX = cute.make_identity_tensor((1, aligned_size))
            row_ptr = cute.make_ptr(
                cutlass.Float32,
                (score.iterator + chunk_start + aligned_prefix).toint(),
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

            # Scalar head (to 32-byte alignment), vector body, scalar tail.
            idx = tidx
            while idx < aligned_prefix:
                raw = score[chunk_start + idx]
                if self.is_partial_match(raw, pattern, SHIFT=shift):
                    self.chunk_visit(
                        raw, chunk_start + idx, threshold_bin, collect, s_histogram,
                        s_output, s_cand_vals, s_cand_idx, s_cand_count, s_found_topk,
                        STEP=STEP, EMIT=EMIT,
                    )  # fmt: skip
                idx = idx + cutlass.Int32(num_threads)
            for tile_idx in range(gX.shape[2]):
                tXpX_tile = self.predicate_tile(tXcX_tile[None, None, None, tile_idx], aligned_size)
                cute.copy(
                    copy_atom,
                    tXgX[None, None, None, tile_idx],
                    tXrX,
                    pred=tXpX_tile[None, None, None],
                )
                self._fill_oob(tXrX, tXpX_tile[None, None, None], -tXrX.element_type.inf)
                cur_tXcX = tXcX[None, None, None, tile_idx]
                for v in cutlass.range(cute.size(tXrX), unroll_full=True):
                    raw = tXrX[v]
                    matched = tXpX_tile[0, v // vec_width, 0]
                    if cutlass.const_expr(shift != 0):
                        matched = matched & self.is_partial_match(raw, pattern, SHIFT=shift)
                    if matched:
                        col = cutlass.Int32(
                            chunk_start
                            + aligned_prefix
                            + cur_tXcX[v // vec_width][1]
                            + v % vec_width
                        )
                        self.chunk_visit(
                            raw, col, threshold_bin, collect, s_histogram, s_output,
                            s_cand_vals, s_cand_idx, s_cand_count, s_found_topk,
                            STEP=STEP, EMIT=EMIT,
                        )  # fmt: skip
            idx = aligned_prefix + aligned_size + tidx
            while idx < chunk_len:
                raw = score[chunk_start + idx]
                if self.is_partial_match(raw, pattern, SHIFT=shift):
                    self.chunk_visit(
                        raw, chunk_start + idx, threshold_bin, collect, s_histogram,
                        s_output, s_cand_vals, s_cand_idx, s_cand_count, s_found_topk,
                        STEP=STEP, EMIT=EMIT,
                    )  # fmt: skip
                idx = idx + cutlass.Int32(num_threads)
            cute.arch.barrier()

        @cute.jit
        def chunk_visit(
            self,
            raw,
            col,
            threshold_bin,
            collect,
            s_histogram,
            s_output,
            s_cand_vals,
            s_cand_idx,
            s_cand_count,
            s_found_topk,
            STEP: cutlass.Constexpr[int],
            EMIT: cutlass.Constexpr[bool],
        ):
            bin_idx = self.extract_bin(raw, STEP=STEP)
            if cutlass.const_expr(not EMIT):
                cute.arch.atomic_add(
                    ptr=(s_histogram.iterator + bin_idx).llvm_ptr,
                    val=cutlass.Int32(1),
                    sem="relaxed",
                    scope="sys",
                )
            else:
                if bin_idx < threshold_bin:
                    pos = cute.arch.atomic_add(
                        ptr=(s_found_topk.iterator).llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="sys",
                    )
                    s_output[pos] = col
                elif bin_idx == threshold_bin:
                    if collect:
                        pos = cute.arch.atomic_add(
                            ptr=(s_cand_count.iterator).llvm_ptr,
                            val=cutlass.Int32(1),
                            sem="relaxed",
                            scope="sys",
                        )
                        s_cand_vals[pos] = raw
                        s_cand_idx[pos] = col

        # --------------------------------------------------------------
        # Guess path: sampled threshold, one collect pass, verified count
        # --------------------------------------------------------------
        @cute.jit
        def collect_at_or_above(
            self, score, chunk_start, chunk_len, tau, s_cand_vals, s_cand_idx, s_cand_count, tidx
        ):
            """Append every chunk column whose ordered key is ``<= tau`` (score at or above
            the guess) to the candidate list; the count keeps counting past the capacity."""

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            cand_cap = cutlass.const_expr(self.candidate_capacity)
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

            row_addr = (score.iterator + chunk_start).toint()
            misaligned_bytes = cutlass.Int32(row_addr % cutlass.Int64(32))
            aligned_prefix = cutlass.Int32(0)
            if misaligned_bytes != cutlass.Int32(0):
                aligned_prefix = (cutlass.Int32(32) - misaligned_bytes) // cutlass.Int32(4)
                if aligned_prefix > chunk_len:
                    aligned_prefix = chunk_len
            aligned_size = (
                (chunk_len - aligned_prefix) // cutlass.Int32(vec_width)
            ) * cutlass.Int32(vec_width)
            idX = cute.make_identity_tensor((1, aligned_size))
            row_ptr = cute.make_ptr(
                cutlass.Float32,
                (score.iterator + chunk_start + aligned_prefix).toint(),
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

            # Scalar head and tail: lanes diverge, so plain atomics.
            idx = tidx
            while idx < aligned_prefix:
                raw = score[chunk_start + idx]
                if self.to_ordered(raw) <= tau:
                    pos = cute.arch.atomic_add(
                        ptr=(s_cand_count.iterator).llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="sys",
                    )
                    if pos < cutlass.Int32(cand_cap):
                        s_cand_vals[pos] = raw
                        s_cand_idx[pos] = chunk_start + idx
                idx = idx + cutlass.Int32(num_threads)
            # Vector body: every lane runs every element slot, so warp-aggregated appends.
            for tile_idx in range(gX.shape[2]):
                tXpX_tile = self.predicate_tile(tXcX_tile[None, None, None, tile_idx], aligned_size)
                cute.copy(
                    copy_atom,
                    tXgX[None, None, None, tile_idx],
                    tXrX,
                    pred=tXpX_tile[None, None, None],
                )
                cur_tXcX = tXcX[None, None, None, tile_idx]
                for v in cutlass.range(cute.size(tXrX), unroll_full=True):
                    raw = tXrX[v]
                    take = tXpX_tile[0, v // vec_width, 0] & (self.to_ordered(raw) <= tau)
                    pos = self.warp_append(s_cand_count, cutlass.Int32(0), take)
                    if take:
                        if pos < cutlass.Int32(cand_cap):
                            s_cand_vals[pos] = raw
                            s_cand_idx[pos] = cutlass.Int32(
                                chunk_start
                                + aligned_prefix
                                + cur_tXcX[v // vec_width][1]
                                + v % vec_width
                            )
            idx = aligned_prefix + aligned_size + tidx
            while idx < chunk_len:
                raw = score[chunk_start + idx]
                if self.to_ordered(raw) <= tau:
                    pos = cute.arch.atomic_add(
                        ptr=(s_cand_count.iterator).llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="sys",
                    )
                    if pos < cutlass.Int32(cand_cap):
                        s_cand_vals[pos] = raw
                        s_cand_idx[pos] = chunk_start + idx
                idx = idx + cutlass.Int32(num_threads)
            cute.arch.barrier()

        @cute.jit
        def resolve_list(
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
            s_sort_keys,
            tidx,
        ):
            """Narrow the SMEM candidate list (``s_cand_count[0]`` entries) toward its exact
            top-k: winners into ``s_output``, the survivors left for ``finish_candidates``.

            Rounds of a 2048-bin equal-width histogram over a key window ``[lo, lo + 2048 <<
            shift)``: the first window is the list's own key range; each round emits the bins
            below the threshold bin for ``top_k - found``, compacts the threshold bin to the
            list front and, while it holds more than ``refine_cutoff`` entries, narrows the
            window to that bin (``shift`` drops by 11 bits, so at most three rounds reach
            single keys). The survivors then take the rank sort (ties to the lower column), or
            the column-digit radix refine above ``refine_cutoff`` bit-identical keys. One window
            over [kmin, kmax] alone is too coarse on served logits: with the k-th score
            negative and positive outliers in the list, the ordered keys span the sign
            boundary and most of the list falls into one bin.
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            num_bins = cutlass.const_expr(self.num_bins)
            top_k = cutlass.const_expr(self.top_k)
            cutoff = cutlass.const_expr(self.refine_cutoff)
            slots = cutlass.const_expr(self.candidate_slots)

            count = s_cand_count[0]
            kmin, kmax = self.key_range_of_list(s_histogram, s_cand_vals, count, tidx)
            span = kmax - kmin
            shift = cutlass.Uint32(0)
            while (span >> shift) >= cutlass.Uint32(num_bins):
                shift = shift + cutlass.Uint32(1)
            lo = kmin
            stage_vals = cute.make_rmem_tensor(cute.make_layout((slots,)), cutlass.Float32)
            stage_cols = cute.make_rmem_tensor(cute.make_layout((slots,)), cutlass.Int32)
            stage_bins = cute.make_rmem_tensor(cute.make_layout((slots,)), cutlass.Int32)
            go = cutlass.Boolean(True)
            while go:
                for s in cutlass.range_constexpr(slots):
                    i = tidx + cutlass.Int32(s * num_threads)
                    value = s_cand_vals[i]
                    stage_vals[s] = value
                    stage_cols[s] = s_cand_idx[i]
                    bin_idx = cutlass.Int32(0)
                    if i < count:
                        bin_idx = cutlass.Int32((self.to_ordered(value) - lo) >> shift)
                    stage_bins[s] = bin_idx
                # Every thread has read the list (and, first round, the key-range scratch
                # inside s_histogram).
                cute.arch.barrier()
                hist_idx = tidx
                while hist_idx < cutlass.Int32(num_bins):
                    s_histogram[hist_idx] = cutlass.Int32(0)
                    hist_idx = hist_idx + cutlass.Int32(num_threads)
                if tidx == 0:
                    s_cand_count[1] = cutlass.Int32(0)
                cute.arch.barrier()
                for s in cutlass.range_constexpr(slots):
                    i = tidx + cutlass.Int32(s * num_threads)
                    self.histogram_add(s_histogram, stage_bins[s], i < count)
                cute.arch.barrier()
                self.find_threshold_bin(
                    s_histogram, s_warp_sums, s_threshold_bin, s_final_bin_size,
                    s_found_threshold, cutlass.Int32(top_k) - s_found_topk[0], tidx,
                )  # fmt: skip
                threshold_bin = s_threshold_bin[0]
                for s in cutlass.range_constexpr(slots):
                    i = tidx + cutlass.Int32(s * num_threads)
                    valid = i < count
                    below = valid & (stage_bins[s] < threshold_bin)
                    pos = self.warp_append(s_found_topk, cutlass.Int32(0), below)
                    if below:
                        s_output[pos] = stage_cols[s]
                    keep = valid & (stage_bins[s] == threshold_bin)
                    pos = self.warp_append(s_cand_count, cutlass.Int32(1), keep)
                    if keep:
                        s_cand_vals[pos] = stage_vals[s]
                        s_cand_idx[pos] = stage_cols[s]
                cute.arch.barrier()
                if tidx == 0:
                    s_cand_count[0] = s_cand_count[1]
                cute.arch.barrier()
                count = s_cand_count[0]
                if (count <= cutlass.Int32(cutoff)) | (shift == cutlass.Uint32(0)):
                    go = cutlass.Boolean(False)
                else:
                    lo = lo + (cutlass.Uint32(threshold_bin) << shift)
                    if shift > cutlass.Uint32(11):
                        shift = shift - cutlass.Uint32(11)
                    else:
                        shift = cutlass.Uint32(0)

        @cute.jit
        def select_topk_row_guess(
            self,
            score,
            length,
            dst,
            rank,
            s_histogram,
            s_output,
            s_cand_vals,
            s_cand_idx,
            s_threshold_bin,
            s_cand_count,
            s_final_bin_size,
            s_found_topk,
            s_found_threshold,
            s_warp_sums,
            s_sort_keys,
            s_total,
            tidx,
        ):
            """One-pass select from a sampled threshold; True (cluster-uniformly) on a miss.

            Every CTA reads the same ``_GUESS_SAMPLES`` evenly spaced scores of the row and
            selects, exactly, the ordered key ``tau`` above which ``_GUESS_TARGET`` columns of
            the whole row are expected (two 11-bit histogram rounds; the low 10 key bits
            are rounded up). Each CTA then collects every column of its chunk with key
            ``<= tau`` into its own list; the peers then push theirs into rank 0's list
            (a range reserved with one DSMEM atomic each). If the cluster collected between
            ``top_k`` and ``candidate_capacity`` columns, they include the row's top-k
            (every column left out scores below every collected one), and rank 0 narrows
            its list (``resolve_list``); the caller finishes it on rank 0
            (``finish_candidates``). Otherwise nothing has been written and the caller runs
            the histogram split.
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            num_bins = cutlass.const_expr(self.num_bins)
            top_k = cutlass.const_expr(self.top_k)
            cand_cap = cutlass.const_expr(self.candidate_capacity)
            ranks = cutlass.const_expr(self.cluster)
            samples = cutlass.const_expr(_GUESS_SAMPLES)
            per_thread = cutlass.const_expr(_GUESS_SAMPLES // 512)
            target = cutlass.const_expr(_GUESS_TARGET)

            # --- sampled threshold (same in every CTA) -----------------------
            keys = cute.make_rmem_tensor(cute.make_layout((per_thread,)), cutlass.Uint32)
            for s in cutlass.range_constexpr(per_thread):
                j = tidx + cutlass.Int32(s * num_threads)
                keys[s] = self.to_ordered(score[j * length // cutlass.Int32(samples)])
            want = (cutlass.Int32(samples * target) + length - cutlass.Int32(1)) // length
            if want > cutlass.Int32(samples):
                want = cutlass.Int32(samples)
            hist_idx = tidx
            while hist_idx < cutlass.Int32(num_bins):
                s_histogram[hist_idx] = cutlass.Int32(0)
                hist_idx = hist_idx + cutlass.Int32(num_threads)
            if tidx == 0:
                s_cand_count[1] = cutlass.Int32(0)
            cute.arch.barrier()
            for s in cutlass.range_constexpr(per_thread):
                cute.arch.atomic_add(
                    ptr=(s_histogram.iterator + cutlass.Int32(keys[s] >> 21)).llvm_ptr,
                    val=cutlass.Int32(1),
                    sem="relaxed",
                    scope="sys",
                )
            cute.arch.barrier()
            self.find_threshold_bin(
                s_histogram, s_warp_sums, s_threshold_bin, s_final_bin_size, s_found_threshold,
                want, tidx,
            )  # fmt: skip
            high = cutlass.Uint32(s_threshold_bin[0])
            hist_idx = tidx
            while hist_idx < cutlass.Int32(num_bins):
                s_histogram[hist_idx] = cutlass.Int32(0)
                hist_idx = hist_idx + cutlass.Int32(num_threads)
            cute.arch.barrier()
            above = cutlass.Int32(0)
            for s in cutlass.range_constexpr(per_thread):
                top = keys[s] >> 21
                if top < high:
                    above = above + cutlass.Int32(1)
                elif top == high:
                    cute.arch.atomic_add(
                        ptr=(
                            s_histogram.iterator + cutlass.Int32((keys[s] >> 10) & 0x7FF)
                        ).llvm_ptr,
                        val=cutlass.Int32(1),
                        sem="relaxed",
                        scope="sys",
                    )
            for offset in cutlass.range_constexpr(5):
                above = above + cute.arch.shuffle_sync_bfly(above, 1 << offset)
            if cute.arch.lane_idx() == cutlass.Int32(0):
                cute.arch.atomic_add(
                    ptr=(s_cand_count.iterator + cutlass.Int32(1)).llvm_ptr,
                    val=above,
                    sem="relaxed",
                    scope="sys",
                )
            cute.arch.barrier()
            self.find_threshold_bin(
                s_histogram, s_warp_sums, s_threshold_bin, s_final_bin_size, s_found_threshold,
                want - s_cand_count[1], tidx,
            )  # fmt: skip
            tau = (high << 21) | (cutlass.Uint32(s_threshold_bin[0]) << 10) | cutlass.Uint32(0x3FF)

            # --- one collect pass over this CTA's chunk -----------------------
            # Rank 0's ``s_total`` is also the cluster's reservation counter into rank 0's
            # list; it is zeroed before the first cluster arrive and peers add to it only
            # after the matching wait.
            chunk = (length + cutlass.Int32(ranks - 1)) // cutlass.Int32(ranks)
            chunk = (chunk + cutlass.Int32(7)) // cutlass.Int32(8) * cutlass.Int32(8)
            chunk_start = rank * chunk
            chunk_len = length - chunk_start
            if chunk_len > chunk:
                chunk_len = chunk
            if chunk_len < cutlass.Int32(0):
                chunk_len = cutlass.Int32(0)
            if tidx == 0:
                s_total[0] = cutlass.Int32(0)
            cute.arch.barrier()
            prims.barrier_cluster_arrive()
            self.collect_at_or_above(
                score, chunk_start, chunk_len, tau, s_cand_vals, s_cand_idx, s_total, tidx
            )
            prims.barrier_cluster_wait()

            # --- peers push their candidates into rank 0's list ---------------
            total_addr = cutlass.Int32(s_total.iterator.toint())
            vals_addr = cutlass.Int32(s_cand_vals.iterator.toint())
            idx_addr = cutlass.Int32(s_cand_idx.iterator.toint())
            if rank != cutlass.Int32(0):
                count = s_total[0]
                if tidx == 0:
                    s_found_threshold[0] = _atom_add_peer_u32(total_addr, cutlass.Int32(0), count)
                cute.arch.barrier()
                base = s_found_threshold[0]
                if count > cutlass.Int32(cand_cap):
                    count = cutlass.Int32(cand_cap)
                i = tidx
                while i < count:
                    pos = base + i
                    if pos < cutlass.Int32(cand_cap):
                        _st_peer_u32(
                            vals_addr + pos * cutlass.Int32(4),
                            cutlass.Int32(0),
                            s_cand_vals[i].bitcast(cutlass.Int32),
                        )
                        _st_peer_u32(
                            idx_addr + pos * cutlass.Int32(4), cutlass.Int32(0), s_cand_idx[i]
                        )
                    i = i + cutlass.Int32(num_threads)
            # Publish the pushes (release) and see them on rank 0 (acquire).
            prims.barrier_cluster_arrive()
            prims.barrier_cluster_wait()
            total = s_total[0]
            if rank != cutlass.Int32(0):
                total = _ld_peer_u32(total_addr, cutlass.Int32(0))
            hit = (total >= cutlass.Int32(top_k)) & (total <= cutlass.Int32(cand_cap))
            # No CTA reads a peer after this point.
            prims.barrier_cluster_arrive_relaxed()
            if hit & (rank == cutlass.Int32(0)):
                i = tidx
                while i < cutlass.Int32(top_k):
                    s_output[i] = cutlass.Int32(-1)
                    i = i + cutlass.Int32(num_threads)
                if tidx == 0:
                    s_cand_count[0] = total
                    s_found_topk[0] = cutlass.Int32(0)
                cute.arch.barrier()
                self.resolve_list(
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
                    s_sort_keys,
                    tidx,
                )
            prims.barrier_cluster_wait()
            return not hit

        # --------------------------------------------------------------
        # The split select of one row
        # --------------------------------------------------------------
        @cute.jit
        def select_topk_row_cluster(
            self,
            score,
            length,
            dst,
            rank,
            s_histogram,
            s_merged,
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
            """Select ``top_k`` columns of ``score[0:length]`` with the whole cluster.

            Returns True (cluster-uniformly) when every step overflowed and rank 0
            must run the one-CTA schedule on the row instead.
            """

            num_threads = cutlass.const_expr(self.num_threads_per_cta)
            num_bins = cutlass.const_expr(self.num_bins)
            top_k = cutlass.const_expr(self.top_k)
            cand_cap = cutlass.const_expr(self.candidate_capacity)
            cutoff = cutlass.const_expr(self.refine_cutoff)
            ranks = cutlass.const_expr(self.cluster)

            # Chunks of a multiple of 8 columns so every chunk but the first
            # starts 32-byte aligned when the row does.
            chunk = (length + cutlass.Int32(ranks - 1)) // cutlass.Int32(ranks)
            chunk = (chunk + cutlass.Int32(7)) // cutlass.Int32(8) * cutlass.Int32(8)
            chunk_start = rank * chunk
            chunk_len = length - chunk_start
            if chunk_len > chunk:
                chunk_len = chunk
            if chunk_len < cutlass.Int32(0):
                chunk_len = cutlass.Int32(0)

            if tidx == 0:
                s_pattern[0] = cutlass.Uint32(0)
                s_found_topk[0] = cutlass.Int32(0)
            cute.arch.barrier()

            hist_addr = cutlass.Int32(s_histogram.iterator.toint())
            found_addr = cutlass.Int32(s_found_topk.iterator.toint())
            go = cutlass.Boolean(True)
            start_digit = cutlass.Int32(0)
            for step in cutlass.range_constexpr(4):
                if go:
                    if cutlass.const_expr(step == 1):
                        # The fp16 coarse bin is not a prefix of the fp32 digits:
                        # restart from the whole row (as the one-CTA schedule).
                        if tidx == 0:
                            s_found_topk[0] = cutlass.Int32(0)
                    if cutlass.const_expr(step == 2):
                        if tidx == 0:
                            s_pattern[0] = cutlass.Uint32(s_threshold_bin[0] & 0x7FF) << 21
                    if cutlass.const_expr(step == 3):
                        if tidx == 0:
                            s_pattern[0] = s_pattern[0] | (
                                cutlass.Uint32(s_threshold_bin[0] & 0x7FF) << 10
                            )
                    if tidx == 0:
                        s_cand_count[0] = cutlass.Int32(0)
                    hist_idx = tidx
                    while hist_idx < cutlass.Int32(num_bins):
                        s_histogram[hist_idx] = cutlass.Int32(0)
                        hist_idx = hist_idx + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    pattern = s_pattern[0]

                    self.chunk_pass(
                        score, chunk_start, chunk_len, pattern, cutlass.Int32(0),
                        cutlass.Boolean(False), s_histogram, s_output, s_cand_vals,
                        s_cand_idx, s_cand_count, s_found_topk, tidx,
                        STEP=step, EMIT=False,
                    )  # fmt: skip

                    # Publish the chunk histograms (and found counts); merge.
                    prims.barrier_cluster_arrive()
                    prims.barrier_cluster_wait()
                    hist_idx = tidx
                    while hist_idx < cutlass.Int32(num_bins):
                        total = cutlass.Int32(0)
                        for peer in cutlass.range_constexpr(ranks):
                            total = total + _ld_peer_u32(
                                hist_addr + hist_idx * cutlass.Int32(4), cutlass.Int32(peer)
                            )
                        s_merged[hist_idx] = total
                        hist_idx = hist_idx + cutlass.Int32(num_threads)
                    found = cutlass.Int32(0)
                    for peer in cutlass.range_constexpr(ranks):
                        found = found + _ld_peer_u32(found_addr, cutlass.Int32(peer))
                    cute.arch.barrier()
                    # No peer reads this CTA's histogram / count after this point.
                    prims.barrier_cluster_arrive_relaxed()
                    prims.barrier_cluster_wait()

                    self.find_threshold_bin(
                        s_merged,
                        s_warp_sums,
                        s_threshold_bin,
                        s_final_bin_size,
                        s_found_threshold,
                        cutlass.Int32(top_k) - found,
                        tidx,
                    )
                    threshold_bin = s_threshold_bin[0]
                    collect = s_final_bin_size[0] <= cutlass.Int32(cand_cap)
                    self.chunk_pass(
                        score, chunk_start, chunk_len, pattern, threshold_bin, collect,
                        s_histogram, s_output, s_cand_vals, s_cand_idx, s_cand_count,
                        s_found_topk, tidx, STEP=step, EMIT=True,
                    )  # fmt: skip
                    if collect:
                        go = cutlass.Boolean(False)
                        start_digit = cutlass.Int32(step if step < 3 else 4)

            tied = go
            if not tied:
                # Publish the emitted lists and candidate lists.
                prims.barrier_cluster_arrive()
                prims.barrier_cluster_wait()
                cand_count_addr = cutlass.Int32(s_cand_count.iterator.toint())
                vals_addr = cutlass.Int32(s_cand_vals.iterator.toint())
                idx_addr = cutlass.Int32(s_cand_idx.iterator.toint())
                total_found = cutlass.Int32(0)
                offset = cutlass.Int32(0)
                for peer in cutlass.range_constexpr(ranks):
                    peer_found = _ld_peer_u32(found_addr, cutlass.Int32(peer))
                    total_found = total_found + peer_found
                    if cutlass.const_expr(peer >= 1):
                        if cutlass.Int32(peer) < rank:
                            offset = offset + peer_found
                own_found = s_found_topk[0]
                if rank != cutlass.Int32(0):
                    # Ranks 1.. fill [0, total_found - found_0) in rank order.
                    i = tidx
                    while i < own_found:
                        dst[offset + i] = s_output[i]
                        i = i + cutlass.Int32(num_threads)
                    prims.barrier_cluster_arrive_relaxed()
                    prims.barrier_cluster_wait()
                else:
                    # Gather the peers' candidates after this CTA's own.
                    count = s_cand_count[0]
                    for peer in cutlass.range_constexpr(1, ranks):
                        peer_count = _ld_peer_u32(cand_count_addr, cutlass.Int32(peer))
                        i = tidx
                        while i < peer_count:
                            s_cand_vals[count + i] = cutlass.Int32(
                                _ld_peer_u32(vals_addr + i * cutlass.Int32(4), cutlass.Int32(peer))
                            ).bitcast(cutlass.Float32)
                            s_cand_idx[count + i] = _ld_peer_u32(
                                idx_addr + i * cutlass.Int32(4), cutlass.Int32(peer)
                            )
                            i = i + cutlass.Int32(num_threads)
                        count = count + peer_count
                    # Own emitted columns move to [total_found - own_found, total_found)
                    # so the candidates resolve into [total_found, top_k).
                    shift_by = total_found - own_found
                    moved = cute.make_rmem_tensor(
                        cute.make_layout((top_k // num_threads,)), cutlass.Int32
                    )
                    for s in cutlass.range_constexpr(top_k // num_threads):
                        i = tidx + cutlass.Int32(s * num_threads)
                        moved[s] = s_output[i]
                    cute.arch.barrier()
                    for s in cutlass.range_constexpr(top_k // num_threads):
                        i = tidx + cutlass.Int32(s * num_threads)
                        if i < own_found:
                            s_output[shift_by + i] = moved[s]
                    if tidx == 0:
                        s_found_topk[0] = total_found
                        s_cand_count[0] = count
                    cute.arch.barrier()
                    # Done reading the peers.
                    prims.barrier_cluster_arrive_relaxed()

                    if count <= cutlass.Int32(cutoff):
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
                        i = total_found + tidx
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
                    i = shift_by + tidx
                    while i < cutlass.Int32(top_k):
                        dst[i] = s_output[i]
                        i = i + cutlass.Int32(num_threads)
                    prims.barrier_cluster_wait()
            return tied

        @cute.kernel
        def histogram_topk_decode_cluster_kernel(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            output_indices: cute.Tensor,
        ):
            bidx, _, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            ranks = cutlass.const_expr(self.cluster)
            next_n = cutlass.const_expr(self.next_n)
            row = bidx // cutlass.Int32(ranks)
            rank = bidx % cutlass.Int32(ranks)

            req = row // cutlass.Int32(next_n)
            nn = row % cutlass.Int32(next_n)
            seq_len = seq_lens[req]
            length = seq_len - cutlass.Int32(next_n) + nn + cutlass.Int32(1)
            num_columns = cute.size(logits, mode=[1])
            if length > num_columns:
                length = num_columns
            if length < cutlass.Int32(0):
                length = cutlass.Int32(0)

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
            # Merged (cluster-wide) histogram; the local one stays readable by peers.
            s_merged = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((self.num_bins,), order=(0,)),
                byte_alignment=128,
            )
            # Guess path: this CTA's collect count; rank 0's is the cluster's.
            s_total = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )

            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()

            # Cluster-uniform: every rank of the cluster sees the same row length. Code
            # order is the point: the two common paths (long rows' guess, short rows'
            # step 0) and their shared tail first, the rare ones (the split, a rerun of
            # the whole one-CTA schedule) after; see ``select_topk_row_head``.
            solo = length <= cutlass.Int32(self.split_min)
            fallback = cutlass.Boolean(False)
            rerun = cutlass.Boolean(False)
            tail = cutlass.Boolean(False)
            if not solo:
                fallback = self.select_topk_row_guess(
                    score,
                    length,
                    dst,
                    rank,
                    s_histogram,
                    s_output,
                    s_cand_vals,
                    s_cand_idx,
                    s_threshold_bin,
                    s_cand_count,
                    s_final_bin_size,
                    s_found_topk,
                    s_found_threshold,
                    s_warp_sums,
                    s_sort_keys,
                    s_total,
                    tidx,
                )
                tail = (not fallback) & (rank == cutlass.Int32(0))
            else:
                if rank == cutlass.Int32(0):
                    rerun = self.select_topk_row_head(
                        score,
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
                    tail = (not rerun) & (length > cutlass.Int32(self.top_k))
            if tail:
                self.finish_candidates(
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
                    s_warp_sums,
                    s_sort_keys,
                    tidx,
                )
            if fallback:
                rerun = self.select_topk_row_cluster(
                    score,
                    length,
                    dst,
                    rank,
                    s_histogram,
                    s_merged,
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
            if rerun:
                if rank == cutlass.Int32(0):
                    self.select_topk_row(
                        score,
                        cutlass.Int32(0),
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

            if cutlass.const_expr(self.pdl):
                cute.arch.barrier()
                if tidx == 0:
                    cute.arch.griddepcontrol_launch_dependents()

        @cute.jit
        def __call__(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            output_indices: cute.Tensor,
            stream: cuda.CUstream,
        ):
            rows = cute.size(logits, mode=[0])
            self.histogram_topk_decode_cluster_kernel(
                logits,
                seq_lens,
                output_indices,
            ).launch(
                grid=(rows * self.cluster, 1, 1),
                block=(self.num_threads_per_cta, 1, 1),
                cluster=(self.cluster, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )

    return IndexerTopKDecodeCluster


def compile_indexer_topk_decode_cluster(
    device_id: int,
    next_n: int,
    cluster: int,
    pdl: bool = False,
    split_min: int = _CLUSTER_SPLIT_MIN,
    top_k: int = _DEFAULT_TOP_K,
) -> RawTopKLauncher:
    """Compile the cluster-split kernel (see ``_make_decode_cluster_topk_class``)."""

    if cluster < 2:
        raise ValueError(f"cluster must be at least 2, got {cluster}")
    if split_min < top_k:
        raise ValueError(f"split_min must be at least top_k={top_k}, got {split_min}")
    key = (device_id, next_n, cluster, bool(pdl), int(split_min))
    if key in _CLUSTER_COMPILE_CACHE:
        return _CLUSTER_COMPILE_CACHE[key]
    kernel_cls = _make_decode_cluster_topk_class()
    topk_kernel = kernel_cls(top_k=top_k)
    topk_kernel.next_n = next_n
    topk_kernel.pdl = bool(pdl)
    topk_kernel.cluster = int(cluster)
    topk_kernel.split_min = int(split_min)
    sym_num_rows = cute.sym_int32()
    fake_logits = make_fake_tensor(
        cutlass.Float32,
        (sym_num_rows, cute.sym_int32()),
        stride=(cute.sym_int64(), 1),
        assumed_align=4,
    )
    fake_seq_lens = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int32(),), assumed_align=4)
    fake_output = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows, top_k),
        stride_order=(1, 0),
        assumed_align=4,
    )
    compiled = cute.compile(
        topk_kernel,
        fake_logits,
        fake_seq_lens,
        fake_output,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    _CLUSTER_COMPILE_CACHE[key] = compiled
    return compiled


def compile_indexer_topk_decode(
    device_id: int = 0,
    logits_dtype: torch.dtype | None = None,
    max_num_columns: int = 8192,
    top_k: int = _DEFAULT_TOP_K,
    next_n: int = _DEFAULT_NEXT_N,
    pdl: bool = False,
    refine_cutoff: int | None = None,
) -> CompiledIndexerTopKDecode:
    """Compile the specialized GLM decode top-k path.

    ``pdl=True`` builds the Programmatic-Dependent-Launch variant (launch
    attribute + ``griddepcontrol`` wait/trigger, see the module docstring).  It
    is only useful when the kernel is launched right after the producer of
    ``logits``; the default variant is unchanged.

    ``refine_cutoff`` overrides the helper's ``_REFINE_CUTOFF`` (largest
    threshold bin the rank sort handles before the radix refine takes over);
    ``None`` keeps the helper default.  It is part of the compile cache key.
    """

    if logits_dtype is None:
        logits_dtype = torch.float32
    if logits_dtype != torch.float32:
        raise TypeError(f"decode top-k expects torch.float32 logits, got {logits_dtype}")
    if max_num_columns <= top_k:
        raise ValueError(
            "compile_indexer_topk_decode expects rows longer than top_k; got "
            f"max_num_columns={max_num_columns}, top_k={top_k}"
        )
    if top_k != _DEFAULT_TOP_K:
        raise ValueError(f"decode top-k expects top_k=2048, got {top_k}")
    if max_num_columns > _histogram.MAX_COLUMNS:
        raise ValueError(
            f"decode top-k separates tied columns below {_histogram.MAX_COLUMNS}; "
            f"got max_num_columns={max_num_columns}"
        )
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if refine_cutoff is None:
        refine_cutoff = _histogram._REFINE_CUTOFF
    if refine_cutoff < 0:
        raise ValueError(f"refine_cutoff must be non-negative, got {refine_cutoff}")

    # The generated function keeps the physical row pitch symbolic.  The typed
    # handle nevertheless binds its warmed logical capacity and all other
    # specialization fields, so a caller cannot pair a 2048-wide kernel with a
    # smaller output or a different speculative-decode grouping.
    key = (
        device_id,
        str(logits_dtype),
        max_num_columns,
        top_k,
        next_n,
        bool(pdl),
        int(refine_cutoff),
    )
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    kernel_cls = _make_decode_histogram_topk_class()
    topk_kernel = kernel_cls(top_k=top_k, refine_cutoff=int(refine_cutoff))
    topk_kernel.next_n = next_n
    topk_kernel.pdl = bool(pdl)

    sym_num_rows = cute.sym_int32()
    sym_num_columns = cute.sym_int32()
    sym_batch = cute.sym_int32()
    sym_logits_stride = cute.sym_int64()
    fake_logits = make_fake_tensor(
        cutlass.Float32,
        (sym_num_rows, sym_num_columns),
        stride=(sym_logits_stride, 1),
        assumed_align=4,
    )
    fake_seq_lens = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=4,
    )
    fake_output = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows, top_k),
        stride_order=(1, 0),
        assumed_align=4,
    )

    compiled = cute.compile(
        topk_kernel,
        fake_logits,
        fake_seq_lens,
        fake_output,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    launcher = CompiledIndexerTopKDecode(
        _raw=compiled,
        device_id=device_id,
        logits_dtype=logits_dtype,
        max_num_columns=max_num_columns,
        top_k=top_k,
        next_n=next_n,
        pdl=bool(pdl),
    )
    _COMPILE_CACHE[key] = launcher
    return launcher


def warmup_indexer_topk_decode(
    max_num_columns: int = 8192,
    top_k: int = _DEFAULT_TOP_K,
    next_n: int = _DEFAULT_NEXT_N,
    device_id: int | None = None,
    pdl: bool = False,
    cluster: bool = True,
    cluster_pdl: bool | None = None,
) -> CompiledIndexerTopKDecode:
    """Compile and launch once before TRT-LLM CUDA graph capture starts.

    With ``cluster`` and ``max_num_columns > CLUSTER_MIN_COLUMNS`` the handle also
    carries the cluster-split launchers (``_CLUSTER_BY_ROWS``), each compiled and
    launched here. ``cluster_pdl`` overrides ``pdl`` for the cluster launchers only.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to warm up indexer_topk_decode")
    if device_id is None:
        device_id = torch.cuda.current_device()
    with torch.cuda.device(device_id):
        compiled = compile_indexer_topk_decode(
            device_id=device_id,
            logits_dtype=torch.float32,
            max_num_columns=max_num_columns,
            top_k=top_k,
            next_n=next_n,
            pdl=pdl,
        )
        if cluster and max_num_columns > CLUSTER_MIN_COLUMNS:
            compiled = replace(
                compiled,
                cluster_variants=tuple(
                    (
                        max_rows,
                        size,
                        compile_indexer_topk_decode_cluster(
                            device_id, next_n, size, pdl=pdl if cluster_pdl is None else cluster_pdl
                        ),
                    )
                    for max_rows, size in _CLUSTER_BY_ROWS
                ),
            )

        # One launch of every launcher: the largest row count that selects it.
        rows_per_launcher = [max(next_n, 1)]
        for max_rows, _, _ in compiled.cluster_variants:
            rows_per_launcher.append(max(next_n, max_rows // next_n * next_n))
        for rows in sorted(set(rows_per_launcher)):
            logits = torch.zeros((rows, max_num_columns), dtype=torch.float32, device="cuda")
            seq_lens = torch.full(
                (rows // next_n,), max_num_columns, dtype=torch.int32, device="cuda"
            )
            output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
            indexer_topk_decode_run(
                logits,
                seq_lens,
                output,
                top_k=top_k,
                next_n=next_n,
                compiled_fn=compiled,
            )
        torch.cuda.synchronize()
    return compiled


def _current_cuda_stream(torch, device: torch.device) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _require_cuda_contiguous(torch, tensor: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def indexer_topk_decode_run(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output_indices: torch.Tensor,
    compiled_fn: CompiledIndexerTopKDecode,
    top_k: int = _DEFAULT_TOP_K,
    next_n: int = _DEFAULT_NEXT_N,
) -> torch.Tensor:
    """Run native CuTe decode top-k for GLM sparse indexer logits."""

    if not isinstance(compiled_fn, CompiledIndexerTopKDecode):
        raise TypeError("compiled_fn must be a CompiledIndexerTopKDecode handle")
    if compiled_fn.top_k != top_k or compiled_fn.next_n != next_n:
        raise ValueError(
            "compiled TopK specialization mismatch: "
            f"handle=(top_k={compiled_fn.top_k}, next_n={compiled_fn.next_n}), "
            f"call=(top_k={top_k}, next_n={next_n})"
        )
    if compiled_fn.logits_dtype != logits.dtype:
        raise TypeError(
            "compiled TopK logits dtype mismatch: "
            f"handle={compiled_fn.logits_dtype}, call={logits.dtype}"
        )
    if compiled_fn.device_id != logits.device.index:
        raise ValueError(
            "compiled TopK device mismatch: "
            f"handle=cuda:{compiled_fn.device_id}, call={logits.device}"
        )
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be {torch.float32}, got {logits.dtype}")
    if not logits.is_cuda:
        raise TypeError("logits must be a CUDA tensor")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if logits.shape[1] <= 0:
        raise ValueError("logits must have a positive logical column count")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if logits.stride(1) != 1 or logits.stride(0) < logits.shape[1]:
        raise ValueError(
            "logits must have unit column stride and a row stride no smaller "
            f"than the logical column count, got shape={tuple(logits.shape)} "
            f"stride={tuple(logits.stride())}"
        )
    _require_cuda_contiguous(torch, seq_lens, "seq_lens", torch.int32)
    _require_cuda_contiguous(torch, output_indices, "output_indices", torch.int32)
    if logits.device != seq_lens.device or logits.device != output_indices.device:
        raise ValueError("logits, seq_lens, and output_indices must share one device")
    if logits.shape[0] % next_n != 0:
        raise ValueError(f"logits rows must be divisible by next_n={next_n}, got {logits.shape[0]}")
    batch = logits.shape[0] // next_n
    if seq_lens.shape != (batch,):
        raise ValueError(f"seq_lens must have shape [{batch}], got {tuple(seq_lens.shape)}")
    if output_indices.shape != (logits.shape[0], top_k):
        raise ValueError(
            f"output_indices must have shape [{logits.shape[0]},{top_k}], "
            f"got {tuple(output_indices.shape)}"
        )
    storage_identities = {
        logits.untyped_storage().data_ptr(),
        seq_lens.untyped_storage().data_ptr(),
        output_indices.untyped_storage().data_ptr(),
    }
    if len(storage_identities) != 3:
        raise ValueError("logits, seq_lens, and output_indices need disjoint storage")
    if logits.shape[0] == 0:
        return output_indices

    if logits.shape[1] > compiled_fn.max_num_columns:
        raise ValueError(
            "logits columns exceed the compiled TopK capacity: "
            f"columns={logits.shape[1]}, capacity={compiled_fn.max_num_columns}"
        )

    _, launcher = compiled_fn.launcher_for(logits.shape[0])
    launcher(
        logits,
        seq_lens,
        output_indices,
        _current_cuda_stream(torch, logits.device),
    )
    return output_indices
