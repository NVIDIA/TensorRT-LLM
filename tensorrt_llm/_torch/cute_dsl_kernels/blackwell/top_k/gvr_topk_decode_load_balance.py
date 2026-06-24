# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""GVR Top-K hybrid multi-CTA + single-CTA load-balance variant.

Breaks the T_max floor on large-BS varlen workloads by splitting long
rows across ``cluster_size`` CTAs (DSMEM cooperation) and packing
short rows one-per-CTA inside the same launch.

Pipeline (single CUDA Graph capture):
  1. ``GvrTopKLBPrepareKernel`` — 1-block classifier; writes
     ``order_row`` (long request_ids first, short after) and
     ``counters = [n_long_req, n_short_req]``.
  2. ``GvrTopKLBKernel.main_kernel`` — one launch; each cluster reads
     counters + order_row to pick a branch (long: cs CTAs / row via
     ``_cluster_kernel.run_one_row``; short: cs CTAs / cluster, each
     CTA on its own row via ``_single_kernel.run_one_row``).

Per-row body comes from ``GvrTopKKernel.run_one_row`` (parent file).
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.utils.smem_allocator import SmemAllocator

from .block_scan import block_prefix_sum_kernel
from .gvr_topk_decode import GvrTopKKernel


class GvrTopKLBPrepareKernel:
    """Single-block prepare kernel: partition requests into long / short.

    Reads ``seq_lens[batch_size]`` and produces:
      - ``order_row[num_threads]`` int32: request_ids, long group first
      - ``counters[2]``     int32: [n_long_req, n_short_req]

    The output ``order_row`` is request-level (not row-level); the main
    kernel expands to row indices via ``req_id * next_n + nn``.
    """

    def __init__(
        self,
        # Compared in scan-length space (= seq_lens / compress_ratio),
        # not raw seq_lens. Default 64K ≈ B200 cs=4 break-even
        # (T_row > 3.2us). TODO: adaptive once we have more workloads.
        long_threshold: int = 64 * 1024,
        # 1 = DSv3.2, 4 = DSv4. Classifier divides seq_lens by this so
        # the threshold stays in scan-length space across both.
        compress_ratio: int = 1,
        # Upper bound on runtime batch_size; 1 thread per request slot,
        # tidx >= batch_size contributes 0 to the prefix sum. Cap 1024
        # comes from block_prefix_sum_kernel (num_warps <= 32).
        num_threads: int = 1024,
    ):
        # block_prefix_sum_kernel: num_threads % 32 == 0, num_warps > 1
        # and a power of 2 → num_threads ∈ {64, 128, 256, 512, 1024}.
        assert num_threads % 32 == 0, f"num_threads must be a multiple of 32; got {num_threads}"
        assert 64 <= num_threads <= 1024, f"num_threads must be in [64, 1024]; got {num_threads}"
        assert (num_threads & (num_threads - 1)) == 0, (
            f"num_threads must be a power of 2 (so num_warps is a power of "
            f"2 per block_prefix_sum_kernel); got {num_threads}"
        )
        assert compress_ratio in (1, 4), (
            f"compress_ratio must be 1 (DSv3.2) or 4 (DSv4); got {compress_ratio}"
        )
        self.long_threshold = long_threshold
        self.compress_ratio = compress_ratio
        self.num_threads = num_threads
        self.num_warps = num_threads // 32

    @cute.kernel
    def prepare_kernel(
        self,
        seq_lens: cute.Tensor,  # [batch_size] int32
        order_row: cute.Tensor,  # [num_threads] int32 output
        counters: cute.Tensor,  # [2] int32 output
        batch_size: cutlass.Int32,  # actual batch size, runtime
    ):
        tidx, _, _ = cute.arch.thread_idx()

        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        long_threshold = cutlass.const_expr(self.long_threshold)
        compress_ratio = cutlass.const_expr(self.compress_ratio)

        # SMEM: warp_sums scratch for the block prefix sum +
        # broadcast slot for n_long_total. The per-thread is_long flag
        # stays in a register (used by prefix-sum input and step-3
        # branch).
        smem = SmemAllocator()
        s_warp_sums = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=128,
        )
        s_n_long_total = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=64,
        )

        # Step 1: classify request `tidx` (tidx >= batch_size → 0).
        # Divide seq_lens by compress_ratio so the comparison is in
        # scan-length space (where long_threshold is defined).
        is_long_val = cutlass.Int32(0)
        if tidx < batch_size:
            seq = seq_lens[tidx]
            if cutlass.const_expr(compress_ratio != 1):
                seq = seq // cutlass.Int32(compress_ratio)
            if seq > cutlass.Int32(long_threshold):
                is_long_val = cutlass.Int32(1)

        # Step 2: block prefix sum of is_long_val. block_prefix_sum
        # always stores cross-warp scan in warp_sums, so
        # warp_sums[num_warps - 1] = n_long_total — read directly
        # below (need_total_sum=False skips the duplicate return).
        pos_long, _ = block_prefix_sum_kernel(
            is_long_val,
            s_warp_sums,
            tidx,
            num_threads,
            num_warps,
            barrier_id=1,
            need_total_sum=False,
        )
        # Broadcast n_long_total via SMEM so step 3/4 don't re-read
        # warp_sums.
        if tidx == cutlass.Int32(0):
            s_n_long_total[0] = s_warp_sums[num_warps - 1]
        # barrier_id 0 here vs 1 inside block_prefix_sum — distinct IDs
        # let the named-bar hardware pipeline them.
        cute.arch.barrier()
        n_long_total = s_n_long_total[0]

        # Step 3: scatter request_id into order_row. pos_long is
        # inclusive for is_long=1 threads, exclusive for 0; subtract
        # is_long_val to get the exclusive position uniformly.
        # ``dst`` is pre-initialized because CuTe DSL forbids first
        # binding a variable inside a dynamic if branch.
        dst = cutlass.Int32(0)
        if tidx < batch_size:
            excl_pos_long = pos_long - is_long_val
            if is_long_val == cutlass.Int32(1):
                dst = excl_pos_long
            else:
                excl_pos_short = tidx - excl_pos_long
                dst = n_long_total + excl_pos_short
            order_row[dst] = tidx

        # Step 4: scalars.
        if tidx == cutlass.Int32(0):
            counters[0] = n_long_total
            counters[1] = batch_size - n_long_total

    @cute.jit
    def __call__(
        self,
        seq_lens: cute.Tensor,
        order_row: cute.Tensor,
        counters: cute.Tensor,
        batch_size: cutlass.Int32,
        stream,
    ):
        self.prepare_kernel(seq_lens, order_row, counters, batch_size).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


class GvrTopKLBKernel:
    """Hybrid multi-CTA + single-CTA top-level kernel.

    Holds two ``GvrTopKKernel`` instances reusing the same
    ``run_one_row`` body:
      - ``_cluster_kernel`` (cs=``cluster_size``): long-row branch
      - ``_single_kernel``  (cs=1):                short-row branch

    Single launch at hw cluster shape ``cluster_size``. Branch choice
    is per-cluster (all CTAs in a cluster share ``cluster_id``), so
    the long-branch cluster sync is deadlock-free.
    """

    def __init__(
        self,
        dtype,
        top_k: int,
        next_n: int,
        num_threads: int,
        compress_ratio: int = 1,
        return_output_values: bool = False,
        # Long-branch cluster size. cs=4 is B200 break-even
        # (T_row > 3.2us recoups DSMEM sync); cs=2 cheaper sync less
        # parallelism; cs=8 GPC-bound, only profitable on very long rows.
        cluster_size: int = 4,
        # Runtime batch_size upper bound. Grid is fixed at
        # ``max_batch_size * next_n * cluster_size`` for graph capture;
        # surplus clusters early-exit at kernel entry.
        max_batch_size: int = 1024,
        # Passthrough knobs for the underlying GvrTopKKernel instances.
        enable_unroll_4: bool = True,
        enable_phase3_unroll: bool = False,
        use_constant_hint: bool = False,
        min_blocks_per_mp: int = 3,
        use_256bit_load: bool = True,
        enable_warp_parallel_reduce: Optional[bool] = None,
    ):
        assert cluster_size in (2, 4, 8), (
            f"cluster_size must be 2, 4, or 8 (GPC-bound); got {cluster_size}"
        )
        # max_batch_size doubles as GvrTopKLBPrepareKernel.num_threads,
        # so the block_prefix_sum_kernel constraint (pow2 in [64,1024])
        # applies here too.
        assert 64 <= max_batch_size <= 1024 and (max_batch_size & (max_batch_size - 1)) == 0, (
            f"max_batch_size must be a power of 2 in [64, 1024] "
            f"(block_prefix_sum_kernel constraint); got {max_batch_size}"
        )
        assert top_k > 0, f"top_k must be > 0; got {top_k}"
        assert next_n > 0, f"next_n must be > 0; got {next_n}"

        # Two GvrTopKKernel instances with different compile-time cs;
        # SMEM / mbarrier layouts differ, per-row body is identical.
        common_kwargs = dict(
            dtype=dtype,
            top_k=top_k,
            next_n=next_n,
            num_threads=num_threads,
            enable_unroll_4=enable_unroll_4,
            enable_phase3_unroll=enable_phase3_unroll,
            use_constant_hint=use_constant_hint,
            min_blocks_per_mp=min_blocks_per_mp,
            use_256bit_load=use_256bit_load,
            enable_warp_parallel_reduce=enable_warp_parallel_reduce,
            compress_ratio=compress_ratio,
            return_output_values=return_output_values,
        )
        self._cluster_kernel = GvrTopKKernel(cluster_size=cluster_size, **common_kwargs)
        self._single_kernel = GvrTopKKernel(cluster_size=1, **common_kwargs)
        # Prepare is decoupled from main: callers run it once per
        # decode step (seq_lens is layer-invariant). Use
        # GvrTopKLBPrepareKernel directly.

        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        self.num_threads = num_threads
        self.return_output_values = return_output_values
        self.cluster_size = cluster_size
        self.max_batch_size = max_batch_size

        # Grid fixed at worst-case for graph capture; counters gates
        # the live clusters at runtime (dead overhead < 1us on B200).
        self._grid_clusters = max_batch_size * next_n
        self._grid_ctas = self._grid_clusters * cluster_size

    @cute.kernel
    def main_kernel(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seq_lens: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
        order_row: cute.Tensor,  # [B_max] int32 (request_ids; long first)
        counters: cute.Tensor,  # [2] int32 [n_long_req, n_short_req]
    ):
        bidx, _, _ = cute.arch.block_idx()
        cluster_size = cutlass.const_expr(self.cluster_size)
        next_n = cutlass.const_expr(self.next_n)

        cluster_id = bidx // cluster_size
        pos_in_cluster = bidx % cluster_size

        n_long_req = counters[0]
        n_short_req = counters[1]
        n_long_clusters = n_long_req * cutlass.Int32(next_n)
        n_short_rows = n_short_req * cutlass.Int32(next_n)
        # ceil(n_short_rows / cluster_size)
        n_short_clusters = (n_short_rows + cutlass.Int32(cluster_size - 1)) // cutlass.Int32(
            cluster_size
        )
        total_clusters = n_long_clusters + n_short_clusters

        # Pre-init: CuTe DSL forbids first-binding a variable inside a
        # dynamic if branch, and `return` is disallowed in
        # @cute.kernel bodies (hence the ``if alive`` style).
        row_idx = cutlass.Int32(0)

        if cluster_id < total_clusters:
            if cluster_id < n_long_clusters:
                # Long branch: 1 cluster cooperates on 1 long row.
                long_row_idx = cluster_id  # 0 .. n_long_clusters - 1
                if cutlass.const_expr(next_n == 1):
                    req_id = order_row[long_row_idx]
                    row_idx = req_id
                else:
                    req_offset = long_row_idx // cutlass.Int32(next_n)
                    nn = long_row_idx % cutlass.Int32(next_n)
                    req_id = order_row[req_offset]
                    row_idx = req_id * cutlass.Int32(next_n) + nn
                self._cluster_kernel.run_one_row(
                    row_idx,
                    input_data,
                    pre_idx,
                    seq_lens,
                    output_values,
                    output_indices,
                )
            else:
                # Short branch: cluster's cs CTAs each handle 1 row.
                short_cluster_id = cluster_id - n_long_clusters
                short_row_idx = short_cluster_id * cutlass.Int32(cluster_size) + pos_in_cluster
                # Short branch skips cluster sync, so a partial cluster
                # (n_short_rows % cs != 0) is safe.
                if short_row_idx < n_short_rows:
                    if cutlass.const_expr(next_n == 1):
                        req_id = order_row[n_long_req + short_row_idx]
                        row_idx = req_id
                    else:
                        req_offset = short_row_idx // cutlass.Int32(next_n)
                        nn = short_row_idx % cutlass.Int32(next_n)
                        req_id = order_row[n_long_req + req_offset]
                        row_idx = req_id * cutlass.Int32(next_n) + nn
                    self._single_kernel.run_one_row(
                        row_idx,
                        input_data,
                        pre_idx,
                        seq_lens,
                        output_values,
                        output_indices,
                    )

    @cute.jit
    def __call__(
        self,
        input_data: cute.Tensor,
        pre_idx: cute.Tensor,
        seq_lens: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
        order_row: cute.Tensor,
        counters: cute.Tensor,
        stream,
    ):
        """Launch the main kernel.

        ``order_row`` and ``counters`` must already be populated by
        ``GvrTopKLBPrepareKernel`` for the current ``seq_lens``. Run
        prepare once per decode step (seq_lens is layer-invariant)
        and reuse the outputs across all per-layer calls:

            prep(seq_lens, order_row, counters)
            for layer in layers:
                main(logits[layer], ..., order_row, counters)

        Grid is fixed for graph capture; surplus clusters early-exit.
        """
        self.main_kernel(
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
            order_row,
            counters,
        ).launch(
            grid=(self._grid_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(self.cluster_size, 1, 1),
            stream=stream,
        )


__all__ = ["GvrTopKLBKernel", "GvrTopKLBPrepareKernel"]
