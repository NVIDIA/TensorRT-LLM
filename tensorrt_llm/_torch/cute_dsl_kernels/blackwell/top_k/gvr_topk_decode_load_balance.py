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
"""GVR Top-K load-balance variant (Idea C).

Targets large-BS varlen workloads where the single-CTA-per-row mapping in
``gvr_topk_decode.py`` leaves the wall time bound by the longest row
(``T_max``). LJF host-sort can close the "long row dispatched late" gap
but not the floor; this kernel additionally splits each long row across
``cluster_size=4`` CTAs that cooperate via DSMEM, while short rows are
processed one-per-CTA inside the same launch.

Pipeline (single CUDA Graph capture):
  1. ``GvrTopKLBPrepareKernel`` — 1-block kernel that classifies each
     request as long (seq_len > threshold) vs short via a block prefix
     sum, then writes:
       order_row[B_max]     : long request_ids first, then short ones
       counters[2]          : [n_long_req, n_short_req]
  2. ``GvrTopKLBKernel.main_kernel`` — single launch with
     ``cluster_size=4``; each cluster reads counters + order_row to map
     bidx -> (row_id, branch). Long clusters delegate to a cs=4
     ``GvrTopKKernel`` instance's ``run_one_row``; short clusters
     delegate to a cs=1 instance's ``run_one_row`` with 4 CTAs each
     handling a different row.

The two ``GvrTopKKernel`` instances share the per-row body (see
``run_one_row`` extracted in the parent file) so this file does not
duplicate the GVR algorithm itself.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.utils.smem_allocator import SmemAllocator

from .block_scan import block_prefix_sum_kernel
from .gvr_topk_decode import GvrTopKKernel

# Compile-time threshold: seq_len > LONG_THRESHOLD => long row (cs=4 path).
# Derived from cluster cs=4 break-even ≈ T_row > 3.2us ≈ 64K elements on B200.
# See ``gvr-topk-load-balance/design/idea_c_design.md`` §3.
LONG_THRESHOLD_DEFAULT = 64 * 1024


class GvrTopKLBPrepareKernel:
    """Single-block prepare kernel: partition requests into long / short.

    Reads ``seq_lens[B]`` and produces:
      - ``order_row[B_max]`` int32: request_ids, long group first
      - ``counters[2]``     int32: [n_long_req, n_short_req]

    The output ``order_row`` is request-level (not row-level); the main
    kernel expands to row indices via ``req_id * next_n + nn``.
    """

    # Upper bound on B (= batch_size). The grid is fixed (1, 1, 1) and the
    # block has B_max threads, one per request slot. Threads with tidx >= B
    # contribute 0 to the prefix sum (effectively skipped).
    MAX_NUM_REQUESTS = 1024

    def __init__(
        self,
        long_threshold: int = LONG_THRESHOLD_DEFAULT,
        num_threads: int = MAX_NUM_REQUESTS,
    ):
        assert num_threads % 32 == 0
        # block_prefix_sum_kernel needs num_warps > 1 (its cross-warp scan
        # uses one warp scanning across num_warps lanes).
        assert num_threads >= 64
        assert num_threads <= self.MAX_NUM_REQUESTS
        self.long_threshold = long_threshold
        self.num_threads = num_threads
        self.num_warps = num_threads // 32

    @cute.kernel
    def prepare_kernel(
        self,
        seq_lens: cute.Tensor,  # [B] int32
        order_row: cute.Tensor,  # [B_max] int32 output
        counters: cute.Tensor,  # [2] int32 output
        B: cutlass.Int32,  # actual batch size, runtime
    ):
        tidx, _, _ = cute.arch.thread_idx()

        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        long_threshold = cutlass.const_expr(self.long_threshold)

        # SMEM: per-thread is_long flag (used twice — for the prefix scan
        # input and again at scatter time to choose the long vs short
        # slot — so cache it) + warp_sums scratch for block_prefix_sum.
        smem = SmemAllocator()
        s_is_long = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_threads,), order=(0,)),
            byte_alignment=128,
        )
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

        # Step 1: classify request `tidx`. Threads with tidx >= B contribute 0.
        is_long_val = cutlass.Int32(0)
        if tidx < B:
            seq = seq_lens[tidx]
            if seq > cutlass.Int32(long_threshold):
                is_long_val = cutlass.Int32(1)
        s_is_long[tidx] = is_long_val
        cute.arch.barrier()

        # Step 2: block prefix sum over s_is_long -> position within long group.
        # ``need_total_sum=True`` makes warp_sums[num_warps-1] hold n_long_total
        # AFTER the cross-warp scan; capture it from thread 0 before warp_sums
        # gets reused below.
        pos_long, _ = block_prefix_sum_kernel(
            is_long_val,
            s_warp_sums,
            tidx,
            num_threads,
            num_warps,
            barrier_id=1,
            need_total_sum=False,
        )
        # warp_sums[num_warps-1] now contains the inclusive prefix sum of all
        # warp-sums i.e. n_long_total. Publish to all threads via SMEM scalar.
        if tidx == cutlass.Int32(0):
            s_n_long_total[0] = s_warp_sums[num_warps - 1]
        cute.arch.barrier()
        n_long_total = s_n_long_total[0]

        # Step 3: scatter request_id into order_row.
        #   pos_long is *exclusive* if is_long_val == 0 (this thread didn't
        #   contribute) or *inclusive* if is_long_val == 1. Convert to the
        #   exclusive form by subtracting is_long_val so the long group
        #   indices run 0 .. n_long_total - 1.
        # ``dst`` is pre-initialized because CuTe DSL forbids first-defining
        # a Python variable inside a dynamic if branch and using it after.
        dst = cutlass.Int32(0)
        if tidx < B:
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
            counters[1] = B - n_long_total

    @cute.jit
    def __call__(
        self,
        seq_lens: cute.Tensor,
        order_row: cute.Tensor,
        counters: cute.Tensor,
        B: cutlass.Int32,
        stream,
    ):
        self.prepare_kernel(seq_lens, order_row, counters, B).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


class GvrTopKLBKernel:
    """Idea C top-level kernel: prepare + heterogeneous main launch.

    Holds two ``GvrTopKKernel`` instances with different compile-time
    ``cluster_size`` to reuse the per-row body (``run_one_row``):
      - ``_cluster_kernel`` (cs=4) drives the long-row branch
      - ``_single_kernel``  (cs=1) drives the short-row branch

    Main kernel launches once with ``cluster_size=4`` (the hardware
    cluster shape); each cluster is homogeneous in branch choice (all 4
    CTAs share the same ``cluster_id`` and therefore take the same
    if/else path), which keeps any cluster sync in the long branch
    deadlock-free.
    """

    CLUSTER_SIZE = 4

    def __init__(
        self,
        dtype,
        top_k: int,
        next_n: int,
        num_threads: int,
        compress_ratio: int = 1,
        return_output_values: bool = False,
        long_threshold: int = LONG_THRESHOLD_DEFAULT,
        max_B: int = 1024,
        # Passthrough knobs for the underlying GvrTopKKernel instances.
        enable_unroll_4: bool = True,
        enable_phase3_unroll: bool = False,
        use_constant_hint: bool = False,
        min_blocks_per_mp: int = 3,
        use_256bit_load: bool = True,
        enable_warp_parallel_reduce: Optional[bool] = None,
    ):
        # Two underlying GvrTopKKernel instances. Each gets its own
        # compile-time cluster_size; SMEM / mbarrier layouts differ
        # between the two but the per-row body method is identical.
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
        self._cluster_kernel = GvrTopKKernel(cluster_size=self.CLUSTER_SIZE, **common_kwargs)
        self._single_kernel = GvrTopKKernel(cluster_size=1, **common_kwargs)
        self._prepare = GvrTopKLBPrepareKernel(
            long_threshold=long_threshold,
            num_threads=min(max_B, GvrTopKLBPrepareKernel.MAX_NUM_REQUESTS),
        )

        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        self.num_threads = num_threads
        self.return_output_values = return_output_values
        self.long_threshold = long_threshold
        self.max_B = max_B

        # Grid sizing for the main kernel under CUDA Graph:
        # worst case is all rows long (B_max * next_n long clusters, each
        # one cluster) → CTAs = max_B * next_n * cluster_size. Any cluster
        # beyond ``n_long + ceil(n_short / cs)`` early-exits.
        self._max_grid_clusters = max_B * next_n
        self._max_grid_ctas = self._max_grid_clusters * self.CLUSTER_SIZE

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
        cluster_size = cutlass.const_expr(self.CLUSTER_SIZE)
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

        # Pre-init row_idx because CuTe DSL forbids first-defining a Python
        # variable inside one branch of a dynamic if/else and using it
        # later (see ``spike_cross_instance_jit.py``).
        row_idx = cutlass.Int32(0)

        if cluster_id >= total_clusters:
            # Dead cluster (max-grid padding). All 4 CTAs of this cluster
            # take this branch → no peer waiting on us.
            return

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
            # Short branch: cluster's 4 CTAs each handle 1 short row.
            short_cluster_id = cluster_id - n_long_clusters
            short_row_idx = short_cluster_id * cutlass.Int32(cluster_size) + pos_in_cluster
            if short_row_idx >= n_short_rows:
                # Cluster tail padding. Sibling CTAs in this cluster may
                # have valid rows; that's fine — short-branch CTAs do not
                # touch any cluster sync so a sibling early-exit is safe.
                return
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
        # Step 1: prepare (1 block).
        B = seq_lens.shape[0]
        self._prepare(seq_lens, order_row, counters, cutlass.Int32(B), stream=stream)

        # Step 2: main (fixed max grid, dead-cluster early-exit).
        # Worst case: all rows are long -> n_long_clusters = max_B * next_n
        # consumes the full ``_max_grid_ctas`` budget. Smaller batches see
        # idle padding clusters early-exit on the ``cluster_id >= total``
        # check at kernel entry; dead-CTA overhead is < 1 us on B200
        # (verified via spike microbench).
        self.main_kernel(
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
            order_row,
            counters,
        ).launch(
            grid=(self._max_grid_ctas, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(self.CLUSTER_SIZE, 1, 1),
            stream=stream,
        )


__all__ = ["GvrTopKLBKernel", "GvrTopKLBPrepareKernel", "LONG_THRESHOLD_DEFAULT"]
