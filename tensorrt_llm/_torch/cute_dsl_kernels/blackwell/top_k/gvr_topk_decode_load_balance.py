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
"""GVR Top-K hybrid multi-CTA and single-CTA load-balance variant.

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
       order_row[max_batch_size] : long request_ids first, then short ones
       counters[2]               : [n_long_req, n_short_req]
  2. ``GvrTopKLBKernel.main_kernel`` — single launch with a configurable
     ``cluster_size`` (default 4, also supports 2/8); each cluster reads
     counters + order_row to map bidx -> (row_id, branch). Long clusters
     delegate to a cs=cluster_size ``GvrTopKKernel`` instance's
     ``run_one_row``; short clusters delegate to a cs=1 instance's
     ``run_one_row`` with ``cluster_size`` CTAs each handling a
     different row.

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
        long_threshold: int = 64 * 1024,
        # Upper bound on the runtime batch_size. The grid is fixed
        # (1, 1, 1) and the block has ``num_threads`` lanes, one per
        # request slot; threads with tidx >= batch_size contribute 0 to
        # the prefix sum (effectively skipped). 1024 is the largest power
        # of two supported by ``block_prefix_sum_kernel`` (its cross-warp
        # scan caps num_warps at 32).
        num_threads: int = 1024,
    ):
        # block_prefix_sum_kernel constraints (see block_scan.py:75-79):
        #   - num_threads % 32 == 0           (divisible by warp size)
        #   - num_warps > 1                   (cross-warp scan needs ≥ 2)
        #   - num_warps is a power of 2       (warp-scan iteration count)
        # Combined: num_threads ∈ {64, 128, 256, 512, 1024}.
        assert num_threads % 32 == 0, f"num_threads must be a multiple of 32; got {num_threads}"
        assert 64 <= num_threads <= 1024, f"num_threads must be in [64, 1024]; got {num_threads}"
        assert (num_threads & (num_threads - 1)) == 0, (
            f"num_threads must be a power of 2 (so num_warps is a power of "
            f"2 per block_prefix_sum_kernel); got {num_threads}"
        )
        self.long_threshold = long_threshold
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

        # SMEM: warp_sums scratch for the block prefix sum + single-int
        # scratch to broadcast n_long_total across threads. The per-thread
        # is_long flag is kept purely in a register (``is_long_val`` below)
        # — both consumers (block_prefix_sum input + the step-3 long/short
        # branch) read it directly without going through SMEM.
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

        # Step 1: classify request `tidx`. Threads with tidx >= batch_size
        # contribute 0 (their seq_lens slot is unset / out of range).
        is_long_val = cutlass.Int32(0)
        if tidx < batch_size:
            seq = seq_lens[tidx]
            if seq > cutlass.Int32(long_threshold):
                is_long_val = cutlass.Int32(1)

        # Step 2: block prefix sum over is_long_val -> position within long group.
        # ``block_prefix_sum_kernel`` always writes the cross-warp scan into
        # ``warp_sums`` (its Step 3, regardless of ``need_total_sum``), so
        # ``warp_sums[num_warps - 1]`` holds n_long_total after the call.
        # We read it directly from SMEM below, hence ``need_total_sum=False``
        # (skip the redundant return-value plumbing).
        pos_long, _ = block_prefix_sum_kernel(
            is_long_val,
            s_warp_sums,
            tidx,
            num_threads,
            num_warps,
            barrier_id=1,
            need_total_sum=False,
        )
        # Publish n_long_total (= warp_sums[num_warps-1]) to all threads via
        # an SMEM scalar so the step-3 scatter and step-4 counters write can
        # use it without re-reading warp_sums.
        if tidx == cutlass.Int32(0):
            s_n_long_total[0] = s_warp_sums[num_warps - 1]
        # barrier_id 0 (default) here, vs barrier_id=1 above inside
        # block_prefix_sum_kernel. Distinct IDs is intentional — the two
        # barriers protect independent SMEM regions and the named-bar
        # hardware can pipeline them.
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
    """Hybrid multi-CTA + single-CTA top-level kernel: prepare + main launch.

    Holds two ``GvrTopKKernel`` instances with different compile-time
    ``cluster_size`` to reuse the per-row body (``run_one_row``):
      - ``_cluster_kernel`` (cs=4) drives the long-row branch
      - ``_single_kernel``  (cs=1) drives the short-row branch

    Main kernel launches once with the configured ``cluster_size`` (the
    hardware cluster shape); each cluster is homogeneous in branch choice
    (all CTAs in a cluster share the same ``cluster_id`` and therefore
    take the same if/else path), which keeps any cluster sync in the
    long branch deadlock-free.
    """

    def __init__(
        self,
        dtype,
        top_k: int,
        next_n: int,
        num_threads: int,
        compress_ratio: int = 1,
        return_output_values: bool = False,
        # Cluster size for the long branch. cs=4 is the default
        # break-even pick on B200 (T_row > 3.2us recoups DSMEM sync);
        # cs=2 trades less parallelism for cheaper sync, cs=8 is GPC-
        # constrained and only profitable on very long rows. The same
        # value is used as the hardware cluster shape at launch (and
        # is exposed via ``self.cluster_size`` to the main kernel).
        cluster_size: int = 4,
        # Worst-case batch size this kernel will ever see at runtime.
        # The grid is fixed at ``max_batch_size * next_n * cluster_size``
        # CTAs so CUDA Graph capture sees a single shape; the runtime
        # ``counters`` (from prepare) tells the kernel how many of those
        # clusters are actually alive — the rest early-exit on the
        # ``cluster_id >= total_clusters`` check at kernel entry.
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
        # ``max_batch_size`` is shared with ``GvrTopKLBPrepareKernel`` (it is
        # the prepare kernel's ``num_threads``), so the same
        # block_prefix_sum_kernel constraints apply: power of 2 in
        # [64, 1024]. Validate here so callers that don't go through the
        # prepare wrapper still get a clear error.
        assert 64 <= max_batch_size <= 1024 and (max_batch_size & (max_batch_size - 1)) == 0, (
            f"max_batch_size must be a power of 2 in [64, 1024] "
            f"(block_prefix_sum_kernel constraint); got {max_batch_size}"
        )
        # Sanity-check downstream-shared args up-front so a bogus value
        # surfaces here instead of inside the child ``GvrTopKKernel``
        # construction or the JIT trace.
        assert top_k > 0, f"top_k must be > 0; got {top_k}"
        assert next_n > 0, f"next_n must be > 0; got {next_n}"

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
        self._cluster_kernel = GvrTopKKernel(cluster_size=cluster_size, **common_kwargs)
        self._single_kernel = GvrTopKKernel(cluster_size=1, **common_kwargs)
        # NOTE: prepare kernel is intentionally NOT held here — it is
        # decoupled from the main kernel so callers can run it once per
        # decode step (``seq_lens`` is invariant across layers) and reuse
        # the metadata across all per-layer GVR Top-K invocations. Use
        # ``GvrTopKLBPrepareKernel`` directly.

        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        self.num_threads = num_threads
        self.return_output_values = return_output_values
        self.cluster_size = cluster_size
        self.max_batch_size = max_batch_size

        # Grid is FIXED at the worst-case ``max_batch_size * next_n``
        # long clusters (each consuming ``cluster_size`` CTAs) so CUDA
        # Graph capture sees a single static shape. The runtime
        # ``counters`` (from prepare) gates how many of these clusters
        # actually run — surplus clusters early-exit on the
        # ``cluster_id >= total_clusters`` check at kernel entry; dead-
        # cluster overhead is < 1 us on B200.
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

        # Pre-init row_idx because CuTe DSL forbids first-defining a Python
        # variable inside one branch of a dynamic if/else and using it
        # later. Also: dynamic ``return`` isn't allowed inside @cute.kernel
        # bodies, so the dead-cluster / short-tail filters are written as
        # ``if alive`` blocks with implicit fall-through to the end of the
        # kernel rather than early exits.
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
                # Short branch: cluster's cluster_size CTAs each handle 1 short row.
                short_cluster_id = cluster_id - n_long_clusters
                short_row_idx = short_cluster_id * cutlass.Int32(cluster_size) + pos_in_cluster
                # Short cluster tail (last cluster's trailing CTAs when
                # n_short_rows isn't a multiple of cluster_size). Sibling
                # CTAs in this cluster may have valid rows; that's fine —
                # short-branch CTAs do not touch any cluster sync so the
                # partial cluster is safe.
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
        """Launch the main kernel only.

        ``order_row`` and ``counters`` MUST already be populated by a
        prior call to ``GvrTopKLBPrepareKernel`` for the current
        ``seq_lens``. They are decoupled because ``seq_lens`` is
        invariant across the per-layer GVR Top-K calls within one decode
        step, so the prepare cost (~1-2 us) can be amortized over many
        layers. The caller is expected to:

            prep(seq_lens, order_row, counters)   # once per decode step
            for layer in layers:
                main(logits[layer], pre_idx[layer], ..., order_row, counters)

        Grid is fixed at the worst-case ``max_batch_size * next_n`` long
        clusters (each consuming ``cluster_size`` CTAs) for CUDA Graph
        compatibility. When fewer rows are long at runtime the surplus
        clusters early-exit on the ``cluster_id >= total_clusters``
        check inside the kernel. Dead-CTA overhead is < 1 us on B200.
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
