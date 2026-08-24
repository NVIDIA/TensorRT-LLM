# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helix softmax-stat output for the Blackwell CuTe DSL MLA kernels."""

from typing import Optional

import cutlass
import cutlass.cute as cute

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from .mla_helpers import LOG2_E


class _MLASoftmaxStatsKernel:
    """Convert base-2 LSE to an equivalent Helix ``(max, sum)`` pair."""

    num_threads = 128

    @cute.jit
    def __call__(
        self,
        lse: cute.Tensor,
        softmax_stats: cute.Tensor,
        cache_seqs: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(lse, softmax_stats, cache_seqs).launch(
            grid=[lse.shape[1], lse.shape[2], 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        lse: cute.Tensor,
        softmax_stats: cute.Tensor,
        cache_seqs: cute.Tensor,
    ) -> None:
        head_idx, _, _ = cute.arch.thread_idx()
        query_idx, batch_idx, _ = cute.arch.block_idx()

        if head_idx < lse.shape[0]:
            if cache_seqs[batch_idx] > 0:
                # Helix only relies on exp(max) * sum. CuTe MLA already
                # materializes log2(partition), so (LSE * ln(2), 1) is an
                # exact equivalent without retaining the online row max/sum.
                softmax_stats[head_idx, query_idx, batch_idx, 0] = (
                    lse[head_idx, query_idx, batch_idx] / LOG2_E
                )
                softmax_stats[head_idx, query_idx, batch_idx, 1] = 1.0
            else:
                softmax_stats[head_idx, query_idx, batch_idx, 0] = -cutlass.Float32.inf
                softmax_stats[head_idx, query_idx, batch_idx, 1] = 0.0


class MLAWithSoftmaxStats:
    """Compile MLA and its Helix stats epilogue into one host launch."""

    def __init__(self, mla) -> None:
        self.mla = mla
        self.stats = _MLASoftmaxStatsKernel()

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        softmax_stats: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        self.mla(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            o,
            lse,
            workspace,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        )
        self.stats(lse, softmax_stats, cache_seqs, stream)
