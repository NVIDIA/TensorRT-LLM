# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DSpark rolling-window attention for Blackwell.

The DSpark draft attends to a small, fixed rolling window and the current draft
block. This kernel consumes the two sources separately so the hot path never
materializes a concatenated KV tensor or a gather-index tensor.
"""

import cutlass
import cutlass.cute as cute

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


class DSparkAttentionKernel:
    """One-CTA-per-query fused MQA attention with an attention sink."""

    # A single warp owns one (request, query, head). Each lane keeps 16
    # dimensions of a production 512-wide head in registers, avoiding a
    # shared-memory reduction and two CTA barriers per attended token.
    num_threads = cute.arch.WARP_SIZE
    log2_e = 1.4426950408889634

    def __init__(
        self,
        window_size: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scale: float,
    ):
        if head_dim % self.num_threads != 0:
            raise ValueError(
                f"DSparkAttentionKernel head_dim must be divisible by {self.num_threads}; "
                f"got {head_dim}"
            )
        self.window_size = window_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.elements_per_thread = head_dim // self.num_threads
        self.softmax_scale = softmax_scale

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        main_kv: cute.Tensor,
        block_kv: cute.Tensor,
        kv_cache: cute.Tensor,
        slots: cute.Tensor,
        start_pos: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink, output).launch(
            grid=[q.shape[0], self.block_size, self.num_heads],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _exp(self, value: cutlass.Float32):
        return cute.math.exp2(value * self.log2_e, fastmath=True)

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        main_kv: cute.Tensor,
        block_kv: cute.Tensor,
        kv_cache: cute.Tensor,
        slots: cute.Tensor,
        start_pos: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        request_idx, query_idx, head_idx = cute.arch.block_idx()

        slot = cutlass.Int32(slots[request_idx])
        position = cutlass.Int32(start_pos[request_idx])
        write_pos = position % self.window_size

        # Exactly one CTA persists the captured-context KV. Attention CTAs read
        # main_kv directly for write_pos, so no inter-CTA synchronization is
        # needed before the newly written row becomes visible on the next step.
        if query_idx == 0 and head_idx == 0:
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = tidx + item * self.num_threads
                kv_cache[slot, write_pos, dim] = main_kv[request_idx, dim]

        q_values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
        accum = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
        for item in cutlass.range_constexpr(self.elements_per_thread):
            dim = tidx + item * self.num_threads
            q_values[item] = cutlass.Float32(q[request_idx, query_idx, head_idx, dim])
            accum[item] = cutlass.Float32(0.0)

        running_max = -cutlass.Float32.inf
        running_sum = cutlass.Float32(0.0)

        # Physical cache order is irrelevant to attention. Before the window
        # fills, slot c is valid iff c <= start_pos; once full all rows are valid.
        for context_idx in cutlass.range(self.window_size, unroll=1):
            if context_idx <= position:
                partial = cutlass.Float32(0.0)
                values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    dim = tidx + item * self.num_threads
                    value = cutlass.Float32(0.0)
                    if context_idx == write_pos:
                        value = cutlass.Float32(main_kv[request_idx, dim])
                    else:
                        value = cutlass.Float32(kv_cache[slot, context_idx, dim])
                    values[item] = value
                    partial += q_values[item] * value

                score = cute.arch.warp_reduction_sum(partial) * self.softmax_scale
                new_max = cute.arch.fmax(running_max, score)
                old_scale = cutlass.Float32(0.0)
                if running_max != -cutlass.Float32.inf:
                    old_scale = self._exp(running_max - new_max)
                weight = self._exp(score - new_max)
                running_sum = running_sum * old_scale + weight
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    accum[item] = accum[item] * old_scale + weight * values[item]
                running_max = new_max

        # The current draft block is non-causal: every query sees every block KV.
        for block_idx in cutlass.range_constexpr(self.block_size):
            partial = cutlass.Float32(0.0)
            values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(block_kv[request_idx, block_idx, dim])
                values[item] = value
                partial += q_values[item] * value

            score = cute.arch.warp_reduction_sum(partial) * self.softmax_scale
            new_max = cute.arch.fmax(running_max, score)
            old_scale = self._exp(running_max - new_max)
            weight = self._exp(score - new_max)
            running_sum = running_sum * old_scale + weight
            for item in cutlass.range_constexpr(self.elements_per_thread):
                accum[item] = accum[item] * old_scale + weight * values[item]
            running_max = new_max

        sink_weight = self._exp(cutlass.Float32(attn_sink[head_idx]) - running_max)
        inv_denom = cutlass.Float32(1.0) / (running_sum + sink_weight)
        for item in cutlass.range_constexpr(self.elements_per_thread):
            dim = tidx + item * self.num_threads
            output[request_idx, query_idx, head_idx, dim] = (accum[item] * inv_denom).to(
                output.element_type
            )
