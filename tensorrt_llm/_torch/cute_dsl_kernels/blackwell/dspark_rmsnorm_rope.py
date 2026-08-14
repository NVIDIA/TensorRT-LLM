# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DSpark RMSNorm and adjacent-pair RoPE for Blackwell."""

import cutlass
import cutlass.cute as cute

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


class DSparkRMSNormRoPEKernel:
    """Apply optional RMSNorm and last-dimension RoPE with one warp per row."""

    num_threads = cute.arch.WARP_SIZE

    def __init__(
        self,
        hidden_dim: int,
        rope_dim: int,
        num_heads: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        inverse_rope: bool,
    ):
        if hidden_dim % self.num_threads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by {self.num_threads}; got {hidden_dim}"
            )
        if rope_dim < 0 or rope_dim > hidden_dim or rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even and in [0, {hidden_dim}]; got {rope_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive; got {num_heads}")
        self.hidden_dim = hidden_dim
        self.rope_dim = rope_dim
        self.nope_dim = hidden_dim - rope_dim
        self.rope_pairs = rope_dim // 2
        self.num_heads = num_heads
        self.eps = eps
        self.apply_weight = apply_weight
        self.apply_rmsnorm = apply_rmsnorm
        self.inverse_rope = inverse_rope
        if self.nope_dim % self.num_threads != 0:
            raise ValueError(
                f"nope_dim must be divisible by {self.num_threads}; got {self.nope_dim}"
            )
        if self.rope_pairs % self.num_threads != 0:
            raise ValueError(
                f"rope pairs must be divisible by {self.num_threads}; got {self.rope_pairs}"
            )
        self.elements_per_thread = hidden_dim // self.num_threads
        self.nope_elements_per_thread = self.nope_dim // self.num_threads
        self.pairs_per_thread = self.rope_pairs // self.num_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(x, weight, freqs, output).launch(
            grid=[x.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        inverse_rms = cutlass.Float32(1.0)
        if cutlass.const_expr(self.apply_rmsnorm):
            sum_sq = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim])
                sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.hidden_dim + self.eps)

        for item in cutlass.range_constexpr(self.nope_elements_per_thread):
            dim = tidx + item * self.num_threads
            value = cutlass.Float32(x[row, dim]) * inverse_rms
            if cutlass.const_expr(self.apply_weight):
                value *= cutlass.Float32(weight[dim])
            output[row, dim] = value.to(output.element_type)

        if cutlass.const_expr(self.rope_pairs > 0):
            freq_row = row // self.num_heads
            for item in cutlass.range_constexpr(self.pairs_per_thread):
                pair = tidx + item * self.num_threads
                real_dim = self.nope_dim + pair * 2
                imag_dim = real_dim + 1
                real = cutlass.Float32(x[row, real_dim]) * inverse_rms
                imag = cutlass.Float32(x[row, imag_dim]) * inverse_rms
                if cutlass.const_expr(self.apply_weight):
                    real *= cutlass.Float32(weight[real_dim])
                    imag *= cutlass.Float32(weight[imag_dim])
                cos = cutlass.Float32(freqs[freq_row, pair, 0])
                sin = cutlass.Float32(freqs[freq_row, pair, 1])
                if cutlass.const_expr(self.inverse_rope):
                    sin = -sin
                output[row, real_dim] = (real * cos - imag * sin).to(output.element_type)
                output[row, imag_dim] = (imag * cos + real * sin).to(output.element_type)
