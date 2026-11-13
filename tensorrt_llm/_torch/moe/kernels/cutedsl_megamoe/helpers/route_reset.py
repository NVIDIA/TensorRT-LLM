# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Reset combine slots excluded by the current expert routing."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64


class ClearInvalidRoutes:
    def __init__(self, tokens, topk, experts, data_bytes, scale_bytes=0):
        if data_bytes % 4 or scale_bytes % 4:
            raise ValueError("Combine rows must contain whole 32-bit words.")
        self.tokens, self.topk, self.experts = tokens, topk, experts
        self.data_words, self.scale_words = data_bytes // 4, scale_bytes // 4

    @cute.jit
    def __call__(
        self,
        indices: cute.Tensor,
        partials: cute.Tensor,
        scales: Optional[cute.Tensor],
        stream,
    ):
        self._kernel(indices, partials, scales).launch(
            grid=(min(256, (self.tokens * self.topk + 3) // 4), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _kernel(
        self, indices: cute.Tensor, partials: cute.Tensor, scales: Optional[cute.Tensor]
    ):
        lane = cute.arch.thread_idx()[0] % 32
        warp = cute.arch.thread_idx()[0] // 32
        first = cute.arch.block_idx()[0] * 4 + warp
        stride = cute.arch.grid_dim()[0] * 4
        data = cute.recast_ptr(partials.iterator, dtype=Int32)
        for route in cutlass.range(first, indices.shape[0] * self.topk, stride):
            token, slot = route // self.topk, route % self.topk
            expert = indices[token, slot]
            if (expert < 0) | (expert >= self.experts):
                data_base = Int64(token) * (
                    partials.stride[0] * partials.element_type.width // 32
                ) + Int64(slot) * (
                    partials.stride[1] * partials.element_type.width // 32
                )
                for offset in cutlass.range(lane, self.data_words, 32):
                    data[data_base + offset] = Int32(0)
                if cutlass.const_expr(scales is not None):
                    sf = cute.recast_ptr(scales.iterator, dtype=Int32)
                    scale_base = Int64(token) * (
                        scales.stride[0] * scales.element_type.width // 32
                    ) + Int64(slot) * (
                        scales.stride[1] * scales.element_type.width // 32
                    )
                    for offset in cutlass.range(lane, self.scale_words, 32):
                        sf[scale_base + offset] = Int32(0)
