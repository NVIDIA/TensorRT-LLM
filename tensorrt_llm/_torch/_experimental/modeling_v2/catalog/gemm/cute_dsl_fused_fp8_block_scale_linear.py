# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental low-M fused K128 activation quantization and FP8 linear."""

import torch

from .._op import Arch, Cell, OpWrapper
from .cute_dsl_fp8_block_scale_linear import cute_dsl_fp8_block_scale_linear as baseline

# (M, N, K) -> (Split-K slices, warps per CTA).
_TACTICS = {
    (4, 1024, 6144): (24, 8),
    (4, 1536, 6144): (16, 8),
    (4, 6144, 768): (1, 8),
    (4, 6144, 1024): (4, 8),
}


@torch.library.custom_op("trtllm::cute_dsl_fused_fp8_block_scale_linear", mutates_args=())
def _fused(input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.fused_fp8_quant_gemv import fused_fp8_quant_gemv

    if input.data_ptr() % 16 or input.stride(0) % 8:
        return baseline(input.contiguous(), weight, weight_scale)
    split_k, warps = _TACTICS[(input.shape[0], weight.shape[0], input.shape[1])]
    return fused_fp8_quant_gemv(input, weight, weight_scale, split_k=split_k, warps=warps)


@_fused.register_fake
def _fake(input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return input.new_empty((input.shape[0], weight.shape[0]))


class CuteDslFusedFp8BlockScaleLinear(OpWrapper):
    """Low-M SM107 GEMM with K128 weight exponents replicated over K32 slots."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(f"SM107 fused FP8 M={m}, N={n}, K={k}", dict(m=m, n=n, k=k)) for m, n, k in _TACTICS
    )

    def __call__(
        self, input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        m, k = input.shape
        if (m, weight.shape[0], k) in _TACTICS and input.stride(-1) == 1 and weight.is_contiguous():
            return _fused(input, weight, weight_scale)
        return baseline(input, weight, weight_scale)

    def reference(
        self, input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        return baseline.reference(input, weight, weight_scale)


cute_dsl_fused_fp8_block_scale_linear = CuteDslFusedFp8BlockScaleLinear()
