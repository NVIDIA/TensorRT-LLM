# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental low-M fused K128 activation quantization and FP8 linear."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    cute_dsl_fp8_block_scale_linear as baseline,
)

# (M, N, K) -> (Split-K slices, warps per CTA) for the decode target's per-rank FP8 projections
# at TP8 and TP4, M = 1, 2, 4, 8, 12 and 16. Each cell holds the fastest tactic at the kernel's
# default prefetch depth on VR200. A shape is absent at an M where no tactic beat the MXFP8
# kernel by 3% with both L2-resident and streamed weights, and runs that kernel there: wq_b,
# dense down and 6144x6144 above M=8, and TP4 dense gate/up from M=4.
_TACTICS = {
    # TP8 wq
    (1, 1024, 6144): (12, 8),
    (2, 1024, 6144): (12, 8),
    (4, 1024, 6144): (24, 8),
    (8, 1024, 6144): (24, 4),
    (12, 1024, 6144): (24, 8),
    (16, 1024, 6144): (24, 8),
    # TP8 shared-expert gate/up
    (1, 1536, 6144): (16, 4),
    (2, 1536, 6144): (16, 4),
    (4, 1536, 6144): (24, 4),
    (8, 1536, 6144): (24, 4),
    (12, 1536, 6144): (24, 8),
    (16, 1536, 6144): (24, 8),
    # TP8 and TP4 indexer wq_b
    (1, 4096, 6144): (12, 4),
    (2, 4096, 6144): (12, 4),
    (4, 4096, 6144): (12, 4),
    (8, 4096, 6144): (12, 8),
    # TP8 shared-expert down
    (1, 6144, 768): (1, 4),
    (2, 6144, 768): (1, 4),
    (4, 6144, 768): (1, 4),
    (8, 6144, 768): (1, 4),
    (12, 6144, 768): (6, 8),
    (16, 6144, 768): (6, 4),
    # TP8 wo
    (1, 6144, 1024): (1, 4),
    (2, 6144, 1024): (1, 4),
    (4, 6144, 1024): (1, 4),
    (8, 6144, 1024): (4, 4),
    (12, 6144, 1024): (4, 8),
    (16, 6144, 1024): (4, 8),
    # TP8 dense down
    (1, 6144, 3072): (4, 8),
    (2, 6144, 3072): (4, 8),
    (4, 6144, 3072): (4, 8),
    (8, 6144, 3072): (6, 8),
    # TP8 dense gate/up, TP4 dense down
    (1, 6144, 6144): (12, 4),
    (2, 6144, 6144): (12, 4),
    (4, 6144, 6144): (12, 4),
    (8, 6144, 6144): (12, 8),
    # TP4 wq
    (1, 2048, 6144): (16, 4),
    (2, 2048, 6144): (16, 4),
    (4, 2048, 6144): (16, 4),
    (8, 2048, 6144): (12, 8),
    (12, 2048, 6144): (24, 8),
    (16, 2048, 6144): (24, 8),
    # TP4 shared-expert gate/up
    (1, 3072, 6144): (8, 8),
    (2, 3072, 6144): (8, 8),
    (4, 3072, 6144): (8, 8),
    (8, 3072, 6144): (12, 8),
    (12, 3072, 6144): (24, 8),
    (16, 3072, 6144): (12, 8),
    # TP4 shared-expert down
    (1, 6144, 1536): (4, 4),
    (2, 6144, 1536): (6, 4),
    (4, 6144, 1536): (6, 4),
    (8, 6144, 1536): (6, 4),
    (12, 6144, 1536): (6, 8),
    (16, 6144, 1536): (6, 8),
    # TP4 wo
    (1, 6144, 2048): (8, 4),
    (2, 6144, 2048): (8, 8),
    (4, 6144, 2048): (8, 8),
    (8, 6144, 2048): (4, 8),
    (12, 6144, 2048): (8, 8),
    (16, 6144, 2048): (8, 8),
    # TP4 dense gate/up
    (1, 12288, 6144): (12, 4),
    (2, 12288, 6144): (12, 4),
}


@torch.library.custom_op("trtllm::cute_dsl_fused_fp8_block_scale_linear", mutates_args=())
def _fused(input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    from ._fused_fp8_quant_gemv import fused_fp8_quant_gemv

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
