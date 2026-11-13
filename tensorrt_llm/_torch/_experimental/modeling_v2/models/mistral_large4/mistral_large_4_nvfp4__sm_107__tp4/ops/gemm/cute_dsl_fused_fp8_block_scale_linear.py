# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental low-M fused K128 activation quantization and FP8 linear."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fp8_block_scale_linear import (
    cute_dsl_fp8_block_scale_linear as baseline,
)

# The decode target's per-rank FP8 projections at TP8 and TP4: (N, K) -> (Split-K slices, warps
# per CTA). One tactic per (N, K) serves every M in _ROWS, on the assumption that streaming the
# weights dominates at these M. The TP8 rows are the fastest probed M=4 tactics at the kernel's
# default prefetch depth. The TP4-only rows are not probed: each keeps the warps and, as nearly
# as K allows, the K128 slots per CTA of the nearest probed shape with the same N or K.
_NK_TACTICS = {
    (1024, 6144): (24, 8),  # TP8 wq
    (1536, 6144): (16, 4),  # TP8 shared-expert gate/up
    (4096, 6144): (12, 4),  # TP8 and TP4 indexer wq_b
    (6144, 768): (1, 4),  # TP8 shared-expert down
    (6144, 1024): (1, 4),  # TP8 wo
    (6144, 3072): (4, 8),  # TP8 dense down
    (6144, 6144): (12, 4),  # TP8 dense gate/up, TP4 dense down
    (2048, 6144): (16, 4),  # TP4 wq
    (3072, 6144): (12, 4),  # TP4 shared-expert gate/up
    (6144, 1536): (2, 4),  # TP4 shared-expert down
    (6144, 2048): (2, 4),  # TP4 wo
    (12288, 6144): (12, 4),  # TP4 dense gate/up
}
_ROWS = (1, 2, 4, 8, 12, 16)
# (M, N, K) -> (Split-K slices, warps per CTA).
_TACTICS = {(m, n, k): tactic for (n, k), tactic in _NK_TACTICS.items() for m in _ROWS}


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
