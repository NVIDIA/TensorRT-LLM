# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The general target's fused Kimi K3 routing + MXFP8 input quantization of an attention-DP routed call (up to 64
rows): one C++ launch for the noaux_tc top-16 routing and the latent's MXFP8 quantization."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
#: Largest row count the op takes.
MAX_ROWS = 64


class K31NoauxTcMxfp8Quant(OpWrapper):
    """`torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant`: ``(topk_ids int32 [M, 16], topk_weights bf16 [M, 16],
    x_fp8 [M, 3584], x_sf uint8 [M, 112])`` of FP32 router logits ``[M, 896]``, the FP32 correction bias and the
    BF16 latent ``x [M, 3584]``, M in 1..64: the one-group DeepSeek-V3 top-16 (``sigmoid + bias`` keys, the unbiased
    sigmoid renormalized and scaled) and ``mxfp8_quantize`` with linear UE8M0 scales.

    896 experts, top-16, one expert group and the 3584-wide latent are compile-time constants; the op raises on any
    other shape, dtype or layout (see :func:`serves`)."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(rows=r, scale=1.0))
        for r, why in (
            (1, "DEP decode, one sequence per rank"),
            (8, "DEP decode, 8 rows per rank"),
            (64, "the op's largest row count"),
        )
    )

    def __call__(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        x: torch.Tensor,
        routed_scaling_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
            router_logits, e_score_correction_bias, x, routed_scaling_factor
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        x: torch.Tensor,
        routed_scaling_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The unfused chain: ``noaux_tc_op`` (FP32 weights) cast to BF16, and ``mxfp8_quantize``."""
        weights, ids = torch.ops.trtllm.noaux_tc_op(
            router_logits, e_score_correction_bias, 1, 1, TOP_K, routed_scaling_factor
        )
        xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=512)
        return ids.to(torch.int32), weights.to(torch.bfloat16), xq, xsf.view(x.shape[0], -1)


k31_noaux_tc_mxfp8_quant = K31NoauxTcMxfp8Quant()


def serves(router_logits: torch.Tensor, bias: torch.Tensor, x: torch.Tensor) -> bool:
    """Whether :data:`k31_noaux_tc_mxfp8_quant` takes this call: FP32 contiguous logits ``[M, 896]`` and bias, a BF16
    contiguous ``x [M, 3584]``, M in 1..64."""
    m = router_logits.shape[0] if router_logits.dim() == 2 else -1
    return (
        router_logits.dim() == 2
        and router_logits.shape[1] == NUM_EXPERTS
        and router_logits.dtype == torch.float32
        and router_logits.is_contiguous()
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and x.dim() == 2
        and x.shape == (m, HIDDEN)
        and 0 < m <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def fused_noaux_quant(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    x: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(topk_ids, topk_weights, x_fp8, x_sf)`` of one attention-DP routed call from
    :data:`k31_noaux_tc_mxfp8_quant`, or None where it does not take the call (:func:`serves`)."""
    if not serves(router_logits, bias, x):
        return None
    return k31_noaux_tc_mxfp8_quant(router_logits, bias, x, routed_scaling_factor)
