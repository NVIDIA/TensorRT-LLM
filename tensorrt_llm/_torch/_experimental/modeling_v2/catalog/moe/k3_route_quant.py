# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's top-16 routing and MXFP8 latent quantization at decode size, ``trtllm::k3_route_quant``: the CuTe DSL
form of ``trtllm::kimi_k3_noaux_tc_mxfp8_quant``, the same four outputs bit for bit, for up to 64 tokens."""

from typing import Tuple

import torch

# Importing the op module registers trtllm::k3_route_quant; CuTe DSL is imported on its first call.
from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import (
    op as _k3_route_quant_op,  # noqa: F401
)

__all__ = ["k3_route_quant"]


def k3_route_quant(
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    latent: torch.Tensor,
    routed_scaling_factor: float,
    early_trigger: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(topk_ids, topk_weights, quantized, scales)`` for ``router_logits`` (fp32 ``[M, 896]``) and ``latent`` (bf16
    ``[M, 3584]``), 1 <= M <= 64: the 16 experts with the largest sigmoid + ``e_score_correction_bias`` (int32
    ``[M, 16]``), their unbiased sigmoids renormalized times ``routed_scaling_factor`` (bf16 ``[M, 16]``), and the
    latent as MXFP8 (float8_e4m3fn ``[M, 3584]``, one UE8M0 scale per 32 columns, uint8 ``[M, 112]``).

    ``early_trigger``: let the next kernel launch (as a programmatic dependent) as soon as every CTA has passed its own
    grid-dependency wait, before the outputs are written; for a dependent that waits for this whole grid before
    reading them, as ``trtllm::k3_moe`` does."""
    return torch.ops.trtllm.k3_route_quant(
        router_logits,
        e_score_correction_bias,
        latent,
        routed_scaling_factor,
        early_trigger=early_trigger,
    )
