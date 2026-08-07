# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MoE routing module.

Structural mirror of HF ``KimiMoEGate`` (see
``model_config/modeling_kimi.py:710``). K3 routing inherits DeepSeek-V3's
``noaux_tc`` topology but pins the following K3 config choices (see
``configuration_kimi_k3.py``):

* ``moe_router_activation_func = "sigmoid"`` — per-expert sigmoid, not softmax.
* ``e_score_correction_bias`` (per-expert) — bias added *only* for
  top-k *selection*; the returned ``topk_weight`` samples the *raw*
  sigmoid ``scores``, not the bias-adjusted ``scores_for_choice``.
* ``moe_renormalize = True`` — for ``top_k > 1``, divide ``topk_weight``
  by ``sum + 1e-20`` before scaling.
* ``routed_scaling_factor`` — final multiplicative scale.
* ``num_expert_group = 1``, ``topk_group = 1`` — K3 config disables the
  grouped top-k branch used by DeepSeek. The gate still handles the
  grouped branch when a caller flips those config knobs.

Parameter names and shapes match HF ``KimiMoEGate`` so the model's normal
weight loader can use the checkpoint names directly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..fused_moe.routing import DeepSeekV3MoeRoutingMethod


class KimiK3MoEGate(nn.Module):
    """K3 gate weights and routing method for ``ConfigurableMoE``."""

    def __init__(
        self,
        config: Any,
        *,
        logits_gemm_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size

        assert self.moe_router_activation_func in ("sigmoid", "softmax"), (
            "K3 MoE gate supports sigmoid or softmax scoring only"
        )

        # Same parameter shapes / names as HF ``KimiMoEGate``.
        #
        # ``logits_gemm_dtype=torch.bfloat16`` stores the gate weight in
        # bf16 and runs the logits GEMM as a single bf16xbf16 kernel with
        # fp32 accumulate/output (``trtllm::dsv3_router_gemm_op``). The K3
        # checkpoint stores this weight in bf16, so the fp32 master was an
        # exact upcast and bf16 storage is lossless; this removes the
        # per-layer bf16->fp32 input cast + fp32 splitK-reduce that ran
        # inside the decode CUDA graph (~5 us x 92 layers per step).
        # Default ``None`` keeps the legacy fp32 GEMM (module parity tests).
        weight_dtype = logits_gemm_dtype or torch.float32
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=weight_dtype)
        )
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Routing logits ``[num_tokens, num_experts]``, fp32, pre-sigmoid.

        Used when the MoE block is hosted under ``ConfigurableMoE``: the
        post-linear gate math (sigmoid, bias-for-selection, renormalize,
        ``routed_scaling_factor``) runs inside the wrapper's routing method
        per chunk; only the gate GEMM stays here, keeping the checkpoint
        parameter mapping identity.
        """
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        if self.weight.dtype == torch.bfloat16 and hidden_2d.dtype == torch.bfloat16:
            # Single bf16xbf16 -> fp32 GEMM (fp32 accumulate); no input
            # upcast kernel, no fp32 splitK-reduce. K3's 896 experts miss
            # the op's specialized 256-expert kernels and take its cublas
            # path, which is the point here (one fused kernel).
            return torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_2d.contiguous(),
                self.weight.t(),
                bias=None,
                out_dtype=torch.float32,
            )
        return torch.nn.functional.linear(
            hidden_2d.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        """Return the shared DeepSeekV3 router used by ConfigurableMoE."""
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi K3 ConfigurableMoE routing requires sigmoid scores.")
        if not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 ConfigurableMoE routing requires top-k weight renormalization."
            )
        return DeepSeekV3MoeRoutingMethod(
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            is_fused=True,
        )
