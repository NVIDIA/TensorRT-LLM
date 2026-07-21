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

Parameter names / shapes match HF ``KimiMoEGate`` so
:func:`copy_hf_moe_gate_weights` is identity name mapping.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import nn


class KimiK3MoEGate(nn.Module):
    """K3 MoE routing — structural mirror of HF ``KimiMoEGate``.

    Positive path reproduces HF ``KimiMoEGate.forward`` at
    ``modeling_kimi.py:747-803`` byte-identically under K3's
    ``sigmoid`` scoring, top-k over the full expert set, raw sigmoid
    weights, renormalization + scaling profile.

    Three mutation flags gate the negative controls required by AC6:

    * ``softmax_routing_mutation`` — softmax over experts instead of
      per-expert sigmoid.
    * ``biased_weights_mutation`` — gather ``topk_weight`` from the
      bias-adjusted scores rather than the raw sigmoid scores.
    * ``omit_renormalize_mutation`` — skip the renormalize step even
      when the config asks for it.
    """

    def __init__(
        self,
        config: Any,
        *,
        softmax_routing_mutation: bool = False,
        biased_weights_mutation: bool = False,
        omit_renormalize_mutation: bool = False,
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
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts))

        self.softmax_routing_mutation = softmax_routing_mutation
        self.biased_weights_mutation = biased_weights_mutation
        self.omit_renormalize_mutation = omit_renormalize_mutation

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.softmax_routing_mutation:
            return logits.softmax(dim=1)
        if self.moe_router_activation_func == "sigmoid":
            return logits.sigmoid()
        return logits.softmax(dim=1)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        logits = torch.nn.functional.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        scores = self._score(logits)
        scores = scores.view(bsz * seq_len, -1)

        # Bias is applied for *selection*, not for the returned weight.
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(bsz * seq_len, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        else:
            tmp_scores = scores_for_choice

        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        # Positive contract: gather from raw ``scores`` (not bias-adjusted).
        weight_source = scores_for_choice if self.biased_weights_mutation else scores
        topk_weight = weight_source.gather(1, topk_idx)

        if self.top_k > 1 and self.moe_renormalize and not self.omit_renormalize_mutation:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


def copy_hf_moe_gate_weights(
    hf: nn.Module,
    k3: KimiK3MoEGate,
):
    """Copy parameters from HF ``KimiMoEGate`` into ``k3``.

    Identity name mapping (``weight`` + ``e_score_correction_bias``).
    Returns a ``{name: (shape, dtype)}`` provenance dict.
    """
    src_params = dict(hf.named_parameters())
    dst_params = dict(k3.named_parameters())
    missing_on_k3 = sorted(set(src_params) - set(dst_params))
    missing_on_hf = sorted(set(dst_params) - set(src_params))
    if missing_on_k3:
        raise KeyError(f"copy_hf_moe_gate_weights: HF params missing on K3: {missing_on_k3}")
    if missing_on_hf:
        raise KeyError(f"copy_hf_moe_gate_weights: K3 params missing on HF: {missing_on_hf}")
    provenance = {}
    for name, src in src_params.items():
        dst = dst_params[name]
        if src.shape != dst.shape:
            raise ValueError(
                f"shape mismatch for {name}: HF {tuple(src.shape)} vs K3 {tuple(dst.shape)}"
            )
        with torch.no_grad():
            dst.data.copy_(src.data.to(dtype=dst.dtype, device=dst.device))
        provenance[name] = (tuple(src.shape), str(src.dtype))
    return provenance
