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

from ..fused_moe.routing import DeepSeekV3MoeRoutingMethod, Deepseekv3RoutingImpl


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
        logits_gemm_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
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
            torch.empty((self.num_experts, self.gating_dim), dtype=weight_dtype, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts, device=device))

        self.softmax_routing_mutation = softmax_routing_mutation
        self.biased_weights_mutation = biased_weights_mutation
        self.omit_renormalize_mutation = omit_renormalize_mutation

        # Fast path: the fused ``noaux_tc`` routing kernel computes exactly K3's
        # production routing contract in one launch -- per-expert sigmoid,
        # ``e_score_correction_bias`` added for *selection* only, top-k weights
        # sampled from the raw sigmoid scores, renormalized by ``sum + 1e-20``,
        # then scaled by ``routed_scaling_factor``. Route through the shared
        # ``Deepseekv3RoutingImpl`` (same op DeepSeek-V3 uses) when the config is
        # eligible and none of the parity-breaking mutation controls are active.
        # The eager path below stays the reference for those controls, for
        # softmax scoring, for ``moe_renormalize=False``, and for grouped /
        # oversized configs the kernel does not support.
        self._routing_impl = Deepseekv3RoutingImpl(
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            is_fused=True,
        )
        # Bounds mirror the n_group == 1 branch of
        # ``Deepseekv3RoutingImpl.noaux_tc`` (num_experts <= 1024, top_k <= 32);
        # staying inside them guarantees the fused kernel branch is taken (never
        # the impl's own PyTorch fallback, whose grouped path differs from K3's).
        self._use_fused_routing = (
            self.moe_router_activation_func == "sigmoid"
            and self.num_expert_group == 1
            and self.moe_renormalize
            and self.top_k > 1
            and self.num_experts <= 1024
            and self.top_k <= 32
            and not softmax_routing_mutation
            and not biased_weights_mutation
            and not omit_renormalize_mutation
        )

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.softmax_routing_mutation:
            return logits.softmax(dim=1)
        if self.moe_router_activation_func == "sigmoid":
            return logits.sigmoid()
        return logits.softmax(dim=1)

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
        if (
            self.softmax_routing_mutation
            or self.biased_weights_mutation
            or self.omit_renormalize_mutation
        ):
            raise ValueError(
                "Kimi K3 routing mutation flags are reference-test controls "
                "and cannot be used by ConfigurableMoE."
            )
        return DeepSeekV3MoeRoutingMethod(
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            is_fused=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(hidden_states)
        # ``compute_logits`` flattens to [num_tokens, num_experts]; derive
        # the token count from it so any input rank works.
        num_tokens = logits.shape[0]

        # ``trtllm::noaux_tc_op`` is a CUDA-only custom op; CPU inputs
        # (reference / parity tests) fall through to the eager path below,
        # which stays the routing-contract reference on every device.
        if self._use_fused_routing and logits.is_cuda:
            # One fused kernel replaces the sigmoid -> (+bias) -> top-k ->
            # gather -> renormalize -> scale chain below. ``noaux_tc`` returns
            # (weights, indices); return the eager dtype contract -- int64
            # indices (as ``torch.topk`` yields) and fp32 weights -- so every
            # downstream consumer is byte-for-byte unaffected by the swap.
            topk_weight, topk_idx = self._routing_impl.noaux_tc(
                logits, self.e_score_correction_bias.float()
            )
            return topk_idx.to(torch.int64), topk_weight.to(torch.float32)

        scores = self._score(logits)
        scores = scores.view(num_tokens, -1)

        # Bias is applied for *selection*, not for the returned weight.
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(num_tokens, self.num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    num_tokens,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(num_tokens, -1)
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
) -> dict[str, tuple[tuple[int, ...], str]]:
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
